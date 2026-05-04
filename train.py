"""
LinAtt trainer entry point. Hydra-driven.

Examples:
    # default = gated_deltanet_200m
    accelerate launch train.py
    # other archs
    accelerate launch train.py model=delta_net_200m
    accelerate launch train.py model=mamba2_200m
    # override anything
    accelerate launch train.py model=delta_net_200m train.lr=4e-4 train.max_steps=20000
"""

from __future__ import annotations

import logging
import os

import hydra
import torch
import fla  # noqa: F401  -- registers fla.* model_types with HF Auto*
import fla_patches  # noqa: F401  -- LinAtt-local fixups over installed fla
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from lm_eval_callback import LMEvalCallback

logger = logging.getLogger(__name__)


def collate(features, pad_id):
    input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    return {'input_ids': input_ids, 'labels': input_ids.clone()}


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # Rank-aware logging: only the main process prints INFO; others stay at WARNING.
    # accelerate launch sets LOCAL_RANK and RANK env vars; on a single-machine
    # setup they coincide. Falls back to 0 if running without distributed launcher.
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
    is_main = (rank == 0)
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        force=True,
    )
    if is_main:
        logger.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # ---- W&B setup (HF Trainer reads these env vars) ----
    os.environ['WANDB_PROJECT'] = cfg.wandb.project
    if cfg.wandb.entity:
        os.environ['WANDB_ENTITY'] = cfg.wandb.entity
    os.environ['WANDB_MODE'] = cfg.wandb.mode

    # ---- Tokenizer ----
    tok = AutoTokenizer.from_pretrained(
        cfg.data.tokenizer, trust_remote_code=True,
        add_bos_token=True, add_eos_token=False,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- Model from config (random init for from-scratch training) ----
    hf_kwargs = OmegaConf.to_container(cfg.model.hf_kwargs, resolve=True)
    hf_cfg = AutoConfig.for_model(**hf_kwargs)
    model = AutoModelForCausalLM.from_config(hf_cfg, dtype=torch.bfloat16)
    if is_main:
        n_train, n_all = model.num_parameters(only_trainable=True), model.num_parameters()
        logger.info(f'Params: trainable={n_train:,} / total={n_all:,} ({n_train / n_all:.2%})')

    # ---- Dataset (must be pre-tokenized via prepare.py) ----
    if not os.path.exists(cfg.data.cache_path):
        raise FileNotFoundError(
            f'Tokenized dataset not found at {cfg.data.cache_path}. '
            f'Run: python prepare.py'
        )
    dataset = load_from_disk(cfg.data.cache_path).shuffle(seed=cfg.seed)
    if is_main:
        logger.info(f'Dataset: {dataset}')

    # ---- LR scheduler kwargs ----
    lr_sched_kwargs = {}
    if cfg.train.lr_scheduler_type == 'cosine_with_min_lr':
        lr_sched_kwargs = {'min_lr_rate': cfg.train.min_lr_rate}

    # ---- Eval steps ----
    # Callback fires lm-eval AND triggers a checkpoint save at each of these.
    eval_steps = [int(cfg.train.max_steps * f) for f in cfg.eval.fractions]

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        seed=cfg.seed,
        per_device_train_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        max_steps=cfg.train.max_steps,
        learning_rate=cfg.train.lr,
        warmup_steps=cfg.train.warmup_steps,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        lr_scheduler_kwargs=lr_sched_kwargs,
        weight_decay=cfg.train.weight_decay,
        adam_beta1=cfg.train.adam_beta1,
        adam_beta2=cfg.train.adam_beta2,
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optim,
        bf16=cfg.train.bf16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        ddp_find_unused_parameters=cfg.train.ddp_find_unused_parameters,
        logging_steps=cfg.train.logging_steps,
        save_strategy='no',  # callback drives saves at eval steps
        save_total_limit=cfg.train.save_total_limit,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        dataloader_pin_memory=cfg.train.dataloader_pin_memory,
        dataloader_persistent_workers=cfg.train.get('dataloader_persistent_workers', False),
        report_to='wandb',
        include_num_input_tokens_seen=True,
    )

    callback = LMEvalCallback(
        eval_steps=eval_steps,
        tasks=list(cfg.eval.tasks),
        tokenizer=tok,
        batch_size=cfg.eval.batch_size,
    )

    trainer = Trainer(
        model=model,
        args=args,
        processing_class=tok,
        train_dataset=dataset,
        data_collator=lambda f: collate(f, tok.pad_token_id),
        callbacks=[callback],
    )

    result = trainer.train()
    trainer.save_model()
    tok.save_pretrained(args.output_dir)
    trainer.log_metrics('train', result.metrics)
    trainer.save_metrics('train', result.metrics)
    trainer.save_state()


if __name__ == '__main__':
    main()
