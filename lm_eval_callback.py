"""
Inline lm-eval-harness callback for HF Trainer.

Fires at the configured global steps, evaluates the live (in-memory) model on
rank 0 only, broadcasts a barrier so other ranks wait, and logs results to
W&B with the current step. Avoids ckpt-save + subprocess overhead.

Notes:
- Pass `trust_remote_code=True` not needed since we wrap the model directly.
- Uses lm_eval.simple_evaluate with HFLM around the live model.
- DDP-safe: only rank 0 runs the eval, others barrier-wait.
"""

from __future__ import annotations

import gc
import logging

import torch
import torch.distributed as dist
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class LMEvalCallback(TrainerCallback):
    def __init__(self, eval_steps, tasks, tokenizer, batch_size: int = 16):
        self.eval_steps = sorted(set(int(s) for s in eval_steps))
        self.tasks = list(tasks)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self._fired = set()

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step not in self.eval_steps or step in self._fired:
            return
        self._fired.add(step)

        # Tell HF Trainer to save the checkpoint right after this step.
        # Trainer writes to {output_dir}/checkpoint-{step}/ via its own logic
        # (handles DDP correctly, respects save_total_limit, atomic).
        control.should_save = True

        model = kwargs.get('model')
        if model is None:
            return
        bare = getattr(model, 'module', model)  # unwrap DDP

        is_main = state.is_world_process_zero
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if is_main:
            try:
                self._run(bare, state, args)
            except Exception as e:
                logger.exception(f'lm-eval at step {step} failed: {e}')

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        return control

    def _run(self, model, state, args):
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        was_training = model.training
        model.eval()
        torch.cuda.empty_cache()
        gc.collect()

        device = next(model.parameters()).device
        lm = HFLM(pretrained=model, tokenizer=self.tokenizer,
                  batch_size=self.batch_size, device=str(device))

        with torch.inference_mode():
            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=self.tasks,
                batch_size=self.batch_size,
                device=str(device),
            )

        # Log to W&B (HF Trainer initialised wandb already if report_to=wandb)
        try:
            import wandb
            if wandb.run is not None:
                flat = {}
                for task, metrics in results['results'].items():
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            flat[f'eval/{task}/{k}'] = v
                flat['train/global_step'] = state.global_step
                wandb.log(flat, step=state.global_step)
        except ImportError:
            pass

        # Print a concise summary line
        summary = ' | '.join(
            f"{t}={(m.get('acc,none') or m.get('acc') or m.get('perplexity,none') or m.get('perplexity') or 'n/a')}"
            for t, m in results['results'].items()
        )
        logger.info(f'[step {state.global_step}] {summary}')

        if was_training:
            model.train()
        torch.cuda.empty_cache()
        gc.collect()
