"""Run lm-eval-harness on the checkpoints saved by train_llm.py.

Hydra-driven, same config as training. Resolves the run from model+data:
    python eval_ckpt.py model=transformer_340m data=slimpajama_15bt
-> output_dir = runs/<run_name>, then evals EVERY step_*.pt in ascending step
order, saves <output_dir>/eval_step_N.json, and logs eval/* into the SAME W&B
run as training (run id embedded in each checkpoint).

Reads from the config: output_dir, eval.tasks, eval.batch_size, eval.device.
Checkpoints are self-contained (embed hf_kwargs + tokenizer + wandb_run_id), so
the model is rebuilt and loaded exactly; per-task isolation keeps one failing
task from killing the rest.
"""
import glob
import json
import os
import re

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa: F401  -- registers fla model_types
import fla_patches  # noqa: F401  -- kata + SDPA shim


def step_of(path):
    m = re.search(r"step_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_model(path, device, hf_kwargs, tokenizer_name, run_id_fallback):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Prefer values embedded in the checkpoint (self-contained); fall back to the
    # hydra config for older checkpoints saved before they were embedded.
    hf_kwargs = ckpt.get("hf_kwargs", hf_kwargs)
    tokenizer_name = ckpt.get("tokenizer", tokenizer_name)
    run_id = ckpt.get("wandb_run_id", run_id_fallback)
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model(**hf_kwargs))
    model.load_state_dict(ckpt["model"])
    model = model.to(device, dtype=torch.bfloat16).eval()
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return (model, tok, int(ckpt.get("step", -1)),
            int(ckpt.get("tokens_seen", 0)), run_id)


@hydra.main(version_base=None, config_path="configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    output_dir = cfg.output_dir
    tasks = list(cfg.eval.tasks)
    batch_size = int(cfg.eval.batch_size)
    device = cfg.eval.get("device", "cuda:0")
    # fallbacks for checkpoints that don't embed these (saved before that change)
    hf_kwargs_cfg = OmegaConf.to_container(cfg.model.hf_kwargs, resolve=True)
    tokenizer_cfg = cfg.data.tokenizer
    run_id_cfg = cfg.wandb.get("run_id", None)  # +wandb.run_id=<id> to override

    paths = sorted(glob.glob(os.path.join(output_dir, "step_*.pt")), key=step_of)
    if not paths:
        raise SystemExit(f"no step_*.pt in {output_dir}")
    print(f"output_dir={output_dir}  tasks={tasks}  batch_size={batch_size}  "
          f"device={device}")
    print(f"evaluating {len(paths)} checkpoints in order: "
          f"{[os.path.basename(p) for p in paths]}")

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    wandb_run = None  # opened lazily from the first checkpoint's embedded run id

    for path in paths:
        model, tok, step, tokens, run_id = load_model(
            path, device, hf_kwargs_cfg, tokenizer_cfg, run_id_cfg)
        print(f"\n=== {os.path.basename(path)}  step={step}  tokens={tokens/1e9:.2f}B ===")

        if wandb_run is None and cfg.wandb.mode != "disabled" and run_id:
            import wandb
            wandb_run = wandb.init(id=str(run_id), resume="must")
            print(f"  logging eval/* into W&B run {run_id}")

        lm = HFLM(pretrained=model, tokenizer=tok,
                  batch_size=batch_size, device=device)

        out_path = os.path.join(output_dir, f"eval_step_{step}.json")
        all_metrics = {}
        if os.path.exists(out_path):  # resume a partial eval
            try:
                all_metrics = json.load(open(out_path))
            except Exception:
                all_metrics = {}

        for task in tasks:
            if task not in all_metrics:
                try:
                    with torch.inference_mode():
                        res = lm_eval.simple_evaluate(
                            model=lm, tasks=[task],
                            batch_size=batch_size, device=device,
                        )
                    all_metrics[task] = res["results"].get(task, {})
                    json.dump(all_metrics, open(out_path, "w"), indent=2, default=str)
                    m = all_metrics[task]
                    head = (m.get("acc,none") or m.get("perplexity,none")
                            or m.get("word_perplexity,none") or "done")
                    print(f"  {task} -> {head}")
                except Exception as e:
                    print(f"  {task} FAILED: {e}")
                    continue
            # log into the training run at this checkpoint's step.
            if wandb_run is not None and task in all_metrics:
                flat = {f"eval/{task}/{k}": v
                        for k, v in all_metrics[task].items()
                        if isinstance(v, (int, float))}
                if flat:
                    wandb_run.log(flat, step=step)

        del model, lm
        torch.cuda.empty_cache()
        print(f"  saved -> {out_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
