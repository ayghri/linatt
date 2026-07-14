"""Run lm-eval-harness on the checkpoints saved by train_llm.py.

Hydra-driven, same config as training. The model architecture + tokenizer come
from the config (model=..., data=...); the checkpoint holds only weights.

    python eval_ckpt.py model=transformer_340m data=slimpajama_15bt
-> output_dir = runs/<run_name>, evals EVERY step_*.pt in ascending step order,
   saves <output_dir>/eval_step_N.yaml. Per-task isolation: one failing task
   doesn't kill the rest. Reads: output_dir, eval.tasks, eval.batch_size, eval.device.
"""

import glob
import json
import os
import re

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa: F401  -- registers fla model_types
import fla_patches  # noqa: F401  -- kata + SDPA shim


def step_of(path):
    m = re.search(r"step_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def to_plain(obj):
    """Normalize lm-eval results (numpy scalars etc.) to yaml-safe Python types."""
    return json.loads(json.dumps(obj, default=str))


def load_model(path, device, hf_kwargs, tokenizer_name):
    # Base architecture from the hydra config, then OVERRIDE with the config the
    # checkpoint was trained with (if saved) -- otherwise post-training arch changes
    # (e.g. full-head -> per-group RoPE, which has no weights so it loads silently)
    # corrupt eval with abnormally high PPL. Old checkpoints w/o a saved config use
    # the yaml as-is; set the matching flags there manually to match how it trained.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = ckpt.get("config")
    if saved:
        hf_kwargs = {**hf_kwargs, **saved}
        print(f"[eval] architecture overridden by checkpoint-saved config ({path})")
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model(**hf_kwargs))
    model.load_state_dict(ckpt["model"])
    model = model.to(device, dtype=torch.bfloat16).eval()
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return (
        model,
        tok,
        int(ckpt.get("step", -1)),
        int(ckpt.get("tokens_seen", 0)),
    )


@hydra.main(version_base=None, config_path="configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    output_dir = cfg.output_dir
    tasks = list(cfg.eval.tasks)
    batch_size = int(cfg.eval.batch_size)
    device = cfg.eval.get("device", "cuda:0")
    hf_kwargs = OmegaConf.to_container(cfg.model.hf_kwargs, resolve=True)
    tokenizer = cfg.data.tokenizer

    paths = sorted(
        glob.glob(os.path.join(output_dir, "step_*.pt")), key=step_of, reverse=True
    )
    if not paths:
        raise SystemExit(f"no step_*.pt in {output_dir}")
    print(
        f"output_dir={output_dir}  tasks={tasks}  batch_size={batch_size}  device={device}"
    )
    print(
        f"evaluating {len(paths)} checkpoints in order: "
        f"{[os.path.basename(p) for p in paths]}"
    )

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    for path in paths:
        model, tok, step, tokens = load_model(path, device, hf_kwargs, tokenizer)
        print(
            f"\n=== {os.path.basename(path)}  step={step}  tokens={tokens / 1e9:.2f}B ==="
        )
        lm = HFLM(
            pretrained=model,
            tokenizer=tok,
            batch_size=batch_size,
            device=device,
        )

        out_path = os.path.join(output_dir, f"eval_step_{step}.yaml")
        results = {"step": step, "tokens_seen": tokens, "tasks": {}}
        if os.path.exists(out_path):  # resume a partial eval
            try:
                results = yaml.safe_load(open(out_path)) or results
                results.setdefault("tasks", {})
            except Exception:
                pass

        for task in tasks:
            if task in results["tasks"]:
                print(f"  {task}: cached, skip")
                continue
            try:
                with torch.inference_mode():
                    res = lm_eval.simple_evaluate(
                        model=lm,
                        tasks=[task],
                        batch_size=batch_size,
                        device=device,
                    )
                m = to_plain(res["results"].get(task, {}))
                results["tasks"][task] = m
                yaml.safe_dump(results, open(out_path, "w"), sort_keys=False)
                head = (
                    m.get("acc,none")
                    or m.get("perplexity,none")
                    or m.get("word_perplexity,none")
                    or "done"
                )
                print(f"  {task} -> {head}")
            except Exception as e:
                print(f"  {task} FAILED: {e}")

        del model, lm
        torch.cuda.empty_cache()
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
