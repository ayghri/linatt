"""Run the Gated-DeltaNet-style RECALL benchmark suite on train_llm.py checkpoints.

Kept SEPARATE from eval_ckpt.py (the general lm-eval run): this evaluates only the
recall/retrieval tasks and writes to <output_dir>/eval_recall_step_N.yaml, so recall
scores never mix with the eval_step_N.yaml commonsense/perplexity numbers.

    python eval_recall.py model=kata_quadratic_m1_340m data=slimpajama_15bt \
        output_dir=runs/kata_m1_340m

Suite (matches the GDN paper):
  RULER S-NIAH-1/2/3     -> niah_single_1/2/3    (passkey / number / uuid, generative)
  MK-NIAH (GDN-2)        -> niah_multikey_1, niah_multivalue, niah_multiquery
  recall-intensive       -> swde, fda, squad_completion, triviaqa, nq_open, drop (Arora'24b)

Requires `pip install wonderwords nltk` (NIAH generators). By default evals the LATEST
checkpoint only (recall tasks are generative/slow); set `+eval.recall_all=true` for all.
Override the task list with `eval.recall_tasks=[...]`; set NIAH lengths with e.g.
`+metadata.max_seq_lengths=[1024,2048,4096,8192]`.
"""

import glob
import os

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

import fla  # noqa: F401  -- registers fla model_types
import fla_patches  # noqa: F401  -- kata + SDPA shim
from eval_ckpt import load_model, step_of, to_plain

RECALL_TASKS = [
    # RULER single needle-in-a-haystack  (= GDN S-NIAH-1/2/3)
    "niah_single_1", "niah_single_2", "niah_single_3",
    # multi-key / multi-value / multi-query NIAH  (GDN-2)
    "niah_multikey_1", "niah_multivalue", "niah_multiquery",
    # recall-intensive real-world  (Arora'24b / Based)
    "swde", "fda", "squad_completion", "triviaqa", "nq_open", "drop",
]


@hydra.main(version_base=None, config_path="configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    output_dir = cfg.output_dir
    batch_size = int(cfg.eval.batch_size)
    device = cfg.eval.get("device", "cuda:0")
    hf_kwargs = OmegaConf.to_container(cfg.model.hf_kwargs, resolve=True)
    tokenizer = cfg.data.tokenizer
    tasks = list(cfg.eval.get("recall_tasks", RECALL_TASKS))
    do_all = bool(cfg.eval.get("recall_all", False))

    paths = sorted(
        glob.glob(os.path.join(output_dir, "step_*.pt")), key=step_of, reverse=True
    )
    if not paths:
        raise SystemExit(f"no step_*.pt in {output_dir}")
    if not do_all:
        paths = paths[:1]  # latest checkpoint only (recall eval is generative/slow)
    print(
        f"[recall] output_dir={output_dir}  tasks={tasks}  "
        f"ckpts={[os.path.basename(p) for p in paths]}"
    )

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    for path in paths:
        model, tok, step, tokens = load_model(path, device, hf_kwargs, tokenizer)
        print(f"\n=== {os.path.basename(path)}  step={step}  tokens={tokens/1e9:.2f}B ===")
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, device=device)

        out_path = os.path.join(output_dir, f"eval_recall_step_{step}.yaml")
        results = {"step": step, "tokens_seen": tokens, "tasks": {}}
        if os.path.exists(out_path):  # resume a partial recall eval
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
                        model=lm, tasks=[task], batch_size=batch_size, device=device
                    )
                m = to_plain(res["results"].get(task, {}))
                results["tasks"][task] = m
                yaml.safe_dump(results, open(out_path, "w"), sort_keys=False)
                head = (
                    m.get("acc,none")
                    or m.get("exact_match,none")
                    or m.get("contains,none")
                    or m.get("score,none")
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
