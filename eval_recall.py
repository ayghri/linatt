"""Run the Gated-DeltaNet-style RECALL benchmark suite on train_llm.py checkpoints.

Kept SEPARATE from eval_ckpt.py (the general lm-eval run): this evaluates only the
recall/retrieval tasks and writes to <output_dir>/eval_recall_step_N.yaml, so recall
scores never mix with the eval_step_N.yaml commonsense/perplexity numbers.

    python eval_recall.py model=kata_quadratic_m1_340m output_dir=runs/kata_m1_340m

Only the model config is needed. The tokenizer defaults to the 340m training tokenizer
(TinyLlama/TinyLlama_v1.1); override with `+tokenizer=...` for a differently-trained model.
No data config is required (recall datasets come from lm-eval, not the training shards).

Suite: the Arora'24b (Based) cloze/completion recall tasks -- swde, fda, squad_completion.
They use next-word-prediction formatting, so they work on BASE (non-instruction-tuned)
models, matching the GDN paper's setup. NIAH/RULER and the standard triviaqa/nq/drop are
QA/instruction-format -> ~0 on base models, so they are NOT in the default set; add cloze
versions later, or override with eval.recall_tasks=[...].

Evaluates only the LATEST checkpoint, writing each task's result to disk as it finishes
(a crash never loses prior tasks; a re-run retries only the missing/failed ones).
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

DEFAULT_TOKENIZER = "TinyLlama/TinyLlama_v1.1"   # the tokenizer the 340m models trained with

# Arora'24b (Based) cloze/completion recall tasks: next-word-prediction formatting that
# works on BASE (non-instruction-tuned) models, matching the GDN paper's setup. These are
# the only lm-eval recall tasks in that format. NIAH (RULER, QA-style) and the standard
# triviaqa/nq_open/drop are instruction/QA-format -> ~0 on base models; add cloze versions
# of those later (override with eval.recall_tasks=[...]).
RECALL_TASKS = ["swde", "fda", "squad_completion"]


@hydra.main(version_base=None, config_path="configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    output_dir = cfg.output_dir
    batch_size = int(cfg.eval.batch_size)
    device = cfg.eval.get("device", "cuda:0")
    hf_kwargs = OmegaConf.to_container(cfg.model.hf_kwargs, resolve=True)
    # recall datasets come from lm-eval, NOT our training data -> no data config needed.
    # Default to the 340m training tokenizer; override with `+tokenizer=...` if different.
    tokenizer = cfg.get("tokenizer", None) or DEFAULT_TOKENIZER
    tasks = list(cfg.eval.get("recall_tasks", RECALL_TASKS))
    # RULER/NIAH build synthetic haystacks at these token lengths using the tokenizer,
    # both passed to lm-eval via `metadata` (the GDN paper swept 1K-8K).
    niah_lengths = list(cfg.eval.get("niah_lengths", [1024, 2048, 4096, 8192]))
    metadata = {"tokenizer": tokenizer, "max_seq_lengths": niah_lengths}

    # LATEST checkpoint only -- recall eval is generative/slow and we only care about the
    # final model.
    paths = sorted(
        glob.glob(os.path.join(output_dir, "step_*.pt")), key=step_of, reverse=True
    )
    if not paths:
        raise SystemExit(f"no step_*.pt in {output_dir}")
    path = paths[0]

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    model, tok, step, tokens = load_model(path, device, hf_kwargs, tokenizer)
    # KATA M1 (parallel_kata_attn) caches conv/RoPE state but NOT the attention K/V, so HF's
    # incremental decode has each new token attend only to itself -> wrong generation -> 0.0
    # on every generative task. HFLM hardcodes use_cache=True, so wrap generate() to force
    # use_cache=False: each step re-attends over the whole sequence (correct, ~O(T^2)/step
    # slower). Remove once the SPD attention gets a real KV cache.
    _orig_generate = model.generate
    def _full_recompute_generate(*a, **kw):
        kw["use_cache"] = False
        return _orig_generate(*a, **kw)
    model.generate = _full_recompute_generate
    print(
        f"[recall] {os.path.basename(path)}  step={step}  "
        f"tokens={tokens/1e9:.2f}B  tasks={tasks}  (use_cache=False: no attn KV-cache)"
    )
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, device=device)

    out_path = os.path.join(output_dir, f"eval_recall_step_{step}.yaml")
    results = {"step": step, "tokens_seen": tokens, "tasks": {}}
    if os.path.exists(out_path):  # resume: reuse finished tasks, RETRY failed ones
        try:
            results = yaml.safe_load(open(out_path)) or results
            results.setdefault("tasks", {})
        except Exception:
            pass
    # create the file UP FRONT so it exists even if every task fails.
    yaml.safe_dump(results, open(out_path, "w"), sort_keys=False)
    print(f"  writing -> {out_path}")

    for task in tasks:
        prev = results["tasks"].get(task)
        if prev is not None and "error" not in prev:
            print(f"  {task}: cached, skip")
            continue
        try:
            with torch.inference_mode():
                res = lm_eval.simple_evaluate(
                    model=lm, tasks=[task], batch_size=batch_size,
                    device=device, metadata=metadata,
                )
            m = to_plain(res["results"].get(task, {}))
            head = (
                m.get("acc,none") or m.get("exact_match,none")
                or m.get("contains,none") or m.get("score,none") or "done"
            )
            print(f"  {task} -> {head}")
        except Exception as e:
            m = {"error": str(e)[:300]}
            print(f"  {task} FAILED: {e}")
        results["tasks"][task] = m
        # save after EVERY task (success or failure) -> a crash never loses prior results.
        yaml.safe_dump(results, open(out_path, "w"), sort_keys=False)

    del model, lm
    torch.cuda.empty_cache()
    print(f"  done -> {out_path}")


if __name__ == "__main__":
    main()
