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

DEFAULT_TOKENIZER = (
    "TinyLlama/TinyLlama_v1.1"  # the tokenizer the 340m models trained with
)


def _apply_yarn(model, scale, orig_max_pos, beta_fast=32, beta_slow=1):
    """YaRN context extension: rewrite each RoPE inv_freq (NTK-by-parts) so a model trained at
    orig_max_pos can process scale*orig_max_pos tokens. High-freq dims extrapolate (kept), low-freq
    dims interpolate (inv_freq/scale), smooth ramp between. (KATA-SPD, not softmax -> no mscale.)"""
    import math

    def yarn_inv_freq(dim, base):
        d2 = -(dim // -2)
        freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32)[:d2] / dim)
        inv_extrap = 1.0 / freqs
        inv_interp = 1.0 / (scale * freqs)
        find = lambda nr: (
            (dim * math.log(orig_max_pos / (nr * 2 * math.pi))) / (2 * math.log(base))
        )
        low, high = (
            max(math.floor(find(beta_fast)), 0),
            min(math.ceil(find(beta_slow)), dim - 1),
        )
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(d2, dtype=torch.float32) - low) / (high - low), 0, 1
        )
        mask = 1 - ramp  # 1=extrapolate (high freq), 0=interpolate
        return inv_interp * (1 - mask) + inv_extrap * mask

    n = 0
    for layer in model.model.layers:
        rot = getattr(getattr(layer, "attn", None), "rotary", None)
        if rot is None:
            continue
        rot.inv_freq.copy_(yarn_inv_freq(rot.dim, rot.base).to(rot.inv_freq.device))
        rot._seq_len_cached = 0  # force cos/sin recompute with new freqs
        rot._cos_cached = rot._sin_cached = rot._cos_k_cached = rot._sin_k_cached = None
        n += 1
    return n


# Arora'24b (Based) cloze/completion recall tasks: next-word-prediction formatting that
# works on BASE (non-instruction-tuned) models, matching the GDN paper's setup. These are
# the only lm-eval recall tasks in that format. NIAH (RULER, QA-style) and the standard
# triviaqa/nq_open/drop are instruction/QA-format -> ~0 on base models; add cloze versions
# of those later (override with eval.recall_tasks=[...]).
# GDN paper's recall suite: SWDE / SQD / FDA / TQA / NQ / Drop.
#   swde, fda, squad_completion (=SQD), drop -> in-context (answer lives in the context);
#       swde/fda/squad use Arora'24b cloze formatting, base-model friendly.
#   triviaqa (=TQA), nq_open (=NQ)           -> WARNING: lm-eval ships the CLOSED-BOOK
#       versions (question only, NO passage). They test PARAMETRIC memory, not in-context
#       recall, so expect low/uninformative numbers at 340M. Included for paper parity;
#       to truly match the paper's TQA/NQ, swap in the Arora'24b in-context versions
#       (passage + cloze query) from the Based repo.
# All IN-CONTEXT (answer lives in the provided passage) -- the GDN/Based recall suite.
# based_triviaqa/based_nq are the Arora'24 in-context cloze versions (recall_tasks/), NOT
# lm-eval's stock triviaqa/nq_open which are CLOSED-BOOK (rc.nocontext) -> ~0 at 340M.
RECALL_TASKS = ["swde", "fda", "squad_completion", "drop", "based_triviaqa", "based_nq"]
_CUSTOM_TASK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "recall_tasks"
)


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
    from lm_eval.tasks import TaskManager

    # metadata (tokenizer + niah lengths) MUST go to the TaskManager -- when a custom task_manager
    # is passed to simple_evaluate, its metadata= arg is ignored, so NIAH loses its tokenizer.
    task_manager = TaskManager(include_path=_CUSTOM_TASK_DIR, metadata=metadata)

    model, tok, step, tokens = load_model(path, device, hf_kwargs, tokenizer)
    # KATA (parallel_kata_attn) caches conv/RoPE state but NOT the attention K/V, so HF's
    # incremental decode has each new token attend only to itself -> wrong generation -> 0.0
    # on every generative task. Force full recompute for KATA only (correct, ~O(T^2)/step
    # slower); GDN/transformer cache correctly on transformers 4.57. HFLM hardcodes
    # use_cache=True and spreads gen_kwargs after it, so we wrap generate() rather than pass
    # a kwarg. Remove once the SPD attention gets a real KV cache.
    no_cache = getattr(model.config, "model_type", "") == "kata"
    if no_cache:
        _orig_generate = model.generate
        model.generate = lambda *a, **kw: _orig_generate(
            *a, **{**kw, "use_cache": False}
        )
    yarn = cfg.eval.get("rope_yarn", None)  # extend RoPE ctx (e.g. 4 -> 2048->8192)
    if yarn:
        n = _apply_yarn(model, float(yarn), 2048)
        print(
            f"  [yarn] scale={yarn} applied to {n} rotaries (2048 -> {int(2048 * float(yarn))} ctx)"
        )
    print(
        f"[recall] {os.path.basename(path)}  step={step}  tokens={tokens / 1e9:.2f}B  "
        f"tasks={tasks}{'  (KATA: use_cache=False, no attn KV-cache)' if no_cache else ''}"
    )
    # NIAH haystacks (up to max(niah_lengths)) exceed the 2048 train ctx. HFLM defaults max_length
    # to the model's max_position_embeddings (2048) and LEFT-TRUNCATES longer inputs to ~1920 tokens
    # -> the needle gets chopped off -> 4K/8K scores collapse to the truncation-survival probability,
    # NOT real recall. Set max_length to fit the whole haystack so the model sees the full context
    # (GDN/linear-attn extrapolate; RoPE models like KATA still process it, just OOD at long range).
    max_len = (
        max(niah_lengths) + 256
        if any("niah" in t or "ruler" in t for t in tasks)
        else None
    )
    lm = HFLM(
        pretrained=model,
        tokenizer=tok,
        batch_size=batch_size,
        device=device,
        max_length=max_len,
    )

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
                    model=lm,
                    tasks=[task],
                    batch_size=batch_size,
                    device=device,
                    metadata=metadata,
                    task_manager=task_manager,
                    limit=cfg.eval.get("limit", None),
                )
            m = to_plain(res["results"].get(task, {}))
            head = (
                m.get("acc,none")
                or m.get("exact_match,none")
                or m.get("contains,none")
                or m.get("score,none")
                or "done"
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
