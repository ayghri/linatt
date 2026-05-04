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
import json
import logging
from pathlib import Path

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

        # Run each task in isolation so one failing task (e.g. dataset script
        # loader removed in datasets>=4) doesn't take down the rest. Each task
        # is logged to W&B as soon as it succeeds, so a crash mid-suite still
        # leaves partial scores recorded.
        all_metrics = {}
        for task in self.tasks:
            try:
                with torch.inference_mode():
                    res = lm_eval.simple_evaluate(
                        model=lm,
                        tasks=[task],
                        batch_size=self.batch_size,
                        device=str(device),
                    )
                task_metrics = res['results'].get(task, {})
                all_metrics[task] = task_metrics
                self._log_task_to_wandb(task, task_metrics, state.global_step)
                # Print headline metric so progress is visible in the trainer log too.
                headline = (task_metrics.get('acc,none')
                            or task_metrics.get('perplexity,none')
                            or task_metrics.get('word_perplexity,none')
                            or 'done')
                logger.info(f'[step {state.global_step}] {task} -> {headline} (logged to W&B)')
            except Exception as e:
                logger.warning(f'[step {state.global_step}] task {task!r} failed: {e}')

        summary = ' | '.join(
            f"{t}={(m.get('acc,none') or m.get('acc') or m.get('perplexity,none') or m.get('perplexity') or 'n/a')}"
            for t, m in all_metrics.items()
        )
        logger.info(f'[step {state.global_step}] {summary}')

        # Persist alongside the run so you can read scores without W&B.
        # Trainer hasn't saved the checkpoint dir yet (will after this hook
        # returns due to control.should_save=True), so we write to the
        # parent output_dir under an `eval/` subfolder.
        try:
            eval_dir = Path(args.output_dir) / 'eval'
            eval_dir.mkdir(parents=True, exist_ok=True)
            out = eval_dir / f'step-{state.global_step}.json'
            with out.open('w') as f:
                json.dump({
                    'global_step': state.global_step,
                    'tasks': all_metrics,
                }, f, indent=2, default=str)
            logger.info(f'[step {state.global_step}] wrote {out}')
        except Exception as e:
            logger.warning(f'[step {state.global_step}] eval json dump failed: {e}')

        if was_training:
            model.train()
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _log_task_to_wandb(task, metrics, step):
        """Push one task's metrics to W&B and flush.

        Each call uses the same `step=`, which W&B merges into the same
        x-axis point. Different tasks land in different keys
        (`eval/<task>/...`), so no collision. `commit=True` (default) flushes
        immediately so the next task's eval doesn't have to finish first.
        """
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        flat = {f'eval/{task}/{k}': v for k, v in metrics.items()
                if isinstance(v, (int, float))}
        if flat:
            wandb.log(flat, step=step, commit=True)
