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

        # Per-task isolation: one failing task (e.g. dataset loader removed in
        # datasets>=4) doesn't take down the rest. File: rewritten after every
        # task so a mid-suite crash still leaves the completed scores on disk.
        # W&B: batched once at the end so all tasks land at the same train
        # step without hitting wandb's "can't log same step twice" rule.
        eval_dir = Path(args.output_dir) / 'eval'
        eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = eval_dir / f'step-{state.global_step}.json'

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
                # Persist immediately — every successful task is on disk
                # before the next one starts.
                with out_path.open('w') as f:
                    json.dump({'global_step': state.global_step,
                               'tasks': all_metrics}, f, indent=2, default=str)
                headline = (task_metrics.get('acc,none')
                            or task_metrics.get('perplexity,none')
                            or task_metrics.get('word_perplexity,none')
                            or 'done')
                logger.info(f'[step {state.global_step}] {task} -> {headline} '
                            f'(saved to {out_path.name})')
            except Exception as e:
                logger.warning(f'[step {state.global_step}] task {task!r} failed: {e}')

        # Single batched W&B log at the end of the eval cycle.
        self._log_all_to_wandb(all_metrics, state.global_step)
        logger.info(f'[step {state.global_step}] {len(all_metrics)} task(s) logged to W&B')

        # Aggressive teardown of HFLM + lm-eval refs. Without this, KV-cache
        # tensors allocated during loglikelihood scoring linger in PyTorch's
        # caching allocator at non-training shapes, fragmenting the pool and
        # slowing training step times for hundreds of steps after.
        del lm
        try:
            del res
        except UnboundLocalError:
            pass

        if was_training:
            model.train()
        gc.collect()
        torch.cuda.empty_cache()
        # Force the caching allocator to release segments back to the driver.
        # This is more aggressive than empty_cache alone and clears the
        # fragmentation eval introduced. Costs nothing other than the next
        # allocation having to go through cudaMalloc once.
        try:
            torch.cuda.synchronize()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    @staticmethod
    def _log_all_to_wandb(all_metrics, step):
        """Single batched W&B log for the whole eval cycle.

        Done once at the end so multiple tasks land at the same train step
        without hitting wandb's "can't go backward" rule (which silently drops
        repeated `wandb.log(..., step=N, commit=True)` calls at the same N).
        """
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        flat = {}
        for task, metrics in all_metrics.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    flat[f'eval/{task}/{k}'] = v
        if flat:
            wandb.log(flat, step=step, commit=True)
