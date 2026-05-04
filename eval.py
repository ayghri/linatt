"""
Standalone lm-eval-harness runner for a saved checkpoint.

Each task runs in isolation, scores write to disk + W&B as soon as the task
finishes. One task crashing (e.g. dataset loader removed) does not take
down the rest of the suite.

Usage:
    python eval.py +ckpt=runs/gated_deltanet_200m_fineweb_edu_10bt
    python eval.py +ckpt=runs/.../checkpoint-19073 eval.batch_size=32

W&B logging is on by default and creates a new run named eval_<ckpt-name>
under the configured project/entity. Disable with `wandb.mode=disabled`.

To attach scores to an existing training run instead, pass its W&B run id:
    python eval.py +ckpt=... +wandb_run_id=15kttze8
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import hydra
import fla  # noqa: F401
import fla_patches  # noqa: F401
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _init_wandb(cfg, ckpt):
    """Start a W&B run for this eval pass. Returns the run, or None if disabled."""
    if cfg.wandb.mode == 'disabled':
        return None
    try:
        import wandb
    except ImportError:
        logger.warning('wandb not installed; skipping W&B logging')
        return None

    if cfg.wandb.entity:
        os.environ.setdefault('WANDB_ENTITY', cfg.wandb.entity)
    os.environ.setdefault('WANDB_PROJECT', cfg.wandb.project)
    os.environ['WANDB_MODE'] = cfg.wandb.mode

    if 'wandb_run_id' in cfg:
        # Resume an existing run so eval scores append to its history.
        return wandb.init(id=str(cfg.wandb_run_id), resume='must')
    return wandb.init(name=f'eval_{ckpt.name}', config={'ckpt': str(ckpt)})


def _log_task(wandb_run, task, metrics):
    if wandb_run is None:
        return
    flat = {f'eval/{task}/{k}': v for k, v in metrics.items()
            if isinstance(v, (int, float))}
    if flat:
        wandb_run.log(flat, commit=True)


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    if 'ckpt' not in cfg:
        raise SystemExit('Pass the checkpoint dir: python eval.py +ckpt=<path>')
    ckpt = Path(cfg.ckpt)
    if not ckpt.exists():
        raise SystemExit(f'ckpt not found: {ckpt}')

    import lm_eval

    wandb_run = _init_wandb(cfg, ckpt)
    out = ckpt / 'lm_eval.json'
    all_metrics: dict = {}
    # If a partial JSON already exists from a previous interrupted run, resume.
    if out.exists():
        try:
            all_metrics = json.loads(out.read_text())
            logger.info(f'Resuming: {len(all_metrics)} task(s) already in {out}')
        except Exception:
            all_metrics = {}

    model_args = f'pretrained={ckpt},dtype=bfloat16,trust_remote_code=True'

    for task in list(cfg.eval.tasks):
        if task in all_metrics:
            logger.info(f'task {task!r} already evaluated, skipping')
            _log_task(wandb_run, task, all_metrics[task])
            continue
        try:
            logger.info(f'-> running task {task!r}')
            res = lm_eval.simple_evaluate(
                model='hf',
                model_args=model_args,
                tasks=[task],
                batch_size=cfg.eval.batch_size,
            )
            metrics = res['results'].get(task, {})
            all_metrics[task] = metrics
            # Persist after every task so a crash doesn't lose previous progress.
            with out.open('w') as f:
                json.dump(all_metrics, f, indent=2, default=str)
            _log_task(wandb_run, task, metrics)
            headline = (metrics.get('acc,none')
                        or metrics.get('perplexity,none')
                        or metrics.get('word_perplexity,none')
                        or 'done')
            logger.info(f'   {task} -> {headline}  (saved + logged)')
        except Exception as e:
            logger.warning(f'   task {task!r} FAILED: {e}')

    if wandb_run is not None:
        wandb_run.finish()
    logger.info(f'Wrote {out}')
    print(json.dumps(all_metrics, indent=2, default=str))


if __name__ == '__main__':
    main()
