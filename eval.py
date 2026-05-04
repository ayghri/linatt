"""
Standalone lm-eval-harness runner for a saved checkpoint.

Useful to (a) re-run eval on a final ckpt with a larger batch_size, or
(b) eval a ckpt produced by save_steps if the inline callback was disabled.

Usage:
    python eval.py +ckpt=runs/gated_deltanet_200m_fineweb_edu_10bt
    python eval.py +ckpt=runs/.../checkpoint-19073 eval.batch_size=32
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import fla  # noqa: F401
import fla_patches  # noqa: F401
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    if 'ckpt' not in cfg:
        raise SystemExit('Pass the checkpoint dir: python eval.py +ckpt=<path>')
    ckpt = Path(cfg.ckpt)
    if not ckpt.exists():
        raise SystemExit(f'ckpt not found: {ckpt}')

    import lm_eval
    results = lm_eval.simple_evaluate(
        model='hf',
        model_args=f'pretrained={ckpt},dtype=bfloat16,trust_remote_code=True',
        tasks=list(cfg.eval.tasks),
        batch_size=cfg.eval.batch_size,
    )

    out = ckpt / 'lm_eval.json'
    with out.open('w') as f:
        json.dump(results['results'], f, indent=2, default=str)
    logger.info(f'Wrote {out}')
    print(json.dumps(results['results'], indent=2, default=str))


if __name__ == '__main__':
    main()
