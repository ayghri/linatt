"""Two checks:
  1. fla's internal shift+CE (labels=x) == explicit next-token CE from logits.
  2. One smoke step of the manual DDP trainer (only under torchrun).

Run shift-check only (plain python):
    CUDA_VISIBLE_DEVICES=0 python verify_loss.py
Run shift-check + trainer smoke (DDP):
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 verify_loss.py
"""
import os

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

import fla  # noqa: F401
import fla_patches  # noqa: F401
from train_llm import DDPLLMPretrainer

HERE = os.path.dirname(os.path.abspath(__file__))
config = OmegaConf.load(os.path.join(HERE, "conf_llm", "verify.yaml"))
print(OmegaConf.to_yaml(config))

cfg = AutoConfig.for_model(
    model_type="gated_deltanet",
    hidden_size=64,
    num_hidden_layers=4,
    num_heads=1,
    vocab_size=128,
    fuse_cross_entropy=True,
)
model = AutoModelForCausalLM.from_config(cfg).cuda().eval()

# ---- check 1: shift+CE equivalence ----
x = torch.randint(0, 128, (1, config.seq_len)).cuda()
loss_model = model(input_ids=x, labels=x).loss
logits = model(input_ids=x).logits
loss_manual = F.cross_entropy(
    logits[:, :-1].reshape(-1, 128),
    x[:, 1:].reshape(-1),
)
print(f"loss_model={loss_model.item():.6f}  loss_manual={loss_manual.item():.6f}")
assert abs(loss_model.item() - loss_manual.item()) < 1e-4, "shift mismatch"
print("shift+CE OK")

# ---- check 2: trainer smoke step (needs torchrun env) ----
if "LOCAL_RANK" in os.environ:
    dataset = [
        {"input_ids": torch.randint(0, 128, (config.seq_len,))} for _ in range(256)
    ]
    trainer = DDPLLMPretrainer(model, dataset, config)
    trainer.train()
    print("trainer smoke OK")
else:
    print("skip trainer smoke (run under torchrun to enable)")
