"""fp32 master weights + bf16 autocast: what dtype does each thing actually run in?
Probes: param storage, hidden states (layer output), mixer/KV path, logits, loss,
and grads after backward. Expectation of mixed precision:
  params/grads = fp32 ; matmul activations / hidden / kv = bf16 ; loss accum = fp32.
"""

import torch
import fla  # noqa
import fla_patches  # noqa
from transformers import AutoConfig, AutoModelForCausalLM

cfg = AutoConfig.for_model(
    model_type="gated_deltanet",
    hidden_size=128,
    num_hidden_layers=2,
    num_heads=2,
    head_dim=64,
    vocab_size=256,
    max_position_embeddings=128,
    fuse_cross_entropy=False,  # keep logits so we can read their dtype
)
model = AutoModelForCausalLM.from_config(cfg).cuda()  # fp32 master
print(f"param storage dtype: {next(model.parameters()).dtype}")

seen = {}


def hook(name):
    def f(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t):
            seen[name] = t.dtype

    return f


model.model.layers[0].register_forward_hook(hook("layer0_out (hidden)"))
model.model.layers[0].attn.register_forward_hook(hook("mixer_out (kv path)"))
model.lm_head.register_forward_hook(hook("lm_head_out (logits)"))

x = torch.randint(0, 256, (2, 64)).cuda()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(input_ids=x, labels=x)
    loss = out.loss

loss.backward()

print("\n--- activation dtypes under autocast(bf16) ---")
for k, v in seen.items():
    print(f"  {k:>26}: {v}")
print(f"  {'logits (out.logits)':>26}: {out.logits.dtype}")
print(f"  {'loss':>26}: {loss.dtype}")

p = next(model.parameters())
print("\n--- after backward ---")
print(f"  {'param':>26}: {p.dtype}")
print(f"  {'param.grad':>26}: {p.grad.dtype}")
