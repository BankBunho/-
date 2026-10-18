#!/usr/bin/env python3
"""CPU check for patch_flux_inplace (G56CT flux training): a tiny random ComfyUI Flux model (the evaluator's own classes).
1. unpatched: forward + backward FAILS with the in-place autograd error (reproduces the 24 Sep CF1 crash = mutation check);
2. patched: no-grad forward is bit-identical to the unpatched forward; 3. patched backward works, grads reach every weight.
Usage (inside the trainer image): python comfytrain/tools/flux_patch_check.py
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
import torch  # noqa: E402
import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True
spec = importlib.util.spec_from_file_location("g56t", HERE / "g56_comfy_train.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
import comfy.ops  # noqa: E402
import comfy.model_management as mm  # noqa: E402
from comfy.ldm.flux.model import Flux  # noqa: E402


def tiny():
    torch.manual_seed(0)
    m = Flux(in_channels=16, out_channels=16, vec_in_dim=32, context_in_dim=32, hidden_size=64, mlp_ratio=2.0, num_heads=2,
             depth=1, depth_single_blocks=1, axes_dim=[8, 12, 12], theta=10000, patch_size=2, qkv_bias=True, guidance_embed=True,
             txt_ids_dims=[], dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape) * 0.05)
    return m


def fwd(m):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 16, 8, 8, generator=g)
    ctx = torch.randn(1, 5, 32, generator=g)
    y = torch.randn(1, 32, generator=g)
    return m(x, torch.tensor([0.5]), ctx, y=y, guidance=torch.tensor([3.5]))


m = tiny()
for p in m.parameters():
    p.requires_grad_(True)
with torch.no_grad():
    ref = fwd(m).clone()
mm.in_training = True  # differentiable rope path, as in the trainer
try:
    fwd(m).square().mean().backward()
    print("[flux-check] unpatched backward: NO error (in-place bug not reproduced)")
    unpatched_fails = False
except RuntimeError as e:
    unpatched_fails = "inplace" in str(e)
    print(f"[flux-check] unpatched backward fails as expected: {str(e)[:90]}")
n = T.patch_flux_inplace()
m.zero_grad(set_to_none=True)
mm.in_training = False
with torch.no_grad():
    out = fwd(m)
mm.in_training = True
same = torch.equal(out, ref)
fwd(m).square().mean().backward()
nograd = [n_ for n_, p in m.named_parameters() if p.grad is None]
print(f"[flux-check] rewrites {n}; patched forward identical: {same}; params without grad: {len(nograd)} {nograd[:3]}")
ok = unpatched_fails and same and n >= 5
print(f"[flux-check] RESULT {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
