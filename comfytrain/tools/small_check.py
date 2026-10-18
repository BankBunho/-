#!/usr/bin/env python3
"""G56CT_SMALL / G56CT_DIRECT check, CPU only: a tiny random krea2 and ideogram4 DiT built through ComfyUI's real loader
(comfy.sd.load_diffusion_model_state_dict -> model detection -> ModelPatcher), driven by the trainer's own attach / export /
exact-patch code.

Checks per family (fp32 weights, so every merge path is exact up to float error):
  1. every small / direct delta gets a gradient through the live training wrappers and hooks;
  2. the exported file passes the evaluator's strict key check (verify_lora_loadable);
  3. forward(live wrappers) == forward(trainer exact_patch_apply) == forward(independent evaluator-style load:
     comfy.lora.load_lora -> ModelPatcher.add_patches(1.0) -> load_models_gpu on a fresh copy of the base);
  4. exact_patch_restore / small_disable really restore: base forward == fresh unpatched model;
  5. mutations must fail: a sign-flipped small delta changes the evaluator forward; a bias diff exported under '.diff'
     is rejected (shape mismatch / not loaded).
Usage (inside the trainer image): python comfytrain/tools/small_check.py [krea2|ideogram4 ...]
"""
import importlib.util
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
os.environ.setdefault("G56CT_INIT", "gauss")

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "3")))


def load_trainer():
    spec = importlib.util.spec_from_file_location("g56t", HERE / "g56_comfy_train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True  # no GPU needed (runs beside a training job)
T = load_trainer()
import comfy.lora  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402

EXTRA = {"krea2": (r"^first$|^last\.linear$|^tmlp\.", r"^first$|^last\.linear$"),
         "ideogram4": (r"^final_layer\.(linear|adaln_modulation)$|^input_proj$", r"^final_layer\.|^input_proj$")}


def tiny_sd(fam):
    gen = torch.Generator().manual_seed(0)
    if fam == "krea2":
        from comfy.ldm.krea2.model import SingleStreamDiT
        m = SingleStreamDiT(features=256, tdim=256, txtdim=160, heads=2, kvheads=2, multiplier=4, layers=2, patch=2,
                            channels=16, txtlayers=2, txtheads=20, txtkvheads=20, dtype=torch.float32,
                            operations=comfy.ops.disable_weight_init)
    else:
        from comfy.ldm.ideogram4.model import Ideogram4Transformer2DModel
        m = Ideogram4Transformer2DModel(num_layers=1, dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for _, p in m.named_parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * (0.02 if p.dim() >= 2 else 0.1))
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def inputs(fam):
    g = torch.Generator().manual_seed(2)
    if fam == "krea2":
        return torch.randn(1, 16, 8, 8, generator=g), torch.tensor([0.3]), torch.randn(1, 5, 2 * 160, generator=g)
    return torch.randn(1, 128, 4, 4, generator=g), torch.tensor([0.3]), torch.randn(1, 3, 53248, generator=g) * 0.1


def fwd(mp, x, t, ctx):
    return mp.model.diffusion_model(x, t, context=ctx).float()


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def build(sd):
    return comfy.sd.load_diffusion_model_state_dict({k: v.clone() for k, v in sd.items()}, model_options={})


def check(fam):
    print(f"==== {fam}", flush=True)
    base_sd = tiny_sd(fam)
    x, t, ctx = inputs(fam)
    # reference: fresh unpatched model
    mp0 = build(base_sd)
    mm.load_models_gpu([mp0], force_full_load=True)
    with torch.no_grad():
        out_base = fwd(mp0, x, t, ctx)
    mm.unload_all_models()

    mp = build(base_sd)
    mp.model.requires_grad_(False)
    extra, direct = EXTRA[fam]
    T.CFG["direct"] = direct
    inc = f"(?:{T.FAMILIES[fam]['include']})|(?:{extra})"
    adapters, params = T.attach_lora(mp, inc, 4, 4.0)
    smalls, sp = T.attach_small(mp, T.SMALL_DEFAULT[fam], 256)
    adapters.update(smalls)
    n_direct = sum(isinstance(a, T.DirectDelta) for a in adapters.values())
    kinds = {}
    for a in smalls.values():
        kinds[a.key.rsplit(".", 1)[-1] + ("/fn" if a.via_fn else "/hook")] = kinds.get(a.key.rsplit(".", 1)[-1] + ("/fn" if a.via_fn else "/hook"), 0) + 1
    print(f"lora {len(adapters) - len(smalls) - n_direct}, direct {n_direct}, small {len(smalls)} {kinds}", flush=True)
    mm.load_models_gpu([mp], force_full_load=True)
    for a in adapters.values():
        if isinstance(a, T.SmallDelta) and not a.via_fn:
            a.move_to(a.module._parameters[a.pname].device)
    c = {"mp": mp, "adapters": adapters}
    with torch.no_grad():  # zero deltas == base (gauss init: up = 0)
        z = fwd(mp, x, t, ctx)
    print(f"zero-delta vs base rel {rel(z, out_base):.2e}", flush=True)
    assert rel(z, out_base) < 1e-6
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for a in adapters.values():
            if isinstance(a, (T.DirectDelta, T.SmallDelta)):
                a.delta.copy_(torch.randn(a.delta.shape, generator=g) * 0.05)
            else:
                a.lora_up.weight.copy_(torch.randn(a.lora_up.weight.shape, generator=g) * 0.05)
    mm.in_training = True  # differentiable rope path (the trainer trains with this flag)
    try:
        fwd(mp, x, t, ctx).square().mean().backward()
    finally:
        mm.in_training = False
    with torch.no_grad():
        out_train = fwd(mp, x, t, ctx)
    nograd = [k for k, a in adapters.items() if isinstance(a, (T.SmallDelta, T.DirectDelta))
              and (a.delta.grad is None or float(a.delta.grad.abs().sum()) == 0.0)]
    print(f"deltas without gradient: {nograd[:5]} ({len(nograd)})", flush=True)
    assert not nograd
    out_train = out_train.detach()
    print(f"trained vs base rel {rel(out_train, out_base):.3e}", flush=True)
    assert rel(out_train, out_base) > 1e-3

    sdx = T.lora_state_dict(adapters)
    n_p, missing = T.verify_lora_loadable(mp, sdx)
    print(f"export {len(sdx)} keys, patches {n_p}, missing {missing[:3]}", flush=True)
    assert not missing

    with torch.no_grad():
        saved = T.exact_patch_apply(c, sdx, mm.get_torch_device())
        out_exact = fwd(mp, x, t, ctx)
        T.exact_patch_restore(saved)
        out_back = fwd(mp, x, t, ctx)
        undo = T.small_disable(c)
        mods = [comfy.utils.get_attr(mp.model, k) for k, a in adapters.items() if not isinstance(a, T.SmallDelta)]
        wf = [list(m.weight_function) for m in mods]
        for m in mods:
            m.weight_function = []
        out_off = fwd(mp, x, t, ctx)
        for m, f in zip(mods, wf):
            m.weight_function = f
        T.small_restore(undo)
        out_on = fwd(mp, x, t, ctx)
    print(f"exact vs live rel {rel(out_exact, out_train):.2e}; restore rel {rel(out_back, out_train):.2e}; "
          f"all-off vs base rel {rel(out_off, out_base):.2e}; re-on vs live {rel(out_on, out_train):.2e}", flush=True)
    assert rel(out_exact, out_train) < 1e-5 and rel(out_back, out_train) < 1e-7
    assert rel(out_off, out_base) < 1e-6 and rel(out_on, out_train) < 1e-7
    mm.unload_all_models()

    def evaluator(sd_file):
        mp2 = build(base_sd)
        key_map = comfy.lora.model_lora_keys_unet(mp2.model, {})
        lost = []

        class H(logging.Handler):
            def emit(self, r):
                if r.getMessage().startswith(("lora key not loaded", "WARNING SHAPE MISMATCH", "ERROR")):
                    lost.append(r.getMessage()[:120])
        h = H()
        logging.getLogger().addHandler(h)
        try:
            patches = comfy.lora.load_lora(sd_file, key_map)
            mp2.add_patches(patches, 1.0, 1.0)
            mm.load_models_gpu([mp2], force_full_load=True)
            with torch.no_grad():
                out = fwd(mp2, x, t, ctx)
        finally:
            logging.getLogger().removeHandler(h)
            mm.unload_all_models()
        return out, lost

    out_eval, lost = evaluator(sdx)
    print(f"evaluator-path vs live rel {rel(out_eval, out_train):.2e} (lost {lost[:2]})", flush=True)
    assert not lost and rel(out_eval, out_train) < 1e-5

    # mutations
    k0 = next(k for k in sdx if k.endswith((".diff_b",)) or (k.endswith(".diff") and k in {a.key for a in smalls.values()}))
    bad = dict(sdx)
    bad[k0] = -bad[k0]
    out_m1, _ = evaluator(bad)
    print(f"mutation sign-flip {k0}: rel {rel(out_m1, out_train):.2e} (must be > 0)", flush=True)
    assert rel(out_m1, out_train) > 1e-7
    kb = next((k for k in sdx if k.endswith(".diff_b")), None)
    if kb:
        bad = {(k[: -len(".diff_b")] + ".diff" if k == kb else k): v for k, v in sdx.items()}
        bad.pop(kb[: -len(".diff_b")] + ".lora_up.weight", None)  # make the '.diff' the only weight patch of that module
        bad.pop(kb[: -len(".diff_b")] + ".lora_down.weight", None)
        bad.pop(kb[: -len(".diff_b")] + ".alpha", None)
        out_m2, lost2 = evaluator(bad)
        print(f"mutation bias-as-weight {kb}: lost {lost2[:1]} rel {rel(out_m2, out_train):.2e}", flush=True)
        assert lost2 or rel(out_m2, out_train) > 1e-7
    print(f"{fam}: OK", flush=True)


if __name__ == "__main__":
    fams = sys.argv[1:] or ["krea2", "ideogram4"]
    for f in fams:
        check(f)
    print("ALL OK")
