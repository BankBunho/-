#!/usr/bin/env python3
"""Pre-training key check (drop_unloadable), CPU only: tiny random ideogram4 / krea2 DiTs built through ComfyUI's real
loader (comfy.sd.load_diffusion_model_state_dict -> ModelPatcher), LoRA attached by the trainer's own attach_lora.

Per family:
  1. complete key map: nothing is dropped;
  2. unmapped target (the evaluator's key map lacks one attached module): drop_unloadable detaches exactly that adapter
     (adapters dict, the patcher's pending weight wrapper, its params), the identity file written afterwards has 0 missing
     keys under the evaluator's strict load (verify_lora_loadable), and the loaded model runs without the dropped wrapper;
  3. mutations must fail: no drop (the old behaviour: the final check finds missing keys after the identity was saved),
     and a drop that leaves the patcher's wrapper in place.
Plus: in main() the key check runs before the fallback identity LoRA is written.
Usage (inside the trainer image): python comfytrain/tools/keycheck_check.py [ideogram4|krea2 ...]
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
os.environ.setdefault("G56CT_INIT", "gauss")

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "3")))
import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True
spec = importlib.util.spec_from_file_location("g56kc", HERE / "g56_comfy_train.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
T.log = lambda *a, **k: None
import comfy.lora  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402


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


def attached(fam, base_sd):
    mp = comfy.sd.load_diffusion_model_state_dict({k: v.clone() for k, v in base_sd.items()}, model_options={})
    mp.model.requires_grad_(False)
    adapters, params = T.attach_lora(mp, T.FAMILIES[fam]["include"], 4, 4.0)
    return mp, adapters, params


def scenario(fam, base_sd, drop=True):
    """Unmapped target -> checks dict (all True = correct behaviour)."""
    mp, adapters, params = attached(fam, base_sd)
    names = sorted(adapters)
    ghost = names[len(names) // 2]
    ghost_ad = adapters[ghost]
    orig = comfy.lora.model_lora_keys_unet

    def km(model, key_map=None):  # the evaluator's key map without the ghost module
        m = orig(model, {})
        return {k: v for k, v in m.items() if v != f"{ghost}.weight"}
    comfy.lora.model_lora_keys_unet = km
    ok = {}
    try:
        before = T.verify_lora_loadable(mp, T.lora_state_dict(adapters))[1]
        ok["unmapped_detected_before"] = len(before) > 0
        dropped = T.drop_unloadable(mp, adapters) if drop else []
        ok["dropped_exactly_ghost"] = dropped == [ghost] and ghost not in adapters and len(adapters) == len(names) - 1
        ok["wrapper_detached"] = not any(f is ghost_ad for f in mp.weight_wrapper_patches.get(f"{ghost}.weight", []))
        live = {id(p) for a in adapters.values() for p in T.adapter_params(a)}  # main(): params filtered the same way
        ok["params_filtered"] = len([p for p in params if id(p) in live]) == len(params) - len(T.adapter_params(ghost_ad))
        tmp = Path(tempfile.mkdtemp())
        T.save_lora(T.lora_state_dict(adapters), tmp)  # = the fallback identity LoRA written after the check
        from safetensors.torch import load_file
        n_patch, missing = T.verify_lora_loadable(mp, load_file(str(tmp / "last.safetensors")))
        ok["identity_file_0_missing"] = not missing and n_patch == len(adapters)
        mm.load_models_gpu([mp], force_full_load=True)
        mod = mp.model.get_submodule(ghost)
        ok["ghost_not_applied"] = not any(f is ghost_ad for f in mod.weight_function)
        x, t, ctx = inputs(fam)
        with torch.no_grad():
            out = mp.model.diffusion_model(x, t, context=ctx).float()
        ok["forward_runs"] = bool(torch.isfinite(out).all())
        mm.unload_all_models()
    finally:
        comfy.lora.model_lora_keys_unet = orig
    return ok, ghost


def check(fam):
    print(f"==== {fam}", flush=True)
    base_sd = tiny_sd(fam)
    mp, adapters, _ = attached(fam, base_sd)
    n = len(adapters)
    clean = T.drop_unloadable(mp, adapters) == [] and len(adapters) == n
    print(f"[keycheck] {fam}: {n} LoRA targets, complete key map -> nothing dropped: {clean}", flush=True)
    ok, ghost = scenario(fam, base_sd)
    print(f"[keycheck] {fam}: unmapped target {ghost} -> {ok}", flush=True)
    good = clean and all(ok.values())
    # mutations: the check must notice them
    m1, _ = scenario(fam, base_sd, drop=False)
    orig_detach = T.detach_wrapper
    T.detach_wrapper = lambda *a, **k: None
    try:
        m2, _ = scenario(fam, base_sd)
    finally:
        T.detach_wrapper = orig_detach
    caught = not all(m1.values()) and not all(m2.values())
    print(f"[keycheck] {fam}: mutation no-drop caught {not all(m1.values())} "
          f"(failed: {[k for k, v in m1.items() if not v]}); mutation wrapper-left caught {not all(m2.values())} "
          f"(failed: {[k for k, v in m2.items() if not v]})", flush=True)
    return good and caught


def order_in_main():
    src = (HERE / "g56_comfy_train.py").read_text()
    body = src[src.index("\ndef main():"):]
    i, j = body.find("drop_unloadable(mp, adapters"), body.find("fallback: identity LoRA written before training")
    ok = 0 <= i < j
    print(f"[keycheck] main(): key check before the fallback identity save: {ok}", flush=True)
    return ok


if __name__ == "__main__":
    fams = sys.argv[1:] or ["ideogram4", "krea2"]
    ok = all([check(f) for f in fams]) and order_in_main()
    print(f"[keycheck] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)
