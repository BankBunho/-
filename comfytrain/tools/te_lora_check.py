#!/usr/bin/env python3
"""G56CT_TE_LORA export check, CPU only (no GPU needed).

1. keys: the trainer's offline-rebuilt TE (--rebuild) and the evaluator's Comfy-Org TE file have the same state-dict keys
   and give the same TE LoRA keys; a file = trainer TE keys + DiT keys of the family's real DiT structure (built on the meta
   device from the real key/shape list) passes the evaluator's apply_lora (copied from G.O.D) with 0 unmatched entries and
   one clip patch per TE adapter.
2. numerics: zero adapters -> te_encode == CLIPTextEncode (bitwise); our exact TE merge (te_exact_patch_apply + te_encode)
   == the evaluator path (LoraLoader -> CLIPTextEncode on the candidate clip) bitwise; reports the gaps train-forward
   (bypass, no requantization) vs evaluator and the zero-delta patch tax (CPU numerics: merge in fp32, the A100 merges in
   fp16 -> magnitudes indicative only).

python te_lora_check.py --family krea2 --comfy-te /calib/qwen3vl_4b_fp8_scaled.safetensors --out /w/out [--rebuild]
"""
import argparse
import gc
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
os.environ.setdefault("G56CT_TE_LORA", "1")

CAPTION = ("A flat vector illustration of a red fox sitting under a crescent moon, bold outlines, teal night sky, "
           "the words 'NIGHT OWL CAFE' in white serif letters at the bottom")


def apply_lora_evaluator(model, clip, name):
    """Verbatim logic of G.O.D validator/evaluation/image_flow_adapter.py ImageFlowAdapter.apply_lora (origin/main)."""
    import nodes
    missing = []

    class CaptureMissing(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if message.startswith(("lora key not loaded:", "NOT LOADED ")):
                missing.append(message)

    handler = CaptureMissing()
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        patched, patched_clip = nodes.LoraLoader().load_lora(model, clip, name, 1.0, 1.0)
    finally:
        root_logger.removeHandler(handler)
    if missing or not patched.patches:
        raise ValueError(
            f"Incomplete LoRA application: {len(missing)} unmatched entries; {len(patched.patches)} model patches"
        )
    return patched, patched_clip


def dit_keys(fam, krea2_ckpt):
    """{key: (dtype, shape)} of the evaluator's DiT file: ideogram4 = baked Comfy-Org table, krea2 = raw.safetensors header."""
    if fam == "ideogram4":
        return {k: (v[0], v[1]) for k, v in json.load(open(HERE / "assets" / "dm_ideogram4_fp8_scaled.json"))["keys"].items()}
    with open(krea2_ckpt, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return {k: (v["dtype"], v["shape"]) for k, v in h.items() if k != "__metadata__"}


def meta_dit(fam, krea2_ckpt):
    """ModelPatcher around the family's real DiT module tree on the meta device (keys/shapes from the real file)."""
    import torch
    import comfy.model_detection
    import comfy.model_patcher
    dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "F8_E4M3": torch.bfloat16}
    sd = {k: torch.empty(shape, dtype=dt.get(d, torch.bfloat16), device="meta") for k, (d, shape) in dit_keys(fam, krea2_ckpt).items()
          if not k.endswith((".comfy_quant", ".weight_scale", ".input_scale", ".scale_weight"))}
    cfg = comfy.model_detection.model_config_from_unet(sd, "")
    cfg.set_inference_dtype(torch.bfloat16, None)
    model = cfg.get_model(sd, "", device=torch.device("meta"))
    return comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def dit_lora(T, fam, mp, rank=4):
    """DiT part in the trainer's export format (same module selection as attach_lora), random small values."""
    import re
    import torch
    g = torch.Generator().manual_seed(1)
    rx = re.compile(T.FAMILIES[fam]["include"])
    sd = {}
    for full, m in mp.model.named_modules():
        if not full.startswith("diffusion_model.") or not rx.match(full[len("diffusion_model."):]):
            continue
        w = getattr(m, "weight", None)
        if not hasattr(m, "weight_function") or w is None or len(w.shape) < 2:
            continue
        sd[f"{full}.lora_up.weight"] = torch.randn(w.shape[0], rank, generator=g) * 1e-3
        sd[f"{full}.lora_down.weight"] = torch.randn(rank, w.shape[1:].numel(), generator=g) * w.shape[1:].numel() ** -0.5
        sd[f"{full}.alpha"] = torch.tensor(float(rank))
    return sd


def flat(c):
    out = {"cond": c[0][0]}
    for k, v in c[0][1].items():
        if hasattr(v, "shape"):
            out[k] = v
    return out


def same(a, b):
    a, b = flat(a), flat(b)
    return set(a) == set(b) and all(torch_equal(a[k], b[k]) for k in a)


def torch_equal(x, y):
    import torch
    return torch.equal(x.detach().cpu(), y.detach().cpu())


def gap(a, b):
    x, y = flat(a)["cond"].detach().float().cpu(), flat(b)["cond"].detach().float().cpu()
    return f"max|d| {float((x - y).abs().max()):.3e} relL2 {float((x - y).norm() / (y.norm() + 1e-12)):.3e}"


def cpu_bootstrap():
    """No GPU in the test container: ComfyUI must start in --cpu mode (before comfy.model_management is imported)."""
    import torch
    if not torch.cuda.is_available():
        import comfy.cli_args
        comfy.cli_args.args.cpu = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["ideogram4", "krea2"])
    ap.add_argument("--comfy-te", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rebuild", action="store_true", help="also build the trainer's own TE (offline rebuild) and compare keys")
    ap.add_argument("--krea2-ckpt", default="/cache/models/krea--Krea-2-Raw/raw.safetensors")
    a = ap.parse_args()
    import torch
    cpu_bootstrap()
    import comfy.model_management as mm
    import folder_paths
    import nodes
    import g56_comfy_train as T
    torch.manual_seed(0)
    torch.set_grad_enabled(False)  # numerics only (a graph through an un-checkpointed TE keeps every fp32 weight copy)
    cpu = torch.device("cpu")
    fam = a.family
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"family": fam}
    t0 = time.time()

    # ---- 1a. trainer-side TE (offline rebuild, exactly what the trainer loads) -> keys
    if a.rebuild:
        ours, rep = T.load_clip(fam, cpu)
        keys_ours = set(ours.cond_stage_model.state_dict().keys())
        ad_ours, _, info_ours = T.attach_te_lora(ours, fam, T.CFG)
        lora_keys_ours = set(ad_ours)
        print(f"[check] trainer TE rebuilt ({rep.get('fp8_layers')} fp8 layers, fingerprint_ok={rep.get('fingerprint_ok')}): "
              f"{len(keys_ours)} state-dict keys, {len(lora_keys_ours)} TE adapters, layers {info_ours['layers']} ({time.time() - t0:.0f}s)", flush=True)
        del ours, ad_ours
        gc.collect()
        mm.unload_all_models()
        gc.collect()

    # ---- 1b. evaluator TE, loaded exactly like ImageFlowAdapter.load_clip (CLIPLoader)
    folder_paths.add_model_folder_path("text_encoders", str(Path(a.comfy_te).parent))
    folder_paths.add_model_folder_path("loras", str(out))
    clip = nodes.CLIPLoader().load_clip(Path(a.comfy_te).name, fam)[0]
    keys_eval = set(clip.cond_stage_model.state_dict().keys())
    print(f"[check] evaluator TE {Path(a.comfy_te).name}: {len(keys_eval)} state-dict keys ({time.time() - t0:.0f}s)", flush=True)
    if a.rebuild:
        res["state_dict_keys_identical"] = keys_ours == keys_eval
        print(f"[check] state-dict keys trainer TE == evaluator TE: {keys_ours == keys_eval} "
              f"(only ours {sorted(keys_ours - keys_eval)[:3]}, only eval {sorted(keys_eval - keys_ours)[:3]})", flush=True)

    base_text = nodes.CLIPTextEncode().encode(clip, CAPTION)[0]
    base_empty = nodes.CLIPTextEncode().encode(clip, "")[0]
    mm.unload_all_models()

    ad, wrappers, info = T.attach_te_lora(clip, fam, T.CFG)
    te = {"clip": clip, "tokens": {}, "guidance": None, "adapters": ad, "wrappers": wrappers}
    c = {"te": te, "adapters": dict(ad)}
    res["te_adapters"] = len(ad)
    res["te_layers"] = info["layers"]
    if a.rebuild:
        res["lora_keys_identical"] = set(ad) == lora_keys_ours
        print(f"[check] TE LoRA keys trainer TE == evaluator TE: {set(ad) == lora_keys_ours}", flush=True)
    print(f"[check] {len(ad)} TE adapters on layers {info['layers']} of {info['n_layers']} (taps < {info['tap_cut']}), e.g. {sorted(ad)[0]}", flush=True)

    # ---- 2a. zero adapters (gauss init, up = 0): the per-step encode path == Comfy's CLIPTextEncode, bit for bit
    z_text, z_empty = T.te_encode(te, CAPTION, cpu), T.te_encode(te, "", cpu)
    res["zero_adapters_equal_base"] = same(z_text, base_text) and same(z_empty, base_empty)
    print(f"[check] zero TE adapters: te_encode == CLIPTextEncode (caption, ''): {res['zero_adapters_equal_base']}", flush=True)

    # ---- trained-like adapter values; gradient flows (full-size model, one backward on the caption)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for a_ in ad.values():
            a_.lora_up.weight.copy_(torch.randn(a_.lora_up.weight.shape, generator=g) * 0.01)
    n_text = T.te_encode(te, CAPTION, cpu)
    print(f"[check] bypass (training forward) vs base: {gap(n_text, base_text)}", flush=True)

    # ---- export = DiT keys of the real DiT structure + TE keys; evaluator apply_lora
    mp = meta_dit(fam, a.krea2_ckpt)
    lora_sd = dit_lora(T, fam, mp)
    n_dit = sum(1 for k in lora_sd if k.endswith(".lora_up.weight"))
    lora_sd.update(T.lora_state_dict(ad, fp32=True))
    name = f"{fam}_te_check.safetensors"
    T.save_lora(lora_sd, out / "tmp")
    os.replace(out / "tmp" / "last.safetensors", out / name)
    print(f"[check] exported {len(lora_sd)} tensors ({n_dit} DiT modules + {len(ad)} TE modules) -> {out / name}", flush=True)
    patched, cand = apply_lora_evaluator(mp, clip, name)
    res["apply_lora_model_patches"] = len(patched.patches)
    res["apply_lora_clip_patches"] = len(cand.patcher.patches)
    res["apply_lora_ok"] = len(cand.patcher.patches) == len(ad) and len(patched.patches) == n_dit
    print(f"[check] evaluator apply_lora: OK, model patches {len(patched.patches)} (DiT modules {n_dit}), clip patches "
          f"{len(cand.patcher.patches)} (TE adapters {len(ad)}) -> candidate_conditioning re-encoded: {bool(cand.patcher.patches)}", flush=True)
    n_patch, n_clip, missing = T.verify_lora_loadable_te(mp, clip, lora_sd)
    res["trainer_verify"] = {"model_patches": n_patch, "clip_patches": n_clip, "missing": len(missing)}
    print(f"[check] trainer verify_lora_loadable_te: model {n_patch} clip {n_clip} missing {missing[:2]}", flush=True)
    # a TE key the evaluator cannot map must be caught (strictness is real)
    bad = dict(lora_sd)
    bad[T.TE_PREFIX + "qwen3vl_x.transformer.model.layers.0.self_attn.q_proj.alpha"] = torch.tensor(8.0)
    _, _, missing_bad = T.verify_lora_loadable_te(mp, clip, bad)
    res["unknown_key_detected"] = bool(missing_bad)
    print(f"[check] mutation (one unknown TE key) detected: {bool(missing_bad)} {missing_bad[:1]}", flush=True)

    # ---- 2b. evaluator encode path (CLIPTextEncode on the candidate clip: merge + fp8 requant + SR at load)
    for w in wrappers:
        w.active = False
    t1 = time.time()
    e_text = nodes.CLIPTextEncode().encode(cand, CAPTION)[0]
    e_empty = nodes.CLIPTextEncode().encode(cand, "")[0]
    mm.unload_all_models()
    for w in wrappers:
        w.active = True
    print(f"[check] evaluator-path encode done ({time.time() - t1:.0f}s); eval vs base: {gap(e_text, base_text)}", flush=True)

    # ---- 2c. our exact TE path (used for every holdout eval) must equal it bit for bit
    t1 = time.time()
    saved = T.te_exact_patch_apply(c, lora_sd, cpu)
    x_text, x_empty = T.te_encode(te, CAPTION, cpu), T.te_encode(te, "", cpu)
    T.te_exact_restore(te, saved)
    res["exact_equals_evaluator"] = same(x_text, e_text) and same(x_empty, e_empty)
    print(f"[check] te_exact_patch_apply+te_encode == evaluator path (caption, ''): {res['exact_equals_evaluator']} "
          f"({len(saved)} layers merged, {time.time() - t1:.0f}s)", flush=True)
    r_text = T.te_encode(te, CAPTION, cpu)
    res["restore_ok"] = same(r_text, n_text)
    print(f"[check] after restore the training forward is back: {res['restore_ok']}", flush=True)

    # ---- 2c'. the exported file merged by Comfy's own LoRA math in fp32 WITHOUT requantization (comfy.lora.calculate_weight,
    # the lowvram-patch function) must give the training forward (bypass, alpha/rank) up to fp32 rounding: same delta,
    # same scale, same orientation -> the train/eval gap above is the requantization alone
    import comfy.lora
    import comfy.utils
    key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
    patches = comfy.lora.load_lora({k: v for k, v in lora_sd.items() if k.startswith(T.TE_PREFIX)}, key_map, log_missing=False)
    tmp = clip.patcher.clone()
    tmp.add_patches(patches, 1.0, 1.0)
    saved = []
    for key in tmp.patches:
        m = comfy.utils.get_attr(clip.cond_stage_model, key.rsplit(".", 1)[0])
        w = m.weight.to(dtype=torch.float32)  # the TE forward's own dequantization (cast_bias_weight -> fp32)
        w = w.dequantize() if hasattr(w, "dequantize") else w
        saved.append((m, m.weight))
        m.weight = torch.nn.Parameter(comfy.lora.calculate_weight(tmp.patches[key], w, key, intermediate_dtype=torch.float32),
                                      requires_grad=False)
    for w_ in wrappers:
        w_.active = False
    f_text = T.te_encode(te, CAPTION, cpu)
    T.te_exact_restore(te, saved)
    d = flat(f_text)["cond"].float() - flat(n_text)["cond"].float()
    rel = float(d.norm() / flat(n_text)["cond"].float().norm())
    res["fp32_merge_vs_bypass_relL2"] = rel
    res["scale_consistent"] = rel < 1e-4
    print(f"[check] exported delta merged in fp32 without requant (comfy calculate_weight) vs training bypass: relL2 {rel:.2e} "
          f"-> consistent: {res['scale_consistent']}", flush=True)
    res["train_vs_eval"] = gap(n_text, e_text)
    print(f"[check] train forward (bypass, no requant) vs evaluator (requant+SR): {res['train_vs_eval']}  "
          f"[delta itself: {gap(e_text, base_text)}]", flush=True)

    # ---- 2d. zero-delta file: every TE layer still requantized ("patch tax")
    zero_sd = {k: (torch.zeros_like(v) if k.endswith(".lora_up.weight") else v) for k, v in lora_sd.items()}
    saved = T.te_exact_patch_apply(c, zero_sd, cpu)
    t_text, t_empty = T.te_encode(te, CAPTION, cpu), T.te_encode(te, "", cpu)
    T.te_exact_restore(te, saved)
    res["zero_delta_tax"] = {"caption": gap(t_text, base_text), "empty": gap(t_empty, base_empty)}
    print(f"[check] zero-delta TE LoRA vs base (patch tax, CPU fp32 merge): caption {res['zero_delta_tax']['caption']}; "
          f"'' {res['zero_delta_tax']['empty']}", flush=True)
    ok = all(res.get(k, True) for k in ("state_dict_keys_identical", "lora_keys_identical", "zero_adapters_equal_base",
                                        "apply_lora_ok", "unknown_key_detected", "exact_equals_evaluator", "restore_ok",
                                        "scale_consistent")) \
        and res["trainer_verify"]["missing"] == 0
    res["ok"] = ok
    print(f"[check] RESULT {fam}: {'PASS' if ok else 'FAIL'} {json.dumps(res)} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
