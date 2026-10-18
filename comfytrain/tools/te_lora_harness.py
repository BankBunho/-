#!/usr/bin/env python3
"""G56CT_TE_LORA harness, CPU only: a tiny REAL text encoder (the family's ComfyUI TE class with the real weights of its
first N decoder layers, loaded from the evaluator's Comfy-Org file) + a tiny fake DiT, driving the trainer's own
train_loop / _run_one / holdout / export code.

--mode grad : gradient flow through the actual Comfy encode path (te_encode -> SDClipModel -> Llama2_), zero adapters ==
              CLIPTextEncode bit for bit, TE block checkpointing on == off, process_conds of the real DiT class keeps the
              graph (meta device).
--mode off  : TE LoRA off: the trainer at --old vs the current trainer -> identical evals and checkpoint bytes.
--mode on   : TE LoRA on: TE adapters get gradients and move, holdout evals re-encode through the exact TE merge, export
              carries the TE keys, drop check, warm start with/without TE keys, OOM fallback (DiT ckpt, then TE freeze).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))

CAPTIONS = ["a red fox under a crescent moon, flat vector, teal sky", "minimal logo of a coffee cup, black on white",
            "watercolor harbor at dawn with fishing boats", "isometric pixel art bakery storefront with a neon sign",
            "product photo of a green glass bottle on marble", "poster with the words SUMMER JAZZ in bold yellow letters",
            "a knight in ornate armor, oil painting, dramatic light", "cozy reading nook, warm lamp, soft film grain"]
TE_NAME = {"krea2": "qwen3vl_4b", "ideogram4": "qwen3vl_8b"}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tiny_te(fam, path, layers):
    import comfy.sd
    return comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=None, clip_type=getattr(comfy.sd.CLIPType, fam.upper()),
                              model_options={f"{TE_NAME[fam]}_model_config": {"num_hidden_layers": layers}})


class FakeMP:
    """What attach_lora / set_checkpointing need from a ModelPatcher; wrappers are applied at once (= load_models_gpu)."""

    def __init__(self, model):
        self.model = model

    def add_weight_wrapper(self, key, fn):
        import comfy.utils
        mod, pname = key.rsplit(".", 1)  # '<module>.weight' -> weight_function, '<module>.bias' -> bias_function
        m = comfy.utils.get_attr(self.model, mod)
        setattr(m, f"{pname}_function", list(getattr(m, f"{pname}_function", [])) + [fn])


def fake_dit(d_te, c_lat=4, hid=32, seed=0):
    """One krea2-named block (attn.wq on the latent, attn.wk on the text conditioning, mlp.down back to the latent)."""
    import torch
    import comfy.ops
    g = torch.Generator().manual_seed(seed)
    L = comfy.ops.manual_cast.Linear
    root = torch.nn.Module()
    root.diffusion_model = torch.nn.Module()
    blk = torch.nn.Module()
    blk.attn = torch.nn.Module()
    blk.mlp = torch.nn.Module()
    blk.attn.wq, blk.attn.wk, blk.mlp.down = L(c_lat, hid), L(d_te, hid), L(hid, c_lat)
    for m in (blk.attn.wq, blk.attn.wk, blk.mlp.down):
        with torch.no_grad():
            m.weight.copy_(torch.randn(m.weight.shape, generator=g) * m.weight.shape[1] ** -0.5)
            m.bias.zero_()
    root.diffusion_model.blocks = torch.nn.ModuleList([blk])
    return root


class FakeInner:
    """BaseModel.extra_conds contract used by process_conds (cross_attn -> c_crossattn)."""
    model_sampling = None  # only read for start/end_percent conds (none here)

    def extra_conds(self, **kw):
        import comfy.conds
        out = {}
        if kw.get("cross_attn") is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(kw["cross_attn"])
        return out


class FakeWrap:
    """The guider as seen by the trainer: .conds['positive'], .inner_model, __call__(x, sigma) -> denoised (flow x0)."""

    def __init__(self, root, predict_hook=None):
        self.root = root
        self.inner_model = FakeInner()
        self.conds = {"positive": []}
        self.train_calls = 0
        self.predict_hook = predict_hook

    def __call__(self, x, ts, **kw):
        import torch
        if torch.is_grad_enabled():  # training forwards only (holdout scoring runs under no_grad)
            self.train_calls += 1
            if self.predict_hook is not None:
                self.predict_hook(self.train_calls)
        blk = self.root.diffusion_model.blocks[0]
        ctx = self.conds["positive"][0]["model_conds"]["c_crossattn"].cond.to(x.device, torch.float32)
        B, C, H, W = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        v = blk.mlp.down(torch.nn.functional.silu(blk.attn.wq(h) + blk.attn.wk(ctx).mean(dim=1, keepdim=True)))
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x - ts.view(-1, 1, 1, 1).to(x.dtype) * v


def items_data(n=8, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    items = [{"name": f"i{i}", "caption": CAPTIONS[i % len(CAPTIONS)], "scaled": torch.randn(1, 4, 8, 8, generator=g),
              "sha": hashlib.sha256(f"img{i}".encode()).hexdigest()} for i in range(n)]
    return items


def build_and_train(T, out_dir, clip, fam, te_on, exact=False, predict_hook=None, cfg_over=None, seed=0):
    import torch
    import nodes
    torch.manual_seed(seed)
    random.seed(seed)
    cpu = torch.device("cpu")
    items = items_data()
    cfg = dict(T.CFG)
    cfg.update(max_steps=12, eval_first=6, eval_every=6, eval_every_late=6, eval_late_from=1000, holdout_strata=2,
               holdout_noises=1, eval_batch=2, seeds=1, lr=1e-3, warmup=2, ema=0.9, save_fp32=True,
               delta_avg=0, scale_search="", rebalance=0.0, anchor=0.0, rank=4, alpha=4.0, clip=1.0)
    if te_on:
        cfg.update(te_lora=True, te_rank=4, te_alpha=4.0, te_lr=5e-4, te_layers="", te_grad_ckpt=True, te_init="gauss",
                   te_drop_check=True)
    cfg.update(cfg_over or {})
    # conditioning: real tiny TE (CLIPTextEncode) when a clip is given, else fixed random tensors of the same kind
    prompts = sorted(set(it["caption"] for it in items))
    for w in getattr(clip, "_harness_wrappers", []):  # a previous run's bypass must not stay on the shared tiny TE
        w.orig.__self__.__dict__.pop("forward", None)
    if clip is not None:
        raw = {p: nodes.CLIPTextEncode().encode(clip, p)[0] for p in prompts + [""]}
    else:
        g = torch.Generator().manual_seed(7)
        raw = {p: [[torch.randn(1, 5 + len(p) % 7, 64, generator=g), {"pooled_output": None}]] for p in prompts + [""]}
    d_te = raw[""][0][0].shape[-1]
    root = fake_dit(d_te)
    mp = FakeMP(root)
    wrap = FakeWrap(root, predict_hook)
    raws, pidx = [], {}
    for p in prompts + [""]:
        pidx[p] = len(raws)
        raws.extend(raw[p])
    te = te_slots = None
    if te_on:
        te_slots = [len(raws), len(raws) + 1]
        raws.extend([raws[pidx[""]], raws[pidx[""]]])
    lat0 = items[0]["scaled"]
    conds = [T.processed_cond(wrap, [r], lat0, cpu) for r in raws]
    wrap.conds["positive"] = conds
    for it in items:
        it["cond_idx"] = pidx[it["caption"]]
    hold_ids = set() if cfg.get("harness_all_data") else {0, 3, 5}  # all-data mode (no holdout) on request
    train = [it for i, it in enumerate(items) if i not in hold_ids]
    hold = [it for i, it in enumerate(items) if i in hold_ids]
    T.CFG.update({k: v for k, v in cfg.items() if k in T.CFG})
    adapters, params = T.attach_lora(mp, T.FAMILIES["krea2"]["include"], cfg["rank"], cfg["alpha"])
    te_params = []
    if te_on:
        te = {"clip": clip, "tokens": {}, "guidance": None}
        te["adapters"], te["wrappers"], info = T.attach_te_lora(clip, fam, cfg)
        clip._harness_wrappers = te["wrappers"]
        adapters.update(te["adapters"])
        T.te_to_device(te, cpu)
        te_params = [p for a in te["adapters"].values() for p in (a.lora_up.weight, a.lora_down.weight)]
        T.set_te_checkpointing(T.te_llama(clip)[2], cfg["te_grad_ckpt"])
    c = {"exact": exact, "cfg": cfg, "params": params + te_params, "adapters": adapters, "train": train, "holdout": hold,
         "conds": conds, "empty_idx": pidx[""], "deadline": time.time() + 1e9,
         "out_dir": Path(out_dir), "mp": mp, "grad_ckpt_on": False}
    if te_on:
        c.update(te=te, te_params=te_params, te_slots=te_slots)
    t0 = time.time()
    res = T.train_loop(c, wrap, conds, {}, cpu)
    return c, res, time.time() - t0


def sd_digest(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def strip_t(evals):
    return [{k: v for k, v in e.items() if k != "t"} for e in evals]


def mode_off(a):
    """Trainer at --old vs the current trainer with the TE LoRA off: same evals, same checkpoint."""
    os.environ.pop("G56CT_TE_LORA", None)
    out = {}
    for tag, path in (("old", a.old), ("new", str(HERE / "g56_comfy_train.py"))):
        T = load_module(path, f"g56_{tag}")
        c, res, dt = build_and_train(T, Path(a.out) / f"off_{tag}", None, "krea2", te_on=False)
        f = Path(a.out) / f"off_{tag}" / "last.safetensors"
        out[tag] = {"evals": strip_t(res["evals"]), "best": {k: v for k, v in res["best"].items() if k != "t"},
                    "sd": sd_digest(res["best_sd"]), "file": hashlib.sha256(f.read_bytes()).hexdigest(), "steps": res["steps"],
                    "seconds": round(dt, 1)}
        print(f"[harness] off {tag}: steps {res['steps']} evals {len(res['evals'])} best {out[tag]['best']} "
              f"sd {out[tag]['sd'][:16]} file {out[tag]['file'][:16]}", flush=True)
    same = all(out["old"][k] == out["new"][k] for k in ("evals", "best", "sd", "file", "steps"))
    print(f"[harness] RESULT off: {'PASS' if same else 'FAIL'} (old == new with G56CT_TE_LORA unset: evals, best, state dict, "
          f"file bytes)", flush=True)
    return same


def mode_grad(a, fam, path):
    import torch
    import nodes
    import comfy.samplers
    import comfy.sampler_helpers
    import comfy.model_management as mm
    os.environ["G56CT_TE_LORA"] = "1"
    T = load_module(str(HERE / "g56_comfy_train.py"), "g56_grad")
    cpu = torch.device("cpu")
    clip = tiny_te(fam, path, a.layers)
    cm, inner, llama = T.te_llama(clip)
    print(f"[harness] tiny {fam} TE: {len(llama.layers)} real decoder layers from {Path(path).name}", flush=True)
    cap = CAPTIONS[0]
    base = nodes.CLIPTextEncode().encode(clip, cap)[0]
    base_e = nodes.CLIPTextEncode().encode(clip, "")[0]
    mm.unload_all_models()
    ad, wrappers, info = T.attach_te_lora(clip, fam, T.CFG)
    te = {"clip": clip, "tokens": {}, "guidance": None, "adapters": ad, "wrappers": wrappers}
    ok = {}
    z = T.te_encode(te, cap, cpu)
    ze = T.te_encode(te, "", cpu)
    ok["zero_equal"] = torch.equal(z[0][0], base[0][0]) and torch.equal(ze[0][0], base_e[0][0])
    print(f"[harness] {len(ad)} adapters on layers {info['layers']} (taps < {info['tap_cut']}); zero adapters: te_encode == "
          f"CLIPTextEncode bitwise: {ok['zero_equal']} (cond {tuple(z[0][0].shape)})", flush=True)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for x in ad.values():
            x.lora_up.weight.copy_(torch.randn(x.lora_up.weight.shape, generator=g) * 0.01)

    def grads(ckpt):
        T.set_te_checkpointing(llama, ckpt)
        for x in ad.values():
            x.zero_grad(set_to_none=True)
        out = T.te_encode(te, cap, cpu)
        loss = out[0][0].float().square().mean()
        loss.backward()
        return out, {k: (x.lora_up.weight.grad.clone(), x.lora_down.weight.grad.clone()) for k, x in ad.items()}

    n_on, g_on = grads(True)
    n_off, g_off = grads(False)
    nz = [k for k, (gu, gd) in g_on.items() if float(gu.norm()) > 0 and float(gd.norm()) > 0]
    ok["all_adapters_nonzero_grad"] = len(nz) == len(ad)
    ok["ckpt_equal"] = all(torch.equal(g_on[k][0], g_off[k][0]) and torch.equal(g_on[k][1], g_off[k][1]) for k in ad) \
        and torch.equal(n_on[0][0], n_off[0][0])
    d = (n_on[0][0].detach() - base[0][0]).norm() / base[0][0].norm()
    ok["cond_changes"] = float(d) > 0
    print(f"[harness] nonzero adapters: cond rel change {float(d):.3e}; adapters with nonzero up AND down grads "
          f"{len(nz)}/{len(ad)}; grads + cond with TE checkpointing == without (bitwise): {ok['ckpt_equal']}", flush=True)
    gl = {}
    for k, (gu, gd) in g_on.items():
        layer = int(k.split(".layers.")[1].split(".")[0])
        gl[layer] = gl.get(layer, 0.0) + float(gu.norm()) + float(gd.norm())
    print(f"[harness] grad norm per TE layer: { {k: round(v, 6) for k, v in sorted(gl.items())} }", flush=True)
    # the graph survives Comfy's own conditioning processing of the real DiT class (meta device) and its input cast
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from te_lora_check import meta_dit
    mp = meta_dit(fam, a.krea2_ckpt)
    for x in ad.values():
        x.zero_grad(set_to_none=True)
    raw = T.te_encode(te, cap, cpu)
    lat = torch.zeros(1, 16 if fam == "krea2" else 32, 8, 8)
    pc = comfy.samplers.process_conds(mp.model, torch.zeros_like(lat), {"positive": comfy.sampler_helpers.convert_cond(raw)},
                                      cpu, lat, None, 42)["positive"][0]
    ctx = pc["model_conds"]["c_crossattn"].cond
    ctx = mm.cast_to_device(ctx, cpu, torch.bfloat16)  # BaseModel._apply_model's context cast
    ctx.float().square().mean().backward()
    n2 = sum(1 for x in ad.values() if x.lora_up.weight.grad is not None and float(x.lora_up.weight.grad.norm()) > 0)
    ok["process_conds_graph"] = n2 == len(ad)
    print(f"[harness] real {type(mp.model).__name__}.extra_conds + process_conds + context cast keep the graph: "
          f"{n2}/{len(ad)} adapters get gradients", flush=True)
    res = all(ok.values())
    print(f"[harness] RESULT grad {fam}: {'PASS' if res else 'FAIL'} {ok}", flush=True)
    return res


def mode_on(a, fam, path):
    import torch
    import comfy.model_management as mm
    from safetensors.torch import load_file
    os.environ["G56CT_TE_LORA"] = "1"
    T = load_module(str(HERE / "g56_comfy_train.py"), "g56_on")
    calls = {"te_exact": 0, "drop": 0}
    orig_exact, orig_drop = T.te_exact_patch_apply, T.te_drop_check

    def cnt_exact(*x, **k):
        calls["te_exact"] += 1
        return orig_exact(*x, **k)

    def cnt_drop(*x, **k):
        calls["drop"] += 1
        return orig_drop(*x, **k)
    T.te_exact_patch_apply, T.te_drop_check = cnt_exact, cnt_drop
    # the fake DiT has no Comfy key map: its (unchanged, production-tested) exact DiT merge is stubbed; the TE merge is real
    T.exact_patch_apply = lambda c_, sd_, dev, **k: []  # (do_eval passes strict=)
    clip = tiny_te(fam, path, a.layers)
    mm.unload_all_models()
    ok = {}
    # run 1: plain TE LoRA training (non-exact DiT like krea2; TE part always exact)
    init_te = None
    c, res, dt = build_and_train(T, Path(a.out) / f"on_{fam}", clip, fam, te_on=True)
    ta = T.te_adapters(c)
    moved = sum(1 for x in ta.values() if float(x.lora_up.weight.detach().abs().max()) > 0)
    ok["te_adapters_moved"] = moved == len(ta)
    ok["evals_ran"] = len(res["evals"]) >= 4 and calls["te_exact"] >= len(res["evals"])
    f = Path(a.out) / f"on_{fam}" / "last.safetensors"
    sd = load_file(str(f))
    n_te = sum(1 for k in sd if k.startswith(T.TE_PREFIX) and k.endswith(".lora_up.weight"))
    ok["drop_check_ran"] = calls["drop"] == 1
    kept = n_te == len(ta)
    ok["export_consistent"] = kept or n_te == 0
    print(f"[harness] on {fam}: {res['steps']} steps in {dt:.0f}s, evals {len(res['evals'])} (TE exact merges {calls['te_exact']}), "
          f"TE adapters moved {moved}/{len(ta)}, best {res['best']}, file TE modules {n_te} ({'kept' if kept else 'dropped by the drop check'})", flush=True)
    for e in res["evals"]:
        print(f"[harness]   eval step {e['step']} {e['tag']}: {e['score']:.6f} (text {e['text']:.6f} no_text {e['no_text']:.6f})", flush=True)
    # run 2: exact-DiT branch of do_eval (DiT merge stubbed as above) + injected OOMs -> DiT ckpt, then TE freeze
    snap = {}

    def hook(n):
        if n in (3, 8):
            raise torch.cuda.OutOfMemoryError("injected")
        if n == 9:  # first forward after the TE freeze
            snap.update({k: v.detach().clone() for k, v in T.lora_state_dict(T.te_adapters(ctx2[0]), fp32=True).items()})
    ctx2 = [None]
    orig_run = T._run_one

    def run_one(c_, *x, **k):
        ctx2[0] = c_
        return orig_run(c_, *x, **k)
    T._run_one = run_one
    c2, res2, dt2 = build_and_train(T, Path(a.out) / f"oom_{fam}", clip, fam, te_on=True, exact=True, predict_hook=hook, seed=1)
    after = T.lora_state_dict(T.te_adapters(c2), fp32=True)
    ok["oom_fallback"] = bool(c2.get("te_frozen")) and c2["grad_ckpt_on"] and bool(snap) and all(torch.equal(snap[k], after[k]) for k in snap)
    ok["exact_branch_evals"] = len(res2["evals"]) >= 4
    print(f"[harness] OOM injected twice: DiT ckpt on {c2['grad_ckpt_on']}, TE frozen {bool(c2.get('te_frozen'))}, TE delta "
          f"unchanged after freeze {bool(snap) and all(torch.equal(snap[k], after[k]) for k in snap)}; exact-branch evals "
          f"{len(res2['evals'])}, {res2['steps']} steps", flush=True)
    r = all(ok.values())
    print(f"[harness] RESULT on {fam}: {'PASS' if r else 'FAIL'} {ok}", flush=True)
    return r


def cpu_bootstrap():
    """No GPU in the test container: ComfyUI must start in --cpu mode (before comfy.model_management is imported)."""
    import torch
    if not torch.cuda.is_available():
        import comfy.cli_args
        comfy.cli_args.args.cpu = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["grad", "off", "on"])
    ap.add_argument("--family", default="krea2", choices=["krea2", "ideogram4"])
    ap.add_argument("--comfy-te", default=None)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--old", default=None)
    ap.add_argument("--out", default="/tmp/te_harness")
    ap.add_argument("--krea2-ckpt", default="/cache/models/krea--Krea-2-Raw/raw.safetensors")
    a = ap.parse_args()
    cpu_bootstrap()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.mode == "off":
        ok = mode_off(a)
    elif a.mode == "grad":
        ok = mode_grad(a, a.family, a.comfy_te)
    else:
        ok = mode_on(a, a.family, a.comfy_te)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
