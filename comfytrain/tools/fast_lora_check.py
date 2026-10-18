#!/usr/bin/env python3
"""G56CT_FAST_LORA check, CPU only (no GPU, no model files): the factored LoRA backward the trainer installs with
G56CT_FAST_LORA=1 (fast_lora_install / fast_lora_guard) against the old weight-function path (knob off).

default  with no G56CT_* env the trainer's FAST_LORA is off, and a default te_lora_harness run with the base frozen (as
   main() does) installs nothing (no fast state, 0 fast forwards). knobs_check cannot see this knob: its harness leaves
   the base trainable, where the fast path refuses anyway (29 Sep review: a default-ON mutant passed knobs_check).
ops  one LoRA Linear per ComfyUI op class the image families train on (rank 16, alpha 32 -> s = 2; nonzero up and down):
   bf16 disable_weight_init (krea2 / z-image / flux bf16), manual_cast bf16, manual_cast fp32 weight -> bf16 compute,
   manual_cast float8_e4m3fn -> bf16 (qwen-image fp8 checkpoint), mixed_precision_ops QuantizedTensor fp8 scale 1
   (ideogram4 evalq; fp8 compute disabled as in the trainer), fp8_ops fp8 weight (the class G56CT_FP8_COMPUTE=1 picks):
   a) forward byte-identical; grad x / up / down: rel L2 <= 2e-2 and cos >= 0.9998 (bf16 compute, both paths round);
      the fast path really ran (1 call); knob on with the switch off is bit-identical to the knob off;
   b) bytes autograd keeps for backward (torch.autograd.graph.saved_tensors_hooks; parameters and LoRA factors excluded,
      one count per storage) drop by out*in*2 (+ at most 64 B: LoraDiff's 0-dim scale) and no out x in tensor is kept;
   c) a token-shared (DC) input component (|c| = 10 |rest| per token) with an output gradient centred over the tokens
      (attention keys: softmax shift invariance): the fast grad_U / grad_D error against an fp64 straight-through
      reference <= 2x the old path's (the old path forms gy^T x in fp32 first; a bf16 tokens x r intermediate was 12x
      worse, 29 Sep review), and without the DC component too;
   d) a LoRA Linear that also carries a SMALL bias delta (bias_function): no fast call, every output and gradient incl.
      the bias delta's bit-identical to FAST=0;
   e) the CUDA-only GEMM pieces emulated on CPU: _mm32_probe on the backward's operand layouts accepts an exact
      mm(out_dtype=float32) and rejects a rounding or missing one; the hi + lo split keeps an fp32 intermediate to 1e-5;
models  tiny real ComfyUI DiTs of every image family with the trainer's production LoRA set: krea2 SingleStreamDiT
   (2 blocks), flux (1 double + 1 single block, patch_flux_inplace), qwen-image (2 blocks, full module set), z-image
   NextDiT with z-image modulation (include '.*': embedders, refiners, final layer), ideogram4 as the trainer loads it
   (mixed_precision_ops, every Linear an fp8 QuantizedTensor with scale 1); bf16 compute: forward output bit-identical;
   every factor grad within rel 5e-2 / cos >= 0.999 (whole-model: rounding differences propagate), aggregate rel <= 2e-2
   (a module whose output never reaches the loss - qwen's last text-stream block - has no gradient on any path); saved
   bytes drop by sum(out*in*2) over the LoRA Linears whose input requires grad (the old path keeps W' only there; + at
   most 8 B per module); the fast path under the trainer's non-reentrant block checkpointing (set_checkpointing) gives
   the same grads as without (rel <= 1e-6);
guard  the runtime guard on the te_lora_harness fake DiT (fp32, rank 4 alpha 8 -> s = 2, 40 steps, base frozen as in
   main()): passes with local statistics from every LoRA module, the model outputs of its two passes are bit-identical,
   factors / grads / global RNG / checkpointing state are restored; the FAST=1 run's evals within rel 1e-4 of the FAST=0
   run; trainable base weights (the harness default): no module may take the fast path -> the guard refuses and the run
   is byte-identical to FAST=0; one module failing its first-use check (_fast_verify) stays on the old path while the
   guard passes on the others;
deep  the guard at production depth: krea2 SingleStreamDiT with 28 blocks (width 128) and the production module set
   (include + include_extra: 264 modules) at rank 16 / 64, 2 / 12 text layers, adaLN shift 0 / 10, several seeds; flux
   19 double + 38 single blocks (228 modules): PASS with the local statistics well inside the tolerance (min cos >=
   0.9999, max |ratio - 1| <= 5e-3). Before, the strict test compared the two passes' whole-model gradients and failed
   2 of 3 seeds at rank 16 (bf16 rounding of grad_x propagated through the depth, 29 Sep review);
fallback  an exception in a fast forward / backward switches the path off and raises FastLoRAFallback: on a tiny krea2
   with block checkpointing off and on, the step redone on the old path (after _fast_retry, as _run_one does, while the
   exception is still referenced - ComfyUI's calc_cond_batch keeps it in a list) gives exactly the FAST=0 gradients (the
   former in-forward fallback made the checkpoint recomputation fail with CheckpointError, and so does a redo without
   releasing the dead checkpoint frame's saved-tensor hooks); a harness training run with an injected failure in a fast
   forward / backward finishes all its steps (the step is redone, 1 retry, the path stays off);
merge  G56CT_FAST_MERGE (default off; only with FAST_LORA): _fm_merge == comfy.ops.cast_bias_weight + LoraDiff bit for bit
   (W' and bias) on every op class x s in {1, 2, 0.75} x |sUD| in {1e-3, 1} x |W| (256 x 384), and a forward + backward
   through the armed, verified module == FM=0; tiny models of every family, 20 AdamW steps from the gauss init (s = 2,
   armed after step 1 as in _run_one): every loss and the final factors == FM=0, every module with a gradient verified ok;
   the harness run (guard + _run_one arming) == FAST=1 FM=0 byte for byte, no merge during the guard or step 1; a wrong
   merge (fixture: truncation instead of rounding) is caught by the per-module check (modules off, results == FM=0);
   _fm_verify: a zero delta proves nothing (pending, off after _FM_TRIVIAL_MAX checks);
release  review B1 (1 Oct): an OOM / FastLoRAFallback raised inside loss.backward() in a harness training run, in each of
   _run_one's four retry handlers (per step / accumulated, OOM / fallback) and with the block checkpointed (OOM, then a
   fallback with checkpointing on): the saved tensors of the failed forward (saved_tensors_hooks, weakrefs) alive at the
   retry's first forward <= a normal step's at its next forward (was 4 vs 1: den / v / loss kept the dead graph); every
   step made;
release_raise  review A F3 (1 Oct): _fast_release raising in the guard's except -> the guard returns False and the run is
   byte-identical to FAST=0; raising in _fast_retry -> the step is still redone and every step made;
mutations (source level, on a copy of the trainer text): each must make the check named in MUTATIONS fail first.
Usage (trainer image or CPU env with COMFY_ROOT): python comfytrain/tools/fast_lora_check.py [--mutation NAME] [--quick]
  --mutation NAME : run the checks on that mutant only, stop at the first failing one (must print RESULT FAIL)
  --quick         : skip the mutation runs
  --only a,b      : run only these checks (default,ops,models,guard,merge,deep,fallback,release,release_raise)
"""
import argparse
import hashlib
import os
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER = HERE.parent / "g56_comfy_train.py"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
for _k in [k for k in os.environ if k.startswith("G56CT_")]:
    os.environ.pop(_k)

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
import te_lora_harness as H  # noqa: E402

H.cpu_bootstrap()
import comfy.ops  # noqa: E402
import comfy.model_management as mm  # noqa: E402

# name -> ([(pattern, replacement), ...] each pattern exactly once in the trainer, the check that must fail first)
MUTATIONS = {
    # autograd keeps W' again (the memory the knob exists to save)
    "save_w": ([("ctx.save_for_backward(x, up, down)", "ctx.save_for_backward(x, up, down, w_merged)")], "ops"),
    # grad_x without the low-rank term (gy U)(s D)
    "no_lowrank_gx": ([("gx = torch.mm(gyu.to(dt), (d * s).to(dt)).addmm_(gy2, _fast_base_weight(m, dt)).reshape(x.shape)",
                        "gx = torch.mm(gy2, _fast_base_weight(m, dt)).reshape(x.shape)")], "ops"),
    # s = 1 in the backward
    "no_scale": ([("s = (ad.alpha / ad.rank).detach()", "s = (ad.alpha / ad.alpha).detach()")], "ops"),
    # the knob on by default (knobs_check does not see it: 29 Sep review)
    "default_on": ([('fast_lora=env("FAST_LORA", False, bool)', 'fast_lora=env("FAST_LORA", True, bool)')], "default"),
    # a Linear with a bias function (SMALL bias delta) takes the fast path: the bias delta gets no gradient
    "elig_bias": ([("len(fns) != 1 or fns[0] is not ad or len(m.bias_function) > 0 or w is None",
                    "len(fns) != 1 or fns[0] is not ad or w is None")], "ops"),
    # the tokens x r intermediates rounded to the compute dtype before the token contraction (b2fc680 precision)
    "lowp_mid": ([("gu = _mm32_hl(gy2.t(), _mm32(x2, d.to(dt).t())).mul_(s)",
                   "gu = _mm32_hl(gy2.t(), _mm32(x2, d.to(dt).t()).to(dt)).mul_(s)"),
                  ("gd = _mm32_hl(gyu.t(), x2).mul_(s)", "gd = _mm32_hl(gyu.to(dt).t(), x2).mul_(s)")], "ops"),
    # the strict test on the two passes' whole-model gradients per group (the b2fc680 guard design)
    "guard_strict_e2e": ([("if not fin or not lo <= ratio <= hi:",
                           "if not fin or not lo <= ratio <= hi or cos < 0.999 or abs(ratio - 1.0) > 0.01:")], "deep"),
    # no local statistics collected (the guard then has no strict test)
    "guard_no_local": ([("st.probe = local if on else None", "st.probe = None")], "guard"),
    # a failing fast forward falls back inside the forward (b2fc680): breaks checkpoint recomputation
    "fallback_swallow": ([('raise FastLoRAFallback(f"fast lora forward failed in {name}: {type(e).__name__}: '
                           '{str(e)[:120]}") from e', "pass")], "fallback"),
    # the redone step without releasing the failed checkpoint frame: its saved-tensor hooks are still pushed
    "no_release": ([("    n = _fast_release(e)\n", "    n = 0\n")], "fallback"),
    # a module that _fast_verify left on the old path makes the guard fail (it has a gradient but no local check)
    "no_exempt": ([('    unchecked = sorted(with_grad - local.get("_modules", set()) - set(exempt))',
                    '    unchecked = sorted(with_grad - local.get("_modules", set()))')], "guard"),
    # ---- G56CT_FAST_MERGE
    # the knob on by default (krea2 trains with FAST_LORA by default: the merge would change production)
    "fm_default_on": ([('fast_merge=env("FAST_MERGE", False, bool)', 'fast_merge=env("FAST_MERGE", True, bool)')], "default"),
    # the scale alpha / rank skipped
    "fm_noscale": ([("    if s is not None:\n        diff.mul_(s)\n", "    pass\n")], "merge"),
    # U @ D in the compute dtype instead of fp32
    "fm_bf16diff": ([("    diff = ad.lora_up.weight @ ad.lora_down.weight\n    if s is not None:",
                      "    diff = (ad.lora_up.weight.to(dt) @ ad.lora_down.weight.to(dt)).float()\n    if s is not None:")],
                    "merge"),
    # fused W + s * diff (one rounding less)
    "fm_fused": ([("    if s is not None:\n        diff.mul_(s)\n    w = torch.empty(W.shape, dtype=dt, device=W.device)\n"
                   "    torch.add(W, diff.reshape(W.shape), out=w)\n",
                   "    w = torch.empty(W.shape, dtype=dt, device=W.device)\n"
                   "    torch.add(W, diff.reshape(W.shape), alpha=1.0 if s is None else float(s), out=w)\n")], "merge"),
    # W + s U D in the GEMM epilogue (addmm)
    "fm_addmm": ([("    diff = ad.lora_up.weight @ ad.lora_down.weight\n    if s is not None:\n        diff.mul_(s)\n"
                   "    w = torch.empty(W.shape, dtype=dt, device=W.device)\n    torch.add(W, diff.reshape(W.shape), out=w)\n",
                   "    w = torch.addmm(W.float(), ad.lora_up.weight, ad.lora_down.weight,\n"
                   "                    alpha=1.0 if s is None else float(s)).to(dt)\n")], "merge"),
    # W' cached across steps
    "fm_cache": ([("    W = _fast_base_weight(m, dt)\n    diff = ad.lora_up.weight @ ad.lora_down.weight\n",
                   "    if getattr(m, '_g56_fmc', None) is not None:\n        return m._g56_fmc\n"
                   "    W = _fast_base_weight(m, dt)\n    diff = ad.lora_up.weight @ ad.lora_down.weight\n"),
                  ("    return w, b, W\n", "    m._g56_fmc = (w, b, W)\n    return w, b, W\n")], "merge"),
    # armed at install: the merge runs in the guard and in step 1 (delta 0 there: its checks prove nothing)
    "fm_arm_install": ([("    st.fm_on = bool(CFG.get(\"fast_merge\"))\n    return True\n",
                         "    st.fm_on = bool(CFG.get(\"fast_merge\"))\n    st.fm_armed = st.fm_on\n    return True\n")],
                       "merge"),
    # a zero-delta check (W' == W) accepted as proof
    "fm_trivial": ([("        elif torch.equal(w_ref, W):\n", "        elif False:\n")], "merge"),
    # no comparison with Comfy's merge (every checked module goes ok)
    "fm_noverify": ([("        same = torch.equal(w2, w_ref) and (b2 is None) == (b_ref is None) and (b2 is None or "
                      "torch.equal(b2, b_ref))\n", "        same = True\n")], "merge"),
    # ---- retries (1 Oct reviews)
    # the four retry handlers keep den / v / loss (gsum / ssum): the failed step's graph is alive in the retry (B1)
    "keep_graph": ([("                den = v = loss = gsum = ssum = None\n", "                pass\n"),
                    ("                den = v = loss = gsum = ssum = None  # the failed group's graph",
                     "                pass  # the failed group's graph"),
                    ("                den = v = loss = None\n", "                pass\n"),
                    ("                den = v = loss = None  # the failed step's graph",
                     "                pass  # the failed step's graph")], "release"),
    # _fast_release unguarded in the guard's except and in _fast_retry (review A F3)
    "release_unguarded": ([("        try:  # both passes run checkpointed: a raise inside a block must not leave its hooks "
                            "behind\n            _fast_release(e)\n        except Exception as e2:",
                            "        _fast_release(e)\n        if False:"),
                           ("    try:\n        n = _fast_release(e)\n    except Exception as e2:",
                            "    n = _fast_release(e)\n    if False:")], "release_raise"),
}
CHECKS = ("default", "ops", "models", "guard", "merge", "deep", "fallback", "release", "release_raise")
BF = torch.bfloat16
OPS = ["bf16 disable_weight_init", "manual_cast bf16", "manual_cast fp32->bf16", "manual_cast fp8->bf16 (qwen fp8)",
       "QuantizedTensor fp8 (ideogram4 evalq)", "fp8_ops fp8->bf16 (G56CT_FP8_COMPUTE=1 class)"]
LOGS = []


def say(*a):
    print("[fast-lora]", *a, flush=True)


def load_trainer(mutation=None):
    """The trainer module from its source text, optionally with one mutation (every pattern must occur exactly once)."""
    src = TRAINER.read_text()
    if mutation:
        for a, b in MUTATIONS[mutation][0]:
            if src.count(a) != 1:
                raise SystemExit(f"[fast-lora] mutation {mutation}: pattern occurs {src.count(a)}x, not exactly once: {a}")
            src = src.replace(a, b)
    mod = types.ModuleType(f"g56fl_{mutation or 'code'}")
    mod.__file__ = str(TRAINER)
    exec(compile(src, f"{TRAINER}{f' [mutation {mutation}]' if mutation else ''}", "exec"), mod.__dict__)
    mod.log = lambda *a, **k: LOGS.append(" ".join(str(x) for x in a))
    mod._cfg0 = dict(mod.CFG)
    calls = {"n": 0}
    orig = mod._fast_forward

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    mod._fast_forward = counted  # the installed wrappers look it up at call time
    mod._fast_calls = calls
    return mod


def reset_cfg(T, **over):
    T.CFG.clear()
    T.CFG.update(T._cfg0)
    T.CFG.update(init="gauss", **over)


def skey(t):
    try:
        return t.untyped_storage().data_ptr()
    except Exception:  # noqa: BLE001  (tensor subclasses)
        return t.data_ptr()


def run_saving(fn, keep):
    """fn() under saved_tensors_hooks: {storage: (bytes, shape, dtype)} of what autograd keeps for backward, without the
    storages in `keep` (parameters, LoRA factors)."""
    seen = {}

    def pack(t):
        k = skey(t)
        if k not in keep and k not in seen:
            try:
                nb = t.untyped_storage().nbytes()
            except Exception:  # noqa: BLE001
                nb = t.numel() * t.element_size()
            seen[k] = (nb, tuple(t.shape), t.dtype)
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, seen


def nbytes(seen):
    return sum(v[0] for v in seen.values())


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))


def wrap_root(lin):
    root = torch.nn.Module()
    root.diffusion_model = torch.nn.Module()
    blk = torch.nn.Module()
    blk.attn = torch.nn.Module()
    blk.attn.wq = lin
    root.diffusion_model.blocks = torch.nn.ModuleList([blk])
    return root


def build_linear(kind, I, O, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(O, I, generator=g) * I ** -0.5
    b = torch.randn(O, generator=g) * 0.01
    if kind == OPS[4]:  # the trainer disables fp8 compute -> pick_operations lists the fp8 formats as disabled
        ops = comfy.ops.mixed_precision_ops({}, compute_dtype=BF, disabled={"float8_e4m3fn", "float8_e5m2", "nvfp4", "mxfp8"})
        lin = ops.Linear(I, O, bias=True)
        lin.load_state_dict({"weight": w.clamp(-448, 448).to(torch.float8_e4m3fn), "weight_scale": torch.tensor(1.0),
                             "comfy_quant": torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8),
                             "bias": b.to(BF)})
        assert isinstance(lin.weight, comfy.ops.QuantizedTensor), "QuantizedTensor weight expected"
    else:
        cls = {OPS[0]: comfy.ops.disable_weight_init.Linear, OPS[5]: comfy.ops.fp8_ops.Linear}.get(kind, comfy.ops.manual_cast.Linear)
        wdt = {OPS[0]: BF, OPS[1]: BF, OPS[2]: torch.float32, OPS[3]: torch.float8_e4m3fn, OPS[5]: torch.float8_e4m3fn}[kind]
        lin = cls(I, O, bias=True, dtype=BF)
        lin.weight = torch.nn.Parameter(w.to(wdt), requires_grad=False)
        lin.bias = torch.nn.Parameter(b.to(torch.float32 if wdt == torch.float32 else BF), requires_grad=False)
    lin.requires_grad_(False)
    return wrap_root(lin)


def attach(T, root, fast, include, rank, alpha, up_std, seed=1):
    reset_cfg(T, fast_lora=fast)
    mp = H.FakeMP(root)
    ads, params = T.attach_lora(mp, include, rank, alpha)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for k in sorted(ads):
            a = ads[k]
            a.lora_up.weight.copy_(torch.randn(a.lora_up.weight.shape, generator=g) * up_std)
            a.lora_down.weight.copy_(torch.randn(a.lora_down.weight.shape, generator=g) * a.lora_down.weight.shape[1] ** -0.5)
    return mp, ads, params


def keep_of(root, params):
    return {skey(p) for p in list(root.parameters()) + list(params)}


# ---------------------------------------------------------------------------------------------------------- default
BASE = dict(case_cycle=True, sigma_mode="mid", max_steps=40, alpha=8.0)  # harness rank 4 -> s = 2


def harness_run(T, fast, tag, guard_probe=None, freeze=True, inject=None, verify_fail=None, extra=None, predict_hook=None,
                on_inject=None):
    """te_lora_harness training run. fast=None: the trainer's own default (no override). freeze: the fake DiT's base
    weights / biases frozen before the LoRA attaches, as main() does (mp.model.requires_grad_(False)); the harness itself
    leaves them trainable (the fast path then refuses). inject: (where, n[, kind]) or a list of them: the n-th fast forward
    / backward after the guard passed (the training loop's calls; counted while the path is on) raises - kind 'error'
    (default; RuntimeError -> FastLoRAFallback) or 'oom' (torch.cuda.OutOfMemoryError, passed on as it is);
    on_inject(where, n, kind) runs just before. verify_fail: this module's first-use check (_fast_verify) fails (it keeps
    the old path, st.bad)."""
    out = Path(tempfile.mkdtemp(prefix=f"fastlora_{tag}_"))
    reset_cfg(T)
    orig_guard, orig_attach, orig_fwd, orig_bwd = T.fast_lora_guard, T.attach_lora, T._fast_forward, T._fast_backward
    orig_verify = T._fast_verify
    if verify_fail is not None:
        T._fast_verify = lambda m, *a, **k: False if m is T._fl_target else orig_verify(m, *a, **k)
    if guard_probe is not None:
        T.fast_lora_guard = lambda *a, **k: guard_probe(orig_guard, *a, **k)
    if freeze or verify_fail is not None:
        def attach_frozen(mp, *a, **k):
            if freeze:
                mp.model.requires_grad_(False)
            if verify_fail is not None:
                T._fl_target = mp.model.get_submodule(verify_fail)
            return orig_attach(mp, *a, **k)
        T.attach_lora = attach_frozen
    if inject is not None:
        plan = list(inject) if isinstance(inject, list) else [inject]
        cnt = {"forward": 0, "backward": 0}

        def maybe_raise(where, st):
            if str(st.verdict).startswith("pass") and st.on:
                cnt[where] += 1
                for p in plan:
                    if p[0] == where and p[1] == cnt[where]:
                        kind = p[2] if len(p) > 2 else "error"
                        if on_inject is not None:
                            on_inject(where, cnt[where], kind)
                        if kind == "oom":
                            raise torch.cuda.OutOfMemoryError(f"injected {where} out of memory")
                        raise RuntimeError(f"injected {where} failure")
        if any(p[0] == "forward" for p in plan):
            def fwd_inj(m, ad, x, args, kwargs, rec):
                maybe_raise("forward", rec[2])
                return orig_fwd(m, ad, x, args, kwargs, rec)
            T._fast_forward = fwd_inj
        if any(p[0] != "forward" for p in plan):
            def bwd_inj(rec, *a, **k):
                maybe_raise("backward", rec[2])
                return orig_bwd(rec, *a, **k)
            T._fast_backward = bwd_inj
    n0 = T._fast_calls["n"]
    over = dict(BASE) if fast is None else dict(BASE, fast_lora=fast)
    over.update(extra or {})
    try:
        c, res, _ = H.build_and_train(T, out, None, "krea2", te_on=False, cfg_over=over, predict_hook=predict_hook)
    finally:
        T.fast_lora_guard, T.attach_lora, T._fast_forward, T._fast_backward = orig_guard, orig_attach, orig_fwd, orig_bwd
        T._fast_verify = orig_verify
    f = out / "last.safetensors"
    return {"evals": H.strip_t(res["evals"]), "sd": H.sd_digest(res["best_sd"]), "steps": res["steps"],
            "file": hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None,
            "st": getattr(c["mp"], "_g56_fast", None), "calls": T._fast_calls["n"] - n0}


def check_default(T, ref):
    d0 = T._cfg0.get("fast_lora") is False and T._cfg0.get("fast_merge") is False
    r = harness_run(T, None, "default")
    same = all(r[k] == ref[k] for k in ("file", "sd", "evals", "steps"))
    good = d0 and r["st"] is None and r["calls"] == 0 and same
    say(f"default: CFG fast_lora {T._cfg0.get('fast_lora')!r} fast_merge {T._cfg0.get('fast_merge')!r} with no env; "
        f"default harness run (base frozen) fast state {r['st'] is not None}, fast forwards {r['calls']}, byte-identical "
        f"to FAST=0 {same} -> {'ok' if good else 'FAIL'}")
    return good


# ---------------------------------------------------------------------------------------------------------- ops
def check_ops(T, ref=None):
    I, O = 256, 384
    inc = T.FAMILIES["krea2"]["include"]
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(2, 48, I, generator=g)).to(BF)
    gy = torch.randn(2, 48, O, generator=g)
    ok = True

    def run(kind, fast, switch_off=False):
        root = build_linear(kind, I, O)
        mp, ads, params = attach(T, root, fast, inc, 16, 32.0, 0.05)
        if switch_off:
            mp._g56_fast.on = False
        lin = root.diffusion_model.blocks[0].attn.wq
        xx = x.clone().requires_grad_(True)
        n0 = T._fast_calls["n"]
        y, seen = run_saving(lambda: lin(xx), keep_of(root, params))
        (y.float() * gy).sum().backward()
        ad = ads["diffusion_model.blocks.0.attn.wq"]
        return {"y": y.detach(), "gx": xx.grad, "gu": ad.lora_up.weight.grad.clone(), "gd": ad.lora_down.weight.grad.clone(),
                "seen": seen, "calls": T._fast_calls["n"] - n0, "st": getattr(mp, "_g56_fast", None),
                "installed": "forward_comfy_cast_weights" in lin.__dict__}

    for kind in OPS:
        r0, r1, roff = run(kind, False), run(kind, True), run(kind, True, switch_off=True)
        y_same = torch.equal(r1["y"], r0["y"])
        errs = {k: (rel(r1[k], r0[k]), cosine(r1[k], r0[k])) for k in ("gx", "gu", "gd")}
        grads_ok = all(e <= 2e-2 and c >= 0.9998 for e, c in errs.values())
        off_same = all(torch.equal(roff[k], r0[k]) for k in ("y", "gx", "gu", "gd")) and nbytes(roff["seen"]) == nbytes(r0["seen"])
        drop = nbytes(r0["seen"]) - nbytes(r1["seen"])
        want = O * I * 2
        wshape = any(v[1] == (O, I) for v in r1["seen"].values())
        bytes_ok = want <= drop <= want + 64 and not wshape
        ran = r1["calls"] == 1 and r1["st"] is not None and r1["st"].on and not r1["st"].bad and r1["installed"] \
            and not r0["installed"] and r0["st"] is None and roff["calls"] == 0
        good = y_same and grads_ok and off_same and bytes_ok and ran
        ok &= good
        say(f"op {kind}: forward identical {y_same}; rel/cos x {errs['gx'][0]:.2e}/{errs['gx'][1]:.6f} up "
            f"{errs['gu'][0]:.2e}/{errs['gu'][1]:.6f} down {errs['gd'][0]:.2e}/{errs['gd'][1]:.6f}; switch-off == knob-off "
            f"{off_same}; saved bytes old {nbytes(r0['seen'])} new {nbytes(r1['seen'])} drop {drop} (out*in*2 = {want}), "
            f"out x in kept {wshape}; fast path ran {ran} -> {'ok' if good else 'FAIL'}")
    ok &= check_dc(T)
    ok &= check_smallbias(T)
    ok &= check_mm32(T)
    return ok


def check_mm32(T):
    """ops e): the CUDA-only GEMM pieces, emulated on CPU. _mm32_probe on the backward's operand layouts returns True for
    an exact out_dtype mm and a reason for a rounding (compute-dtype result) or a missing one; _mm32_hl keeps an fp32
    tokens x r intermediate with a large DC part to <= 1e-5 (the same intermediate rounded to bf16 first: > 1e-4)."""
    real = torch.mm

    def exact(a, b, *, out_dtype=None, out=None):
        return real(a, b) if out_dtype is None else real(a.float(), b.float()).to(out_dtype)

    def rounding(a, b, *, out_dtype=None, out=None):
        return real(a, b) if out_dtype is None else real(a, b).to(out_dtype)

    def missing(a, b, *, out_dtype=None, out=None):
        if out_dtype is not None:
            raise NotImplementedError("aten::mm.dtype on CPU")
        return real(a, b)
    res = {}
    try:
        for tag, fake in (("exact", exact), ("rounding", rounding), ("missing", missing)):
            torch.mm = fake
            res[tag] = T._mm32_probe(BF, torch.device("cpu"))
    finally:
        torch.mm = real
    g = torch.Generator().manual_seed(3)
    a = torch.randn(64, 2048, generator=g).to(BF)                 # gy^T (out x tokens)
    b = torch.randn(2048, 16, generator=g) * 0.5 + 10.0           # x D^T (tokens x r), fp32, token-shared part 20x
    ref = a.double() @ b.double()
    e_hl, e_lo = rel(T._mm32_hl(a, b), ref), rel(T._mm32(a, b.to(BF)), ref)
    good = res["exact"] is True and res["rounding"] == "mismatch" and str(res["missing"]).startswith("unavailable") \
        and e_hl <= 1e-5 and e_lo > 1e-4
    say(f"mm32 helpers (CUDA pieces emulated): probe exact {res['exact']!r} / rounding {res['rounding']!r} / missing "
        f"{str(res['missing'])[:40]!r}; hi+lo split rel err {e_hl:.1e} (bf16 intermediate {e_lo:.1e}) -> "
        f"{'ok' if good else 'FAIL'}")
    return good


def check_dc(T):
    """ops c): factor-gradient precision with a token-shared input component and a token-centred output gradient."""
    I, O, N = 512, 384, 1024
    ok = True
    for dc in (0.0, 10.0):
        g = torch.Generator().manual_seed(11)
        n = torch.randn(1, N, I, generator=g)
        c = torch.randn(1, 1, I, generator=g) * dc
        x = (c + n).to(BF)
        gy = torch.randn(1, N, O, generator=g)
        if dc > 0:
            gy = gy - gy.mean(dim=1, keepdim=True)
        gy = gy.to(BF)
        res, ref = {}, None
        for fast in (False, True):
            root = build_linear(OPS[0], I, O)
            mp, ads, params = attach(T, root, fast, T.FAMILIES["krea2"]["include"], 16, 32.0, 0.05)
            lin = root.diffusion_model.blocks[0].attn.wq
            ad = ads["diffusion_model.blocks.0.attn.wq"]
            n0 = T._fast_calls["n"]
            lin(x.clone().requires_grad_(True)).backward(gy)
            res[fast] = (ad.lora_up.weight.grad.double(), ad.lora_down.weight.grad.double(), T._fast_calls["n"] - n0)
            if not fast:  # fp64 straight-through reference (gradient of the forward with the bf16 W' held fixed)
                U, D = ad.lora_up.weight.detach().double(), ad.lora_down.weight.detach().double()
                G = gy.reshape(-1, O).double().t() @ x.reshape(-1, I).double()
                s = float(ad.alpha) / ad.rank
                ref = (s * G @ D.t(), s * U.t() @ G)
        e = {f: (rel(res[False][i], ref[i]), rel(res[True][i], ref[i])) for i, f in enumerate(("up", "down"))}
        worst = max(e[f][1] / max(e[f][0], 1e-12) for f in e)
        good = worst <= 2.0 and res[True][2] == 1 and res[False][2] == 0
        ok &= good
        say(f"precision |DC| = {dc:g} x |rest|{' (token-centred gy)' if dc else ''}: rel err vs fp64 old/fast up "
            f"{e['up'][0]:.2e}/{e['up'][1]:.2e} down {e['down'][0]:.2e}/{e['down'][1]:.2e}; worst fast/old {worst:.2f}x "
            f"(<= 2) -> {'ok' if good else 'FAIL'}")
    return ok


def check_smallbias(T):
    """ops d): LoRA + SMALL bias delta on one Linear."""
    I, O = 64, 48
    g = torch.Generator().manual_seed(5)
    x = torch.randn(2, 16, I, generator=g).to(BF)
    gy = torch.randn(2, 16, O, generator=g).to(BF)
    res = {}
    for fast in (False, True):
        root = build_linear(OPS[0], I, O)
        mp, ads, params = attach(T, root, fast, T.FAMILIES["krea2"]["include"], 8, 16.0, 0.05)
        smalls, _ = T.attach_small(mp, r"^blocks\.0\.attn\.wq\.bias$", 256.0)
        sd = list(smalls.values())[0]
        with torch.no_grad():
            sd.delta.copy_(torch.randn(sd.delta.shape, generator=torch.Generator().manual_seed(9)) * 0.1)
        lin = root.diffusion_model.blocks[0].attn.wq
        ad = ads["diffusion_model.blocks.0.attn.wq"]
        n0 = T._fast_calls["n"]
        xx = x.clone().requires_grad_(True)
        y = lin(xx)
        y.backward(gy)
        res[fast] = dict(y=y.detach(), gx=xx.grad, gu=ad.lora_up.weight.grad, gd=ad.lora_down.weight.grad, gb=sd.delta.grad,
                         calls=T._fast_calls["n"] - n0, installed="forward_comfy_cast_weights" in lin.__dict__)
    same = all((res[True][k] is None and res[False][k] is None) or
               (res[True][k] is not None and res[False][k] is not None and torch.equal(res[True][k], res[False][k]))
               for k in ("y", "gx", "gu", "gd", "gb"))
    good = same and res[False]["gb"] is not None and res[True]["calls"] == 0 and res[True]["installed"]
    say(f"LoRA + SMALL bias delta on one Linear: FAST=1 fast calls {res[True]['calls']} (wrapper installed "
        f"{res[True]['installed']}); bias-delta grad FAST=0 {'set' if res[False]['gb'] is not None else 'None'} / FAST=1 "
        f"{'set' if res[True]['gb'] is not None else 'None'}; all outputs / grads bit-identical {same} -> "
        f"{'ok' if good else 'FAIL'}")
    return good


# ---------------------------------------------------------------------------------------------------------- models
def krea2_model(layers=2, features=256, heads=2, kvheads=2, txtlayers=2, shift=0.0, seed=0):
    from comfy.ldm.krea2.model import SingleStreamDiT
    g = torch.Generator().manual_seed(seed)
    m = SingleStreamDiT(features=features, tdim=256, txtdim=160, heads=heads, kvheads=kvheads, multiplier=4, layers=layers,
                        patch=2, channels=16, txtlayers=txtlayers, txtheads=20, txtkvheads=20, dtype=BF,
                        operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_((torch.randn(p.shape, generator=g) * (0.02 if p.dim() >= 2 else 0.1)).to(p.dtype))
            if shift and name.endswith("mod.lin"):  # large token-shared adaLN shifts (pre / post shift chunks 1 and 4)
                F = features
                v = torch.randn(6 * F, generator=g) * 0.1
                v[F:2 * F] = torch.randn(F, generator=g) * shift
                v[4 * F:5 * F] = torch.randn(F, generator=g) * shift
                p.copy_(v.to(p.dtype))
    m.requires_grad_(False)
    root = torch.nn.Module()
    root.diffusion_model = m
    return root


def tiny_krea2(seed=0, layers=2):
    root = krea2_model(layers=layers, seed=seed)
    gi = torch.Generator().manual_seed(2)
    x, ctx = torch.randn(1, 16, 8, 8, generator=gi).to(BF), torch.randn(1, 5, 2 * 160, generator=gi).to(BF)
    return root, (lambda: root.diffusion_model(x, torch.tensor([0.3]), context=ctx))


def flux_model(depth=1, single=1, hidden=64, mlp_ratio=2.0, scale=0.05, seed=0):
    from comfy.ldm.flux.model import Flux
    g = torch.Generator().manual_seed(seed)
    m = Flux(in_channels=16, out_channels=16, vec_in_dim=32, context_in_dim=32, hidden_size=hidden, mlp_ratio=mlp_ratio,
             num_heads=2, depth=depth, depth_single_blocks=single, axes_dim=[8, 12, 12], theta=10000, patch_size=2,
             qkv_bias=True, guidance_embed=True, txt_ids_dims=[], dtype=BF, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for _, p in m.named_parameters():
            p.copy_((torch.randn(p.shape, generator=g) * scale).to(p.dtype))
    m.requires_grad_(False)
    root = torch.nn.Module()
    root.diffusion_model = m
    return root


def tiny_flux(seed=0):
    root = flux_model(seed=seed)
    gi = torch.Generator().manual_seed(1)
    x, ctx, y = (torch.randn(1, 16, 8, 8, generator=gi).to(BF), torch.randn(1, 5, 32, generator=gi).to(BF),
                 torch.randn(1, 32, generator=gi).to(BF))
    return root, (lambda: root.diffusion_model(x, torch.tensor([0.5]), ctx, y=y, guidance=torch.tensor([3.5])))


def _tiny_root(m, seed, scale=0.05, skip=()):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if not isinstance(p, comfy.ops.QuantizedTensor) and not any(name.endswith(s) for s in skip):
                p.copy_((torch.randn(p.shape, generator=g) * scale).to(p.dtype))
    m.requires_grad_(False)
    root = torch.nn.Module()
    root.diffusion_model = m
    return root


def qwen_model(layers=2, seed=0):
    from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel
    return _tiny_root(QwenImageTransformer2DModel(
        patch_size=2, in_channels=64, out_channels=16, num_layers=layers, attention_head_dim=32, num_attention_heads=2,
        joint_attention_dim=48, pooled_projection_dim=32, axes_dims_rope=(8, 12, 12), dtype=BF,
        operations=comfy.ops.disable_weight_init), seed)


def zimage_model(layers=2, refiners=1, seed=0):
    from comfy.ldm.lumina.model import NextDiT  # z-image: z_image_modulation (adaLN input 256 -> dim >= 256)
    return _tiny_root(NextDiT(
        patch_size=2, in_channels=16, dim=256, n_layers=layers, n_refiner_layers=refiners, n_heads=2, n_kv_heads=2,
        multiple_of=32, ffn_dim_multiplier=None, norm_eps=1e-5, qk_norm=True, cap_feat_dim=48, axes_dims=(32, 48, 48),
        axes_lens=(64, 64, 64), z_image_modulation=True, time_scale=1000.0, dtype=BF,
        operations=comfy.ops.disable_weight_init), seed)


def ideogram4_model(layers=2, seed=0):
    """ideogram4 as the trainer loads it (G56CT_BASE=evalq): mixed_precision_ops with fp8 compute disabled, every Linear
    an fp8 e4m3 QuantizedTensor weight with scale 1 and a bf16 bias."""
    from comfy.ldm.ideogram4.model import Ideogram4Transformer2DModel
    ops = comfy.ops.mixed_precision_ops({}, compute_dtype=BF, disabled={"float8_e4m3fn", "float8_e5m2", "nvfp4", "mxfp8"})
    m = Ideogram4Transformer2DModel(in_channels=128, num_layers=layers, num_attention_heads=2, attention_head_dim=32,
                                    intermediate_size=128, adaln_dim=32, llm_features_dim=48, rope_theta=10000,
                                    mrope_section=(4, 6, 6), norm_eps=1e-5, dtype=BF, operations=ops)
    g = torch.Generator().manual_seed(seed + 100)
    for _, lin in m.named_modules():
        if isinstance(lin, ops.Linear):
            sd = {"weight": (torch.randn(lin.out_features, lin.in_features, generator=g) * 0.05).to(torch.float8_e4m3fn),
                  "weight_scale": torch.tensor(1.0),
                  "comfy_quant": torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8)}
            if lin.bias is not None:
                sd["bias"] = (torch.randn(lin.out_features, generator=g) * 0.05).to(BF)
            lin.load_state_dict(sd)
    return _tiny_root(m, seed, skip=(".bias",))


def tiny_qwen(seed=0):
    root = qwen_model(seed=seed)
    gi = torch.Generator().manual_seed(1)
    x, ctx = torch.randn(1, 16, 1, 8, 8, generator=gi).to(BF), torch.randn(1, 5, 48, generator=gi).to(BF)
    return root, (lambda: root.diffusion_model(x, torch.tensor([0.5]), ctx))


def tiny_zimage(seed=0):
    root = zimage_model(seed=seed)
    gi = torch.Generator().manual_seed(1)
    x, ctx = torch.randn(1, 16, 8, 8, generator=gi).to(BF), torch.randn(1, 5, 48, generator=gi).to(BF)
    return root, (lambda: root.diffusion_model(x, torch.tensor([0.5]), ctx, 5))


def tiny_ideogram4(seed=0):
    root = ideogram4_model(seed=seed)
    gi = torch.Generator().manual_seed(1)
    x, ctx = torch.randn(1, 128, 4, 4, generator=gi).to(BF), torch.randn(1, 5, 48, generator=gi).to(BF)
    return root, (lambda: root.diffusion_model(x, torch.tensor([0.5]), ctx))


# family -> (tiny model, the trainer's production LoRA module set for it)
MODELS = {"krea2": (tiny_krea2, lambda T: T.FAMILIES["krea2"]["include"]),
          "flux": (tiny_flux, lambda T: T.FAMILIES["flux"]["include"]),
          "qwen-image": (tiny_qwen, lambda T: T.FAMILY_DEFAULTS["qwen-image"]["include"]),
          "z-image": (tiny_zimage, lambda T: T.FAMILY_DEFAULTS["z-image"]["include"]),
          "ideogram4": (tiny_ideogram4, lambda T: T.FAMILIES["ideogram4"]["include"])}


def run_model(T, fam, fast, ckpt=False):
    maker, include = MODELS[fam]
    root, fwd = maker()
    mp, ads, params = attach(T, root, fast, include(T), 8, 16.0, 0.02)
    if ckpt:
        T.set_checkpointing(mp, True)
    flags, hooks = {}, []

    def note(name):
        def hook(mod, args):  # must return None (a return value would replace the module's input)
            flags.setdefault(name, bool(args[0].requires_grad))
        return hook
    for k in ads:
        hooks.append(root.get_submodule(k).register_forward_pre_hook(note(k)))
    n0 = T._fast_calls["n"]
    mm.in_training = True  # differentiable rope path, as in the trainer
    try:
        out, seen = run_saving(fwd, keep_of(root, params))
        gy = torch.randn(out.shape, generator=torch.Generator().manual_seed(3))
        (out.float() * gy).sum().backward()
    finally:
        mm.in_training = False
        for h in hooks:
            h.remove()
    shapes = {k: tuple(root.get_submodule(k).weight.shape) for k in ads}
    grads = {k: tuple(None if p.grad is None else p.grad.clone() for p in (a.lora_up.weight, a.lora_down.weight))
             for k, a in ads.items()}
    qt = sum(1 for k in ads if isinstance(root.get_submodule(k).weight, comfy.ops.QuantizedTensor))
    return {"grads": grads, "seen": seen, "flags": flags, "shapes": shapes, "calls": T._fast_calls["n"] - n0,
            "st": getattr(mp, "_g56_fast", None), "out": out.detach(), "qt": qt}


def check_models(T, ref=None):
    ok = True
    for fam in MODELS:
        r0, r1, rc = run_model(T, fam, False), run_model(T, fam, True), run_model(T, fam, True, ckpt=True)
        worst, bad, unused = (0.0, 1.0), [], 0
        num = den = 0.0
        ck = 0.0
        for k, (u0, d0) in r0["grads"].items():
            for i, f in ((0, "up"), (1, "down")):
                a, b, c_ = r1["grads"][k][i], r0["grads"][k][i], rc["grads"][k][i]
                zero = [t is None or not bool(t.any()) for t in (a, b, c_)]
                if any(zero):  # a module whose output does not reach the loss (qwen: the last block's text stream):
                    unused += all(zero) and i == 0  # no gradient (None or exactly 0) on every path
                    if not all(zero):
                        bad.append(f"{k.split('diffusion_model.')[-1]}.{f} gradient on some paths only")
                    continue
                e, c = rel(a, b), cosine(a, b)
                worst = (max(worst[0], e), min(worst[1], c))
                num += float((a.double() - b.double()).square().sum())
                den += float(b.double().square().sum())
                ck = max(ck, rel(c_, a))
                if e > 5e-2 or c < 0.999:
                    bad.append(f"{k.split('diffusion_model.')[-1]}.{f} rel {e:.2e} cos {c:.5f}")
        agg = (num / max(den, 1e-300)) ** 0.5
        n = len(r0["grads"])
        need = sum(r0["shapes"][k][0] * r0["shapes"][k][1] * 2 for k in r0["grads"] if r0["flags"].get(k))
        drop = nbytes(r0["seen"]) - nbytes(r1["seen"])
        bytes_ok = need <= drop <= need + 8 * n
        same_out = torch.equal(r0["out"], r1["out"])
        ran = r1["calls"] == n and rc["calls"] >= n and r1["st"].on and not r1["st"].bad and not r1["st"].skipped \
            and r0["st"] is None
        good = not bad and agg <= 2e-2 and ck <= 1e-6 and bytes_ok and ran and same_out and unused < n \
            and (fam != "ideogram4" or r1["qt"] == n)
        ok &= good
        say(f"model {fam}{' fp8 QuantizedTensor' if r1['qt'] else ' bf16'}: {n} LoRA Linears "
            f"({sum(1 for k in r0['grads'] if r0['flags'].get(k))} with a grad-requiring input"
            + (f", {unused} without a gradient on any path" if unused else "") + f"); output fast == old {same_out}; "
            f"factor grads worst rel {worst[0]:.2e} min cos {worst[1]:.6f}, aggregate rel {agg:.2e}"
            + (f", out of tolerance {bad[:3]}" if bad else "") + f"; saved bytes old {nbytes(r0['seen'])} new "
            f"{nbytes(r1['seen'])} drop {drop} (sum out*in*2 = {need}); checkpointed vs not rel {ck:.1e}; fast calls "
            f"{r1['calls']}/{n} (ckpt run {rc['calls']}) -> {'ok' if good else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------------------------------------- guard
def guard_probe_factory(record):
    def probe(orig, c, *a, **k):
        before = [p.detach().clone() for p in c["params"]]
        rng = torch.get_rng_state()
        ck = c["grad_ckpt_on"]
        n0 = sum(1 for m in c["mp"].model.modules() if hasattr(m, "_g56_orig_forward"))
        r = orig(c, *a, **k)
        record["restored"] = (all(torch.equal(p.detach(), q) for p, q in zip(c["params"], before))
                              and all(p.grad is None for p in c["params"]) and torch.equal(torch.get_rng_state(), rng)
                              and c["grad_ckpt_on"] == ck
                              and sum(1 for m in c["mp"].model.modules() if hasattr(m, "_g56_orig_forward")) == n0)
        record["ret"] = r
        return r
    return probe


def check_guard(T, ref):
    rec = {}
    r1 = harness_run(T, True, "fast", guard_probe_factory(rec))
    st = r1["st"]
    verdict = st.verdict if st is not None else None
    stats = (st.stats or {}) if st is not None else {}
    passed = bool(rec.get("ret")) and st is not None and st.on and str(verdict).startswith("pass") \
        and "bit-identical True" in str(verdict) and stats.get("local_modules") == len(st.mods) == 3
    d = max(abs(a["score"] - b["score"]) / abs(b["score"]) for a, b in zip(r1["evals"], ref["evals"])) \
        if len(r1["evals"]) == len(ref["evals"]) and ref["evals"] else float("inf")
    good = passed and rec.get("restored", False) and r1["steps"] == ref["steps"] and d <= 1e-4 and r1["calls"] > 0
    say(f"guard on the harness: {verdict}; state restored {rec.get('restored')}; FAST=1 run {r1['steps']} steps, "
        f"{r1['calls']} fast forwards, max eval rel diff vs FAST=0 {d:.2e} -> {'ok' if good else 'FAIL'}")
    # trainable base weights (harness default): no module may take the fast path -> the guard refuses, run == FAST=0
    rt = harness_run(T, True, "trainable_base", freeze=False)
    refused = rt["st"] is not None and not rt["st"].on and "no module took the fast path" in str(rt["st"].verdict) \
        and rt["calls"] == 0 and all(rt[k] == ref[k] for k in ("file", "sd", "evals", "steps"))
    say(f"guard with trainable base weights: {str(rt['st'].verdict if rt['st'] else None)[:110]}; run byte-identical to "
        f"FAST=0 {all(rt[k] == ref[k] for k in ('file', 'sd', 'evals', 'steps'))} -> {'ok' if refused else 'FAIL'}")
    # one module fails its first-use check (_fast_verify): it keeps the old path, the others stay fast, the guard passes
    tgt = "diffusion_model.blocks.0.attn.wk"
    rv = harness_run(T, True, "verify_fail", verify_fail=tgt)
    sv = rv["st"]
    dv = max(abs(a["score"] - b["score"]) / abs(b["score"]) for a, b in zip(rv["evals"], ref["evals"])) \
        if len(rv["evals"]) == len(ref["evals"]) and ref["evals"] else float("inf")
    partial = sv is not None and sv.on and sv.bad == [tgt] and str(sv.verdict).startswith("pass") \
        and (sv.stats or {}).get("local_modules") == 2 and rv["steps"] == ref["steps"] and dv <= 1e-4 and rv["calls"] > 0
    say(f"guard with a per-module fallback ({tgt.split('diffusion_model.')[-1]} fails _fast_verify): "
        f"{str(sv.verdict if sv else None)[:120]}; fallbacks {sv.bad if sv else None}; max eval rel diff vs FAST=0 {dv:.2e} "
        f"-> {'ok' if partial else 'FAIL'}")
    return good and refused and partial


# ---------------------------------------------------------------------------------------------------------- merge
def fm_attach(T, root, fm, include, rank, alpha, up_std, seed=1):
    """attach() with G56CT_FAST_MERGE set before the LoRA attaches (fast_lora_install reads it)."""
    reset_cfg(T, fast_lora=True, fast_merge=fm)
    mp = H.FakeMP(root)
    ads, params = T.attach_lora(mp, include, rank, alpha)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for k in sorted(ads):
            a = ads[k]
            a.lora_up.weight.copy_(torch.randn(a.lora_up.weight.shape, generator=g) * up_std)
            a.lora_down.weight.copy_(torch.randn(a.lora_down.weight.shape, generator=g) * a.lora_down.weight.shape[1] ** -0.5)
    return mp, ads, params


def check_merge_ops(T):
    """_fm_merge == Comfy's merge (cast_bias_weight + LoraDiff) bit for bit per op class, s and delta size; then a forward
    + backward through the armed module (call 1 verifies, call 2 merges) == FM=0."""
    I, O, R = 256, 384, 16
    inc = T.FAMILIES["krea2"]["include"]
    g = torch.Generator().manual_seed(17)
    x = torch.randn(2, 48, I, generator=g).to(BF)
    gy = torch.randn(2, 48, O, generator=g)
    ok, rows = True, []
    for kind in OPS:
        for s in (1.0, 2.0, 0.75):
            for relsz in (1e-3, 1.0):
                up_std = relsz / (s * R ** 0.5)  # |s U D| ~ relsz x |W| (W ~ N(0, 1/in), down ~ N(0, 1/in))
                root = build_linear(kind, I, O)
                mp, ads, params = fm_attach(T, root, True, inc, R, R * s, up_std)
                lin = root.diffusion_model.blocks[0].attn.wq
                ad = ads["diffusion_model.blocks.0.attn.wq"]
                with torch.no_grad():
                    w0, b0, ost = comfy.ops.cast_bias_weight(lin, x, offloadable=True)
                    w1, b1, W = T._fm_merge(lin, ad, x.dtype, None if s == 1.0 else (ad.alpha / ad.rank).detach())
                    same = torch.equal(w1, w0) and (b1 is None) == (b0 is None) and (b1 is None or torch.equal(b1, b0))
                    changed = int((w0 != W).sum())
                    comfy.ops.uncast_bias_weight(lin, w0, b0, ost)
                outs = {}
                for fm in (False, True):
                    root = build_linear(kind, I, O)
                    mp, ads, params = fm_attach(T, root, fm, inc, R, R * s, up_std)
                    st = mp._g56_fast
                    st.fm_armed = fm
                    lin = root.diffusion_model.blocks[0].attn.wq
                    ad = ads["diffusion_model.blocks.0.attn.wq"]
                    res = []
                    m0 = st.fm_calls
                    for _ in range(2):
                        xx = x.clone().requires_grad_(True)
                        y = lin(xx)
                        (y.float() * gy).sum().backward()
                        res.append((y.detach(), xx.grad, ad.lora_up.weight.grad.clone(), ad.lora_down.weight.grad.clone()))
                        for p in params:
                            p.grad = None
                    outs[fm] = (res, st.fm_calls - m0, dict(st.fm))
                fwd_same = all(torch.equal(a, b) for a, b in zip(outs[True][0][1], outs[False][0][1])) \
                    and all(torch.equal(a, b) for a, b in zip(outs[True][0][0], outs[False][0][0]))
                state = outs[True][2].get("diffusion_model.blocks.0.attn.wq", [None])[0]
                used = outs[True][1] == 1 and state == "ok"
                good = same and fwd_same and changed > 0 and (used if relsz == 1.0 else state in ("ok", "pending"))
                ok &= good
                rows.append(good)
                if not good or (s != 1.0 and relsz == 1.0):
                    say(f"merge op {kind} s {s:g} |sUD| {relsz:g}|W|: W'/bias == Comfy {same} ({changed} weights changed "
                        f"by the delta); FM forward state {state}, merges {outs[True][1]}, fwd+bwd == FM=0 {fwd_same} -> "
                        f"{'ok' if good else 'FAIL'}")
    say(f"merge ops: {sum(rows)}/{len(rows)} (op class x s x delta size) bit-identical to Comfy's merge -> "
        f"{'ok' if ok else 'FAIL'}")
    return ok


def fm_train(T, fam, fm, steps=20, fixture=None):
    """Tiny model of the family, FAST=1, G56CT_FAST_MERGE fm, 20 AdamW steps from the gauss init (up = 0: delta 0 at
    step 1), rank 8 alpha 16 (s = 2), armed after the first optimizer step as _run_one does. fixture: replaces
    _fm_merge. Returns (losses, final factors, fast state)."""
    maker, include = MODELS[fam]
    root, fwd = maker()
    reset_cfg(T, fast_lora=True, fast_merge=fm)
    mp = H.FakeMP(root)
    torch.manual_seed(0)
    ads, params = T.attach_lora(mp, include(T), 8, 16.0)
    opt = torch.optim.AdamW(params, lr=2e-3)
    g = torch.Generator().manual_seed(3)
    orig = T._fm_merge
    if fixture is not None:
        T._fm_merge = fixture
    losses, target = [], None
    mm.in_training = True
    try:
        for i in range(steps):
            out = fwd()
            if target is None:
                target = torch.randn(out.shape, generator=g)
            loss = (out.float() - target).square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.detach().clone())
            if i == 0 and fm:
                mp._g56_fast.fm_armed = True
    finally:
        mm.in_training = False
        T._fm_merge = orig
    facs = {k: (a.lora_up.weight.detach().clone(), a.lora_down.weight.detach().clone()) for k, a in ads.items()}
    return losses, facs, mp._g56_fast


def fm_same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a[0], b[0])) and len(a[0]) == len(b[0]) \
        and all(torch.equal(a[1][k][i], b[1][k][i]) for k in a[1] for i in (0, 1))


def check_merge_models(T):
    ok = True
    for fam in MODELS:
        r0, r1 = fm_train(T, fam, False), fm_train(T, fam, True)
        rep = r1[2].fm_report()
        bad_off = [w for _, w in rep["off_modules"] if not str(w).startswith("the delta changed no weight")]
        same = fm_same(r0, r1)
        good = same and rep["ok"] > 0 and not bad_off and r1[2].fm_calls > 0 and rep["pending"] == 0 \
            and r0[2].fm_calls == 0 and r0[2].fm == {}
        ok &= good
        say(f"merge model {fam}: 20 steps FM=1 vs FM=0 losses + final factors bit-identical {same}; {r1[2].fm_line()[:200]}"
            f" -> {'ok' if good else 'FAIL'}")
    return ok


def check_merge_harness(T, ref_fast=None):
    """The harness run (fast_lora_guard + _run_one's arming after step 1) with FM=1 == FAST=1 FM=0 byte for byte; no
    merge during the guard (4 training forwards: 2 passes x 2 prompts) or step 1 (forwards 5-6)."""
    ref_fast = ref_fast or harness_run(T, True, "fm_ref")
    cur, at = {"n": 0}, []
    orig = T._fm_merge

    def counted(*a, **k):
        at.append(cur["n"])
        return orig(*a, **k)
    T._fm_merge = counted
    try:
        r = harness_run(T, True, "fm_on", extra=dict(fast_merge=True), predict_hook=lambda n: cur.__setitem__("n", n))
    finally:
        T._fm_merge = orig
    st = r["st"]
    rep = st.fm_report() if st is not None else {}
    same = all(r[k] == ref_fast[k] for k in ("file", "sd", "evals", "steps"))
    good = same and rep.get("ok") == 3 and rep.get("off") == 0 and bool(at) and min(at) >= 7 \
        and str(st.verdict).startswith("pass") and st.fm_calls > 0
    say(f"merge harness: FM=1 == FAST=1 FM=0 (file, state dict, evals, steps) {same}; {st.fm_line() if st else None}; "
        f"first merge at training forward {min(at) if at else None} (guard 1-4, step 1 5-6) -> {'ok' if good else 'FAIL'}")
    return good


def check_merge_verify(T):
    """A wrong merge (fixture: W + s U D truncated to the compute dtype instead of rounded) must be caught by the
    per-module check: every module off ('differs'), results == FM=0; and _fm_verify's zero-delta rule."""
    orig = T._fm_merge

    def trunc(m, ad, dt, s):
        W = T._fast_base_weight(m, dt)
        diff = ad.lora_up.weight @ ad.lora_down.weight
        if s is not None:
            diff = diff * s
        f = W.float() + diff.reshape(W.shape)
        w = (f.view(torch.int32) & -65536).view(torch.float32).to(dt)  # bf16: drop the low 16 bits (no rounding)
        b = m.bias.to(dtype=dt) if getattr(m, "bias", None) is not None else None
        return w, b, W
    ok = True
    for fam in ("krea2", "ideogram4"):
        r0, r1 = fm_train(T, fam, False), fm_train(T, fam, True, fixture=trunc)
        rep = r1[2].fm_report()
        caught = rep["ok"] == 0 and rep["off"] > 0 and all("differs" in str(w) for _, w in rep["off_modules"])
        same = fm_same(r0, r1)
        good = caught and same and r1[2].fm_calls == 0
        ok &= good
        say(f"merge verify {fam}: truncating fixture -> {rep['ok']} ok / {rep['off']} off ({rep['off_modules'][:1]}), "
            f"results == FM=0 {same} -> {'ok' if good else 'FAIL'}")
    # zero delta: nothing proven -> pending, off after _FM_TRIVIAL_MAX checks
    root = build_linear(OPS[0], 64, 48)
    mp, ads, params = fm_attach(T, root, True, T.FAMILIES["krea2"]["include"], 8, 8.0, 0.0)  # up = 0
    lin, ad = root.diffusion_model.blocks[0].attn.wq, ads["diffusion_model.blocks.0.attn.wq"]
    x = torch.randn(1, 4, 64).to(BF)
    st = mp._g56_fast
    rs = T._fm_state(st, "m", lin, ad, x)
    states = []
    with torch.no_grad():
        for _ in range(T._FM_TRIVIAL_MAX):
            w0, b0, ost = comfy.ops.cast_bias_weight(lin, x, offloadable=True)
            T._fm_verify(st, "m", lin, ad, x.dtype, w0, b0, rs)
            states.append(rs[0])
    triv = states[:-1] == ["pending"] * (T._FM_TRIVIAL_MAX - 1) and states[-1] == "off"
    say(f"merge verify zero delta: states {states[0]} x{states.count('pending')} then {states[-1]} -> "
        f"{'ok' if triv else 'FAIL'}")
    T._fm_merge = orig
    return ok and triv


def check_merge(T, ref=None):
    parts = {"ops": check_merge_ops, "models": check_merge_models, "harness": check_merge_harness,
             "verify": check_merge_verify}
    ok = True
    for name, fn in parts.items():
        good = fn(T)
        say(f"merge {name}: {'ok' if good else 'FAIL'}")
        ok &= good
        if not good:
            break
    return ok


# ---------------------------------------------------------------------------------------------------------- deep
class DWrap:
    """model_wrap as predict() uses it: .conds['positive'] = [context], __call__(x, ts) -> denoised (flow x0)."""

    def __init__(self, root, call):
        self.root, self.call = root, call
        self.conds = {"positive": []}

    def __call__(self, x, ts, **kw):
        v = self.call(self.root.diffusion_model, x.to(BF), ts, self.conds["positive"][0]).float()
        return x - ts.view(-1, *([1] * (x.dim() - 1))) * v


def deep_setup(T, fam, seed, txtlayers=2, shift=0.0):
    """A model at the family's production depth (small width): (root, LoRA module set, [caption, empty] contexts,
    call(diffusion_model, x, ts, context) -> velocity, a training latent)."""
    gi = torch.Generator().manual_seed(4 + seed)
    cdim = {"krea2": txtlayers * 160, "flux": 32}.get(fam, 48)
    conds = [torch.randn(1, 5, cdim, generator=gi).to(BF), torch.randn(1, 3, cdim, generator=gi).to(BF)]
    lat = (1, 16, 8, 8)
    if fam == "krea2":  # 28 blocks (+ txtfusion), the production set include + include_extra (264 modules)
        root = krea2_model(layers=28, features=128, heads=2, kvheads=1, txtlayers=txtlayers, shift=shift, seed=seed)
        include = f"(?:{T.FAMILIES['krea2']['include']})|(?:{T.FAMILY_DEFAULTS['krea2']['include_extra']})"

        def call(dm, x, ts, ctx):
            return dm(x, ts, context=ctx)
    elif fam == "flux":  # 19 double + 38 single blocks (228 modules)
        root = flux_model(depth=19, single=38, hidden=64, mlp_ratio=4.0, seed=seed)
        include = T.FAMILIES["flux"]["include"]
        ys = {5: torch.randn(1, 32, generator=gi).to(BF), 3: torch.randn(1, 32, generator=gi).to(BF)}

        def call(dm, x, ts, ctx):
            return dm(x, ts, ctx, y=ys[ctx.shape[1]], guidance=torch.tensor([3.5]))
    elif fam == "qwen-image":  # 60 blocks, the full production set (720 modules)
        root, include, lat = qwen_model(layers=60, seed=seed), T.FAMILY_DEFAULTS["qwen-image"]["include"], (1, 16, 1, 8, 8)

        def call(dm, x, ts, ctx):
            return dm(x, ts, ctx)
    elif fam == "z-image":  # 30 layers + 2 + 2 refiners, include '.*'
        root, include = zimage_model(layers=30, refiners=2, seed=seed), T.FAMILY_DEFAULTS["z-image"]["include"]

        def call(dm, x, ts, ctx):
            return dm(x, ts, ctx, ctx.shape[1])
    else:  # ideogram4: 34 layers, fp8 QuantizedTensor weights
        root, include, lat = ideogram4_model(layers=34, seed=seed), T.FAMILIES["ideogram4"]["include"], (1, 128, 4, 4)

        def call(dm, x, ts, ctx):
            return dm(x, ts, ctx)
    return root, include, conds, call, torch.randn(*lat, generator=gi)


def deep_guard(T, fam, rank, seed, txtlayers=2, shift=0.0):
    """fast_lora_guard on a production-depth model: (ok, state, n LoRA modules, seconds)."""
    root, include, conds, call, lat = deep_setup(T, fam, seed, txtlayers, shift)
    reset_cfg(T, fast_lora=True)
    mp = H.FakeMP(root)
    ads, params = T.attach_lora(mp, include, rank, float(rank))
    c = {"mp": mp, "adapters": ads, "params": params, "train": [{"scaled": lat, "cond_idx": 0}], "cfg": dict(T.CFG),
         "empty_idx": 1, "grad_ckpt_on": False}
    t0 = time.time()
    mm.in_training = True
    try:
        ok = T.fast_lora_guard(c, DWrap(root, call), conds, {}, torch.device("cpu"))
    finally:
        mm.in_training = False
    return ok, mp._g56_fast, len(ads), time.time() - t0


# (family, rank, seed, krea2 text layers, krea2 adaLN shift): the review's failing / tightest krea2 cases, flux, and the
# other families at their production rank
DEEP = [("krea2", 16, 0, 2, 0.0), ("krea2", 16, 2, 2, 0.0), ("krea2", 16, 4, 12, 0.0), ("krea2", 64, 5, 12, 0.0),
        ("krea2", 64, 0, 12, 10.0), ("flux", 16, 1, 0, 0.0), ("qwen-image", 32, 0, 0, 0.0), ("z-image", 64, 0, 0, 0.0),
        ("ideogram4", 64, 0, 0, 0.0)]


def check_deep(T, ref=None):
    ok = True
    for fam, rank, seed, txtl, shift in DEEP:
        g_ok, st, n, dt = deep_guard(T, fam, rank, seed, txtl, shift)
        s = st.stats or {}
        margin = s.get("local_min_cos", 0.0) >= 0.9999 and s.get("local_max_ratio_dev", 1.0) <= 5e-3 \
            and s.get("local_modules") == s.get("grad_modules") and s.get("fast_modules") == n == len(st.mods) \
            and s.get("local_modules", 0) >= n - 3  # qwen: the last block's text stream never reaches the loss (3 modules)
        good = bool(g_ok) and margin
        ok &= good
        say(f"deep guard {fam} rank {rank} seed {seed}" + (f" text layers {txtl} adaLN shift {shift:g}" if fam == "krea2"
                                                            else "") + f": {n} LoRA modules, guard "
            f"{'PASS' if g_ok else 'FAIL'} in {dt:.1f}s: {st.verdict} -> {'ok' if good else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------------------------------------- fallback
def fallback_case(T, fast, ckpt, where=None, n_at=9):
    """Tiny krea2 (3 blocks): one step's forward + backward, with an injected failure at the n-th fast forward /
    backward; then what _run_one does (zero the grads, redo the step). Returns (exception, state, grads of the redo)."""
    root, fwd = tiny_krea2(layers=3)
    mp, ads, params = attach(T, root, fast, T.FAMILIES["krea2"]["include"], 8, 16.0, 0.02)
    if ckpt:
        T.set_checkpointing(mp, True)
    gy = None
    orig_fwd, orig_bwd = T._fast_forward, T._fast_backward
    cnt = {"n": 0}

    def maybe_raise():
        cnt["n"] += 1
        if cnt["n"] == n_at:
            raise RuntimeError(f"injected {where} failure")
    if where == "forward":
        def fwd_inj(*a, **k):
            maybe_raise()
            return orig_fwd(*a, **k)
        T._fast_forward = fwd_inj
    elif where == "backward":
        def bwd_inj(*a, **k):
            maybe_raise()
            return orig_bwd(*a, **k)
        T._fast_backward = bwd_inj
    err = None
    mm.in_training = True
    try:
        try:
            out = fwd()
            gy = torch.randn(out.shape, generator=torch.Generator().manual_seed(3))
            (out.float() * gy).sum().backward()
        except Exception as e:  # noqa: BLE001
            err = e  # kept alive during the redo, as ComfyUI's calc_cond_batch does (its results list holds it)
        finally:
            T._fast_forward, T._fast_backward = orig_fwd, orig_bwd
        for p in params:
            p.grad = None
        if isinstance(err, T.FastLoRAFallback):
            T._fast_retry({"mp": mp}, err)  # what _run_one does before it redoes the step
        out = fwd()
        gy = torch.randn(out.shape, generator=torch.Generator().manual_seed(3))
        (out.float() * gy).sum().backward()
    finally:
        mm.in_training = False
    grads = {k: (a.lora_up.weight.grad.clone(), a.lora_down.weight.grad.clone()) for k, a in ads.items()}
    return err, getattr(mp, "_g56_fast", None), grads


def check_fallback(T, ref=None):
    ok = True
    for ckpt in (False, True):
        _, _, g0 = fallback_case(T, False, ckpt)
        for where, n_at in (("forward", 9), ("backward", 5)):
            err, st, g1 = fallback_case(T, True, ckpt, where, n_at)
            same = all(torch.equal(g1[k][i], g0[k][i]) for k in g0 for i in (0, 1))
            good = isinstance(err, T.FastLoRAFallback) and st is not None and not st.on \
                and "injected" in str(st.off_reason) and same
            ok &= good
            say(f"fallback, block checkpointing {'on ' if ckpt else 'off'}, failure in fast {where} #{n_at}: raised "
                f"{type(err).__name__ if err else None}; path off {st is not None and not st.on}; redone step == FAST=0 "
                f"bit for bit {same} -> {'ok' if good else 'FAIL'}")
    for where, n_at in (("forward", 20), ("backward", 20)):
        r = harness_run(T, True, f"inject_{where}", inject=(where, n_at))
        st = r["st"]
        good = r["steps"] == BASE["max_steps"] and st is not None and not st.on and st.retries == 1 \
            and "injected" in str(st.off_reason) and str(st.verdict).startswith("pass")
        ok &= good
        say(f"fallback in training (harness, failure in fast {where} #{n_at} after the guard): {r['steps']} steps, "
            f"retries {st.retries if st else None}, path off {st is not None and not st.on} -> {'ok' if good else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------------------------------------- release
def blk_harness():
    """te_lora_harness with the fake DiT's computation routed through its block's forward, so the trainer's non-reentrant
    block checkpointing (set_checkpointing) really wraps it; the block ends in a gate (ones: the same values) as the real
    blocks' gated residuals, so its last saving op is not a LoRA Linear (if it were, torch's recomputation early stop,
    _StopRecomputationError, would be raised inside the fast forward and turned into a FastLoRAFallback: the guard fails;
    the real families' blocks pass the guard, check deep). Returns the undo."""
    orig_dit, orig_call = H.fake_dit, H.FakeWrap.__call__

    def fake_dit(*a, **k):
        root = orig_dit(*a, **k)
        blk = root.diffusion_model.blocks[0]
        gate = torch.ones(1)

        def fwd(h, ctx, _b=blk, _g=gate):
            return _b.mlp.down(torch.nn.functional.silu(_b.attn.wq(h) + _b.attn.wk(ctx).mean(dim=1, keepdim=True))) * _g
        blk.forward = fwd
        return root

    def call(self, x, ts, **kw):
        if torch.is_grad_enabled():
            self.train_calls += 1
            if self.predict_hook is not None:
                self.predict_hook(self.train_calls)
        ctx = self.conds["positive"][0]["model_conds"]["c_crossattn"].cond.to(x.device, torch.float32)
        B, C, Hh, W = x.shape
        v = self.root.diffusion_model.blocks[0](x.permute(0, 2, 3, 1).reshape(B, Hh * W, C), ctx)
        return x - ts.view(-1, 1, 1, 1).to(x.dtype) * v.reshape(B, Hh, W, C).permute(0, 3, 1, 2)
    H.fake_dit, H.FakeWrap.__call__ = fake_dit, call

    def undo():
        H.fake_dit, H.FakeWrap.__call__ = orig_dit, orig_call
    return undo


def release_run(T, plan, blk=False, extra=None):
    """harness_run (FAST=1) with the injections of plan; what every training forward saves for backward
    (torch.autograd.graph.saved_tensors_hooks: weakrefs, the tensor itself is returned unchanged; parameters and LoRA
    factors excluded) is tagged with the forward's number (predict_hook). Returns (run, [(kind, failed forward F, saved
    tensors of F alive at the retry's first forward F + 1, those of the normal step F - 1 alive at F)], OOM retries)."""
    import weakref
    rec, cur, fails, alive = {}, {"n": 0}, [], {}

    def pack(t):
        if not isinstance(t, torch.nn.Parameter):
            rec.setdefault(cur["n"], []).append(weakref.ref(t))
        return t

    def hook(n):  # before training forward n: what the previous forward saved and is still alive
        alive[n] = sum(1 for w in rec.get(n - 1, []) if w() is not None)
        cur["n"] = n
    undo = blk_harness() if blk else None
    LOGS.clear()
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            r = harness_run(T, True, "release", inject=plan, extra=extra, predict_hook=hook,
                            on_inject=lambda where, n, kind: fails.append((kind, cur["n"])))
    finally:
        if undo is not None:
            undo()
    return r, [(kind, f, alive.get(f + 1), alive.get(f)) for kind, f in fails], \
        sum(1 for x in LOGS if x.startswith("OOM -> gradient checkpointing enabled"))


# (name, injection plan, block checkpointing, cfg): the four retry handlers of _run_one (one optimizer step per case /
# accumulated groups, OOM / FastLoRAFallback) and, with the block checkpointed, an OOM (-> checkpointing) then a
# FastLoRAFallback redo with checkpointing on
RELEASE = [("step oom", [("backward", 13, "oom")], False, None), ("step fallback", [("backward", 13)], False, None),
           ("accum oom", [("backward", 13, "oom")], False, dict(accum=2, agree=0.5)),
           ("accum fallback", [("backward", 13)], False, dict(accum=2, agree=0.5)),
           ("ckpt oom + fallback", [("backward", 13, "oom"), ("backward", 25)], True, None)]


def check_release(T, ref=None):
    """review B1 (1 Oct): an OOM / FastLoRAFallback raised INSIDE loss.backward() leaves the failed step's graph (its saved
    activations) referenced by _run_one's den / v / loss (accumulation: + gsum / ssum) until the retry's forward returns,
    so the retry ran with the dead graph still allocated (a second OOM -> crash -> the identity LoRA shipped). Each retry
    handler on the harness (RELEASE): the saved tensors of the failed forward still alive at the retry's first forward <=
    those of a normal completed step at the next forward; every injection fired, every step made, an OOM retried once with
    checkpointing, a FastLoRAFallback redone once with the path off. Before the fix: 4 vs 1 (block checkpointed: 5 vs 2,
    then 1 vs 0 with checkpointing on)."""
    ok = True
    for name, plan, blk, extra in RELEASE:
        r, res, n_oom = release_run(T, plan, blk, extra)
        st = r["st"]
        n_err = sum(1 for p in plan if len(p) < 3 or p[2] == "error")
        good = r["steps"] == BASE["max_steps"] and len(res) == len(plan) and st is not None \
            and str(st.verdict).startswith("pass") and n_oom == sum(1 for p in plan if p[-1] == "oom") \
            and st.retries == n_err and st.on == (n_err == 0) \
            and all(a is not None and b is not None and a <= b for _, _, a, b in res)
        ok &= good
        say(f"release {name}: " + "; ".join(f"{k} in the backward of training forward {f}: saved tensors alive at the "
                                            f"retry's forward {a} (a normal step's {b})" for k, f, a, b in res)
            + f"; {r['steps']} steps, OOM retries {n_oom}, fallback redos {st.retries if st else None} -> "
            f"{'ok' if good else 'FAIL'}")
    return ok


def check_release_raise(T, ref):
    """review A F3 (1 Oct): _fast_release (generator close / clear_frames) raising inside the guard's except or in
    _fast_retry must not escape: the guard's contract is any exception -> FAST off, and a redo must not become a crash.
    _fast_release replaced by one that raises: a failure in the guard's fast pass -> the guard returns False, the run makes
    every step on the old path, byte-identical to FAST=0; a FastLoRAFallback in training -> the step is redone (1 retry,
    path off), every step made."""
    orig_rel, orig_fwd = T._fast_release, T._fast_forward
    calls, got, first = {"n": 0}, {}, {"done": False}

    def boom(e):
        calls["n"] += 1
        raise RuntimeError("injected _fast_release failure")

    def probe(guard, *a, **k):
        got["ret"] = guard(*a, **k)
        return got["ret"]

    def fwd_once(*a, **k):  # the guard's first fast forward fails
        if not first["done"]:
            first["done"] = True
            raise RuntimeError("injected guard forward failure")
        return orig_fwd(*a, **k)
    ok = True
    T._fast_release, T._fast_forward = boom, fwd_once
    try:
        r = harness_run(T, True, "relraise_guard", guard_probe=probe)
    except Exception as e:  # noqa: BLE001
        r, err = None, f"{type(e).__name__}: {str(e)[:120]}"
    else:
        err = None
    finally:
        T._fast_release, T._fast_forward = orig_rel, orig_fwd
    st = r["st"] if r else None
    same = r is not None and all(r[k] == ref[k] for k in ("file", "sd", "evals", "steps"))
    good = err is None and got.get("ret") is False and calls["n"] == 1 and st is not None and not st.on \
        and str(st.verdict).startswith("FAIL") and same
    ok &= good
    say(f"release raising in the guard's except: guard returned {got.get('ret')}, release calls {calls['n']}, run "
        f"{err or str(r['steps']) + ' steps'}, byte-identical to FAST=0 {same} -> {'ok' if good else 'FAIL'}")
    calls["n"] = 0
    T._fast_release = boom
    try:
        r = harness_run(T, True, "relraise_retry", inject=("backward", 20))
    except Exception as e:  # noqa: BLE001
        r, err = None, f"{type(e).__name__}: {str(e)[:120]}"
    else:
        err = None
    finally:
        T._fast_release = orig_rel
    st = r["st"] if r else None
    good = err is None and r["steps"] == BASE["max_steps"] and calls["n"] == 1 and st is not None and not st.on \
        and st.retries == 1 and "injected" in str(st.off_reason)
    ok &= good
    say(f"release raising in _fast_retry (failure in fast backward #20 after the guard): release calls {calls['n']}, run "
        f"{err or str(r['steps']) + ' steps'}, retries {st.retries if st else None} -> {'ok' if good else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------------------------------------- main
def isolate():
    """Between checks: collect reference cycles (a mutant's failed checkpointed forward keeps checkpoint()'s generator,
    and with it its saved-tensor hooks, alive until the cyclic GC runs - that poisoned the next mutant's first check) and
    pop any saved-tensor hooks still pushed. Returns the number of hooks popped."""
    import gc
    gc.collect()
    n = 0
    top = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
    while top is not None and top(False) is not None:
        torch._C._autograd._pop_saved_tensors_default_hooks()
        n += 1
    return n


def run_checks(T, ref, stop=False, only=None):
    fns = {"default": check_default, "ops": check_ops, "models": check_models, "guard": check_guard, "merge": check_merge,
           "deep": check_deep, "fallback": check_fallback, "release": check_release, "release_raise": check_release_raise}
    fails = []
    for name in CHECKS:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            good = fns[name](T, ref)
        except Exception as e:  # noqa: BLE001  (a crash is a failure of that check)
            say(f"check {name}: {type(e).__name__}: {str(e)[:300]}")
            good = False
        leaked = isolate()
        if leaked:  # the check left saved-tensor hooks behind (a dead checkpoint frame): that is a failure too
            say(f"check {name}: {leaked} saved-tensor hook set(s) still pushed after the check")
            good = False
        say(f"check {name}: {'PASS' if good else 'FAIL'} ({time.time() - t0:.1f}s)")
        if not good:
            fails.append(name)
            if stop:
                break
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutation", choices=sorted(MUTATIONS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = [x for x in a.only.split(",") if x] or None
    T0 = load_trainer()
    T0.patch_flux_inplace()  # flux blocks train out of place (the trainer does this in main)
    ref = harness_run(T0, False, "ref")  # FAST=0 reference of the harness run (the current code, knob off)
    say(f"harness FAST=0 reference: {ref['steps']} steps, {len(ref['evals'])} evals, fast calls {ref['calls']}, "
        f"no fast state {ref['st'] is None}")
    if a.mutation:
        fails = run_checks(load_trainer(a.mutation), ref, stop=True, only=only)
        say(f"mutation {a.mutation}: first failing check {fails[0] if fails else None} (expected "
            f"{MUTATIONS[a.mutation][1]})")
        say(f"RESULT {'FAIL' if fails else 'PASS'}")
        sys.exit(1 if fails else 0)
    ok = ref["st"] is None and ref["calls"] == 0
    fails = run_checks(T0, ref, only=only)
    ok &= not fails
    say(f"checks on the current code: {'PASS' if not fails else 'FAIL ' + str(fails)}")
    if not only:
        # a failing guard (no_scale mutant, s = 2 in the harness) falls back before step 1: byte-identical to FAST=0
        Tm = load_trainer("no_scale")
        rm = harness_run(Tm, True, "guardfail")
        fb = rm["st"] is not None and not rm["st"].on and str(rm["st"].verdict).startswith("FAIL") and rm["calls"] > 0 \
            and all(rm[k] == ref[k] for k in ("file", "sd", "evals", "steps"))
        say(f"guard failure (no_scale mutant): {str(rm['st'].verdict if rm['st'] else None)[:200]}; run byte-identical to "
            f"FAST=0 (file, state dict, evals) {all(rm[k] == ref[k] for k in ('file', 'sd', 'evals', 'steps'))} -> "
            f"{'ok' if fb else 'FAIL'}")
        ok &= fb
    if not a.quick and not only:
        for mut, (_, want) in MUTATIONS.items():
            fails = run_checks(load_trainer(mut), ref, stop=True)
            first = fails[0] if fails else None
            say(f"mutation {mut}: first failing check {first} (expected {want}) -> {'caught' if first == want else 'NOT CAUGHT'}")
            ok &= first == want
    say(f"RESULT {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
