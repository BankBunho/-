#!/usr/bin/env python3
"""G56CT_SAFE check, CPU only, on the te_lora_harness fake DiT (no GPU, no model files). Never run on a GPU before
30 Sep -> treated as broken until this passes.

refuse        save_lora with SAFE=1: a NaN / inf state over a valid file returns False, the file keeps its bytes, no .tmp is
              left; with SAFE=0 the same call writes the non-finite file (the knob is what refuses).
start_clamp   _anchor_start: SAFE=0 == the old float(G56CT_START_TS or T0) (malformed raises); SAFE=1 keeps a value in
              [T0 - 600 s, T0] and falls back to T0 with a warning for T0 + 1000, T0 - 5000, 'abc', 'nan', 'inf'.
equal_on      SAFE=1 vs SAFE=0 on finite runs: holdout default, holdout seeds=2, accum=2, all-data save_every=5,
              seeds_auto=3 + SAVE_RUNS, fixed seeds=2 all-data -> identical evals, best state dict, last.safetensors bytes,
              steps.
nf_step       one training step with a NaN prediction (all-data and holdout): SAFE=1 skips it (steps == max_steps, one
              'non-finite step skipped (loss nan' line, file finite and equal to the returned state); SAFE=0 (the
              injection is real): all-data ships a non-finite file, holdout dies at the next eval (evalcore raises
              'Non-finite denoising MSE'; in production cli then ships the last saved file).
nf_grad       one step with an inf output gradient while the loss stays finite: SAFE=1 skips it (clip_grad_norm_
              error_if_nonfinite), file finite; SAFE=0: as nf_step.
nf_intermittent  every second step non-finite, SAFE_NF_MAX=2: finite steps reset the in-a-row count, the run is not
              abandoned and makes all its steps.
nf_member     seeds_auto=3 + SAVE_RUNS, every step of member 2 non-finite, SAFE_NF_MAX=5: member 2 abandoned, the shipped
              file == save_file(average_loras([run0, run2])) byte for byte; member 1 non-finite instead: == avg(run1, run2)
              (the abandoned first member is not averaged once a member completes); every member non-finite: the run
              still ends normally and ships the first member's (finite) state.
nan_first_eval  holdout mode, the first holdout eval non-finite - a NaN score, or the evaluator core raising on the
              non-finite MSE (what evalcore does): SAFE=1 picks the best FINITE eval (file == its state); SAFE=0 keeps
              the NaN as the best forever (lock-in) / dies.
te_drop       te_drop_check: with TE keys non-finite, DiT only finite -> SAFE=1 drops the TE keys (SAFE=0 keeps them).
save_diet     seeds_auto=3, save_every=5: save_lora calls SAFE=0 = 2 interim + 3 member-end + 3 running + 1 final = 9,
              SAFE=1 = 2 interim + 3 running = 5 ('safe: final reused'); identical file bytes.
margin        virtual clock (1 s per training step; a member-size write costs 100 s; the identity file main() writes
              first is the first measurement): A) one time-bound member, B) a second member slower than planned (4 s
              per step). SAFE=0 finishes the last write of last.safetensors after the deadline; SAFE=1 before it.
oom_swallowed the real exact_patch_apply on the harness DiT, ComfyUI's calculate_weight swallowing a CUDA OOM for one key
              (logs 'ERROR lora <key> ...', weight without delta; review B2): SAFE=1 restores the weights exactly and
              scores the holdout live from then on (one public WARN, evals == a live-scored run); same for a non-OOM text
              and 'Calculate Weight Failed'; SAFE=0 scores the partly merged LoRA as before.
Mutations (source level, on a copy of the trainer text): each must make the check named in MUTATIONS fail first.
Usage (CPU env with COMFY_ROOT): python comfytrain/tools/safe_check.py [--mutation NAME] [--quick] [--only a,b]
"""
import argparse
import hashlib
import math
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

# name -> ([(pattern, replacement), ...] each pattern exactly once in the trainer, the check that must fail first)
MUTATIONS = {
    "save_nocheck": ([('        if not _sd_finite(sd):\n            log("WARN safe: refused non-finite state',
                       '        if False:\n            log("WARN safe: refused non-finite state')], "refuse"),
    "clamp_max": ([("return min(t0, v), None", "return max(t0, v), None")], "start_clamp"),
    "clip_nocheck": ([("return torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True)",
                       "return torch.nn.utils.clip_grad_norm_(params, max_norm)")], "nf_grad"),
    "nf_noreset": ([("        nf_run = 0\n        if fm_arm:", "        if fm_arm:")], "nf_intermittent"),
    "abandoned_avg": ([('        if r.get("abandoned"):\n', "        if False:\n")], "nf_member"),
    "nan_best": ([('if safe and not math.isfinite(r["score"]):  # G56CT_SAFE: never the best',
                   'if False:  # G56CT_SAFE: never the best')], "nan_first_eval"),
    "eval_raise": ([('        if "Non-finite denoising MSE" not in str(e):\n            raise\n',
                     '        if True:\n            raise\n')], "nan_first_eval"),
    "te_drop_nan": ([('drop = r["score"] < full or (c["cfg"].get("safe", False) and not math.isfinite(full) '
                      'and math.isfinite(r["score"]))', 'drop = r["score"] < full')], "te_drop"),
    # the member-end save back in multi mode: an abandoned member's EMA then sits on disk as the "running average" and
    # is shipped (nf_member, first); save_diet counts the extra writes too (--mutation member_save --only save_diet)
    "member_save": ([('if not (safe and c.get("multi")):', "if True:")], "nf_member"),
    "final_rewrite": ([("if self.last is not None and self.last[0] == self.key():", "if False:")], "save_diet"),
    "margin_fixed": ([("margin = max(margin, _save_margin(c))", "margin = max(margin, 0.0)")], "margin"),
    "oom_raise": ([("                    if not safe:\n                        raise\n                    sd_x = None\n",
                    "                    if True:\n                        raise\n                    sd_x = None\n")], "oom_exact"),
    # no log handler in exact_patch_apply: a merge failure ComfyUI swallows is scored as an exact eval (review B2)
    "swallow_accept": ([("    h = MergeLog() if strict else None\n", "    h = None\n")], "oom_swallowed"),
}
CHECKS = ("refuse", "start_clamp", "equal_on", "nf_step", "nf_grad", "nf_intermittent", "nf_member", "nan_first_eval",
          "te_drop", "save_diet", "margin", "oom_exact", "oom_swallowed")
LOGS = []
BASE = dict(case_cycle=True, sigma_mode="mid", max_steps=12)  # harness: holdout {0,3,5}, evals at 6 and 12
ALL = dict(BASE, harness_all_data=True, holdout_frac=0.0)


def say(*a):
    print("[safe]", *a, flush=True)


def load_trainer(mutation=None):
    src = TRAINER.read_text()
    if mutation:
        for a, b in MUTATIONS[mutation][0]:
            if src.count(a) != 1:
                raise SystemExit(f"[safe] mutation {mutation}: pattern occurs {src.count(a)}x, not exactly once: {a!r}")
            src = src.replace(a, b)
    mod = types.ModuleType(f"g56safe_{mutation or 'code'}")
    mod.__file__ = str(TRAINER)
    exec(compile(src, f"{TRAINER}{f' [mutation {mutation}]' if mutation else ''}", "exec"), mod.__dict__)
    mod.log = lambda *a, **k: LOGS.append(" ".join(str(x) for x in a))
    mod._cfg0 = dict(mod.CFG)
    return mod


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def finite_file(p):
    from safetensors.torch import load_file
    return Path(p).exists() and all(bool(torch.isfinite(v).all()) for v in load_file(str(p)).values()
                                    if v.is_floating_point())


def file_digest(p):
    from safetensors.torch import load_file
    return H.sd_digest(load_file(str(p))) if Path(p).exists() else None


def hrun(T, over, tag, safe, nan=None, grad_inf=None, hook=None, wrap_tl=None):
    """One te_lora_harness training run. nan / grad_inf: predicates (run index, attempt index) -> inject a NaN
    prediction / an inf output gradient on the first training pass of that attempt (paired prompts: 2 passes per
    attempt). hook(state, n): FakeWrap predict hook (every training forward). wrap_tl(orig) -> train_loop replacement."""
    T.CFG.clear()
    T.CFG.update(T._cfg0)
    T._SAVE_STATS.clear()
    LOGS.clear()
    out = Path(tempfile.mkdtemp(prefix=f"safe_{tag}_"))
    state = {"run": None, "calls": {}}
    orig_run, orig_pred, orig_tl = T._run_one, T.predict, T.train_loop

    def run_one(c, mw, conds, ea, dev, seed, t_end, run_idx):
        state["run"] = run_idx
        state["calls"][run_idx] = 0
        return orig_run(c, mw, conds, ea, dev, seed, t_end, run_idx)

    def pred(model_wrap, conds, cond_idx, x, sigma_value, extra_args):
        den = orig_pred(model_wrap, conds, cond_idx, x, sigma_value, extra_args)
        if torch.is_grad_enabled() and state["run"] is not None:
            n = state["calls"][state["run"]]
            state["calls"][state["run"]] = n + 1
            att, pas = divmod(n, 2)
            if pas == 0 and nan is not None and nan(state["run"], att):
                den = den * float("nan")
            if pas == 0 and grad_inf is not None and grad_inf(state["run"], att):
                den.register_hook(lambda g: g * float("inf"))
        return den

    T._run_one, T.predict = run_one, pred
    if wrap_tl is not None:
        T.train_loop = wrap_tl(orig_tl)
    try:
        c, res, _ = H.build_and_train(T, out, None, "krea2", te_on=False, cfg_over=dict(over, safe=safe),
                                      predict_hook=(lambda n: hook(state, n)) if hook else None)
    finally:
        T._run_one, T.predict, T.train_loop = orig_run, orig_pred, orig_tl
    f = out / "last.safetensors"
    sd = res.get("best_sd")
    return {"c": c, "res": res, "out": out, "file": sha(f), "fdig": file_digest(f), "usable": T.usable_adapter(out),
            "evals": H.strip_t(res["evals"]), "sd": H.sd_digest(sd) if sd is not None else None,
            "sd_finite": sd is not None and all(bool(torch.isfinite(v).all()) for v in sd.values() if v.is_floating_point()),
            "steps": res["steps"], "logs": list(LOGS), "tmp_left": (out / ".last.safetensors.tmp").exists()}


def nlog(r, text):
    return sum(1 for x in r["logs"] if text in x)


# ---------------------------------------------------------------------------------------------------------- checks
def check_refuse(T, ref=None):
    ok = True
    for safe in (True, False):
        T.CFG.clear()
        T.CFG.update(T._cfg0)
        T.CFG["safe"] = safe
        LOGS.clear()
        d = Path(tempfile.mkdtemp(prefix="safe_refuse_"))
        g = torch.Generator().manual_seed(0)
        good = {"m.lora_up.weight": torch.randn(4, 2, generator=g), "m.lora_down.weight": torch.randn(2, 4, generator=g),
                "m.alpha": torch.tensor(2.0)}
        r0 = T.save_lora(good, d)
        s0 = sha(d / "last.safetensors")
        res = []
        for bad_v in (float("nan"), float("inf"), -float("inf")):
            bad = dict(good)
            up = good["m.lora_up.weight"].clone()
            up[1, 0] = bad_v
            bad["m.lora_up.weight"] = up
            r = T.save_lora(bad, d)
            res.append((r, sha(d / "last.safetensors"), sorted(p.name for p in d.iterdir())))
        if safe:
            good_ = r0 is True and all(r is False and s == s0 and names == ["last.safetensors"] for r, s, names in res) \
                and T.usable_adapter(d) and nlog({"logs": LOGS}, "safe: refused non-finite state") == 3
            say(f"refuse SAFE=1: valid write {r0}, NaN / inf / -inf over it -> {[x[0] for x in res]}, file bytes kept "
                f"{all(x[1] == s0 for x in res)}, folder {res[-1][2]} -> {'ok' if good_ else 'FAIL'}")
        else:
            good_ = res[-1][1] != s0 and not T.usable_adapter(d)
            say(f"refuse SAFE=0 (knob off): the non-finite state is written (file changed {res[-1][1] != s0}, usable "
                f"{T.usable_adapter(d)}) -> {'ok' if good_ else 'FAIL'}")
        ok &= good_
    T.CFG.clear()
    T.CFG.update(T._cfg0)
    return ok


def check_start_clamp(T, ref=None):
    t0 = 1_790_000_000.0
    ok = True
    for raw in (None, str(t0 - 100), str(t0 + 1000), str(t0 - 5000)):  # SAFE=0: the old expression exactly
        want = float(raw if raw is not None else t0)
        got = T._anchor_start(raw, t0, False)
        ok &= got == (want, None)
    try:
        T._anchor_start("abc", t0, False)
        raised = False
    except ValueError:
        raised = True
    ok &= raised
    say(f"start_clamp SAFE=0: == float(G56CT_START_TS or T0), 'abc' raises {raised} -> {'ok' if ok else 'FAIL'}")
    cases = [(None, t0, False), ("", t0, False), (str(t0 - 100), t0 - 100, False), (str(t0), t0, False),
             (str(t0 - 600), t0 - 600, False), (str(t0 + 1000), t0, True), (str(t0 + 0.5), t0, True),
             (str(t0 - 5000), t0, True), (str(t0 - 601), t0, True), ("abc", t0, True), ("nan", t0, True),
             ("inf", t0, True)]
    res = []
    for raw, want, warn in cases:
        got, why = T._anchor_start(raw, t0, True)
        good = got == want and (why is not None) == warn
        res.append(good)
        if not good:
            say(f"  start_clamp SAFE=1 {raw!r}: got {got - t0:+.1f} s from T0, warning {why!r} (want {want - t0:+.1f}, "
                f"warning {warn})")
    ok1 = all(res)
    say(f"start_clamp SAFE=1: {sum(res)}/{len(res)} cases (in [T0-600, T0] kept, future / stale / malformed -> T0 with a "
        f"warning) -> {'ok' if ok1 else 'FAIL'}")
    return ok and ok1


def check_equal_on(T, ref=None):
    rd = Path(tempfile.mkdtemp(prefix="safe_eq_research_"))
    confs = {"holdout": dict(BASE), "holdout_seeds2": dict(BASE, seeds=2), "accum2": dict(BASE, accum=2),
             "alldata_save_every5": dict(ALL, save_every=5),
             "seeds_auto3_save_runs": dict(ALL, seeds_auto=3, save_runs=True, research_dir=str(rd)),
             "alldata_seeds2": dict(ALL, seeds=2)}
    ok = True
    for name, over in confs.items():
        orig = T.exact_patch_apply
        if name == "holdout_seeds2":  # the combos score through the exact DiT merge, which needs a real Comfy key map
            T.exact_patch_apply = lambda c_, sd_, dev, **k: []  # (stub as in te_lora_harness mode on; same in both arms)
        try:
            a, b = hrun(T, over, f"eq0_{name}", False), hrun(T, over, f"eq1_{name}", True)
        finally:
            T.exact_patch_apply = orig
        same = {k: a[k] == b[k] for k in ("evals", "sd", "file", "steps")}
        good = all(same.values()) and a["file"] is not None and b["usable"]
        ok &= good
        say(f"equal_on {name}: steps {a['steps']}/{b['steps']}, evals {len(a['evals'])}, identical "
            f"{[k for k, v in same.items() if v]}" + (f" DIFFERENT {[k for k, v in same.items() if not v]}"
                                                      if not all(same.values()) else "") + f" -> {'ok' if good else 'FAIL'}")
    return ok


def knob_off_outcome(T, over, tag, **inj):
    """SAFE=0 arm of an injection: all-data -> a non-finite file is shipped; holdout -> the next holdout eval raises
    (evalcore: 'Non-finite denoising MSE') and the run dies (production: cli ships the last saved file)."""
    try:
        r0 = hrun(T, over, tag, False, **inj)
    except ValueError as e:
        return "raised" if "Non-finite denoising MSE" in str(e) else f"raised {e}"
    return "non-finite file" if (not r0["usable"] and r0["file"] is not None) else "finite file"


def check_nf_step(T, ref=None):
    ok = True
    for mode, over, want0 in (("all-data", ALL, "non-finite file"), ("holdout", BASE, "raised")):
        inj = (lambda run, att: run == 0 and att == 3)
        r1, o0 = hrun(T, over, f"nfs1_{mode}", True, nan=inj), knob_off_outcome(T, over, f"nfs0_{mode}", nan=inj)
        n = nlog(r1, "safe: non-finite step skipped (loss nan")
        good1 = r1["steps"] == over["max_steps"] and n == 1 and r1["usable"] and r1["sd_finite"] \
            and r1["fdig"] == r1["sd"] and not r1["tmp_left"]
        ok &= good1 and o0 == want0
        say(f"nf_step {mode}: SAFE=1 steps {r1['steps']} (max {over['max_steps']}), 'skipped (loss nan' lines {n}, file "
            f"finite {r1['usable']} == returned state {r1['fdig'] == r1['sd']}; SAFE=0: {o0} (want {want0}) -> "
            f"{'ok' if good1 and o0 == want0 else 'FAIL'}")
    return ok


def check_nf_grad(T, ref=None):
    ok = True
    for mode, over, want0 in (("all-data", ALL, "non-finite file"), ("holdout", BASE, "raised")):
        inj = (lambda run, att: run == 0 and att == 4)
        r1, o0 = hrun(T, over, f"nfg1_{mode}", True, grad_inf=inj), knob_off_outcome(T, over, f"nfg0_{mode}", grad_inf=inj)
        n = nlog(r1, "safe: non-finite step skipped (gradient norm not finite")
        good1 = r1["steps"] == over["max_steps"] and n == 1 and r1["usable"] and r1["sd_finite"] and r1["fdig"] == r1["sd"]
        ok &= good1 and o0 == want0
        say(f"nf_grad {mode}: SAFE=1 steps {r1['steps']}, 'gradient norm not finite' lines {n}, returned state finite "
            f"{r1['sd_finite']}, file == it {r1['fdig'] == r1['sd']}; SAFE=0: {o0} (want {want0}) -> "
            f"{'ok' if good1 and o0 == want0 else 'FAIL'}")
    return ok


def check_nf_intermittent(T, ref=None):
    r = hrun(T, dict(ALL, safe_nf_max=2), "nfi", True, nan=lambda run, att: att % 2 == 1 and att < 13)
    res = r["res"]
    good = r["steps"] == 12 and res.get("abandoned") is False and res.get("nonfinite_steps") == 6 and r["usable"] \
        and nlog(r, "member abandoned") == 0
    say(f"nf_intermittent (every 2nd step non-finite, SAFE_NF_MAX=2): steps {r['steps']}, non-finite {res.get('nonfinite_steps')}, "
        f"abandoned {res.get('abandoned')} -> {'ok' if good else 'FAIL'}")
    return good


def _avg_file(T, paths, tag):
    """save_lora's bytes for the exact average of the given member files (save_file with metadata {})."""
    from safetensors.torch import load_file, save_file
    sds = [load_file(str(p)) for p in paths]
    p = Path(tempfile.mkdtemp(prefix=f"safe_avg_{tag}_")) / "ref.safetensors"
    save_file(T.average_loras(sds) if len(sds) > 1 else sds[0], str(p), metadata={})
    return sha(p)


def check_nf_member(T, ref=None):
    ok = True
    for bad, keep in ((1, (0, 2)), (0, (1, 2)), (None, (0,))):
        rd = Path(tempfile.mkdtemp(prefix="safe_nfm_research_"))
        over = dict(ALL, seeds_auto=3, save_runs=True, research_dir=str(rd), safe_nf_max=5)
        inj = (lambda run, att, _b=bad: _b is None or run == _b)
        r = hrun(T, over, f"nfm_{bad}", True, nan=inj)
        want = _avg_file(T, [rd / f"run{k}.safetensors" for k in keep], f"nfm{bad}")
        n_ab = nlog(r, "safe: member abandoned after 6 non-finite steps in a row")
        tag = r["res"]["best"]["tag"]
        good = r["file"] == want and r["usable"] and n_ab == (1 if bad is not None else 3) \
            and tag == f"auto_avg{len(keep)}" and not r["tmp_left"]
        ok &= good
        what = f"member {bad + 1}" if bad is not None else "every member"
        say(f"nf_member ({what} non-finite, SAFE_NF_MAX=5): abandoned lines {n_ab}, tag {tag}, shipped == "
            f"save_file(average_loras({['run%d' % k for k in keep]})) byte for byte {r['file'] == want} -> "
            f"{'ok' if good else 'FAIL'}")
    return ok


def check_nan_first_eval(T, ref=None):
    """The first holdout eval non-finite, two ways: the score comes back NaN, or the evaluator core raises on the
    non-finite MSE (what evalcore does). SAFE=1: that eval is never the best, the best is the argmin of the finite evals
    and the file is its state. SAFE=0: NaN stays the best forever (lock-in) / the run dies."""
    ok = True
    for how in ("nan", "raise"):
        for safe in (True, False):
            calls = {"n": 0}
            orig = T.holdout_score_te

            def hs(*a, _how=how, **k):
                calls["n"] += 1
                if calls["n"] == 1 and _how == "raise":
                    raise ValueError("Non-finite denoising MSE")
                r = orig(*a, **k)
                if calls["n"] == 1 and r is not None:
                    r = dict(r, score=float("nan"), text=float("nan"), no_text=float("nan"))
                return r
            T.holdout_score_te = hs
            err = None
            try:
                r = hrun(T, BASE, f"nan1_{how}{int(safe)}", safe)
            except ValueError as e:
                err = str(e)
            finally:
                T.holdout_score_te = orig
            if err is not None:
                good = not safe and how == "raise" and "Non-finite denoising MSE" in err
                say(f"nan_first_eval {how} SAFE={int(safe)}: the run raised ({err}) -> "
                    f"{'ok (the crash SAFE removes)' if good else 'FAIL'}")
                ok &= good
                continue
            best = r["res"]["best"]
            fin = [e for e in r["evals"] if math.isfinite(e["score"])]
            amin = min(fin, key=lambda e: e["score"]) if fin else None
            if safe:
                good = amin is not None and best.get("step") == amin["step"] and best.get("tag") == amin["tag"] \
                    and math.isfinite(best.get("score", float("nan"))) and r["fdig"] == r["sd"] and r["usable"] \
                    and len(r["evals"]) == 4 and not math.isfinite(r["evals"][0]["score"])
                say(f"nan_first_eval {how} SAFE=1: evals {[(e['step'], e['tag'], round(e['score'], 6)) for e in r['evals']]}; "
                    f"best {best.get('tag')}@{best.get('step')} = argmin of the finite evals "
                    f"{amin and (amin['tag'], amin['step'])}, file == its state {r['fdig'] == r['sd']} -> "
                    f"{'ok' if good else 'FAIL'}")
            else:
                good = how == "nan" and not math.isfinite(best.get("score", 0.0)) and best.get("step") == r["evals"][0]["step"]
                say(f"nan_first_eval {how} SAFE=0 (knob off): the NaN eval stays the best ({best.get('tag')}@"
                    f"{best.get('step')} score {best.get('score')}) -> {'ok (the lock-in SAFE removes)' if good else 'FAIL'}")
            ok &= good
    return ok


def check_te_drop(T, ref=None):
    """te_drop_check with SAFE: a non-finite evaluator score WITH the TE keys and a finite one without -> the TE keys are
    dropped (x < NaN is False: with SAFE off the non-finite file is kept)."""
    ok = True
    sd = {"diffusion_model.a.lora_up.weight": torch.zeros(2, 2), "text_encoders.b.lora_up.weight": torch.zeros(2, 2)}
    orig = T.score_sd_exact
    for safe in (True, False):
        c = {"cfg": dict(T._cfg0, te_drop_check=True, safe=safe), "holdout": [0], "exact": False,
             "deadline": time.time() + 1e6, "eval_cost": 1.0}
        T.score_sd_exact = lambda c_, mw, cd, ea, dev, s_: {"score": float("nan") if any(
            k.startswith("text_encoders.") for k in s_) else 0.5}
        try:
            sc, tag, out = T.te_drop_check(c, None, None, None, None, (None, "best", sd))
        finally:
            T.score_sd_exact = orig
        dropped = not any(k.startswith("text_encoders.") for k in out)
        good = dropped == safe
        ok &= good
        say(f"te_drop SAFE={int(safe)}: with TE NaN, DiT only 0.5 -> TE keys {'dropped' if dropped else 'kept'} "
            f"(want {'dropped' if safe else 'kept (knob off)'}) -> {'ok' if good else 'FAIL'}")
    return ok


def check_save_diet(T, ref=None):
    over = dict(ALL, seeds_auto=3, save_every=5)
    res = {}
    for safe in (False, True):
        cnt = {"n": 0}
        orig = T.save_lora

        def counted(sd, out_dir, meta=None, _o=orig):
            cnt["n"] += 1
            return _o(sd, out_dir, meta)
        T.save_lora = counted
        try:
            r = hrun(T, over, f"diet{int(safe)}", safe)
        finally:
            T.save_lora = orig
        res[safe] = (cnt["n"], r["file"], nlog(r, "safe: final reused"))
    good = res[False][0] == 9 and res[True][0] == 5 and res[False][1] == res[True][1] and res[True][2] == 1
    say(f"save_diet seeds_auto=3 save_every=5: save_lora calls SAFE=0 {res[False][0]} (want 9 = 2 interim + 3 member-end "
        f"+ 3 running + 1 final), SAFE=1 {res[True][0]} (want 5 = 2 interim + 3 running; 'final reused' "
        f"{res[True][2]}x); same file bytes {res[False][1] == res[True][1]} -> {'ok' if good else 'FAIL'}")
    return good


def margin_run(T, safe, scen):
    """Virtual clock: scen = (seeds_auto, max_steps, deadline D after the train_loop start, {run: s per forward});
    X = 100 s per member-size write (save_file), the identity file written first (as main() with FALLBACK=1).
    Returns (virtual times at which a last.safetensors write completed, deadline, run result)."""
    import safetensors.torch as st
    seeds_auto, max_steps, D, per = scen
    X = 100.0
    clock = [1000.0]
    writes, unit = [], {}
    real_save, real_time = st.save_file, T.time

    def fake_save(tensors, filename, metadata=None):
        clock[0] += X * T._sd_nbytes(tensors) / unit["b"]
        real_save(tensors, filename, metadata=metadata)
        if str(filename).endswith(".last.safetensors.tmp"):
            writes.append(clock[0])

    def wrap_tl(orig):
        def tl(c, *a, **k):
            ident = T.lora_state_dict(c["adapters"], fp32=True)
            unit["b"] = T._sd_nbytes(ident)
            start = clock[0]
            T.save_lora(ident, c["out_dir"])  # main(): the fallback identity LoRA before training
            c["deadline"] = unit["deadline"] = start + D
            return orig(c, *a, **k)
        return tl

    def hook(state, n):
        clock[0] += per.get(state["run"], per[0])

    st.save_file = fake_save
    T.time = types.SimpleNamespace(time=lambda: clock[0])
    try:
        r = hrun(T, dict(ALL, seeds_auto=seeds_auto, max_steps=max_steps), f"margin{int(safe)}", safe, hook=hook,
                 wrap_tl=wrap_tl)
    finally:
        st.save_file, T.time = real_save, real_time
    return writes, unit["deadline"], r


def check_margin(T, ref=None):
    ok = True
    scens = {"A one time-bound member": (2, 10000, 500.0, {0: 0.5}),
             "B second member 4x slower than planned": (2, 100, 800.0, {0: 0.5, 1: 2.0})}
    for name, scen in scens.items():
        res = {}
        for safe in (False, True):
            writes, dl, r = margin_run(T, safe, scen)
            res[safe] = (max(writes) - dl, [round(w - dl, 1) for w in writes], [x.get("steps") for x in
                                                                                 [r["res"]] if x], r["usable"])
        good = res[False][0] > 0 and res[True][0] <= 0 and res[True][3]
        ok &= good
        say(f"margin {name}: last write vs deadline SAFE=0 {res[False][0]:+.1f} s (writes {res[False][1]}), SAFE=1 "
            f"{res[True][0]:+.1f} s (writes {res[True][1]}) -> {'ok' if good else 'FAIL'}")
    return ok


def check_oom_exact(T, ref=None):
    """Holdout mode on the evaluator-exact eval path (c["exact"] forced on; exact_patch_apply replaced by a stand-in that
    raises torch.OutOfMemoryError like the QMODs0 crash of 1 Oct at the first eval). SAFE=1: the run goes on, scores the
    holdout through the live wrappers from then on (c["exact"] False, one WARN), trains every step and ships the best
    finite state (file == best_sd, not the identity fallback). SAFE=0: the run raises (the crash that shipped identity)."""
    ok = True
    for safe in (True, False):
        orig_apply = T.exact_patch_apply
        calls = {"n": 0}

        def apply_oom(*a, **k):
            calls["n"] += 1
            raise torch.OutOfMemoryError("CUDA out of memory (simulated exact merge, safe_check)")

        orig_run = T._run_one

        def run_exact(c, *a, **k):  # exact eval path inside the run only (the base score before it stays harness-native)
            c["exact"] = True
            return orig_run(c, *a, **k)
        T.exact_patch_apply, T._run_one = apply_oom, run_exact
        err = None
        try:
            r = hrun(T, BASE, f"oom{int(safe)}", safe)
        except torch.OutOfMemoryError as e:
            err = str(e)
        finally:
            T.exact_patch_apply, T._run_one = orig_apply, orig_run
        if not safe:
            good = err is not None
            say(f"oom_exact SAFE=0: the run {'raised (the crash SAFE removes)' if good else 'did NOT raise'} -> {'ok' if good else 'FAIL'}")
            ok &= good
            continue
        if err is not None:
            say(f"oom_exact SAFE=1: the run raised ({err[:80]}) -> FAIL")
            ok = False
            continue
        warn = nlog(r, "WARN safe: out of memory in the evaluator-exact holdout eval")
        good = (calls["n"] == 1 and r["c"]["exact"] is False and warn == 1 and r["steps"] == BASE["max_steps"]
                and len(r["evals"]) == 4 and all(math.isfinite(e["score"]) for e in r["evals"])
                and r["fdig"] == r["sd"] and r["usable"] and r["sd_finite"])
        say(f"oom_exact SAFE=1: exact merge attempts {calls['n']}, exact now {r['c']['exact']}, WARN lines {warn}, steps "
            f"{r['steps']}, evals {len(r['evals'])} finite, file == best state {r['fdig'] == r['sd']} -> {'ok' if good else 'FAIL'}")
        ok &= good
    return ok


SWALLOW_KEY = "diffusion_model.blocks.0.mlp.down.weight"  # the last of the harness DiT's three LoRA keys to be patched
SWALLOW_TEXT = {"oom": "CUDA out of memory. Tried to allocate 108.00 MiB. GPU 0 has a total capacity of 79.18 GiB of which "
                       "20.19 MiB is free.", "error": "mat1 and mat2 shapes cannot be multiplied (32x4 and 16x4)"}


def _exact_ready(c):
    """The harness DiT on the evaluator-exact path: Comfy's LoRA key map reads model_config.unet_config, the merge needs a
    real ModelPatcher (FakeMP only applies the weight wrappers)."""
    import comfy.model_patcher
    root, cpu = c["mp"].model, torch.device("cpu")
    root.model_config = types.SimpleNamespace(unet_config={})
    c["mp"].clone = lambda: comfy.model_patcher.ModelPatcher(root, load_device=cpu, offload_device=cpu)
    c["exact"] = True


def _model_state(root):
    """(name, Parameter object, bytes digest, weight / bias function objects) of every parameter of the model."""
    out = []
    for name, m in root.named_modules():
        for pn, p in m._parameters.items():
            if p is not None:
                b = p.detach().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
                out.append((f"{name}.{pn}", id(p), hashlib.sha256(b).hexdigest(),
                            tuple(id(f) for f in (getattr(m, f"{pn}_function", None) or []))))
    return out


def swallow_run(T, safe, how, exact=True):
    """hrun (holdout, BASE) with the holdout eval on the REAL evaluator-exact path and ComfyUI's
    LoRAAdapter.calculate_weight failing the way it does for SWALLOW_KEY: how 'oom' / 'error' -> logs 'ERROR lora <key>
    <text>' and returns the weight without the delta (comfy/weight_adapter/lora.py:284-286), 'none' -> returns None
    (comfy.lora.calculate_weight: 'Calculate Weight Failed'), None -> no failure. Returns (run, exact merge attempts,
    [(exception, weights + weight functions after it == before)], keys patched before the failing one, public log lines,
    root logging handlers added)."""
    import logging
    import comfy.weight_adapter.lora as wal
    orig_cw, orig_apply, orig_run, orig_log = wal.LoRAAdapter.calculate_weight, T.exact_patch_apply, T._run_one, T.log
    calls, undo, seen, pub = {"n": 0}, [], [], []
    n_handlers = len(logging.getLogger().handlers)

    def cw(self, weight, key, *a, **k):
        seen.append(key)
        if how is not None and key == SWALLOW_KEY:
            if how == "none":
                return None
            logging.error("ERROR {} {} {}".format(self.name, key, SWALLOW_TEXT[how]))  # what calculate_weight's except logs
            return weight
        return orig_cw(self, weight, key, *a, **k)

    def apply(c_, sd_, dev, **k):
        calls["n"] += 1
        before = _model_state(c_["mp"].model)
        try:
            return orig_apply(c_, sd_, dev, **k)
        except Exception as e:  # noqa: BLE001
            undo.append((type(e).__name__, _model_state(c_["mp"].model) == before))
            raise

    def run_exact(c_, *a, **k):
        if exact:
            _exact_ready(c_)
        return orig_run(c_, *a, **k)

    def log_(*a, **k):
        if k.get("public"):
            pub.append(" ".join(str(x) for x in a))
        orig_log(*a, **k)
    wal.LoRAAdapter.calculate_weight, T.exact_patch_apply, T._run_one, T.log = cw, apply, run_exact, log_
    try:
        r = hrun(T, BASE, f"swal{int(safe)}_{how}_{int(exact)}", safe)
    finally:
        wal.LoRAAdapter.calculate_weight, T.exact_patch_apply, T._run_one, T.log = orig_cw, orig_apply, orig_run, orig_log
    before_fail = seen.index(SWALLOW_KEY) if SWALLOW_KEY in seen else None
    return r, calls["n"], undo, before_fail, pub, len(logging.getLogger().handlers) - n_handlers


def check_oom_swallowed(T, ref=None):
    """review B2 (1 Oct): ComfyUI's LoRAAdapter.calculate_weight catches every exception (a CUDA OOM too), logs 'ERROR lora
    <key> <error>' and returns the weight WITHOUT the delta (QMODs0__b1546f69.log:27-29: three such lines before the OOM
    that did propagate) -> the exact holdout eval scored a partly merged LoRA. On the real exact path (swallow_run):
    SAFE=1 -> the failed merge restores the weights (same Parameter objects and bytes, weight functions back), the holdout
    switches to the live wrappers (c['exact'] False, one public WARN) and every eval == a run scored live from the start;
    a non-OOM text and 'Calculate Weight Failed' -> the same (ExactMergeError). Controls: no failure -> the exact path is
    used at every eval (c['exact'] True, no WARN); SAFE=0 (knob off) -> the partly merged LoRA is scored as before."""
    ok = True
    live = hrun(T, BASE, "swal_live", True)  # scored through the live wrappers from the start
    for how in (None, "oom", "error", "none"):
        r, n, undo, before, pub, leak = swallow_run(T, True, how)
        warn = [x for x in pub if "in the evaluator-exact holdout eval" in x and x.startswith("WARN safe:")]
        base = r["steps"] == BASE["max_steps"] and r["usable"] and r["fdig"] == r["sd"] and leak == 0 \
            and before is not None and before >= 1
        if how is None:
            good = base and n == len(r["evals"]) == 4 and r["c"]["exact"] is True and not warn and not undo
        else:
            want = ("OutOfMemoryError", "out of memory") if how == "oom" else ("ExactMergeError", "a failed LoRA merge")
            good = base and n == 1 and undo == [(want[0], True)] and r["c"]["exact"] is False and len(warn) == 1 \
                and warn[0].startswith(f"WARN safe: {want[1]} in the evaluator-exact") and r["evals"] == live["evals"] \
                and nlog(r, "exact merge: ComfyUI logged 1 merge failure(s) at " + SWALLOW_KEY) == 1
        ok &= good
        say(f"oom_swallowed SAFE=1 {how or 'no failure'}: exact merge attempts {n}, {before} key(s) patched before the "
            f"failing one, raised + restored exactly {undo}, exact now {r['c']['exact']}, public WARN {len(warn)}, evals == "
            f"live-scored run {r['evals'] == live['evals']}, steps {r['steps']}, handlers left {leak} -> "
            f"{'ok' if good else 'FAIL'}")
    r0, n0, undo0, _, pub0, leak0 = swallow_run(T, False, "oom")
    good0 = r0["c"]["exact"] is True and n0 == 4 and not undo0 and not [x for x in pub0 if x.startswith("WARN safe:")] \
        and r0["evals"] != live["evals"] and leak0 == 0 and r0["steps"] == BASE["max_steps"]
    ok &= good0
    say(f"oom_swallowed SAFE=0 (knob off) oom: exact merge attempts {n0}, exact {r0['c']['exact']}, evals != live "
        f"{r0['evals'] != live['evals']} (the partly merged LoRA is scored, as before) -> {'ok' if good0 else 'FAIL'}")
    return ok


def run_checks(T, stop=False, only=None):
    fns = {"refuse": check_refuse, "start_clamp": check_start_clamp, "equal_on": check_equal_on, "nf_step": check_nf_step,
           "nf_grad": check_nf_grad, "nf_intermittent": check_nf_intermittent, "nf_member": check_nf_member,
           "nan_first_eval": check_nan_first_eval, "te_drop": check_te_drop, "save_diet": check_save_diet,
           "margin": check_margin, "oom_exact": check_oom_exact, "oom_swallowed": check_oom_swallowed}
    fails = []
    for name in CHECKS:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            good = fns[name](T)
        except Exception as e:  # noqa: BLE001  (a crash is a failure of that check)
            import traceback
            say(f"check {name}: {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
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
    ok = T0._cfg0.get("safe") is False and T0._cfg0.get("safe_nf_max") == 20
    say(f"defaults: CFG safe {T0._cfg0.get('safe')!r} safe_nf_max {T0._cfg0.get('safe_nf_max')!r} with no env -> "
        f"{'ok' if ok else 'FAIL'}")
    if a.mutation:
        fails = run_checks(load_trainer(a.mutation), stop=True, only=only)
        say(f"mutation {a.mutation}: first failing check {fails[0] if fails else None} (expected {MUTATIONS[a.mutation][1]})")
        say(f"RESULT {'FAIL' if fails else 'PASS'}")
        sys.exit(1 if fails else 0)
    fails = run_checks(T0, only=only)
    ok &= not fails
    say(f"checks on the current code: {'PASS' if not fails else 'FAIL ' + str(fails)}")
    if not a.quick and not only:
        for mut, (_, want) in MUTATIONS.items():
            fails = run_checks(load_trainer(mut), stop=True)
            first = fails[0] if fails else None
            say(f"mutation {mut}: first failing check {first} (expected {want}) -> {'caught' if first == want else 'NOT CAUGHT'}")
            ok &= first == want
    say(f"RESULT {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
