#!/usr/bin/env python3
"""G56CT_SEEDS=2 + G56CT_FOLDS=1 check (holdout mode, e.g. the qwen-image QF2 arm), CPU only, on the te_lora_harness
fake DiT (no GPU, no model files). Never run on a GPU before 30 Sep -> treated as broken until this passes.

For G56CT_FAST_LORA 0 and 1 (base frozen before the LoRA attaches, as main() does):
  1. two runs; run k holds out its own block of h items (h = the harness holdout size), the two blocks are disjoint,
     each run trains on every item outside its block, every item is trained on by at least one run;
  2. the result is tagged cv_avg2 and the shipped state dict is exactly average_loras() of the two runs' bests;
     last.safetensors holds exactly that state dict;
  3. the second run starts from re-initialised adapters (the two bests differ).
Mutations that must FAIL: m1 no fold rotation (both runs keep the original holdout); m2 the fold branch ships the first
run's best instead of the average.
Usage (CPU env): python comfytrain/tools/folds_check.py
"""
import hashlib
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import te_lora_harness as H  # noqa: E402

TRAINER = HERE.parent / "g56_comfy_train.py"
OVER = dict(case_cycle=True, sigma_mode="mid", max_steps=12, seeds=2, folds=True, alpha=8.0)
M1 = ('            c["holdout"] = [all_items[i] for i in idx]\n'
      '            c["train"] = [it for i, it in enumerate(all_items) if i not in set(idx)]\n',
      '            pass  # MUTANT m1: no fold rotation\n')
M2 = ('        sd = average_loras(sds) if len(sds) > 1 else sds[0]\n        save_lora(sd, c["out_dir"])\n'
      '        log(f"folds: ',
      '        sd = sds[0]  # MUTANT m2: first fold only\n        save_lora(sd, c["out_dir"])\n'
      '        log(f"folds: ')


def load(src_text, name):
    if src_text is None:
        return H.load_module(str(TRAINER), name)
    p = Path(tempfile.mkdtemp(prefix=f"{name}_")) / "g56_comfy_train.py"
    p.write_text(src_text)
    return H.load_module(str(p), name)


def run(T, fast, tag):
    """One harness run with seeds=2 folds=1; returns the per-run records and the result."""
    from safetensors.torch import load_file
    T.CFG.clear()
    T.CFG.update(T._cfg0)
    T.CFG.update(init="gauss")
    recs, orig_run, orig_attach = [], T._run_one, T.attach_lora

    def run_one(c, *a, **k):
        hold = [it["sha"] for it in c["holdout"]]
        train = [it["sha"] for it in c["train"]]
        r = orig_run(c, *a, **k)
        recs.append({"hold": hold, "train": train, "sd": r.get("best_sd")})
        return r

    def attach_frozen(mp, *a, **k):
        mp.model.requires_grad_(False)
        return orig_attach(mp, *a, **k)

    T._run_one, T.attach_lora = run_one, attach_frozen
    out = Path(tempfile.mkdtemp(prefix=f"folds_{tag}_"))
    try:
        c, res, _ = H.build_and_train(T, out, None, "krea2", te_on=False, cfg_over=dict(OVER, fast_lora=fast))
    finally:
        T._run_one, T.attach_lora = orig_run, orig_attach
    f = out / "last.safetensors"
    return recs, res, (load_file(str(f)) if f.exists() else None), c


def check(T, fast, tag):
    recs, res, filesd, c = run(T, fast, tag)
    allsha = {h for r in recs for h in r["hold"] + r["train"]}
    ok, why = True, []

    def req(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            why.append(msg)

    req(len(recs) == 2, f"runs {len(recs)} != 2")
    if len(recs) == 2:
        h0, h1 = set(recs[0]["hold"]), set(recs[1]["hold"])
        req(h0 and h1 and not (h0 & h1), f"holdout blocks not disjoint/non-empty: {len(h0)} {len(h1)} overlap {len(h0 & h1)}")
        for k, r in enumerate(recs):
            req(set(r["train"]) == allsha - set(r["hold"]), f"run {k}: train != all - its holdout")
        req(all(any(s in r["train"] for r in recs) for s in allsha), "an item is never trained on")
        req(recs[0]["sd"] is not None and recs[1]["sd"] is not None, "a run has no best state dict")
        if recs[0]["sd"] is not None and recs[1]["sd"] is not None:
            req(H.sd_digest(recs[0]["sd"]) != H.sd_digest(recs[1]["sd"]), "the two runs' bests are identical")
            avg = T.average_loras([recs[0]["sd"], recs[1]["sd"]])
            req(H.sd_digest(res["best_sd"]) == H.sd_digest(avg), "shipped state dict != exact average of the two bests")
            req(filesd is not None and H.sd_digest(filesd) == H.sd_digest(avg), "last.safetensors != exact average")
    req(res.get("best", {}).get("tag") == "cv_avg2", f"tag {res.get('best', {}).get('tag')} != cv_avg2")
    if fast:
        st = getattr(c["mp"], "_g56_fast", None)
        req(st is not None and str(getattr(st, "verdict", "")).startswith("pass"), f"fast guard verdict {getattr(st, 'verdict', None)}")
    print(f"[folds] {tag}: {'PASS' if ok else 'FAIL'}" + ("" if ok else "  " + "; ".join(why)), flush=True)
    return ok


def main():
    H.cpu_bootstrap()
    for k in list(os.environ):
        if k.startswith("G56CT_"):
            os.environ.pop(k)
    src = TRAINER.read_text()
    for m in (M1, M2):
        assert src.count(m[0]) == 1, f"mutation anchor not unique/present: {m[0][:60]!r}"
    T = load(None, "g56f_cur")
    T._cfg0 = dict(T.CFG)
    ok = check(T, False, "fast0") & check(T, True, "fast1")
    mut_ok = True
    for name, (a, b) in (("m1", M1), ("m2", M2)):
        Tm = load(src.replace(a, b), f"g56f_{name}")
        Tm._cfg0 = dict(Tm.CFG)
        caught = not check(Tm, False, f"mutant {name}")
        print(f"[folds] mutant {name} {'caught' if caught else 'SURVIVED'}", flush=True)
        mut_ok &= caught
    print(f"[folds] RESULT {'PASS' if ok and mut_ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok and mut_ok else 1)


if __name__ == "__main__":
    main()
