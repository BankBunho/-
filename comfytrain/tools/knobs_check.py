#!/usr/bin/env python3
"""CPU regression / smoke check for the trainer knobs added on 24 Sep (SMALL, SAVE_RUNS, BAND_MIN/MAX, CROP_*, ANTITHETIC,
LORA_PLUS), on the fake DiT of te_lora_harness (no GPU, no model files).

1. defaults: the trainer at --old (a commit before these knobs) vs the current trainer -> identical evals, state dict and
   checkpoint bytes (the knobs are off by default);
2. smoke: every new knob runs end to end (case-cycle long enough to revisit (image, band) pairs) and changes the result;
   antithetic visits really reuse -eps.
Usage: python comfytrain/tools/knobs_check.py --old /path/to/old/g56_comfy_train.py
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import te_lora_harness as H  # noqa: E402


def run(T, out, over):
    c, res, dt = H.build_and_train(T, out, None, "krea2", te_on=False, cfg_over=over)
    f = Path(out) / "last.safetensors"
    return {"evals": H.strip_t(res["evals"]), "sd": H.sd_digest(res["best_sd"]),
            "file": hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None, "steps": res["steps"], "res": res,
            "trace": list(c.get("band_trace", []))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--out", default="/tmp/knobs_check")
    a = ap.parse_args()
    H.cpu_bootstrap()
    for k in list(os.environ):
        if k.startswith("G56CT_"):
            os.environ.pop(k)
    ok = True
    base_over = dict(case_cycle=True, sigma_mode="mid", max_steps=40)
    r = {}
    for tag, path in (("old", a.old), ("new", str(HERE.parent / "g56_comfy_train.py"))):
        T = H.load_module(path, f"g56k_{tag}")
        r[tag] = run(T, f"{a.out}/def_{tag}", dict(base_over))
        print(f"[knobs] defaults {tag}: steps {r[tag]['steps']} sd {r[tag]['sd'][:16]} file {str(r[tag]['file'])[:16]}", flush=True)
    same = all(r["old"][k] == r["new"][k] for k in ("evals", "sd", "file", "steps"))
    print(f"[knobs] RESULT defaults old == new: {'PASS' if same else 'FAIL'}", flush=True)
    ok &= same

    T = H.load_module(str(HERE.parent / "g56_comfy_train.py"), "g56k_new2")
    cfg0 = dict(T.CFG)  # build_and_train writes the case's knobs into T.CFG: restore before every case (no leakage)
    ref = r["new"]
    import torch
    e0, e1, e2 = (T.case_noise((1, 4, 8, 8), 0, 3, 5, v) for v in (0, 1, 2))
    anti = torch.equal(e1, -e0) and not torch.allclose(e2, e0) and not torch.allclose(e2, -e0)
    print(f"[knobs] case_noise: visit1 == -visit0 {torch.equal(e1, -e0)}, visit2 independent {anti}", flush=True)
    ok &= anti
    cases = {
        "antithetic": dict(base_over, antithetic=True, max_steps=200),
        "lora_plus4": dict(base_over, lora_plus=4.0),
        "crop": dict(base_over, crop_frac=1.0, crop_min=0.5, crop_maxsig=1.0),
        "crop_paired": dict(base_over, crop_frac=1.0, crop_min=0.5, crop_maxsig=0.5, crop_prompt="paired"),
        "band_max3": dict(base_over, band_max=3),
        "small_bias": dict(base_over, small=r"^blocks\.0\.(attn\.wq|mlp\.down)\.bias$", small_lr=1e-2),
        "ema_extra": dict(base_over, harness_all_data=True, holdout_frac=0.0, ema_extra=(0.995, 0.0),
                          research_dir=f"{a.out}/research"),
        "seeds_auto3": dict(base_over, harness_all_data=True, holdout_frac=0.0, seeds_auto=3, save_runs=True,
                            research_dir=f"{a.out}/research_auto"),
        "final_scale": dict(base_over, harness_all_data=True, holdout_frac=0.0, final_scale=0.9),
        "ema_debias": dict(base_over, harness_all_data=True, holdout_frac=0.0, ema_debias=True),
        "save_every": dict(base_over, harness_all_data=True, holdout_frac=0.0, save_every=7),
        "pair_anti": dict(base_over, pair_anti=True),
        "lora_down_mult8": dict(base_over, lora_down_mult=8.0),
        "spec_init": dict(base_over, harness_all_data=True, holdout_frac=0.0, spec_init=True),
        "spec_blocks_auto2": dict(base_over, harness_all_data=True, holdout_frac=0.0, spec_init=True, spec_blocks=True,
                                  spec_probes=2, seeds_auto=2, save_runs=True, research_dir=f"{a.out}/research_spec"),
        "seeds_fill_inert": dict(base_over, harness_all_data=True, holdout_frac=0.0, seeds_auto=3, seeds_fill=0.6),
        "late_phase": dict(base_over, harness_all_data=True, holdout_frac=0.0, late_frac=0.5, late_bands=4),
        "accum1_inert": dict(base_over, accum=1, accum_band=True, agree=1.0),
        "accum2": dict(base_over, accum=2),
        "accum_band4": dict(base_over, accum=4, accum_band=True),
        "agree4": dict(base_over, accum=4, accum_band=True, agree=1.0),
    }
    for name, over in cases.items():
        T.CFG.clear()
        T.CFG.update(cfg0)
        if name == "small_bias":  # SMALL attaches in main(); here: attach on the harness model via a wrapper around attach_lora
            orig_attach = T.attach_lora

            def attach_plus(mp, inc, rank, alpha, _o=orig_attach, _T=T):
                ads, ps = _o(mp, inc, rank, alpha)
                sm, sp = _T.attach_small(mp, _T.CFG["small"], 256)
                ads.update(sm)
                _T._harness_small = sp
                return ads, ps + sp
            T.attach_lora = attach_plus
        try:
            rr = run(T, f"{a.out}/{name}", over)
        finally:
            if name == "small_bias":
                T.attach_lora = orig_attach
        changed = rr["sd"] != ref["sd"]
        extra = ""
        if name == "small_bias":
            keys = [k for k in rr["res"]["best_sd"] if k.endswith(".diff_b")]
            extra = f" diff_b keys {keys}"
            changed = changed and len(keys) == 2
        if name == "ema_extra":
            files = sorted(p.name for p in Path(f"{a.out}/research").glob("*.safetensors"))
            extra = f" side files {files}"
            changed = files == ["r0_ema0.995.safetensors", "r0_ema0.safetensors"]
        if name == "seeds_auto3":
            files = sorted(p.name for p in Path(f"{a.out}/research_auto").glob("run*.safetensors"))
            from safetensors.torch import load_file
            sd = load_file(f"{a.out}/seeds_auto3/last.safetensors")
            ranks = sorted({v.shape[0] for k, v in sd.items() if k.endswith(".lora_down.weight")})
            extra = f" members {files} shipped ranks {ranks} steps {rr['steps']}"
            changed = len(files) == 3 and ranks == [12] and rr["steps"] == 120  # 3 x rank 4, 3 x 40 steps
        if name == "final_scale":
            from safetensors.torch import load_file
            T.CFG.clear()
            T.CFG.update(cfg0)
            ref_all = run(T, f"{a.out}/final_scale_ref", dict(base_over, harness_all_data=True, holdout_frac=0.0))
            a0 = load_file(f"{a.out}/final_scale_ref/last.safetensors"); a1 = load_file(f"{a.out}/final_scale/last.safetensors")
            ok_s = all(torch.allclose(a1[k], a0[k] * (0.9 if k.endswith(".lora_up.weight") else 1.0)) for k in a0)
            extra = f" shipped = 0.9 x reference ups: {ok_s}"
            changed = ok_s
        if name == "ema_debias":  # the zero-initialised factor (harness: comfy init -> DOWN = 0) must come out as
            # ema / (1 - beta^T) exactly (harness beta = 0.9); the random-initialised factor changes differently
            from safetensors.torch import load_file
            T.CFG.clear()
            T.CFG.update(cfg0)
            run(T, f"{a.out}/ema_debias_ref", dict(base_over, harness_all_data=True, holdout_frac=0.0))
            a0 = load_file(f"{a.out}/ema_debias_ref/last.safetensors"); a1 = load_file(f"{a.out}/ema_debias/last.safetensors")
            f = 1.0 / (1.0 - 0.9 ** rr["steps"])
            ok_d = all(torch.allclose(a1[k], a0[k] * f, rtol=1e-5, atol=1e-8) for k in a0 if k.endswith(".lora_down.weight"))
            up_changed = any(not torch.equal(a1[k], a0[k]) for k in a0 if k.endswith(".lora_up.weight"))
            extra = f" downs (zero-init) = ref x {f:.5f}: {ok_d}; ups (random init) debiased/changed: {up_changed}"
            changed = ok_d and up_changed
        if name == "save_every":  # interim saves must not change training: final file == plain all-data run
            T.CFG.clear()
            T.CFG.update(cfg0)
            ref_sv = run(T, f"{a.out}/save_every_ref", dict(base_over, harness_all_data=True, holdout_frac=0.0))
            same = rr["file"] == ref_sv["file"] and rr["sd"] == ref_sv["sd"]
            extra = f" final file identical to the run without interim saves: {same}"
            changed = same
        if name == "seeds_auto3":
            sd_auto3, file_auto3 = rr["sd"], rr["file"]
        if name == "seeds_fill_inert":  # no time pressure in the harness -> SEEDS_FILL must not change seeds_auto=3
            changed = rr["sd"] == sd_auto3 and rr["file"] == file_auto3
            extra = f" identical to seeds_auto3 without fill: {changed}"
        if name == "late_phase":  # after step max_steps/2 only bands < 4, before it the full range
            early = [b for st, b in rr["trace"] if st < 20]; late = [b for st, b in rr["trace"] if st >= 20]
            ok_l = bool(early) and max(early) >= 4 and bool(late) and max(late) < 4
            extra = f" early bands max {max(early) if early else None}, late bands {sorted(set(late))} -> {ok_l}"
            changed = changed and ok_l
        if name == "accum1_inert":  # ACCUM=1: the grouping / AND-mask knobs must not touch the default path
            changed = rr["sd"] == ref["sd"] and rr["file"] == ref["file"]
            extra = f" identical to the defaults: {changed}"
        if name == "accum_band4":
            sd_band4 = rr["sd"]
        if name == "agree4":  # the mask must change the result of the same grouping without it
            changed = changed and rr["sd"] != sd_band4
            extra = f" differs from accum_band4 without the mask: {rr['sd'] != sd_band4}"
        if name == "spec_init":  # down rows orthonormal & data-dependent (not the gauss draw), up learned from 0
            from safetensors.torch import load_file
            sd = load_file(f"{a.out}/spec_init/last.safetensors")
            dn = [v for k, v in sd.items() if k.endswith(".lora_down.weight")]
            orth = max(float((d @ d.T - torch.eye(d.shape[0])).abs().max()) for d in dn)
            extra = f" max|DD^T-I| after training {orth:.3f}"
        if name == "spec_blocks_auto2":  # two members on different direction blocks -> their down subspaces differ
            from safetensors.torch import load_file
            m0 = load_file(f"{a.out}/research_spec/run0.safetensors"); m1 = load_file(f"{a.out}/research_spec/run1.safetensors")
            k0 = next(k for k in m0 if k.endswith(".lora_down.weight"))
            q0 = torch.linalg.qr(m0[k0].T)[0]; q1 = torch.linalg.qr(m1[k0].T)[0]
            ov = float((q0.T @ q1).pow(2).sum() / q0.shape[1])
            extra = f" member down-subspace overlap {ov:.3f} (blocks -> low)"
            changed = changed and ov < 0.5
        print(f"[knobs] {name}: steps {rr['steps']} changed {changed}{extra}", flush=True)
        ok &= changed
    T.CFG.clear()
    T.CFG.update(cfg0)
    ok &= check_family_defaults(T, cfg0)
    ok &= check_accum_helpers(T)
    ok &= check_fill_member_steps(T)
    print(f"[knobs] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


def check_fill_member_steps(T):
    """SEEDS_FILL decision: B3 R1's real numbers (2 members x 560 steps in 848/846 s, 770 s left) -> a 444-step last
    member; off / too little time / no finished member -> 0; never above the full cap."""
    f = T.fill_member_steps
    cases = [(([848, 846], [560, 560], 770, 560, 0.6), 444), (([848, 846], [560, 560], 770, 560, 0.0), 0),
             (([848, 846], [560, 560], 400, 560, 0.6), 0), (([848, 846], [560, 560], 5000, 560, 0.6), 560),
             (([], [], 770, 560, 0.6), 0), (([900], [560], 770, 0, 0.6), 0)]
    ok = all(f(*a) == want for a, want in cases)
    print(f"[knobs] fill_member_steps: {[f(*a) for a, _ in cases]} == {[w for _, w in cases]} {ok}", flush=True)
    return ok


def check_accum_helpers(T):
    """accum_groups: one pass covers every (image, band) case exactly once, steps hold <= m cases, same_band steps hold
    distinct images of ONE band with no lone single-image chunk; and_mask keeps an element only where the per-case signs
    agree (a large gradient from one case outvoted by the others is dropped)."""
    import random
    import torch
    ok = True
    for n, m, same in ((16, 4, True), (9, 4, True), (7, 3, False), (5, 2, True), (3, 4, True)):
        bp = list(range(16))
        g = T.accum_groups(n, bp, m, same, random.Random(0))
        flat = [c for grp in g for c in grp]
        cover = sorted(flat) == sorted((i, b) for i in range(n) for b in bp)
        sizes = all(1 <= len(grp) <= m for grp in g)
        one = (not same) or all(len({b for _, b in grp}) == 1 and len({i for i, _ in grp}) == len(grp) for grp in g)
        mins = (not same) or min(len(grp) for grp in g) >= n // -(-n // m)
        ok &= cover and sizes and one and mins
        print(f"[knobs] accum_groups n={n} m={m} same_band={same}: cover {cover} sizes {sizes} one-band {one} "
              f"min-chunk {mins}", flush=True)
    per_case = [torch.tensor([1.0, 1.0, -1.0, 1.0, 5.0]), torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0]),
                torch.tensor([1.0, -1.0, -1.0, 2.0, -1.0])]
    gsum = [sum(per_case)]
    ssum = [sum(torch.sign(x) for x in per_case)]
    out1, k1 = T.and_mask(gsum, ssum, 3, 1.0)
    out0, k0 = T.and_mask(gsum, ssum, 3, 0.3)
    m_ok = (torch.equal(out1[0], torch.tensor([3.0, 0.0, -3.0, 0.0, 0.0])) and abs(k1 - 0.4) < 1e-9
            and torch.equal(out0[0], gsum[0]) and abs(k0 - 1.0) < 1e-9)
    print(f"[knobs] and_mask: strict keeps only unanimous elements (one-case outlier dropped) {m_ok}", flush=True)
    return ok and m_ok


def check_family_defaults(T, cfg0):
    """Freeze B0: every FAMILY_DEFAULTS key is a CFG key of the same type (regexes compile), apply_family_defaults
    applies exactly that dict, an explicit env var wins over it, and G56CT_FAMILY_DEFAULTS=0 leaves CFG untouched."""
    import re
    ok = True
    for fam, d in T.FAMILY_DEFAULTS.items():
        bad = [k for k, v in d.items() if k not in cfg0 or type(cfg0[k]) is not type(v)]
        for k in ("include", "include_extra"):
            if k in d:
                re.compile(d[k])
        T.CFG.clear()
        T.CFG.update(cfg0)
        ap = T.apply_family_defaults(fam)
        good = fam in T.FAMILIES and not bad and ap == d and all(T.CFG[k] == v for k, v in d.items())
        print(f"[knobs] family defaults {fam}: {len(d)} keys applied {good}" + (f" bad {bad}" if bad else ""), flush=True)
        ok &= good
    T.CFG.clear()
    T.CFG.update(cfg0)
    os.environ["G56CT_SEEDS_AUTO"] = "2"
    ap = T.apply_family_defaults("ideogram4")
    env_wins = "seeds_auto" not in ap and T.CFG["seeds_auto"] == cfg0["seeds_auto"]
    os.environ.pop("G56CT_SEEDS_AUTO")
    T.CFG.clear()
    T.CFG.update(cfg0)
    os.environ["G56CT_FAMILY_DEFAULTS"] = "0"
    off = T.apply_family_defaults("krea2") == {} and T.CFG == cfg0
    os.environ.pop("G56CT_FAMILY_DEFAULTS")
    T.CFG.clear()
    T.CFG.update(cfg0)
    print(f"[knobs] family defaults: explicit env wins {env_wins}, G56CT_FAMILY_DEFAULTS=0 is a no-op {off}", flush=True)
    return ok and env_wins and off


if __name__ == "__main__":
    main()
