#!/usr/bin/env python3
"""CPU check of a FAMILY_DEFAULTS change against an older trainer: the global CFG defaults must be identical, and the per-family
defaults may differ ONLY in the expected keys (each set to the expected value). Everything else byte-identical.

python comfytrain/tools/family_defaults_check.py --old /path/to/old/g56_comfy_train.py \
    [--expect ideogram4:seeds_fill=0.6] [--mutate ideogram4:seeds_fill]   (mutate = flip that key (bool) or put back the old
    trainer's value (number) in the new trainer before checking: the check must then FAIL)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _value(v):
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v) if any(c in v for c in ".eE") else int(v)
    except ValueError:
        return v


def parse_expect(s):
    out = {}
    for part in filter(None, (s or "").split(";")):
        fam, kv = part.split(":", 1)
        out[fam] = {k: _value(v) for k, v in (x.split("=", 1) for x in kv.split(","))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", default=str(HERE / "g56_comfy_train.py"))
    ap.add_argument("--expect", default="")  # the change of the commit under test (vs --old)
    ap.add_argument("--mutate", default="")
    a = ap.parse_args()
    old, new = load(a.old, "g56fd_old"), load(a.new, "g56fd_new")
    if a.mutate:
        fam, key = a.mutate.split(":")
        cur = new.FAMILY_DEFAULTS[fam].get(key, False)
        new.FAMILY_DEFAULTS[fam][key] = (not cur) if isinstance(cur, bool) else old.FAMILY_DEFAULTS[fam].get(key)
    expect = parse_expect(a.expect)
    ok = True
    if old.CFG != new.CFG:
        diff = sorted(k for k in set(old.CFG) | set(new.CFG) if old.CFG.get(k) != new.CFG.get(k))
        print(f"[famdef] FAIL global CFG defaults differ: {diff}"); ok = False
    else:
        print(f"[famdef] global CFG defaults identical ({len(new.CFG)} keys)")
    MISSING = object()
    for fam in sorted(set(old.FAMILY_DEFAULTS) | set(new.FAMILY_DEFAULTS)):
        o, n = old.FAMILY_DEFAULTS.get(fam, {}), new.FAMILY_DEFAULTS.get(fam, {})
        diff = {k for k in set(o) | set(n) if o.get(k, MISSING) != n.get(k, MISSING)}
        want = expect.get(fam, {})
        good = diff == set(want) and all(n.get(k, MISSING) == v for k, v in want.items())
        print(f"[famdef] {fam}: changed {sorted(diff) or 'nothing'} expected {sorted(want) or 'nothing'} -> {'ok' if good else 'FAIL'}")
        ok &= good
    print("[famdef] RESULT " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
