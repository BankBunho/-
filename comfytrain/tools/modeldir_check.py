#!/usr/bin/env python3
"""model_dir fallback, CPU only (no model files): a fake /cache with
  1. the named folder <org--repo> present (plus others)  -> the named folder (unchanged behaviour);
  2. the named folder absent, exactly one model folder     -> that folder (anonymized cache, like the champion's trainer);
  3. the named folder absent, two model folders            -> the named path (no guess; the loaders fail loudly as before);
  4. dot-folders (e.g. .locks) and files do not count as model folders.
Mutation: the old model_dir (no fallback) must FAIL case 2.
Usage: python comfytrain/tools/modeldir_check.py [--old path/to/old/g56_comfy_train.py]
"""
import argparse
import importlib.util
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.log = lambda *a, **k: None
    return m


def cases(T):
    repo = "some-org/Some-Model"
    res = {}
    for name, dirs, files, want in (
            ("named_present", ["some-org--Some-Model", "other--x"], [], "some-org--Some-Model"),
            ("single_anonymous", ["a1b2c3"], [], "a1b2c3"),
            ("two_folders_no_guess", ["a1b2c3", "d4e5f6"], [], "some-org--Some-Model"),
            ("dot_and_files_ignored", ["a1b2c3", ".locks"], ["README"], "a1b2c3")):
        root = Path(tempfile.mkdtemp())
        for d in dirs:
            (root / "models" / d).mkdir(parents=True)
        for f in files:
            (root / "models" / f).write_text("x")
        T.CACHE = root
        if hasattr(T, "_MODEL_DIRS"):
            T._MODEL_DIRS.clear()
        got = T.model_dir(repo)
        res[name] = got == root / "models" / want
        print(f"[modeldir] {name}: {got.name} (want {want}) {'ok' if res[name] else 'FAIL'}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=None)
    a = ap.parse_args()
    ok = all(cases(load(HERE / "g56_comfy_train.py", "g56md_new")).values())
    if a.old:
        old = cases(load(a.old, "g56md_old"))
        caught = not old["single_anonymous"]
        print(f"[modeldir] mutation (old model_dir without fallback) caught: {caught}", flush=True)
        ok &= caught
    print(f"[modeldir] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
