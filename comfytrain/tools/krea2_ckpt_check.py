#!/usr/bin/env python3
"""krea2 base selection (_krea2_checkpoint), CPU only: a fake /cache with SPARSE checkpoint files (no disk blocks):
  1. raw_named          task folder krea--Krea-2-Raw has raw.safetensors           -> that raw.safetensors (unchanged behaviour);
  2. other_base_cached  task folder krea--Krea-2-Turbo has its single root-level checkpoint and krea--Krea-2-Raw (raw.safetensors
                        + text_encoder/) is still in the cache                   -> the TASK's own checkpoint (evaluator contract:
                        image_artifacts.prepare_base = the task repo's single root-level > 5 GiB .safetensors);
  3. task_folder_absent named folder missing, two cache folders, one with raw.safetensors + text_encoder/ -> that raw (unchanged);
  4. task_folder_empty  task folder without any root checkpoint, raw elsewhere   -> the raw elsewhere (unchanged, now with a WARN);
  5. nothing            no checkpoint anywhere                                     -> FileNotFoundError (unchanged).
Mutation: the old _krea2_checkpoint (cache-wide raw.safetensors before the task's own file) must FAIL case 2.
Usage: python comfytrain/tools/krea2_ckpt_check.py [--old path/to/old/g56_comfy_train.py]
"""
import argparse
import importlib.util
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BIG = 5 * 1024 ** 3 + 4096  # > ROOT_MIN_BYTES, sparse


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.log = lambda *a, **k: None
    return m


def sparse(p, size=BIG):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.truncate(size)


def cases(T):
    res = {}
    specs = (
        ("raw_named", "krea/Krea-2-Raw",
         {"krea--Krea-2-Raw/raw.safetensors": BIG, "krea--Krea-2-Raw/text_encoder/x": 1}, "krea--Krea-2-Raw/raw.safetensors"),
        ("other_base_cached", "krea/Krea-2-Turbo",
         {"krea--Krea-2-Turbo/turbo.safetensors": BIG, "krea--Krea-2-Turbo/text_encoder/x": 1,
          "krea--Krea-2-Raw/raw.safetensors": BIG, "krea--Krea-2-Raw/text_encoder/x": 1}, "krea--Krea-2-Turbo/turbo.safetensors"),
        ("task_folder_absent", "krea/Krea-2-Raw",
         {"a1b2c3/raw.safetensors": BIG, "a1b2c3/text_encoder/x": 1, "d4e5f6/other.safetensors": BIG}, "a1b2c3/raw.safetensors"),
        ("task_folder_empty", "krea/Krea-2-Turbo",
         {"krea--Krea-2-Turbo/README.md": 10, "krea--Krea-2-Raw/raw.safetensors": BIG, "krea--Krea-2-Raw/text_encoder/x": 1},
         "krea--Krea-2-Raw/raw.safetensors"),
        ("nothing", "krea/Krea-2-Turbo", {"krea--Krea-2-Turbo/README.md": 10, "other--x/small.safetensors": 100}, None),
    )
    for name, repo, files, want in specs:
        root = Path(tempfile.mkdtemp())
        for rel, size in files.items():
            sparse(root / "models" / rel, size)
        T.CACHE = root
        if hasattr(T, "_MODEL_DIRS"):
            T._MODEL_DIRS.clear()
        try:
            got = T._krea2_checkpoint(repo)
            got_rel = str(got.relative_to(root / "models"))
        except FileNotFoundError:
            got_rel = None
        res[name] = got_rel == want
        print(f"[krea2ckpt] {name}: {got_rel} (want {want}) {'ok' if res[name] else 'FAIL'}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=None)
    a = ap.parse_args()
    ok = all(cases(load(HERE / "g56_comfy_train.py", "g56k2_new")).values())
    if a.old:
        old = cases(load(a.old, "g56k2_old"))
        caught = not old["other_base_cached"]
        same = all(old[k] for k in ("raw_named", "task_folder_absent", "task_folder_empty", "nothing"))
        print(f"[krea2ckpt] mutation (old: cache-wide raw.safetensors first) caught: {caught}; old agrees elsewhere: {same}", flush=True)
        ok &= caught and same
    print(f"[krea2ckpt] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
