#!/usr/bin/env python3
"""CPU check of the trainer's exit-code policy (cli / usable_adapter in g56_comfy_train.py).

The validator marks a task failed (no upload) when the container exits != 0, but uploads the checkpoint folder when it
kills the container at the deadline. So a crash AFTER a usable adapter was saved must exit 0, and anything else must not.
Cases: crash with a valid adapter -> 0; crash with no / NaN / corrupt adapter -> 1; normal run -> 0; SystemExit(2) with and
without a valid adapter -> 0 / 2; crash before the upload folder is known -> 1.
Usage (inside the trainer image): python comfytrain/tools/exit_check.py [--module path/to/g56_comfy_train.py]
"""
import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path


def load(path):
    spec = importlib.util.spec_from_file_location("g56_exit_check", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=str(Path(__file__).resolve().parents[1] / "g56_comfy_train.py"))
    a = ap.parse_args()
    import torch
    from safetensors.torch import save_file
    T = load(a.module)
    T.log = lambda *x, **k: None
    tmp = Path(tempfile.mkdtemp())

    def adapter(kind):
        d = tmp / kind
        d.mkdir(parents=True, exist_ok=True)
        f = d / "last.safetensors"
        if kind == "valid":
            save_file({"m.lora_up.weight": torch.zeros(4, 2), "m.lora_down.weight": torch.randn(2, 4)}, str(f))
        elif kind == "nan":
            save_file({"m.lora_up.weight": torch.full((4, 2), float("nan")), "m.lora_down.weight": torch.randn(2, 4)}, str(f))
        elif kind == "corrupt":
            f.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{not json")
        return d

    def crash():
        raise RuntimeError("boom")

    def exit2():
        raise SystemExit(2)

    cases = [("crash, valid adapter", "valid", crash, 0), ("crash, no adapter", "none", crash, 1),
             ("crash, NaN adapter", "nan", crash, 1), ("crash, corrupt adapter", "corrupt", crash, 1),
             ("normal run", "none", lambda: None, 0), ("SystemExit(2), valid adapter", "valid", exit2, 0),
             ("SystemExit(2), no adapter", "none", exit2, 2), ("crash before out_dir is known", None, crash, 1)]
    ok = True
    for name, kind, fn, want in cases:
        T._OUT_DIR = None if kind is None else adapter(kind)
        got = T.cli(fn)
        ok &= got == want
        print(f"[exit] {name}: exit {got} (want {want}) {'ok' if got == want else 'FAIL'}", flush=True)
    print(f"[exit] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
