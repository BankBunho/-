#!/usr/bin/env python3
"""CPU check of the task-size conditional family defaults (FAMILY_DEFAULTS_BIG_N, 4 Oct): qwen-image gets steps_per_image 50 + lr 7.5e-5
only on tasks with >= 20 images; small tasks keep the QLF45 recipe; other families untouched; explicit env vars still win;
G56CT_FAMILY_DEFAULTS=0 switches it off. Prints '[bign] RESULT PASS|FAIL'.
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def fresh():
    spec = importlib.util.spec_from_file_location("g56bign", HERE / "g56_comfy_train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(fam, n, env=None):
    for k in ("G56CT_FAMILY_DEFAULTS", "G56CT_STEPS_PER_IMAGE", "G56CT_LR"):
        os.environ.pop(k, None)
    os.environ.update(env or {})
    m = fresh()
    m.apply_family_defaults(fam)
    applied = m.apply_big_n_defaults(fam, n)
    return applied, m.CFG["steps_per_image"], m.CFG["lr"]


def main():
    ok = True
    cases = [
        ("qwen big N=29", ("qwen-image", 29, None), lambda a, spi, lr: spi == 50.0 and abs(lr - 7.5e-5) < 1e-12 and set(a) == {"steps_per_image", "lr"}),
        ("qwen threshold N=20", ("qwen-image", 20, None), lambda a, spi, lr: spi == 50.0 and abs(lr - 7.5e-5) < 1e-12),
        ("qwen small N=19", ("qwen-image", 19, None), lambda a, spi, lr: a == {} and spi == 75.0 and abs(lr - 5e-5) < 1e-12),
        ("qwen small N=9", ("qwen-image", 9, None), lambda a, spi, lr: a == {} and spi == 75.0 and abs(lr - 5e-5) < 1e-12),
        ("krea2 N=100 untouched", ("krea2", 100, None), lambda a, spi, lr: a == {} and spi == 37.0 and abs(lr - 1.5e-4) < 1e-12),
        ("ideogram4 N=100 untouched", ("ideogram4", 100, None), lambda a, spi, lr: a == {} and spi == 35.0),
        ("env LR wins", ("qwen-image", 29, {"G56CT_LR": "1e-4"}), lambda a, spi, lr: spi == 50.0 and abs(lr - 1e-4) < 1e-12 and "lr" not in a),
        ("FAMILY_DEFAULTS=0 off", ("qwen-image", 29, {"G56CT_FAMILY_DEFAULTS": "0"}), lambda a, spi, lr: a == {}),
    ]
    for name, args, pred in cases:
        try:
            a, spi, lr = run(*args)
            good = bool(pred(a, spi, lr))
        except Exception as e:  # noqa: BLE001
            good, a, spi, lr = False, repr(e), None, None
        print(f"[bign] {name}: applied={a} steps_per_image={spi} lr={lr} -> {'ok' if good else 'FAIL'}")
        ok &= good
    print("[bign] RESULT " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
