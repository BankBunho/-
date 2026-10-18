#!/usr/bin/env python3
"""G56CT_WATCHDOG_S check, CPU only: real child processes running the trainer's cli() with a stand-in main() that starts
the watchdog exactly as main() does (_start_watchdog(t_start, hours, out_dir)). hours = 0.01 -> the validator's kill at
+36 s; WATCHDOG_S = 30 -> the hard stop at +6 s.

wd_fire        a usable adapter + a stale .last.safetensors.tmp, main() still busy at +6 s: exit 0 within 15 s, the
               adapter's bytes unchanged, the .tmp removed, 'hard stop before the validator's kill' logged.
wd_no_adapter  no adapter at +6 s: 'no usable adapter' logged, the process keeps running (the validator's kill decides;
               a save may still complete) - alive at +15 s.
wd_done        main() returns at +2 s, the process lingers (teardown) past +6 s: exit 0, no watchdog line.
wd_off         WATCHDOG_S unset: no thread, alive at +15 s with a usable adapter.
Mutations (source level): each must make the named check fail.
Usage (CPU env): python comfytrain/tools/watchdog_check.py [--mutation NAME] [--quick]
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER = HERE.parent / "g56_comfy_train.py"

MUTATIONS = {
    # the hard stop without the usable-adapter check (an empty folder would be 'shipped' with exit 0)
    "wd_nousable": ([("        if usable_adapter(out_dir):\n            try:\n                (Path(out_dir) / \".last.safetensors.tmp\")",
                      "        if True:\n            try:\n                (Path(out_dir) / \".last.safetensors.tmp\")")],
                    "wd_no_adapter"),
    # anchored at the trainer's own deadline instead of the validator's kill: fires at once, before the adapter exists
    "wd_anchor_deadline": ([("    t_fire = t_start + max(1, int(hours * 3600)) - ws\n",
                             "    t_fire = t_start + hours * 3600 * CFG[\"wall_frac\"] - CFG[\"reserve_s\"] - ws\n")],
                           "wd_fire"),
    # the stale partial file left in the upload folder
    "wd_notmp": ([("                (Path(out_dir) / \".last.safetensors.tmp\").unlink()\n",
                   "                pass\n")], "wd_fire"),
    # the thread ignores main()'s end (cli's done event)
    "wd_nodone": ([("        if done.is_set():\n            return\n        if usable_adapter(out_dir):\n",
                    "        if usable_adapter(out_dir):\n")], "wd_done"),
}
CHILD = r'''
import importlib.util, os, sys, time
from pathlib import Path
path, scen, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("g56wd", path)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
T.log = lambda *a, **k: print("LOG", " ".join(str(x) for x in a), flush=True)
import torch  # main() has imported torch long before the hard stop
from safetensors.torch import save_file

def fake_main():
    T._OUT_DIR = out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    T._start_watchdog(t0, 0.01, out)  # the validator kills at +36 s
    print("ARMED", repr(t0), flush=True)
    if scen in ("fire", "done", "off"):
        save_file({"m.lora_up.weight": torch.zeros(4, 2), "m.lora_down.weight": torch.randn(2, 4),
                   "m.alpha": torch.tensor(2.0)}, str(out / "last.safetensors"))
        (out.parent / (out.name + ".ref")).write_bytes((out / "last.safetensors").read_bytes())
        (out / ".last.safetensors.tmp").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00partial")
        print("READY", flush=True)
    if scen == "done":
        time.sleep(2)
        return
    time.sleep(60)

rc = T.cli(fake_main)
print("MAIN_RETURNED", rc, flush=True)
if scen == "done":
    time.sleep(10)  # a slow teardown after main() (e.g. CUDA context): the watchdog must stay quiet
sys.exit(rc)
'''


def say(*a):
    print("[watchdog]", *a, flush=True)


def trainer_path(mutation):
    src = TRAINER.read_text()
    if mutation:
        for a, b in MUTATIONS[mutation][0]:
            if src.count(a) != 1:
                raise SystemExit(f"[watchdog] mutation {mutation}: pattern occurs {src.count(a)}x, not exactly once: {a!r}")
            src = src.replace(a, b)
    d = Path(tempfile.mkdtemp(prefix=f"wd_{mutation or 'code'}_"))
    (d / "g56_comfy_train.py").write_text(src)
    (d / "child.py").write_text(CHILD)
    return d


def run_child(d, scen, env_ws, wait):
    """Start the child; wait up to `wait` s for it to end. Returns (exit code or None if still alive, seconds from the
    watchdog start (ARMED) to the end or the kill, output, out_dir). The torch import before ARMED is not counted."""
    out = Path(tempfile.mkdtemp(prefix=f"wd_out_{scen}_"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("G56CT_")}
    env["G56CT_LOGFILE"] = str(out.parent / f"{out.name}.log")
    if env_ws is not None:
        env["G56CT_WATCHDOG_S"] = str(env_ws)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, str(d / "child.py"), str(d / "g56_comfy_train.py"), scen, str(out)], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        so, _ = p.communicate(timeout=wait)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        so, _ = p.communicate()
        rc = None
    t_end = time.time()
    armed = [float(x.split()[1]) for x in so.splitlines() if x.startswith("ARMED ")]
    return rc, t_end - (armed[0] if armed else t0), so, out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def check_wd_fire(d):
    rc, dt, so, out = run_child(d, "fire", 30, 30)
    good = rc == 0 and 5.5 <= dt <= 15 and "hard stop before" in so and not (out / ".last.safetensors.tmp").exists() \
        and sha(out / "last.safetensors") == sha(out.parent / f"{out.name}.ref") and "MAIN_RETURNED" not in so
    say(f"wd_fire: exit {rc} after {dt:.1f}s (hard stop at +6 s), 'hard stop' logged {'hard stop before' in so}, "
        f".tmp removed {not (out / '.last.safetensors.tmp').exists()}, adapter kept {(out / 'last.safetensors').exists()} "
        f"-> {'ok' if good else 'FAIL'}")
    if not good:
        say("  output: " + so.replace("\n", " | ")[-600:])
    return good


def check_wd_no_adapter(d):
    rc, dt, so, out = run_child(d, "none", 30, 30)
    good = rc is None and dt >= 15 and "no usable adapter" in so and "hard stop before" not in so
    say(f"wd_no_adapter: alive at +15 s {rc is None} (exit {rc}), 'no usable adapter' logged {'no usable adapter' in so} "
        f"-> {'ok' if good else 'FAIL'}")
    if not good:
        say("  output: " + so.replace("\n", " | ")[-600:])
    return good


def check_wd_done(d):
    rc, dt, so, out = run_child(d, "done", 30, 40)
    fired = "hard stop before" in so or "no usable adapter" in so
    good = rc == 0 and "MAIN_RETURNED 0" in so and not fired and dt >= 11
    say(f"wd_done: exit {rc} after {dt:.1f}s (main() returned at +2 s, teardown until +12 s), watchdog fired {fired} "
        f"-> {'ok' if good else 'FAIL'}")
    if not good:
        say("  output: " + so.replace("\n", " | ")[-600:])
    return good


def check_wd_off(d):
    rc, dt, so, out = run_child(d, "off", None, 30)
    good = rc is None and dt >= 15 and "watchdog" not in so
    say(f"wd_off (G56CT_WATCHDOG_S unset): alive at +15 s {rc is None}, watchdog line {'watchdog' in so} -> "
        f"{'ok' if good else 'FAIL'}")
    return good


CHECKS = {"wd_fire": check_wd_fire, "wd_no_adapter": check_wd_no_adapter, "wd_done": check_wd_done,
          "wd_off": check_wd_off}


def run_checks(d, only=None, stop=False):
    fails = []
    for name, fn in CHECKS.items():
        if only and name not in only:
            continue
        try:
            good = fn(d)
        except Exception as e:  # noqa: BLE001
            say(f"check {name}: {type(e).__name__}: {e}")
            good = False
        if not good:
            fails.append(name)
            if stop:
                break
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutation", choices=sorted(MUTATIONS))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.mutation:
        fails = run_checks(trainer_path(a.mutation), only=[MUTATIONS[a.mutation][1]])
        say(f"mutation {a.mutation}: {'caught' if fails else 'NOT CAUGHT'} by {MUTATIONS[a.mutation][1]}")
        say(f"RESULT {'FAIL' if fails else 'PASS'}")
        sys.exit(1 if fails else 0)
    fails = run_checks(trainer_path(None))
    ok = not fails
    say(f"checks on the current code: {'PASS' if ok else 'FAIL ' + str(fails)}")
    if not a.quick:
        for mut, (_, want) in MUTATIONS.items():
            caught = bool(run_checks(trainer_path(mut), only=[want]))
            say(f"mutation {mut}: {want} {'fails -> caught' if caught else 'passes -> NOT CAUGHT'}")
            ok &= caught
    say(f"RESULT {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
