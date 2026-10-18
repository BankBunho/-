#!/usr/bin/env python3
"""Check of ops/docker/bake_comfy_assets.py (the docker-build download step), CPU only, no network: fetch() and main() on
file:// URLs of small synthetic safetensors files (written like the safetensors writer: 8-byte LE header length, JSON
header space-padded to 8 bytes, data). The pinned structure digests are computed here, independently of the script.

match      pinned sha256 == served: True, output exactly "[bake] <name> ok", bytes == served, one download, the structure
           check never runs (the old path).
changed    upstream re-upload: same tensor names/shapes/dtypes, other values / __metadata__ / tensor order, the old file's
           sha256 pinned: True at the first download (no retry) with one WARNING naming the file, the new sha256 and
           "structure verified"; bytes == served.
reject_*   sha256 not pinned and not the pinned structure: False after every try (each one downloads again), nothing at
           the destination, no WARNING. truncated (data cut), trailing (bytes after the data), header_cut (ends inside
           the header), bad_json, huge_len (header length > file), html (an error page), extra / missing / renamed tensor,
           shape (same byte size), dtype (same byte size), overlap (two tensors on one byte range, file size consistent).
te         no structure digest (the flux text encoders, pinned by revision): a structurally identical mismatch fails with
           exactly the old lines (no structure check, no WARNING).
transient  the first two opens raise URLError: retried, True at the third.
lib        (safetensors.numpy importable) files written by the safetensors library itself: the reference passes the
           structure check, a re-upload with other values is accepted, a cut one is not.
cli        main(): only the three VAE entries without --flux-te ("ok" lines exactly), the TE entry too with --flux-te,
           SystemExit("[bake] giving up on <name>") when a file never verifies.
--real DIR the three pinned VAE files in DIR (a real bake output): sha256 == pinned, structure check passes with the pinned
           digest (and safetensors.safe_open gives the same digest); a copy with one data byte changed is accepted with
           a WARNING and still opens; a 4 MB prefix is rejected.
Mutations (source level, on a copy of the script): each must make the named case fail.
Usage: python comfytrain/tools/bake_check.py [--mutation NAME] [--quick] [--real DIR] [--script PATH]
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import types
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parents[1] / "ops" / "docker" / "bake_comfy_assets.py"
# name -> ([(pattern, replacement), ...] each pattern exactly once in the script, the case that must fail)
MUTATIONS = {
    # the old script: any sha256 mismatch fails the build (an upstream re-upload forfeits every task)
    "no_fallback": ([('why = structure_problem(path + ".part", digest) if digest else "sha256 only"',
                      'why = "sha256 only"')], "changed"),
    # structure check disabled: any download of a VAE is accepted
    "no_struct": ([("    try:\n        got = structure_digest(header_tensors(path))\n",
                    "    return None\n    try:\n        got = structure_digest(header_tensors(path))\n")], "reject_truncated"),
    "no_size": ([("    if size != 8 + n + end:\n", "    if False:\n")], "reject_truncated"),
    "no_tiling": ([("        if b != end or e - b != numel * DTYPE_BYTES[dtype]:\n", "        if False:\n")], "reject_overlap"),
    # digest without shapes / dtypes: no longer the format the pinned digests were computed in
    "names_only": ([("""lines = sorted(f"{k}|{','.join(map(str, s))}|{t}" for k, (s, t) in tensors.items())""",
                     """lines = sorted(k for k, (s, t) in tensors.items())""")], "changed"),
    "no_retry": ([("    for attempt in range(tries):\n", "    for attempt in range(1):\n")], "transient"),
    # the structure path taken on a hash match too
    "match_struct": ([("            if got != want:\n", "            if True:\n")], "match"),
}
NBYTES = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8}
SPEC = {"decoder.conv_in.weight": ("F32", (8, 4, 3, 3)), "decoder.conv_in.bias": ("F32", (8,)),
        "decoder.conv_out.bias": ("F32", (8,)), "encoder.norm.weight": ("BF16", (8,)), "quant.scale": ("F16", (2, 2)),
        "bn.num_batches_tracked": ("I64", ())}
VAE = "vae.safetensors"
VERBOSE = [True]


def say(*a):
    print("[bake_check]", *a, flush=True)


def load(mutation=None, script=SCRIPT):
    src = Path(script).read_text()
    if mutation:
        for a, b in MUTATIONS[mutation][0]:
            if src.count(a) != 1:
                raise SystemExit(f"[bake_check] mutation {mutation}: pattern occurs {src.count(a)}x, not exactly once: {a!r}")
            src = src.replace(a, b)
    mod = types.ModuleType(f"g56bake_{mutation or 'code'}")
    mod.__file__ = str(script)
    exec(compile(src, f"{script}{f' [mutation {mutation}]' if mutation else ''}", "exec"), mod.__dict__)
    return mod


def sha(b):
    return hashlib.sha256(b).hexdigest()


def digest_of(spec):  # the documented format, not the script's function
    lines = sorted(f"{k}|{','.join(str(d) for d in shape)}|{dt}" for k, (dt, shape) in spec.items())
    return sha("\n".join(lines).encode())


def st_file(header, data):
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    return len(h).to_bytes(8, "little") + h + data


def st_bytes(spec, seed, metadata=None, order=None):
    rng, header, data = random.Random(seed), ({"__metadata__": metadata} if metadata else {}), b""
    for k in order or sorted(spec):
        dt, shape = spec[k]
        n = NBYTES[dt]
        for d in shape:
            n *= d
        header[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [len(data), len(data) + n]}
        data += rng.randbytes(n)
    return st_file(header, data)


REF = st_bytes(SPEC, 0, {"format": "pt"})
CHANGED = st_bytes(SPEC, 1, {"format": "pt", "modelspec.date": "2026-10-05"}, order=sorted(SPEC, reverse=True))
DIGEST = digest_of(SPEC)


def bad_files():
    n = int.from_bytes(CHANGED[:8], "little")
    raw = st_bytes({k: v for k, v in SPEC.items() if k != "decoder.conv_out.bias"}, 1)
    m = int.from_bytes(raw[:8], "little")
    hdr = json.loads(raw[8:8 + m])
    hdr["decoder.conv_out.bias"] = dict(hdr["decoder.conv_in.bias"])  # same 32-byte range as conv_in.bias
    bad_hdr = b'{"decoder.conv_in.bias":{"dtype":"F32","shape":[8],"data_offsets":[0,32]}'  # no closing brace
    return {"truncated": CHANGED[:-37], "trailing": CHANGED + bytes(8), "header_cut": CHANGED[:8 + n // 2],
            "bad_json": len(bad_hdr).to_bytes(8, "little") + bad_hdr + bytes(32),
            "huge_len": len(CHANGED).to_bytes(8, "little") + CHANGED[8:],
            "html": b"<!DOCTYPE html><html><body>502 Bad Gateway</body></html>\n",
            "extra_tensor": st_bytes({**SPEC, "decoder.extra.weight": ("F32", (2,))}, 1),
            "missing_tensor": st_bytes({k: v for k, v in SPEC.items() if k != "quant.scale"}, 1),
            "renamed": st_bytes({k.replace("conv_in.weight", "conv_in.weight2"): v for k, v in SPEC.items()}, 1),
            "shape": st_bytes({**SPEC, "decoder.conv_in.weight": ("F32", (4, 8, 3, 3))}, 1),
            "dtype": st_bytes({**SPEC, "quant.scale": ("BF16", (2, 2))}, 1),
            "overlap": st_file(hdr, raw[8 + m:])}


@contextlib.contextmanager
def instrumented(B, fail_first=0):
    """Counts downloads (the first `fail_first` raise URLError) and structure checks; captures the script's output."""
    real_open, real_struct, n, out = urllib.request.urlopen, B.structure_problem, {"open": 0, "struct": 0}, io.StringIO()

    def opener(req, *a, **k):
        n["open"] += 1
        if n["open"] <= fail_first:
            raise urllib.error.URLError("simulated transient failure")
        return real_open(req, *a, **k)

    def struct(*a, **k):
        n["struct"] += 1
        return real_struct(*a, **k)
    urllib.request.urlopen, B.structure_problem = opener, struct
    try:
        with contextlib.redirect_stdout(out):
            yield n, out
    finally:
        urllib.request.urlopen, B.structure_problem = real_open, real_struct


def run_fetch(B, served, want, digest, name=VAE, tries=3, fail_first=0, root=None):
    """fetch() of file://<served> into a fresh dir. Returns (result, output lines, downloads, structure checks, installed
    file or None, .part left)."""
    d = Path(tempfile.mkdtemp(prefix="bake_", dir=root))
    src = d / "served.bin"
    if isinstance(served, (bytes, bytearray)):
        src.write_bytes(served)
    else:
        src = Path(served)
    dest = d / Path(name).name  # main() maps "@flux/x" to <flux-te dir>/x
    with instrumented(B, fail_first) as (n, out):
        res = B.fetch(src.as_uri(), name, want, str(dest), digest, tries=tries, wait=0)
    return res, out.getvalue().splitlines(), n["open"], n["struct"], (dest if dest.exists() else None), \
        Path(str(dest) + ".part").exists()


def report(name, good, detail, lines=None):
    if VERBOSE[0]:
        say(f"{name}: {detail} -> {'ok' if good else 'FAIL'}")
        if not good and lines is not None:
            say("  output: " + " | ".join(lines)[-600:])
    return good


def case_match(B):
    res, lines, n, ns, dest, part = run_fetch(B, REF, sha(REF), DIGEST)
    good = res is True and lines == [f"[bake] {VAE} ok"] and n == 1 and ns == 0 and dest.read_bytes() == REF and not part
    return report("match", good, f"result {res}, output {lines}, {n} download(s), {ns} structure check(s)", lines)


def case_changed(B):
    res, lines, n, ns, dest, part = run_fetch(B, CHANGED, sha(REF), DIGEST)
    w = [x for x in lines if x.startswith(f"[bake] WARNING {VAE}:")]
    good = (res is True and n == 1 and len(lines) == 1 and len(w) == 1 and sha(CHANGED) in w[0]
            and "structure verified" in w[0] and dest is not None and dest.read_bytes() == CHANGED and not part)
    return report("changed", good, f"result {res}, {n} download(s), WARNING with the new sha256 + 'structure verified' "
                  f"{len(w) == 1 and sha(CHANGED) in w[0] and 'structure verified' in w[0]}, served bytes installed "
                  f"{dest is not None and dest.read_bytes() == CHANGED}", lines)


def case_reject(B, kind, bad):
    res, lines, n, ns, dest, part = run_fetch(B, bad, sha(REF), DIGEST, tries=3)
    why = [x for x in lines if "structure not verified" in x]
    good = (res is False and n == 3 and len(why) == 3 and dest is None and not any("WARNING" in x for x in lines)
            and sum(f"attempt {i} failed: OSError: sha256 mismatch" in x for i in (1, 2, 3) for x in lines) == 3)
    return report(f"reject_{kind}", good, f"result {res}, {n} download(s) for 3 tries, installed {dest is not None}"
                  + (f" ({why[0].split('verified: ', 1)[-1][:90]})" if why else ""), lines)


def case_te(B):
    name = "@flux/clip_l.safetensors"
    res, lines, n, ns, dest, part = run_fetch(B, CHANGED, sha(REF), None, name=name, tries=3)
    old = [f"[bake] {name} attempt {i} failed: OSError: sha256 mismatch for {name}: {sha(CHANGED)}" for i in (1, 2, 3)]
    good = res is False and lines == old and n == 3 and ns == 0 and dest is None
    return report("te", good, f"result {res}, output == the old script's lines {lines == old}, {ns} structure check(s)", lines)


def case_transient(B):
    res, lines, n, ns, dest, part = run_fetch(B, REF, sha(REF), DIGEST, tries=3, fail_first=2)
    good = (res is True and n == 3 and lines[-1:] == [f"[bake] {VAE} ok"] and len(lines) == 3
            and all("failed: URLError" in x for x in lines[:2]) and dest.read_bytes() == REF)
    return report("transient", good, f"result {res}, {n} download(s), {len(lines) - 1} failed attempt line(s)", lines)


def case_lib(B):
    try:
        import numpy as np
        from safetensors.numpy import save
    except Exception as e:  # noqa: BLE001
        return report("lib", True, f"safetensors.numpy not importable ({type(e).__name__}) - skipped")
    spec = {"decoder.conv_in.weight": ("F32", (8, 4, 3, 3)), "decoder.conv_in.bias": ("F32", (8,)),
            "quant.scale": ("F16", (2, 2)), "bn.num_batches_tracked": ("I64", ())}
    npt = {"F32": np.float32, "F16": np.float16, "I64": np.int64}

    def mk(seed, meta):
        rng = np.random.default_rng(seed)
        return save({k: np.asarray(rng.standard_normal(s) * 100).astype(npt[dt]).reshape(s) for k, (dt, s) in spec.items()},
                    metadata=meta)
    ref, new = mk(0, {"format": "pt"}), mk(1, None)
    p = Path(tempfile.mkdtemp(prefix="bake_lib_")) / "ref.safetensors"
    p.write_bytes(ref)
    prob = B.structure_problem(str(p), digest_of(spec))
    r1, l1, *_ = run_fetch(B, new, sha(ref), digest_of(spec))
    r2, l2, *_ = run_fetch(B, new[:-5], sha(ref), digest_of(spec), tries=1)
    good = prob is None and r1 is True and any("structure verified" in x for x in l1) and r2 is False
    return report("lib", good, f"library-written reference passes {prob is None}, re-upload accepted {r1}, cut one "
                  f"rejected {r2 is False}", l1 + l2)


def run_main(B, argv, fail_first=0):
    with instrumented(B, fail_first) as (n, out):
        try:
            B.main(argv)
            code = None
        except SystemExit as e:
            code = e.code
    return out.getvalue().splitlines(), n["open"], code


def case_cli(B):
    d = Path(tempfile.mkdtemp(prefix="bake_cli_"))
    names = ["flux2-vae.safetensors", "qwen_image_vae.safetensors", "flux_ae.safetensors"]
    files = []
    for i, name in enumerate(names):
        b = st_bytes(SPEC, 10 + i)
        (d / f"src{i}").write_bytes(b)
        files.append(((d / f"src{i}").as_uri(), name, sha(b), DIGEST))
    te = st_bytes(SPEC, 20)
    (d / "te").write_bytes(te)
    saved = B.FILES, B.FLUX_TE_FILES, B.fetch.__defaults__
    policy = B.fetch.__defaults__ == (None, 6, 5)  # the old CLI's retries: 6 tries, sleep 5 s x attempt
    B.FILES, B.FLUX_TE_FILES = files, [((d / "te").as_uri(), "@flux/clip_l.safetensors", sha(te), None)]
    B.fetch.__defaults__ = (None, 2, 0)  # 2 tries, no sleep
    try:
        o1, n1, c1 = run_main(B, ["bake", str(d / "a")])
        ok1 = o1 == [f"[bake] {x} ok" for x in names] and n1 == 3 and c1 is None and sorted(os.listdir(d / "a")) == sorted(names)
        o2, n2, c2 = run_main(B, ["bake", str(d / "b"), "--flux-te", str(d / "flux")])
        ok2 = (o2 == o1 + ["[bake] @flux/clip_l.safetensors ok"] and n2 == 4 and c2 is None
               and (d / "flux" / "clip_l.safetensors").read_bytes() == te)
        B.FILES = [files[0], files[1][:2] + ("0" * 64, "0" * 64), files[2]]
        o3, n3, c3 = run_main(B, ["bake", str(d / "c")])
        ok3 = c3 == "[bake] giving up on qwen_image_vae.safetensors" and n3 == 3 and not (d / "c" / names[2]).exists()
    finally:
        B.FILES, B.FLUX_TE_FILES, B.fetch.__defaults__ = saved
    return report("cli", policy and ok1 and ok2 and ok3, f"retry policy 6 tries / 5 s x attempt {policy}, VAEs only "
                  f"without --flux-te {ok1}, TE too with --flux-te {ok2}, SystemExit giving up {ok3}", o1 + o2 + o3)


def run_checks(B):
    cases = [("match", case_match), ("changed", case_changed), ("te", case_te), ("transient", case_transient),
             ("lib", case_lib), ("cli", case_cli)]
    cases += [(f"reject_{k}", (lambda B, k=k, v=v: case_reject(B, k, v))) for k, v in bad_files().items()]
    fails = []
    for name, fn in cases:
        try:
            good = fn(B)
        except Exception as e:  # noqa: BLE001
            report(name, False, f"{type(e).__name__}: {e}")
            good = False
        if not good:
            fails.append(name)
    return fails


def check_real(B, d):
    """The pinned digests against the real files (run on a bake output dir)."""
    try:
        from safetensors import safe_open
    except Exception:  # noqa: BLE001
        safe_open = None
    try:
        import torch  # noqa: F401
        fw = "pt"
    except Exception:  # noqa: BLE001
        fw = "np"  # no bf16 in numpy: qwen_image_vae then only opens, no tensor read
    ok, tmp = True, Path(tempfile.mkdtemp(prefix="bake_real_", dir=d))
    try:
        for url, name, want, digest in B.FILES:
            p = Path(d) / name
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(8 << 20), b""):
                    h.update(b)
            prob, lib = B.structure_problem(str(p), digest), None
            if safe_open is not None:
                with safe_open(str(p), framework=fw) as f:
                    lib = sha("\n".join(sorted(f"{k}|{','.join(map(str, f.get_slice(k).get_shape()))}|"
                                               f"{f.get_slice(k).get_dtype()}" for k in f.keys())).encode())
            c = tmp / name
            shutil.copyfile(p, c)
            with open(c, "r+b") as f:  # one data byte changed: same structure, another sha256
                n = int.from_bytes(f.read(8), "little")
                f.seek(8 + n + (c.stat().st_size - 8 - n) // 2)
                x = f.read(1)
                f.seek(-1, 1)
                f.write(bytes([x[0] ^ 1]))
            res, lines, nd, ns, dest, part = run_fetch(B, c, want, digest, name=name, tries=1, root=tmp)
            opens = True
            if safe_open is not None and dest is not None:
                with safe_open(str(dest), framework=fw) as f:  # the library's own header / size validation
                    keys = list(f.keys())
                    if fw == "pt" or "BF16" not in {f.get_slice(k).get_dtype() for k in keys}:
                        opens = f.get_tensor(keys[0]) is not None and f.get_tensor(keys[-1]) is not None
            with open(p, "rb") as f:
                (tmp / "prefix").write_bytes(f.read(4 << 20))
            r2, *_ = run_fetch(B, tmp / "prefix", want, digest, name=name, tries=1, root=tmp)
            good = (h.hexdigest() == want and prob is None and lib in (None, digest) and res is True and opens
                    and any("structure verified" in x for x in lines) and r2 is False)
            say(f"real {name}: sha256 pinned {h.hexdigest() == want}, structure check {prob or 'passes'}, safe_open digest "
                f"{'== pinned' if lib == digest else lib}, 1-byte-changed copy accepted with WARNING {res} (opens {opens}), "
                f"4 MB prefix rejected {r2 is False} -> {'ok' if good else 'FAIL'}")
            ok &= good
            for q in tmp.iterdir():
                shutil.rmtree(q) if q.is_dir() else q.unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutation", choices=sorted(MUTATIONS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--real", default=None, help="dir with the three pinned VAE files (a real bake output)")
    ap.add_argument("--script", default=str(SCRIPT))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="bake_check_") as t:
        tempfile.tempdir = t  # every synthetic case's files under one dir, removed at exit
        ok = run(a)
    sys.exit(0 if ok else 1)


def run(a):
    if a.mutation:
        fails = run_checks(load(a.mutation, a.script))
        want = MUTATIONS[a.mutation][1]
        say(f"mutation {a.mutation}: failing cases {fails or 'none'} -> {'caught' if want in fails else 'NOT CAUGHT'} by {want}")
        say(f"RESULT {'FAIL' if fails else 'PASS'}")
        return not fails
    B = load(None, a.script)
    fails = run_checks(B)
    ok = not fails
    say(f"cases on the current code: {'PASS' if ok else 'FAIL ' + str(fails)}")
    if a.real:
        ok &= check_real(B, a.real)
    if not a.quick:
        VERBOSE[0] = False
        for mut, (_, want) in MUTATIONS.items():
            fails = run_checks(load(mut, a.script))
            say(f"mutation {mut}: {want} {'fails -> caught' if want in fails else 'passes -> NOT CAUGHT'} "
                f"(failing: {', '.join(fails) or 'none'})")
            ok &= want in fails
    say(f"RESULT {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    main()
