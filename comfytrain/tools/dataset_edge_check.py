#!/usr/bin/env python3
"""Dataset edge cases, CPU only: synthetic training zips through the trainer's own data path (read_rows = the evaluator's
read_dataset + caption filter, then per image adjust_image(Image.open(raw)) exactly as main() does). For every case: does
it crash, how many rows survive, what the images become. Prints one line per case and a final table.
Usage: python comfytrain/tools/dataset_edge_check.py [--module path/to/g56_comfy_train.py]
"""
import argparse
import importlib.util
import io
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent.parent


def load(path):
    spec = importlib.util.spec_from_file_location("g56_dataset_edge", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.log = lambda *a, **k: print("    [trainer log]", *a)
    return m


def img_bytes(mode="RGB", size=(640, 480), fmt="PNG", exif_orientation=None):
    if mode == "I;16":
        im = Image.new("I;16", size)
    elif mode == "CMYK":
        im = Image.new("CMYK", size, (10, 20, 30, 40))
        fmt = "JPEG"
    else:
        im = Image.new(mode, size)
    b = io.BytesIO()
    kw = {}
    if exif_orientation:
        ex = Image.Exif()
        ex[0x0112] = exif_orientation
        kw["exif"] = ex.tobytes()
    im.save(b, fmt, **kw)
    return b.getvalue()


def zip_of(entries):
    p = Path(tempfile.mkdtemp()) / "train.zip"
    import warnings
    with warnings.catch_warnings(), zipfile.ZipFile(p, "w") as z:
        warnings.simplefilter("ignore")  # duplicate names are the point of one case
        for name, data in entries:
            z.writestr(name, data)
    return p


def base(n=9, ext="png", fmt="PNG", prefix=""):
    out = []
    for i in range(n):
        out.append((f"{prefix}img{i}.{ext}", img_bytes(fmt=fmt)))
        out.append((f"{prefix}img{i}.txt", f"a caption {i}".encode()))
    return out


def cases():
    c = {}
    c["baseline N=9"] = base()
    c["nested folders"] = base(prefix="data/sub/")
    c["__MACOSX + ._ files"] = base() + [(f"__MACOSX/._img{i}.png", b"\x00\x05\x16\x07binary-appledouble") for i in range(9)] \
        + [(f"__MACOSX/._img{i}.txt", b"\x00\x05\x16\x07\xff\xfe binary") for i in range(9)]
    c["uppercase .PNG/.JPG/.JPEG"] = [(f"u{i}.{e}", img_bytes(fmt="PNG" if e == "PNG" else "JPEG")) for i, e in enumerate(["PNG", "JPG", "JPEG"])] \
        + [(f"u{i}.txt", b"cap") for i in range(3)]
    c["webp"] = [("w0.webp", img_bytes(fmt="WEBP")), ("w0.txt", b"cap")]
    for mode in ("RGBA", "LA", "P", "L", "CMYK", "I;16"):
        c[f"mode {mode}"] = [("m0.png" if mode != "CMYK" else "m0.jpg", img_bytes(mode=mode)), ("m0.txt", b"cap")]
    c["EXIF orientation 6"] = [("e0.jpg", img_bytes(fmt="JPEG", exif_orientation=6)), ("e0.txt", b"cap")]
    c["tiny 64x48"] = [("t0.png", img_bytes(size=(64, 48))), ("t0.txt", b"cap")]
    c["huge 6000x4000"] = [("h0.png", img_bytes(size=(6000, 4000))), ("h0.txt", b"cap")]
    c["ratio 1:4"] = [("r0.png", img_bytes(size=(256, 1024))), ("r0.txt", b"cap")]
    c["missing .txt for 2 of 9"] = [e for e in base() if e[0] not in ("img3.txt", "img7.txt")]
    c["empty .txt for 1 of 9"] = [(n, b"   " if n == "img2.txt" else d) for n, d in base()]
    c["non-UTF-8 caption"] = [(n, b"caf\xe9 latin-1" if n == "img4.txt" else d) for n, d in base()]
    c["very long caption (>1000 words)"] = [(n, (" ".join(["word"] * 1500)).encode() if n == "img0.txt" else d) for n, d in base()]
    c["duplicate image content"] = base() + [("dup.png", base()[0][1]), ("dup.txt", b"dup caption")]
    c["undecodable image"] = base() + [("bad.png", b"not really a png"), ("bad.txt", b"cap")]
    c["duplicate archive entries"] = base() + [("img0.png", img_bytes()), ("img0.txt", b"second copy")]
    return c


def run_case(T, name, entries):
    import evalcore
    p = zip_of(entries)
    try:
        rows = T.read_rows(p)
    except Exception as e:  # noqa: BLE001
        return f"CRASH in read_rows: {type(e).__name__}: {str(e)[:80]}"
    if hasattr(T, "decode_items"):  # the trainer's own decode step (main() calls it)
        try:
            items = T.decode_items(rows)
        except Exception as e:  # noqa: BLE001
            return f"CRASH in decode_items: {type(e).__name__}: {str(e)[:80]}"
        return f"ok rows={len(rows)} images={len(items)} sizes={sorted(set(it['image'].size for it in items))[:3]}"
    sizes, errs = [], []  # older trainers: main()'s inline loop
    for n, raw, cap in rows:
        try:
            im = evalcore.adjust_image(Image.open(io.BytesIO(raw)))
            sizes.append(im.size)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{n}: {type(e).__name__}: {str(e)[:60]}")
    if errs:
        return f"CRASH in image decode ({len(errs)}/{len(rows)}): {errs[0]}"
    return f"ok rows={len(rows)} sizes={sorted(set(sizes))[:3]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=str(HERE / "g56_comfy_train.py"))
    ap.add_argument("--old", default=None, help="mutation check: an older trainer must crash on some case")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.module).resolve().parent))
    T = load(a.module)
    n_new = check(T, "new")
    ok = n_new == 0
    if a.old:
        n_old = check(load(a.old), "old")
        print(f"[edge] mutation (old trainer) crashes on {n_old} case(s): {'caught' if n_old > 0 else 'NOT caught'}", flush=True)
        ok &= n_old > 0
    print(f"[edge] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    raise SystemExit(0 if ok else 1)


def check(T, tag):
    res = {}
    for name, entries in cases().items():
        r = run_case(T, name, entries)
        res[name] = r
        print(f"[edge:{tag}] {name:34s} {r}", flush=True)
    n_crash = sum(r.startswith("CRASH") for r in res.values())
    print(f"[edge:{tag}] {len(res)} cases, {n_crash} crash", flush=True)
    return n_crash


if __name__ == "__main__":
    main()
