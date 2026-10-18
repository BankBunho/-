#!/usr/bin/env python3
"""End-to-end CPU smoke of the trainer's main() with G56CT_TE_LORA=1: real DiT (from /cache), real offline-rebuilt TE,
real VAE; only the images are shrunk (64x64 instead of 1024 max side) so a CPU can run DiT steps. Exercises the whole
wiring: TE kept after conditioning, per-step TE conditioning slots, TE adapters on the device, exact holdout with the TE
merge, drop check, final evaluator-style verification on the real model + clip, export.

python te_main_smoke.py --family ideogram4 --model gradients-io-tournaments/ideogram-4-fp8 --out /w/out/smoke
"""
import argparse
import io
import os
import random
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))

CAPTIONS = ["a red fox under a crescent moon, flat vector, teal sky", "minimal logo of a coffee cup, black on white",
            "watercolor harbor at dawn with fishing boats", "isometric pixel art bakery storefront with a neon sign",
            "product photo of a green glass bottle on marble", "poster with the words SUMMER JAZZ in bold yellow letters",
            "a knight in ornate armor, oil painting, dramatic light"]


def make_dataset(path):
    from PIL import Image, ImageDraw
    rnd = random.Random(0)
    with zipfile.ZipFile(path, "w") as z:
        for i, cap in enumerate(CAPTIONS):
            im = Image.new("RGB", (96, 80), tuple(rnd.randrange(256) for _ in range(3)))
            d = ImageDraw.Draw(im)
            for _ in range(4):
                x, y = rnd.randrange(80), rnd.randrange(64)
                d.ellipse([x, y, x + rnd.randrange(8, 30), y + rnd.randrange(8, 30)], fill=tuple(rnd.randrange(256) for _ in range(3)))
            b = io.BytesIO()
            im.save(b, format="PNG")
            z.writestr(f"img{i}.png", b.getvalue())
            z.writestr(f"img{i}.txt", cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="ideogram4")
    ap.add_argument("--model", default="gradients-io-tournaments/ideogram-4-fp8")
    ap.add_argument("--out", default="/w/out/smoke")
    ap.add_argument("--side", type=int, default=64)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = out / "smoke_ds.zip"
    make_dataset(ds)
    for k, v in dict(TE_LORA="1", MAX_STEPS="2", EVAL_FIRST="2", EVAL_EVERY="2", HOLDOUT_STRATA="2", HOLDOUT_NOISES="1",
                     EVAL_TAGS="raw", GRAD_CKPT="on", VERBOSE="1", WARMUP="1", RANK="4", ALPHA="4",
                     LOGFILE=str(out / "g56ct.log"), SUMMARY=str(out / "summary.json")).items():
        os.environ.setdefault("G56CT_" + k, v)
    import torch
    if not torch.cuda.is_available():
        import comfy.cli_args
        comfy.cli_args.args.cpu = True
    from PIL import Image
    import evalcore

    def small(image, side=a.side):  # evaluator geometry at a toy size (multiples of 16)
        return image.convert("RGB").resize((side, side), Image.Resampling.LANCZOS)
    evalcore.adjust_image = small
    import g56_comfy_train as T
    sys.argv = ["g56_comfy_train.py", "--task-id", "smoke", "--model", a.model, "--dataset-zip", str(ds), "--dataset", str(ds),
                "--model-type", a.family, "--expected-repo-name", "smoke", "--hours-to-complete", "4", "--out-dir", str(out / "lora")]
    T.main()
    from safetensors.torch import load_file
    sd = load_file(str(out / "lora" / "last.safetensors"))
    n_te = sum(1 for k in sd if k.startswith(T.TE_PREFIX) and k.endswith(".lora_up.weight"))
    n_dit = sum(1 for k in sd if k.startswith("diffusion_model.") and k.endswith(".lora_up.weight"))
    print(f"[smoke] main() finished: file has {n_dit} DiT + {n_te} TE LoRA modules ({len(sd)} tensors)", flush=True)


if __name__ == "__main__":
    main()
