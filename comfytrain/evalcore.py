"""Evaluator-identical primitives, copied verbatim from the validator (gradients-ai/G.O.D @ 3f3f4d55, Apache-2.0):
validator/evaluation/image_test_data.py (adjust_image, read_dataset, decoded_images) and
validator/evaluation/denoising_mse.py (case_seed, flow_cases, flow_prediction_mse).
Kept byte-identical in behaviour so training data preparation and holdout scoring match the official metric exactly."""
import hashlib
import torch
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image


EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def adjust_image(image):
    w, h = image.size
    if min(w, h) <= 0:
        raise ValueError("Invalid image dimensions")
    size = (1024, int(h / w * 1024)) if w > h else (int(w / h * 1024), 1024)
    image = image.resize(size, Image.Resampling.LANCZOS)
    cw, ch = (v // 16 * 16 for v in size)
    left, top = (size[0] - cw) // 2, (size[1] - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).convert("RGB")


def read_dataset(source, download_path):
    """Read a nested local directory/ZIP or HTTP ZIP without extracting archive paths."""
    if str(source).startswith(("https://", "http://")):
        with urllib.request.urlopen(source, timeout=120) as response, download_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        source = download_path
    source = Path(source)
    rows = []
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                if not path.resolve().is_relative_to(source.resolve()):
                    raise ValueError("Dataset image escapes its directory")
                caption = path.with_suffix(".txt")
                if caption.exists() and not caption.resolve().is_relative_to(source.resolve()):
                    raise ValueError("Dataset caption escapes its directory")
                rows.append((str(path.relative_to(source)), path.read_bytes(), caption.read_text() if caption.exists() else None))
    else:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate archive entries")
            for name in sorted(names):
                if name.lower().endswith(EXTENSIONS):
                    caption = str(Path(name).with_suffix(".txt"))
                    rows.append((name, archive.read(name), archive.read(caption).decode("utf-8") if caption in names else None))
    if not rows:
        raise ValueError("Empty image dataset")
    return rows


def decoded_images(rows):
    result = []
    for name, raw, caption in rows:
        if not caption or not caption.strip():
            raise ValueError("Image evaluation requires a caption for every image")
        image = adjust_image(Image.open(io.BytesIO(raw)))
        digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
        result.append({"name": name, "image": image, "caption": caption, "sha256": digest})
    return result

def case_seed(image_hash: str, stratum: int, noise_index: int, master_seed: int = 42) -> int:
    key = f"{master_seed}:{image_hash}:{stratum}:{noise_index}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)


def flow_cases(latent, image_hash: str, strata: int = 16, noises: int = 4):
    if strata < 1 or noises < 1:
        raise ValueError("Strata and noise counts must be positive")
    if latent.shape[0] != 1 or latent.device.type != "cpu":
        raise ValueError("Cache one CPU latent per image")
    cases = []
    for band in range(strata):
        # Midpoints exclude unstable endpoints while covering uniform flow time.
        sigma = (band + 0.5) / strata
        seeds = [case_seed(image_hash, band, sample) for sample in range(noises)]
        noise = torch.cat(
            [torch.randn(latent.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32) for seed in seeds]
        )
        cases.append(
            {
                "stratum": band,
                "sigma": sigma,
                "seeds": seeds,
                "noisy": (1 - sigma) * latent + sigma * noise,
                "target": noise - latent,
            }
        )
    return cases


def flow_prediction_mse(noisy, denoised, target, sigma):
    if not 0 < sigma < 1:
        raise ValueError("Flow sigma must lie strictly between zero and one")
    if noisy.shape != denoised.shape or target.shape != noisy.shape:
        raise ValueError("Prediction and target dimensions differ")
    # Comfy returns x0 = x_t - sigma * velocity, so recover the velocity target.
    predicted = (noisy.float() - denoised.float()) / sigma
    losses = (predicted - target.float()).square().flatten(1).mean(1)
    if not torch.isfinite(losses).all():
        raise ValueError("Non-finite denoising MSE")
    return losses
