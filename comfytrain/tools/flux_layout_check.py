#!/usr/bin/env python3
"""Flux base layouts, CPU only: a tiny real ComfyUI Flux (hidden 128, one double + one single block) written in every
layout the tournament has shipped, loaded through the trainer's own root_checkpoint + load_base_model
(comfy.sd.load_diffusion_model = the evaluator's UNETLoader prefix detection):
  1. rayonlabs/FLUX.1-dev style: BFL file without key prefix + ae.safetensors (< 5 GiB, must not be picked) + diffusers
     text_encoder folders;
  2. PixelWave style: bf16 + fp8 + nf4 root exports with the model.diffusion_model. prefix (the bf16 one must be picked);
  3. all-in-one: one prefixed file that also carries text_encoders.* and vae.* (only the diffusion weights are loaded).
Per layout: the picked file, model class Flux with the right size, the loaded weights equal the source, the family LoRA
set attaches (double + single blocks) and the pre-training key check drops nothing.
Mutation: picking the smallest root file (nf4 export) must FAIL the PixelWave case.
Usage (inside the trainer image or with COMFY_ROOT): python comfytrain/tools/flux_layout_check.py
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
os.environ.setdefault("G56CT_INIT", "gauss")

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "3")))
import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True
spec = importlib.util.spec_from_file_location("g56fl", HERE / "g56_comfy_train.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
T.log = lambda *a, **k: None
from safetensors.torch import save_file  # noqa: E402

REPO = "some-org/flux-finetune"


def tiny_flux_sd():
    import comfy.ops
    from comfy.ldm.flux.model import Flux
    torch.manual_seed(0)
    m = Flux(image_model="flux", in_channels=16, out_channels=16, vec_in_dim=32, context_in_dim=64, hidden_size=128,
             mlp_ratio=4.0, num_heads=1, depth=1, depth_single_blocks=1, axes_dim=[16, 56, 56], theta=10000, patch_size=2,
             qkv_bias=True, guidance_embed=True, txt_ids_dims=[], dtype=torch.float32, device="cpu",
             operations=comfy.ops.disable_weight_init)
    return {k: (torch.randn_like(v) * 0.02).to(torch.bfloat16) for k, v in m.state_dict().items()}


def small(shape, v):
    return torch.full(shape, v, dtype=torch.float32)


def diffusers_te(d):
    (d / "text_encoder").mkdir()
    save_file({"text_model.final_layer_norm.weight": small((4,), 1.0)}, str(d / "text_encoder" / "model.safetensors"))
    (d / "text_encoder_2").mkdir()
    save_file({"shared.weight": small((10, 4), 2.0)}, str(d / "text_encoder_2" / "model.safetensors"))


def layout(kind, sd):
    root = Path(tempfile.mkdtemp())
    d = root / "models" / REPO.replace("/", "--")
    d.mkdir(parents=True)
    pre = {f"model.diffusion_model.{k}": v for k, v in sd.items()}
    if kind == "bfl":
        save_file(sd, str(d / "flux1-dev.safetensors"))
        save_file({"decoder.conv_in.weight": small((4, 4, 3, 3), 3.0)}, str(d / "ae.safetensors"))
        diffusers_te(d)
        want = "flux1-dev.safetensors"
    elif kind == "pixelwave":
        save_file(pre, str(d / "pixelwave_flux1_dev_bf16_03.safetensors"))
        save_file({k: v.to(torch.float8_e4m3fn) for k, v in pre.items()}, str(d / "pixelwave_flux1_dev_fp8_03.safetensors"))
        save_file({k: torch.zeros(max(1, v.numel() // 4), dtype=torch.uint8) for k, v in pre.items()},
                  str(d / "pixelwave_flux1_dev_nf4_03.safetensors"))
        diffusers_te(d)
        want = "pixelwave_flux1_dev_bf16_03.safetensors"
    else:
        ck = dict(pre)
        ck.update({"text_encoders.clip_l.transformer.text_model.final_layer_norm.weight": small((4,), 5.0),
                   "text_encoders.t5xxl.transformer.shared.weight": small((10, 4), 6.0),
                   "vae.decoder.conv_in.weight": small((4, 4, 3, 3), 7.0)})
        save_file(ck, str(d / "flux1-dev-all-in-one.safetensors"))
        want = "flux1-dev-all-in-one.safetensors"
    return root, want


def check(kind, sd):
    root, want = layout(kind, sd)
    T.CACHE = root
    T._MODEL_DIRS.clear()
    ok = {}
    try:
        ck = T.root_checkpoint(REPO)
        ok["picked"] = ck.name == want
        mp = T.load_base_model("flux", REPO, torch.device("cpu"))
        dm = mp.model.diffusion_model
        ok["flux_class"] = type(dm).__name__ == "Flux" and dm.hidden_size == 128 and len(dm.double_blocks) == 1
        ok["weights_equal"] = torch.equal(dm.state_dict()["img_in.weight"].to(torch.bfloat16).cpu(), sd["img_in.weight"])
        mp.model.requires_grad_(False)
        adapters, _ = T.attach_lora(mp, T.FAMILIES["flux"]["include"], 4, 4.0)
        names = set(adapters)
        ok["lora_targets"] = {"diffusion_model.double_blocks.0.img_attn.qkv",
                              "diffusion_model.single_blocks.0.linear1"} <= names
        ok["key_check_0_dropped"] = T.drop_unloadable(mp, adapters) == [] and len(adapters) == len(names)
    except Exception as e:  # noqa: BLE001
        ok["no_exception"] = False
        print(f"[flux-layout] {kind}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print(f"[flux-layout] {kind}: {ok}", flush=True)
    return all(ok.values()), ok


def main():
    T.ROOT_MIN_BYTES = 100_000  # tiny stand-ins for the >5 GiB root files; ae.safetensors stays below
    sd = tiny_flux_sd()
    res = [check(k, sd)[0] for k in ("bfl", "pixelwave", "all_in_one")]
    orig = T.root_checkpoint

    def smallest(repo):  # mutation: the nf4 export instead of the bf16 one
        d = T.model_dir(repo)
        c = [f for f in d.glob("*.safetensors") if f.stat().st_size > T.ROOT_MIN_BYTES]
        return sorted(c, key=lambda f: f.stat().st_size)[0]
    T.root_checkpoint = smallest
    try:
        caught = not check("pixelwave", sd)[0]
    finally:
        T.root_checkpoint = orig
    print(f"[flux-layout] mutation (smallest root file) caught: {caught}", flush=True)
    ok = all(res) and caught
    print(f"[flux-layout] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
