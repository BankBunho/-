#!/usr/bin/env python3
"""g56 ComfyUI-native image LoRA trainer for the SN56 image tournament.

Trains INSIDE the validator's own evaluation stack (ComfyUI 694815f4, torch 2.9.1): the same model classes, the same
conditioning path (CLIPTextEncode on the evaluator's exact text encoder, rebuilt byte-identically from /cache), the same
VAE posterior-mean latents, the same prediction call (model_wrap(noisy, sigma) through a CFG=1 guider) and the evaluator's
own flow-MSE target. Checkpoints are selected on a held-out split with the evaluator's own case generator.

Validator CLI (unchanged): --task-id --model --dataset-zip --model-type --expected-repo-name --hours-to-complete [--trigger-word]
Output: /app/checkpoints/<task_id>/<expected_repo_name>/last.safetensors  (Comfy LoRA keys: diffusion_model.<m>.lora_up/down.weight, .alpha)

Knobs are environment variables G56CT_* (see CFG below); nothing here downloads anything (container is offline).
"""
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import traceback
import zipfile
from pathlib import Path

T0 = time.time()


def _load_profile():
    """comfytrain/profile.env (KEY=VALUE lines) sets per-entry defaults; explicit environment variables win."""
    f = Path(__file__).resolve().parent / "profile.env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_profile()


VERBOSE = os.environ.get("G56CT_VERBOSE", "0") == "1"
_OUT_DIR = None  # set by main() as soon as the upload folder is known; the crash handler looks for a usable adapter there
LOGFILE = os.environ.get("G56CT_LOGFILE", "/tmp/g56ct.log")


def log(*a, public=False):
    """Container stdout is public (validator Loki). Details go to a private file; stdout gets only heartbeats unless
    G56CT_VERBOSE=1 (local tests)."""
    line = f"[ct {time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    try:
        with open(LOGFILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    if VERBOSE or public:
        print(line if VERBOSE else " ".join(str(x) for x in a), flush=True)


def env(name, default, cast=str):
    v = os.environ.get("G56CT_" + name)
    if v is None or v == "":
        return default
    if cast is bool:
        return v.lower() in ("1", "true", "yes", "on")
    return cast(v)


COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
ASSETS = Path(os.environ.get("G56CT_ASSETS", str(Path(__file__).resolve().parent / "assets")))
BAKED = Path(os.environ.get("G56CT_BAKED", "/opt/g56/baked"))          # Comfy-Org VAEs downloaded at docker build
CACHE = Path(os.environ.get("CACHE_ROOT", "/cache"))
CKPT_ROOT = Path(os.environ.get("G56CT_CKPT_ROOT", "/app/checkpoints"))

# ----------------------------------------------------------------------------------------------------------------------
# Family contracts (mirror validator/evaluation/image_flow_adapter.py FAMILIES + prepare_base)
# ----------------------------------------------------------------------------------------------------------------------
FAMILIES = {
    # te_taps: ComfyUI's tapped-hidden-state list of the family's TE (G56CT_TE_LORA=1 adapts the layers that feed it)
    "ideogram4": dict(clip_type="IDEOGRAM4", te_snapshot="hf_cache/Qwen--Qwen3-VL-8B-Instruct",
                      te_table="te_ideogram4_qwen3vl_8b_fp8_scaled.json", te_map="qwen3vl",
                      vae_file="flux2-vae.safetensors", shift=None, guidance=None,
                      te_taps=("comfy.text_encoders.ideogram4", "IDEOGRAM4_TAP_LAYERS"),
                      include=r"^layers\.\d+\.(attention\.(qkv|o)|adaln_modulation|feed_forward\.(w1|w2|w3))$"),
    "krea2": dict(clip_type="KREA2", te_snapshot="hf_cache/Qwen--Qwen3-VL-4B-Instruct",
                  te_table="te_krea2_qwen3vl_4b_fp8_scaled.json", te_map="qwen3vl",
                  vae_file="qwen_image_vae.safetensors", shift=None, guidance=None,
                  te_taps=("comfy.text_encoders.krea2", "KREA2_TAP_LAYERS"),
                  include=r"^blocks\.\d+\.(attn\.(wq|wk|wv|wo|gate)|mlp\.(gate|up|down))$"),
    # qwen-image: evaluator TE = Comfy-Org qwen_2.5_vl_7b_fp8_scaled (old scale_weight format); the task repo's bf16
    # text_encoder has the same 729 names and RTNE(fp32(W)/scale) reproduces every fp8 byte (verified 2026-09-22)
    "qwen-image": dict(clip_type="QWEN_IMAGE", te_snapshot=None, te_repo_subdir="text_encoder",
                       te_table="te_qwen-image_qwen25vl_7b_fp8_scaled.json", te_map=None,
                       vae_file="qwen_image_vae.safetensors", shift=3.0, guidance=None,
                       include=r"^transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out)|img_mlp\.net\.(0\.proj|2)|txt_mlp\.net\.(0\.proj|2))$"),
    # z-image: the evaluator's TE file (Comfy-Org qwen_3_4b.safetensors, 398 bf16 tensors) is byte-identical to the task
    # repo's text_encoder shards (verified 2026-09-22) -> loaded as-is; VAE = the repo's diffusers VAE (evaluator contract)
    "z-image": dict(clip_type="LUMINA2", te_snapshot=None, te_repo_subdir="text_encoder", te_table=None, te_map=None,
                    vae_file=None, shift=None, guidance=None,
                    include=r"^layers\.\d+\.(attention\.(qkv|out)|feed_forward\.(w1|w2|w3)|adaLN_modulation\.0)$"),
    # flux: evaluator TEs are fp16 files (t5xxl_fp16, clip_l); the task repo carries bf16 copies -> loaded and cast to
    # fp16 (not byte-identical: bf16-rounded weights, ~2^-9 relative). Single-file repos borrow text_encoder(_2) dirs
    # from another cached FLUX snapshot.
    # The flux trainer image bakes the evaluator's own TE files (comfyanonymous/flux_text_encoders t5xxl_fp16 + clip_l,
    # sha256-identical to main) -> loaded exactly like DualCLIPLoader; otherwise fall back to the repo copies.
    "flux": dict(clip_type="FLUX", te_snapshot=None, te_files=("/app/flux/t5xxl_fp16.safetensors", "/app/flux/clip_l.safetensors"),
                 te_dual=("text_encoder_2", "text_encoder"), te_cast="fp16",
                 te_table=None, te_map=None, vae_file="flux_ae.safetensors", shift=None, guidance=3.5,
                 include=r"^(double_blocks\.\d+\.(img_attn\.(qkv|proj)|txt_attn\.(qkv|proj)|img_mlp\.(0|2)|txt_mlp\.(0|2))|single_blocks\.\d+\.(linear1|linear2))$"),
}

# G56CT_SMALL=auto: per-family DiffFit-style set (ours). ideogram4: sandwich-norm gains of every block (pre-norm input
# gains + post-norm branch-output gains) and the per-block adaLN biases (constant scale/gate offsets), final-layer biases.
# krea2: pre/post-norm gains and the learned per-block modulation offsets (mod.lin), last-layer norm/modulation/bias.
# Production defaults per family (freeze, B0): applied in main() to every CFG key whose G56CT_* env var is NOT set, so the
# shipped image runs the winning recipe with no env at all, while experiments can still override any key. Values are
# CFG-typed. Filled at freeze from the measured winners (riset/DECISIONS.md); G56CT_FAMILY_DEFAULTS=0 switches them off
# (research runs that must keep the plain trainer defaults for every key they do not set).
# Shared by every family: evaluator sigma midpoints walked (image, band) pair by pair, gauss init (R1 -1.9% vs ComfyUI's
# init), 97% of the budget minus 120 s (the validator uploads the checkpoint dir even on timeout; saves are atomic) and an
# identity LoRA written before training (never a forfeit).
# 1 Oct (T016 G0 on the H100): G56CT_SAFE never ships a non-finite LoRA, survives an out-of-memory evaluator-exact eval and
# writes less at the end of an ensemble (dad11714 0.25 h: SAFE on 0.111827 = off 0.111827, same s/step, file 23 s earlier);
# G56CT_WATCHDOG_S 60 exits 0 with the saved adapter 60 s before the validator's kill if a run overruns (z-image test: fired
# at the planned second, a trained LoRA was shipped).
_FD_COMMON = dict(init="gauss", sigma_mode="mid", case_cycle=True, wall_frac=0.97, reserve_s=120.0, fallback=True,
                  safe=True, watchdog_s=60.0)
# ideogram4 / krea2: all training images + a step cap per image, and the exact average of as many full-cap members as
# the budget fits (12/12 pairs better; production form R1 -0.18%, b3d4e834 -0.26%, krea2 13e44b2d -0.29% vs the champion)
# Rank 64 at lr 2e-4 x sqrt(16/64) (R1: 3 x r64 -0.10% vs 3 x r16, production B3R64 -0.07%; krea2 B3R64 -0.33%; af4138aa one r64
# member = two r16 members; delta-scale sweeps x1.0 optimal; r128/r256 over-adapt mid/high sigma). SEEDS_FILL 0.75: a last member on a
# shortened schedule when a full one no longer fits (R1 B3F -0.05%; only >= 75% of the cap, the size that was measured).
_FD_ALLDATA = dict(_FD_COMMON, holdout_frac=0.0, holdout_min=0, seeds_auto=4, save_every=100, seeds_fill=0.75,
                   rank=64, alpha=64.0, lr=1e-4)
FAMILY_DEFAULTS = {
    # 1 Oct (T016 I4, H100, paired with our own d6cb290 config): FAST_LORA (1.59 -> 1.34 s/step, 1.18x) gives a 0.75 h
    # R1-size task (b9ac031e, 24 imgs x 35 = 840 steps) a 2nd full member where the old path shipped one: -0.171% / -0.138%
    # per seed (mean -0.154%, -0.11% vs the R1 winner); 7b5416a9 1.0 h (2nd member 0.82 cap -> full) -0.018%; ec70d081
    # 1408x768 40-step gate +0.010% at 1.25 -> 1.06 s/step; full members equal to the old path's (+0.064% / -0.013% /
    # +0.001%). FAST_MERGE stays off here (150-step evidence only, and it changes no member count on these tasks).
    # 3 Oct (WFX, H100): SEEDS_FILL 0.6 - the idle tail after the last full member is filled down to 60% of the cap (was 75%).
    # 4c5361b5 1.0 h (N=45: one full member of 1575 steps, then 1391 s idle) + one 1019-step fill member: -0.264% / -0.232%
    # (two fill seeds, vs the shipped member). Production-form validation T025 (riset/DECISIONS.md, 3 Oct). Only tasks whose
    # idle tail is 60-75% of a member change; every other task plans exactly as before.
    # 4 Oct (LFX, T029): the cosine ends at 45% of the peak LR instead of 10% (qwen's QLF45 lever). Paired with the candidate AND with
    # the 28 Sep entry config (d6cb290) on the same task / seed / machine: 7b5416a9 -0.114% / -0.069% (H100 PCIe, one member),
    # ec70d081 -0.007% / -0.012% (H100 PCIe, two members); production-form SXM 7b5416a9 + R1-size b9ac031e: riset/DECISIONS.md (4 Oct).
    # LR 1.5e-4 + 0.45 (LRMX) and SMN gains on top were worse (+0.007..+0.209%, +0.127%) and stay off.
    "ideogram4": dict(_FD_ALLDATA, steps_per_image=35.0, fast_lora=True, seeds_fill=0.6, lr_final_frac=0.45),
    # krea2 also adapts the embedding / text projection modules (godtrainer's set; 13e44b2d -0.13% alone, stacks with the
    # ensemble)
    # 30 Sep (T016 B1, H100, paired with our own d6cb290 config): FAST_LORA removes the 4:3 OOM -> gradient-checkpointing
    # path (2.44 -> 1.32 s/step, 77 -> 59 GiB), so 790304ac fits 3 members instead of 1 (-0.29% / -0.05% per seed; the
    # 3-member average is -0.21% / -0.20% vs its own members, FAST members = old-path members); dad11714 (2 members already)
    # no change (+0.05% / +0.02%). seeds_auto stays 4: a 5th member cut to 0.80 cap was the worst member and cost +0.05%.
    # 1 Oct (G1, 150 steps): FAST_MERGE builds the same W' (bit-identical per module) without Comfy's copies: 1.32 -> 1.16 s/step
    # (+14%), 0.125520 vs 0.125530.
    # 4 Oct (LR15, T030): peak LR 1.5e-4 (was 1e-4 = 2e-4 x sqrt(16/64)); cosine floor unchanged (0.1). Paired with the candidate AND with
    # the 28 Sep entry config on the same task / seed / machine: dad11714 -0.144% (A100), 790304ac +0.017% (H100 PCIe), 13e44b2d +0.061%
    # (A100), production-form SXM 790304ac in riset/DECISIONS.md (4 Oct; owner rule: mean over all tests < 0, every test <= the previous
    # entry). lr_final_frac 0.45 and a constant LR (1.0) lost on average (+0.051% / +0.147%) and stay off.
    "krea2": dict(_FD_ALLDATA, steps_per_image=37.0, fast_lora=True, fast_merge=True, lr=1.5e-4,
                  include_extra=r"^first$|^last\.linear$|^tmlp\.|^tproj\.|^txtfusion\.|^txtmlp\."),
    # holdout step selection for these three (z-image all-data lost: 8fd01555 +0.13%): every Linear at lr 3e-4 for
    # z-image, lr 3e-4 for flux, rank-8 attention-only at lr 1e-4 for qwen (qwen ae65e0ac -0.68% vs godtrainer's code).
    # z-image also alternates caption / empty per step and evaluates the holdout only every 400 steps: its low-sigma bands
    # need many optimizer steps (CZBE 244984ee -0.33% vs CZ1 -> par with the champion; 8fd01555 low sigma -0.5% vs champion)
    # Rank 64 at lr x sqrt(16/64): wider LoRAs keep improving the low-sigma bands (ideogram4 R1, krea2, z-image all show it;
    # CZBR64 244984ee -0.03% vs r16, low sigma -0.07..-0.19%, now level with the champion)
    # 27 Sep (H1, R1-size benchmark on 244984ee subsets, exact validator contract): all training images + a fixed step cap
    # per image + the exactly averaged ensemble beat the holdout recipe at N=9 (2 seeds: -0.29%, -0.68%) and N=18 (-0.25%);
    # the holdout optimum wandered 67-133 steps/image, the fixed cap is 100.
    # 3 Oct (V7C-1 + v40, paired): FAST_LORA + FAST_MERGE (0.40 -> 0.28 s/step) keep the 2nd ensemble member on slower nodes.
    # H100 PCIe (~1.46x slower than SXM, like slow validator nodes): 0bf3c1b9 FM K~2 vs old path K=1 -> -0.426% / -0.192%;
    # nominal SXM: no harm (2 seeds mean -0.002%, R1 a3ed678c +0.005%), 208/208 merges exact. Explicit env vars still win.
    # 3 Oct (ZSV5): save_every 500. The interim saves of member 1 only (~40 x 2.4 s at 100) made the planner price every later
    # member ~95 s too slow (max(dur) includes member 1), so 0bf3c1b9 1.0 h refused a 3rd member that fits (fill 0.743 < 0.75)
    # and idled 1004 s. At 500: members 1115 / 1097 s, 3rd member 3423 steps; shipped -0.084% vs the first two (same run),
    # -0.136% vs the 100 run; second seed in riset/DECISIONS.md. Training math unchanged; a crash loses <= 500 steps of one member.
    "z-image": dict(_FD_COMMON, include=".*", lr=1.5e-4, rank=64, alpha=64.0, prompt_mode="balanced",
                    holdout_frac=0.0, holdout_min=0, steps_per_image=100.0, seeds_auto=4, seeds_fill=0.75, save_every=500,
                    fast_lora=True, fast_merge=True),
    # 27 Sep: flux all-data + 55 steps/image (= the holdout optimum, 55-58/image) + ensemble: 9159e3dc 2 seeds -0.25% vs the
    # holdout recipe (flux noise +-0.04%), -0.39% vs the loader-fixed champion code; rank 16 kept (r64 tied)
    # 3 Oct (T016, H100 SXM + H100 PCIe worker): FAST_LORA + FAST_MERGE for flux (1.03 -> 0.70 s/step SXM, 1.48 -> 0.97 s/step PCIe;
    # 150-step parity +0.069%, 228/228 merges exact). Every paired comparison lower: PCIe K3 -> K4 (2 seeds), short budget K1 -> K2
    # (2 seeds), nominal SXM; numbers in riset/DECISIONS.md (3 Oct, owner rule: all pairs lower, none worse). Cap and fill unchanged.
    # 4 Oct (MOD, T031): the LoRA also adapts the modulation projections (double_blocks img_mod / txt_mod lin + single_blocks
    # modulation lin: 228 -> 304 modules). Paired with the candidate AND with the 28 Sep entry config on the same task / seed / machine:
    # 9159e3dc -0.036% / -0.083% (H100 PCIe, four members), 40018c85 (fluxunchained base) -0.109% / -0.131% (A100); SXM arm in
    # riset/DECISIONS.md (4 Oct). LR-floor / LR x1.5 doses lost on 40018c85 (+0.18..+0.61%) and stay off.
    "flux": dict(_FD_COMMON, lr=3e-4, holdout_frac=0.0, holdout_min=0, steps_per_image=55.0, seeds_auto=4, seeds_fill=0.75,
                 save_every=100, fast_lora=True, fast_merge=True,
                 include_extra=r"^double_blocks\.\d+\.(img_mod|txt_mod)\.lin$|^single_blocks\.\d+\.modulation\.lin$"),
    # 28 Sep: the full qwen module set (attention incl. the text-stream add projections + image/text MLPs) at rank 32,
    # lr 1e-4 x sqrt(8/32) = 5e-5: 3b6a0a8e 1.5 h -0.15% / -0.15% (seeds 0/1, noise +-0.03%), ae65e0ac 1.0 h -0.54% vs the
    # rank-8 attention-only recipe (3x slower per step, 76 GiB at 1024x544; an OOM switches on gradient checkpointing)
    # 1 Oct (T016 B2, H100, paired with the d6cb290 config): FAST_LORA (no OOM -> checkpointing, 2.23 -> 1.26 s/step, 78 -> 62 GiB)
    # stretches the holdout run to ~3200 steps: b1546f69 -0.13% / -0.33% per seed (text/no_text both better on seed 1).
    # 1 Oct (G1): + FAST_MERGE, 1.26 -> 1.07 s/step (+18%), 0.128099 vs 0.128056 (716 ok / 4 off: modules whose delta changed no
    # weight keep Comfy's merge).
    # 4 Oct (QLF45, T028): the all-data ensemble recipe of ideogram4/krea2 with a raised LR floor - all training images, 75 steps per
    # image, the exact average of as many full members as fit (seeds_auto 4), lr_final_frac 0.45 (cosine down to 45% of the peak
    # instead of 10%: the same steps travel ~1.3x further, and the ensemble average absorbs the extra end-of-run noise); save_every
    # 500 as z-image. Paired with the holdout recipe it replaces AND with the 28 Sep entry config (d6cb290) on the same task, seed and
    # machine: b1546f69 s0 -0.141% / -0.271%, s1 -0.230% / -0.560% (H100 SXM); 9-image pseudo-tasks q09b1546 -0.244% / -0.359%
    # (H100 PCIe), q09b3b6a -0.084% / -0.253% (A100); 3b6a0a8e (N=45, one member fits) and the slow-node floor check: DECISIONS 4 Oct.
    # SEEDS_FILL 0.5 (QLFV5F5): on 3b6a0a8e (N=45, 3375-step members) one member leaves 1937 s = 0.56 of a member idle; the fill
    # member now uses it (the planner change touches only tasks whose idle tail is 50-75% of a member; b1546f69 0.18 and the 9-image
    # pseudo-tasks plan exactly as before). Paired result in riset/DECISIONS.md (4 Oct).
    "qwen-image": dict(_FD_COMMON, rank=32, alpha=32.0, lr=5e-5, fast_lora=True, fast_merge=True,
                       holdout_frac=0.0, holdout_min=0, steps_per_image=75.0, seeds_auto=4, seeds_fill=0.5, save_every=500,
                       lr_final_frac=0.45,
                       include=r"^transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|"
                               r"to_add_out)|img_mlp\.net\.(0\.proj|2)|txt_mlp\.net\.(0\.proj|2))$"),
}


def apply_family_defaults(fam):
    """Fill CFG from FAMILY_DEFAULTS[fam] for keys without an explicit G56CT_<KEY> env var; returns the applied keys."""
    if os.environ.get("G56CT_FAMILY_DEFAULTS", "1") == "0":
        return {}
    applied = {}
    for k, v in FAMILY_DEFAULTS.get(fam, {}).items():
        if os.environ.get("G56CT_" + k.upper()) in (None, ""):
            CFG[k] = v
            applied[k] = v
    return applied


# Task-size conditional defaults (ours, 4 Oct; owner: "K3 qwen hanya untuk task besar, task kecil jangan"): FAMILY_DEFAULTS_BIG_N[fam] =
# (min_images, {key: value}) replaces family defaults once the dataset size is known, on tasks with >= min_images images. Keys with an
# explicit G56CT_<KEY> env var still win; G56CT_FAMILY_DEFAULTS=0 switches this off together with FAMILY_DEFAULTS.
# qwen-image K3D (shorter members: 50 steps/image at lr 7.5e-5, everything else QLF45-F5), paired with QLF45 on the same task / seed /
# machine: N=29 b1546f69 -0.083% (SXM s1) and -0.083% (H100 PCIe slow node), N=45 3b6a0a8e -0.072%; N=9 q09b1546 +0.22% -> small tasks keep
# QLF45 (75 steps/image, lr 5e-5). Threshold 20 sits between the measured sizes (riset/DECISIONS.md, 4 Oct).
FAMILY_DEFAULTS_BIG_N = {
    "qwen-image": (20, dict(steps_per_image=50.0, lr=7.5e-5)),
}


def apply_big_n_defaults(fam, n_images):
    """Apply FAMILY_DEFAULTS_BIG_N[fam] when n_images >= its threshold (keys without an explicit env var); returns the applied keys."""
    if os.environ.get("G56CT_FAMILY_DEFAULTS", "1") == "0":
        return {}
    spec = FAMILY_DEFAULTS_BIG_N.get(fam)
    if not spec or n_images < spec[0]:
        return {}
    applied = {}
    for k, v in spec[1].items():
        if os.environ.get("G56CT_" + k.upper()) in (None, ""):
            CFG[k] = v
            applied[k] = v
    return applied


SMALL_DEFAULT = {
    "ideogram4": r"^layers\.\d+\.((attention|ffn)_norm[12]\.weight|adaln_modulation\.bias)$|^final_layer\.(linear|adaln_modulation)\.bias$",
    "krea2": r"^blocks\.\d+\.((prenorm|postnorm)\.scale|mod\.lin)$|^last\.(norm\.scale|modulation\.lin|linear\.bias)$",
}

CFG = dict(
    rank=env("RANK", 16, int), alpha=env("ALPHA", 16.0, float), lr=env("LR", 2e-4, float),
    lr_final_frac=env("LR_FINAL_FRAC", 0.1, float), warmup=env("WARMUP", 20, int), wd=env("WD", 0.0, float),
    clip=env("CLIP", 1.0, float), ema=env("EMA", 0.99, float), seed=env("SEED", 0, int),
    prompt_mode=env("PROMPT_MODE", "paired"),          # paired: caption+empty on the same noisy sample each step | balanced | single
    empty_weight=env("EMPTY_WEIGHT", 0.5, float),      # paired loss weight of the empty-prompt half (evaluator: 0.5)
    empty_frac=env("EMPTY_FRAC", 0.3, float),          # single mode: probability of the empty prompt
    sigma_mode=env("SIGMA_MODE", "band"),              # band: (b+u)/16 stratified | mid: evaluator midpoints (b+0.5)/16
    bands=env("BANDS", 16, int),
    holdout_frac=env("HOLDOUT_FRAC", 0.1, float), holdout_min=env("HOLDOUT_MIN", 3, int),
    holdout_noises=env("HOLDOUT_NOISES", 1, int), eval_batch=env("EVAL_BATCH", 2, int),
    holdout_strata=env("HOLDOUT_STRATA", 16, int),     # evaluator uses 16; lower only for CPU smoke tests
    eval_every=env("EVAL_EVERY", 50, int), eval_first=env("EVAL_FIRST", 50, int),
    eval_every_late=env("EVAL_EVERY_LATE", 100, int), eval_late_from=env("EVAL_LATE_FROM", 400, int),
    wall_frac=env("WALL_FRAC", 0.90, float), reserve_s=env("RESERVE_S", 150.0, float),
    grad_ckpt=env("GRAD_CKPT", "auto"),                # auto | on | off
    fast_lora=env("FAST_LORA", False, bool),           # ours: factored LoRA backward for the DiT LoRA Linears, the merged
                                                       # weight W' is not kept for backward (fast_lora_install); 0 = old path
    fast_merge=env("FAST_MERGE", False, bool),         # ours (T016/speed): with FAST_LORA, W' = W + s U D built without a copy
                                                       # of W and with one rounding into a compute-dtype buffer (_fm_merge),
                                                       # bit-identical to Comfy's merge; per module verified against it once
                                                       # after the first optimizer step (_fm_verify); 0 = Comfy's merge
    base=env("BASE", "evalq"),                         # ideogram4: evalq (eval-like fp8 grid) | fine (dequantized bf16)
    te_source=env("TE", "rebuild"),                    # rebuild (byte-exact eval TE) | bf16 (snapshot as-is)
    max_steps=env("MAX_STEPS", 0, int), save_fp32=env("SAVE_FP32", True, bool),
    steps_per_image=env("STEPS_PER_IMAGE", 0.0, float),  # ours: all-data step cap = this x training images (unless MAX_STEPS)
    # crash safety (lesson from the env-tournament winner): the validator kills the container at hours_to_complete and
    # uploads whatever the output dir holds; all-data runs used to write only at the end of a run
    save_every=env("SAVE_EVERY", 0, int),              # all-data: also save the current EMA state every N steps
    fallback=env("FALLBACK", False, bool),             # write an identity (zero-delta, loadable) file before training
    stop_after_bad=env("STOP_AFTER_BAD", 0, int),       # >0: stop when this many evals in a row do not improve
    exact_holdout=env("EXACT_HOLDOUT", "auto"),        # auto (fp8 families) | on | off: holdout on evaluator-merged weights
    seeds=env("SEEDS", 1, int),                        # >1: independent runs sharing the budget + exact delta averaging
    folds=env("FOLDS", False, bool),                   # with SEEDS>1: run k holds out fold k (all data used across runs)
    run_stop_bad=env("RUN_STOP_BAD", 3, int),          # multi-seed: end a run after this many non-improving evals
    init=env("INIT", "comfy"),                         # comfy: LoRAAdapter.create_train | gauss: down~N(0,1/in), up=0
    eval_tags=tuple(t for t in env("EVAL_TAGS", "raw,ema").split(",") if t in ("raw", "ema")) or ("raw", "ema"),
    include=env("INCLUDE", "").replace("{OR}", "|"),   # override the family's LoRA module regex (e.g. qwen attention-only);
                                                       # "{OR}" = regex '|' (queue files use '|' as their field separator)
    include_extra=env("INCLUDE_EXTRA", "").replace("{OR}", "|"),  # ours: ADD modules to the family set (e.g. final/input
                                                       # projections that shape per-patch detail), keeping the default set
    beta2=env("BETA2", 0.999, float),                  # AdamW beta2 (the R1 leader uses 0.99)
    edge_bands=env("EDGE_BANDS", 0, int),              # oversample this many lowest+highest sigma bands (leader: krea2 2)
    edge_repeat=env("EDGE_REPEAT", 1, int),            # ... this many times (leader: 3)
    low_bands=env("LOW_BANDS", 0, int),                # ours: oversample ONLY the lowest sigma bands (36-58% of the test loss
    low_repeat=env("LOW_REPEAT", 1, int),              # sits at sigma <= 0.22 and improves least) ... this many times
    # ours: latent crop augmentation for the low/mid sigma bands only. A crop of the latent keeps the generator's fine-detail
    # statistics exactly (no resampling) while breaking whole-image memorization; high sigma (layout) keeps full frames.
    crop_frac=env("CROP_FRAC", 0.0, float),            # probability that an eligible step trains on a random crop
    crop_min=env("CROP_MIN", 0.6, float),              # crop side scale ~ U(crop_min, 1)
    crop_maxsig=env("CROP_MAXSIG", 0.5, float),        # only steps with sigma <= this
    crop_prompt=env("CROP_PROMPT", "empty"),           # empty: crop steps use the empty prompt only (the caption describes
                                                       # the full frame) | paired: keep the normal prompt plan
    antithetic=env("ANTITHETIC", False, bool),         # ours (survey E4): with CASE_CYCLE, every 2nd visit of an (image, band)
                                                       # pair reuses the previous visit's noise with the opposite sign
    lora_plus=env("LORA_PLUS", 1.0, float),            # LoRA+ (survey E5): lr of the LoRA up (B) matrices = this x LR
    lora_down_mult=env("LORA_DOWN_MULT", 1.0, float),  # ours (E0: lora_down never leaves its random init subspace): lr of
                                                       # the LoRA down (A) matrices = this x LR
    late_frac=env("LATE_FRAC", 0.0, float),           # ours: last fraction of the run (steps; time in holdout mode) trains
    late_bands=env("LATE_BANDS", 16, int),             # only sigma bands < LATE_BANDS (low sigma keeps learning after the
                                                       # mid/high bands reached their optimum; needs CASE_CYCLE)
    accum=env("ACCUM", 1, int),                        # ours: cases (image, band) per optimizer step (gradient accumulation;
                                                       # paired/balanced per case, mean over the step's cases)
    accum_band=env("ACCUM_BAND", False, bool),         # ours: the cases of one step = different images at ONE sigma band
    agree=env("AGREE", 0.0, float),                    # ours: AND-mask (Parascandolo et al. 2021) over the step's cases: an
                                                       # element of the summed gradient survives only if |mean sign| >= this
    band_min=env("BAND_MIN", 0, int),                  # research: train only sigma bands band_min <= b < band_max
    band_max=env("BAND_MAX", 0, int),                  # (0 = all bands; diagnostics of what a LoRA can learn per band)
    anchor=env("ANCHOR", 0.0, float),                  # per-step pull back toward the run's starting delta:
                                                       # p += anchor*(p0 - p); 0.005 = half-life ~140 steps. Ours.
    anchor_every=env("ANCHOR_EVERY", 10, int),         # apply the anchor pull every N steps (host-RAM copy)
    delta_avg=env("DELTA_AVG", 0, int),                # >1: average the last N evaluated states at the end of a run
    scale_groups=env("SCALE_GROUPS", ""),              # e.g. "adaln_modulation,attention,feed_forward": greedy per-group
                                                       # strength search after the global one (ours)
    save_runs=env("SAVE_RUNS", False, bool),           # research: all-data ensemble also writes run<k>.safetensors (members)
    seeds_fill=env("SEEDS_FILL", 0.0, float),           # ours: when the next full-cap member no longer fits, a LAST member on a
                                                       # shortened step schedule uses the remaining time if it gets >= this
                                                       # fraction of the full cap (0 = off; B3 R1 left 29% of the budget unused)
    seeds_auto=env("SEEDS_AUTO", 0, int),              # ours (survey E7): all-data mode keeps adding full-cap members (exact
                                                       # delta average) while the remaining time fits one more; <= this many
    final_scale=env("FINAL_SCALE", 1.0, float),        # ours (survey E1): scale of the shipped delta (all adapter keys)
    # ours (LoRA-One-lite; riset/LITERATUR-OPTIMIZER-LORA.md E3): seed deltas differ mainly by the RANDOM input subspace of
    # lora_down (overlap across seeds ~ random, down barely moves). SPEC_INIT: down <- top right singular directions of the
    # mean full-weight gradient at the base (two-pass randomized range finder with the LoRA modules), up = 0 -> delta 0.
    spec_init=env("SPEC_INIT", False, bool),
    spec_blocks=env("SPEC_BLOCKS", False, bool),       # ensemble member k uses block k of the ranked directions (diverse)
    spec_probes=env("SPEC_PROBES", 2, int),            # probe blocks of r directions (oversampling; >= members if blocks)
    pair_anti=env("PAIR_ANTI", False, bool),           # ours (survey X3): paired prompts, the empty half uses -eps (antithetic)
    ema_debias=env("EMA_DEBIAS", False, bool),         # ours (paper notes, EDM2/Adam-style): the EMA starts at the init (up=0),
                                                       # so after T steps it keeps weight beta^T on it -> a (1-beta^T)^2 shrink
                                                       # of the shipped delta (0.90 at 300 steps). Debias: (ema - beta^T p0)/(1-beta^T)
    ema_extra=tuple(float(x) for x in env("EMA_EXTRA", "").split(",") if x.strip()),  # research (survey E2-lite): extra
                                                       # EMA decays tracked beside EMA; all-data runs write ema<d>.safetensors
                                                       # (d=0: the raw final weights) for post-hoc EMA calibration
    research_dir=env("RESEARCH_DIR", ""),              # side outputs (SAVE_RUNS members, EMA_EXTRA) go here, not to out_dir
    rebalance=env("REBALANCE", 0.0, float),            # ours: gain for shifting the paired empty-prompt weight toward the
                                                       # weaker half after each holdout eval (0 = off; 10 = +0.06 at a 0.6% gap)
    case_cycle=env("CASE_CYCLE", False, bool),         # leader skill: walk every (image, band) pair before repeating
    direct=env("DIRECT", "").replace("{OR}", "|"),                          # ours: regex of modules trained as FULL-rank deltas (exported as .diff)
    direct_max_m=env("DIRECT_MAX_M", 8.0, float),      # ... only when the weight has at most this many million params
    # ours, DiffFit/BitFit-style: full deltas on small non-matrix params (biases, norm gains, modulation offsets), e.g.
    # ideogram4 "^layers\.\d+\.((attention|ffn)_norm[12]\.weight|adaln_modulation\.bias)$"; "auto" = family default below
    small=env("SMALL", "").replace("{OR}", "|"),
    small_lr=env("SMALL_LR", 0.0, float),              # 0 -> = LR
    small_max_k=env("SMALL_MAX_K", 256.0, float),      # only params with at most this many thousand elements
    scale_search=env("SCALE_SEARCH", ""),              # e.g. "0.85,0.925,1.075,1.15": rescale the winning delta and
                                                       # keep the best on the holdout (evaluator numerics). Ours.
    # ours: text-encoder LoRA (ideogram4, krea2). The evaluator applies LoraLoader(model, clip, 1.0, 1.0) and re-encodes the
    # caption AND "" with the patched TE, so TE adapters in the same file are scored. Off = trainer unchanged.
    te_lora=env("TE_LORA", False, bool),
    te_rank=env("TE_RANK", 8, int), te_alpha=env("TE_ALPHA", 0.0, float),    # alpha 0 -> = rank (scale 1)
    te_lr=env("TE_LR", 0.0, float),                    # 0 -> 0.5 x LR (kohya convention TE lr = unet lr / 2; a TE delta
                                                       # shifts the conditioning of every DiT block and of the empty prompt)
    te_include=env("TE_INCLUDE", r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$").replace("{OR}", "|"),
    te_layers=env("TE_LAYERS", ""),                    # "a-b": only TE layers a..b (default: all layers feeding a tap)
    te_grad_ckpt=env("TE_GRAD_CKPT", True, bool),      # checkpoint every TE block (fp32 activations; ~2 GB instead of ~30)
    te_init=env("TE_INIT", "gauss"),                   # gauss (PEFT-like: down~N(0,1/in), up=0 -> starts at the base TE) | comfy
    te_drop_check=env("TE_DROP_CHECK", True, bool),    # final pick also scored without the TE keys; keep the better one
    # ours (30 Sep, T016/speed): never ship a non-finite LoRA, less writing at the end of an ensemble. SAFE=1: a training
    # step whose loss or gradient norm is not finite is skipped (no optimizer / EMA update), more than SAFE_NF_MAX such
    # steps in a row abandon the member; a non-finite holdout score never becomes the best; save_lora refuses a
    # non-finite state (the file on disk stays); ensemble members leave the running average on disk and the final
    # average reuses the last running one; all-data members stop early enough for the measured save / average time;
    # a malformed or implausible G56CT_START_TS falls back to the trainer start. 0 = the trainer unchanged.
    safe=env("SAFE", False, bool),
    safe_nf_max=env("SAFE_NF_MAX", 20, int),
    # ours (T016/speed): > 0 -> exit 0 this many seconds before the validator's kill (t_start + hours x 3600) when main()
    # is still running and the upload folder holds a usable adapter (_start_watchdog); 0 = off
    watchdog_s=env("WATCHDOG_S", 0.0, float),
)
if CFG["te_lora"]:
    CFG["te_alpha"] = CFG["te_alpha"] if CFG["te_alpha"] > 0 else float(CFG["te_rank"])
    CFG["te_lr"] = CFG["te_lr"] if CFG["te_lr"] > 0 else 0.5 * CFG["lr"]


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-zip", required=True)
    p.add_argument("--model-type", required=True, choices=sorted(FAMILIES))
    p.add_argument("--expected-repo-name", required=True)
    p.add_argument("--hours-to-complete", required=True, type=float)
    p.add_argument("--trigger-word", default=None)
    p.add_argument("--out-dir", default=None, help="override output dir (tests)")
    p.add_argument("--dataset", default=None, help="local zip/dir override (tests)")
    return p.parse_args()


_MODEL_DIRS = {}


def model_dir(repo):
    """/cache/models/<org--repo>; when that folder is absent and the cache holds exactly ONE model folder (an anonymized
    cache), that folder - like the champion's trainer. With several unknown folders no guess: the named path (loaders fail
    loudly as before)."""
    if repo in _MODEL_DIRS:
        return _MODEL_DIRS[repo]
    root = CACHE / "models"
    d = root / repo.replace("/", "--")
    if not d.is_dir() and root.is_dir():
        dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        if len(dirs) == 1:
            log(f"WARN {d.name} not in {root}; using the only model folder {dirs[0].name}")
            d = dirs[0]
    _MODEL_DIRS[repo] = d
    return d


ROOT_MIN_BYTES = 5 * 1024 ** 3  # evaluator prepare_base: root-level checkpoint files larger than 5 GiB


def root_checkpoint(repo):
    """prepare_base(): the task repo's single root-level >5GB .safetensors (evaluator contract for non-ideogram4)."""
    d = model_dir(repo)
    cands = [f for f in d.glob("*.safetensors") if f.is_file() and f.stat().st_size > ROOT_MIN_BYTES] if d.is_dir() else []
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:  # e.g. PixelWave ships bf16 + fp8 + nf4; the validator still scored such tasks -> prefer bf16/largest
        pick = sorted(cands, key=lambda c: ("bf16" not in c.name.lower(), -c.stat().st_size))[0]
        log(f"WARN {len(cands)} root checkpoints in {d}; using {pick.name}")
        return pick
    raise FileNotFoundError(f"no root-level checkpoint in {d}")


def dataset_zip_path(args):
    if args.dataset:
        return Path(args.dataset)
    z = CACHE / "datasets" / f"{args.task_id}_tourn.zip"
    if z.exists():
        return z
    if args.dataset_zip and Path(args.dataset_zip).exists():
        return Path(args.dataset_zip)
    raise FileNotFoundError(f"dataset zip not found ({z})")


def read_rows(src):
    """Training rows with the evaluator's reader semantics (sorted names, caption = sibling .txt, raw text). Only when the
    evaluator's reader rejects the training zip outright (non-UTF-8 caption bytes, e.g. macOS __MACOSX/._* AppleDouble
    entries, or duplicate archive entries) the same entries are read tolerantly instead of forfeiting the task (such a zip
    could never be a scored test set: the evaluator itself would fail on it)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evalcore import read_dataset
    try:
        rows = read_dataset(str(src), Path("/tmp/_g56ct_ds.zip"))
    except (UnicodeDecodeError, ValueError) as e:
        if not isinstance(e, UnicodeDecodeError) and "Duplicate archive entries" not in str(e):
            raise  # (UnicodeDecodeError is a ValueError subclass: test it first) e.g. "Empty image dataset" stays fatal
        log(f"WARN evaluator reader rejected the training zip ({type(e).__name__}: {str(e)[:80]}); tolerant read")
        rows = read_rows_tolerant(src)
    good = [(n, raw, cap) for n, raw, cap in rows if cap and cap.strip()]
    if len(good) < len(rows):
        log(f"WARN {len(rows) - len(good)} rows without caption dropped")
    return good



def read_rows_tolerant(src):
    """read_dataset for a zip the evaluator's reader rejects: __MACOSX/ and ._* entries skipped, the first copy of a
    duplicated name kept, captions decoded with replacement characters; otherwise the same sorted-name / sibling-.txt rules."""
    from evalcore import EXTENSIONS
    rows, seen = [], set()
    with zipfile.ZipFile(src) as archive:
        names = archive.namelist()
        present = set(names)
        for name in sorted(names):
            leaf = name.rsplit("/", 1)[-1]
            if name.startswith("__MACOSX/") or "/__MACOSX/" in name or leaf.startswith("._") or name in seen:
                continue
            seen.add(name)
            if name.lower().endswith(EXTENSIONS):
                caption = str(Path(name).with_suffix(".txt"))
                text = archive.read(caption).decode("utf-8", errors="replace") if caption in present else None
                rows.append((name, archive.read(name), text))
    if not rows:
        raise ValueError("Empty image dataset")
    return rows


def decode_items(rows):
    """Training images exactly as the evaluator prepares them (adjust_image); an undecodable image is skipped with a WARN
    instead of killing the run (at least one image must remain)."""
    import io
    from PIL import Image
    from evalcore import adjust_image
    items = []
    for name, raw, cap in rows:
        try:
            img = adjust_image(Image.open(io.BytesIO(raw)))
        except Exception as e:  # noqa: BLE001
            log(f"WARN undecodable training image {name} skipped ({type(e).__name__}: {str(e)[:80]})")
            continue
        sha = hashlib.sha256(img.tobytes() + str(img.size).encode()).hexdigest()
        items.append({"name": name, "image": img, "caption": cap, "sha": sha})
    if not items:
        raise ValueError("no decodable training image")
    return items

def safetensors_header(path):
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


# ----------------------------------------------------------------------------------------------------------------------
# Text encoder: rebuild the evaluator's exact fp8_scaled TE from the bf16 snapshot in /cache + baked scale table
# ----------------------------------------------------------------------------------------------------------------------
def load_snapshot_tensors(snap_dir):
    import torch
    from safetensors import safe_open
    snap_dir = Path(snap_dir)
    idx = snap_dir / "model.safetensors.index.json"
    if idx.exists():
        wm = json.load(open(idx))["weight_map"]
        files = sorted(set(wm.values()))
    else:
        files = sorted(p.name for p in snap_dir.glob("*.safetensors"))
    out = {}
    for fn in files:
        with safe_open(str(snap_dir / fn), framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def hf_to_comfy_key(k, mode):
    if mode == "qwen3vl":
        for a, b in (("model.language_model.layers.", "model.layers."), ("model.language_model.embed_tokens.", "model.embed_tokens."),
                     ("model.language_model.norm.", "model.norm.")):
            if k.startswith(a):
                return b + k[len(a):]
    return k


def resolve_snapshot(rel):
    """/cache/hf_cache/<Org--Name> may be a plain dir or an HF cache tree (snapshots/<rev>/...)."""
    base = CACHE / rel
    if (base / "config.json").exists() or list(base.glob("*.safetensors")):
        return base
    snaps = sorted((base / "snapshots").glob("*")) if (base / "snapshots").is_dir() else []
    for s in snaps:
        if (s / "config.json").exists():
            return s
    hub = CACHE / "hf_cache" / ("models--" + rel.split("/", 1)[1])
    if (hub / "snapshots").is_dir():
        for s in sorted((hub / "snapshots").glob("*")):
            if (s / "config.json").exists():
                return s
    raise FileNotFoundError(f"TE snapshot not found: {base}")


def build_eval_te_state_dict(fam, device):
    """Returns (state_dict, report). With te_source=rebuild: fp8 layers quantized exactly like Comfy-Org's file
    (q = RTNE(W/scale), scale from the baked table), plus weight_scale + comfy_quant entries; bf16 tensors unchanged."""
    import torch
    spec = FAMILIES[fam]
    snap = resolve_snapshot(spec["te_snapshot"])
    t0 = time.time()
    raw = load_snapshot_tensors(snap)
    table = json.load(open(ASSETS / spec["te_table"]))
    sd, rep = {}, {"snapshot": str(snap), "fp8_layers": 0, "fingerprint_ok": None}
    for k, v in raw.items():
        sd[hf_to_comfy_key(k, spec["te_map"])] = v
    del raw
    want = set(table["keys"].keys())
    have_w = {k for k in sd}
    need_w = {k for k in want if not k.endswith((".weight_scale", ".comfy_quant"))}
    missing = sorted(need_w - have_w)
    extra = sorted(have_w - need_w)
    rep["missing"], rep["extra"] = missing[:5], extra[:5]
    if missing:
        log(f"WARN TE snapshot lacks {len(missing)} eval keys, e.g. {missing[:3]}")
    for k in extra:  # keys the eval file does not have (e.g. lm_head on tied models) -> drop to match eval exactly
        sd.pop(k, None)
    fp_ok = []
    for k, h in table.get("fingerprints_bf16_sha256", {}).items():
        if k in sd:
            fp_ok.append(hashlib.sha256(sd[k].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == h)
    rep["fingerprint_ok"] = all(fp_ok) if fp_ok else None
    if CFG["te_source"] == "rebuild":
        rep["fp8_layers"] = quantize_like_table(sd, table, device)
        bad = [k for k, (dt, shp) in table["keys"].items() if k in sd and list(sd[k].shape) != list(shp)]
        rep["shape_mismatch"] = bad[:5]
    rep["seconds"] = round(time.time() - t0, 1)
    return sd, rep


def quantize_like_table(sd, table, device):
    """In place: every layer listed in the public table becomes fp8 exactly like the evaluator's file
    (q = RTNE(fp32(W) / scale), clamp 448) + weight_scale + comfy_quant (new format; old-format tables use .scale_weight)."""
    import torch
    n = 0
    for sk, scale in table["weight_scale"].items():
        layer = sk.rsplit(".", 1)[0]
        wk = layer + ".weight"
        if wk not in sd:
            continue
        w = sd[wk].to(device=device, dtype=torch.float32)
        q = (w / torch.tensor(scale, dtype=torch.float32, device=device)).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
        sd[wk] = q.cpu()
        sd[layer + ".weight_scale"] = torch.tensor(scale, dtype=torch.float32)
        conf = table.get("comfy_quant", {}).get(layer, '{"format": "float8_e4m3fn"}')
        sd[layer + ".comfy_quant"] = torch.tensor(list(conf.encode("utf-8")), dtype=torch.uint8)
        n += 1
        del w, q
    return n


def find_repo_subdir(repo, sub):
    d = model_dir(repo) / sub
    if d.is_dir():
        return d
    cands = sorted(p for p in (CACHE / "models").glob(f"*/{sub}") if p.is_dir())
    cands.sort(key=lambda p: ("flux" not in p.parent.name.lower(), p.parent.name))
    if cands:
        log(f"WARN {sub} not in {model_dir(repo)}; using {cands[0]}")
        return cands[0]
    raise FileNotFoundError(f"{sub} not found for {repo}")


def load_clip(fam, device, repo=None):
    import comfy.sd
    import torch
    spec = FAMILIES[fam]
    tf = [os.environ.get(f"G56CT_TE_FILE{i}", f) for i, f in enumerate(spec.get("te_files") or ())]
    if tf and all(Path(f).exists() for f in tf):  # evaluator's exact TE files (DualCLIPLoader path)
        rep = {"files": tf}
        log(f"TE {fam}: {rep}")
        clip = comfy.sd.load_clip(ckpt_paths=tf, embedding_directory=None,
                                  clip_type=getattr(comfy.sd.CLIPType, spec["clip_type"]), model_options={})
        return clip, rep
    if spec.get("te_dual"):
        sds, rep = [], {"sources": []}
        for sub in spec["te_dual"]:
            d = find_repo_subdir(repo, sub)
            sd = load_snapshot_tensors(d)
            if spec.get("te_cast") == "fp16":
                sd = {k: (v.to(torch.float16) if v.is_floating_point() else v) for k, v in sd.items()}
            sds.append(sd)
            rep["sources"].append(str(d))
        log(f"TE {fam}: {rep}")
        clip = comfy.sd.load_text_encoder_state_dicts(sds, embedding_directory=None,
                                                      clip_type=getattr(comfy.sd.CLIPType, spec["clip_type"]), model_options={})
        return clip, rep
    if spec.get("te_repo_subdir"):
        d = find_repo_subdir(repo, spec["te_repo_subdir"])
        sd = load_snapshot_tensors(d)
        rep = {"source": str(d), "tensors": len(sd)}
        if spec.get("te_table") and CFG["te_source"] == "rebuild":
            rep["fp8_layers"] = quantize_like_table(sd, json.load(open(ASSETS / spec["te_table"])), device)
    elif spec["te_snapshot"] is not None:
        sd, rep = build_eval_te_state_dict(fam, device)
    else:
        raise NotImplementedError(f"TE loader for {fam} not implemented in comfytrain yet")
    log(f"TE {fam}: {rep}")
    clip = comfy.sd.load_text_encoder_state_dicts([sd], embedding_directory=None,
                                                  clip_type=getattr(comfy.sd.CLIPType, spec["clip_type"]), model_options={})
    return clip, rep


# ----------------------------------------------------------------------------------------------------------------------
# Diffusion model
# ----------------------------------------------------------------------------------------------------------------------
def ideogram4_state_dict(repo, mode, device):
    """Cache transformer (per-channel fp8) -> Comfy key/dtype layout of the evaluator's Comfy-Org file.
    mode=evalq: every fp8 layer re-cast onto the evaluator's scale-1.0 fp8 grid (99.1% byte-identical, rest 1 ULP);
    mode=fine: dequantized bf16 (precision of the cache file; no fp8)."""
    import torch
    from safetensors import safe_open
    d = model_dir(repo) / "transformer"
    idx = d / "diffusion_pytorch_model.safetensors.index.json"
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if idx.exists() else ["diffusion_pytorch_model.safetensors"]
    table = json.load(open(ASSETS / "dm_ideogram4_fp8_scaled.json"))
    raw = {}
    for fn in files:
        with safe_open(str(d / fn), framework="pt", device="cpu") as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k)
    scales = {k: raw.pop(k) for k in [k for k in raw if k.endswith(".weight_scale")]}
    sd = {}
    n8 = 0
    for k in list(raw.keys()):
        v = raw.pop(k)
        sk = k[: -len(".weight")] + ".weight_scale" if k.endswith(".weight") else None
        if sk and sk in scales and v.dtype == torch.float8_e4m3fn:
            s = scales[sk].to(device=device, dtype=torch.float32)
            w = v.to(device=device).float() * (s.reshape(-1, 1) if s.numel() == v.shape[0] else s)
            layer = k[: -len(".weight")]
            if mode == "evalq":
                sd[k] = w.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).cpu()
                sd[sk] = torch.tensor(1.0, dtype=torch.float32)
                conf = table["comfy_quant"].get(layer, '{"format": "float8_e4m3fn"}')
                sd[layer + ".comfy_quant"] = torch.tensor(list(conf.encode("utf-8")), dtype=torch.uint8)
            else:
                sd[k] = w.to(torch.bfloat16).cpu()
            n8 += 1
            del w, v
        else:
            sd[k] = v
    del raw
    if mode == "evalq":
        want = set(table["keys"])
        miss, extra = sorted(want - set(sd)), sorted(set(sd) - want)
        if miss or extra:
            log(f"WARN ideogram4 key audit: missing {miss[:4]} extra {extra[:4]}")
    log(f"ideogram4 base from cache ({mode}): {n8} fp8 layers converted, {len(sd)} tensors")
    return sd, {"model_type": "ideogram4_cond"}


def load_base_model(fam, repo, device):
    import comfy.sd
    import torch
    if fam == "ideogram4":
        sd, meta = ideogram4_state_dict(repo, CFG["base"], device)
        mp = comfy.sd.load_diffusion_model_state_dict(sd, model_options={}, metadata=meta)
        src = f"cache:{repo} ({CFG['base']})"
    elif fam == "krea2":
        ck = _krea2_checkpoint(repo)
        mp = comfy.sd.load_diffusion_model(str(ck), model_options={})
        src = str(ck)
    else:
        ck = root_checkpoint(repo)
        mp = comfy.sd.load_diffusion_model(str(ck), model_options={})
        src = str(ck)
    if mp is None:
        raise RuntimeError(f"Comfy could not detect the {fam} model")
    spec = FAMILIES[fam]
    if spec["shift"] is not None:
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
        mp = ModelSamplingAuraFlow().patch_aura(mp, spec["shift"])[0]
    import comfy.model_sampling
    sampling = mp.model.model_sampling
    if not isinstance(sampling, comfy.model_sampling.CONST) or getattr(sampling, "noise_scale", 1.0) != 1.0:
        raise ValueError("model is not unit-noise rectified flow (evaluator contract)")
    log(f"base model loaded: {src}  dtype={mp.model.get_dtype()} manual_cast={getattr(mp.model, 'manual_cast_dtype', None)}")
    return mp


def _krea2_checkpoint(repo):
    """krea2 base = the evaluator's prepare_base contract: the task repo's own root-level checkpoint (Krea-2-Raw: raw.safetensors).
    A raw.safetensors in another cache folder only when the task folder has no root checkpoint at all - never before the task's
    own file (2 Oct: a krea2 task on another base, e.g. Krea-2-Turbo, with Krea-2-Raw still cached trained the wrong base)."""
    d = model_dir(repo)
    if (d / "raw.safetensors").exists():
        return d / "raw.safetensors"
    try:
        return root_checkpoint(repo)
    except FileNotFoundError:
        pass
    for c in sorted((CACHE / "models").glob("*")):
        if (c / "raw.safetensors").exists() and (c / "text_encoder").is_dir():
            log(f"WARN krea2: no root-level checkpoint in {d}; using {c / 'raw.safetensors'} from another cache folder")
            return c / "raw.safetensors"
    return root_checkpoint(repo)


def load_vae(fam, repo):
    """Comfy VAE exactly as ImageFlowAdapter.load_vae (fp32, posterior mean); z-image uses diffusers VAE like the evaluator."""
    import torch
    import comfy.sd
    import comfy.utils
    spec = FAMILIES[fam]
    if fam == "z-image":
        from diffusers import AutoencoderKL
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        vae = AutoencoderKL.from_pretrained(str(model_dir(repo)), subfolder="vae", torch_dtype=torch.float32)
        return ("diffusers", vae.eval().requires_grad_(False))
    path = BAKED / spec["vae_file"]
    if not path.exists():
        raise FileNotFoundError(f"baked VAE missing: {path}")
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(path)), dtype=torch.float32)
    vae.first_stage_model.eval().requires_grad_(False)
    reg = getattr(vae.first_stage_model, "regularization", None)
    if reg is not None and getattr(reg, "sample", False):
        raise ValueError("VAE would sample the posterior")
    return ("comfy", vae)


def encode_latent(vae_pair, image, device):
    import numpy as np
    import torch
    kind, vae = vae_pair
    if kind == "comfy":
        pixels = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)
        with torch.inference_mode():
            lat = vae.encode(pixels).float().cpu()
    else:
        arr = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 127.5 - 1.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode():
            vae.to(device=device, dtype=torch.float32)
            lat = vae.encode(t).latent_dist.mode().float().cpu()
    if not torch.isfinite(lat).all():
        raise ValueError("non-finite latent")
    return lat


# ----------------------------------------------------------------------------------------------------------------------
# LoRA adapters (ComfyUI's own LoraDiff via weight wrappers = the path ComfyUI's TrainLoraNode uses)
# ----------------------------------------------------------------------------------------------------------------------
class DirectDelta(torch.nn.Module if False else object):
    """Full-rank trainable delta for a single Linear weight, exported as ComfyUI's `<key>.diff` patch (ours).
    Used for the small sigma-conditioning layers (krea2 tmlp, ideogram4 adaLN) where rank 16 is the binding limit."""

    def __init__(self, weight):
        import torch
        self.delta = torch.nn.Parameter(torch.zeros(weight.shape, dtype=torch.float32, device=weight.device))

    def __call__(self, w):
        return (w.float() + self.delta.to(w.device)).to(w.dtype)

    def to(self, *a, **k):
        return self

    def move_to(self, device):  # ModelPatcher.load -> move_weight_functions (the model sits on CPU when adapters attach)
        self.delta.data = self.delta.data.to(device)  # same Parameter object: optimizer / EMA ids stay valid
        return self.delta.numel() * self.delta.element_size()


class SmallDelta(object):
    """Ours, DiffFit/BitFit-style: a full trainable delta on one small non-matrix parameter (a Linear bias, a norm gain,
    a per-block modulation offset). Comfy-op weights/biases go through the module's weight/bias_function (like the LoRA
    wrappers); plain nn.Parameters (krea2 RMSNorm.scale, *.mod.lin / *.modulation.lin) are swapped for base+delta around
    the module's forward (re-run by checkpoint recomputation). Exported as ComfyUI diff patches (comfy.lora.load_lora):
    '<module>.diff_b' for a bias, '<module>.diff' for a comfy-op 1-D weight, '<module>.<param>.diff' otherwise."""

    def __init__(self, module, pname, full_mod, via_fn):
        import torch
        p = getattr(module, pname)
        self.module, self.pname, self.full_mod, self.via_fn = module, pname, full_mod, via_fn
        self.delta = torch.nn.Parameter(torch.zeros(p.shape, dtype=torch.float32, device=p.device))
        self.enabled = True
        self._orig = None
        if not via_fn:
            module.register_forward_pre_hook(self._pre)
            module.register_forward_hook(self._post)

    def __call__(self, w):  # comfy-op weight/bias function
        return (w.float() + self.delta.to(w.device)).to(w.dtype)

    def _pre(self, module, args):
        if not self.enabled:
            return
        orig = module._parameters[self.pname]
        self._orig = orig
        module._parameters[self.pname] = orig.float() + self.delta.to(orig.device)  # fp32, graph into the delta

    def _post(self, module, args, out):
        if self._orig is not None:
            module._parameters[self.pname] = self._orig
            self._orig = None

    def to(self, *a, **k):
        return self

    def move_to(self, device):
        self.delta.data = self.delta.data.to(device)
        return self.delta.numel() * self.delta.element_size()

    @property
    def key(self):
        if self.pname == "bias":
            return f"{self.full_mod}.diff_b"
        if self.via_fn:
            return f"{self.full_mod}.diff"
        return f"{self.full_mod}.{self.pname}.diff"


def attach_small(mp, regex, max_k):
    """SmallDelta on every diffusion-model parameter whose name (without 'diffusion_model.') matches `regex` (re.match),
    that is not a matrix weight (LoRA / DIRECT territory) and has at most max_k thousand elements."""
    rx = re.compile(regex)
    out, params = {}, []
    for full, m in mp.model.named_modules():
        if not full.startswith("diffusion_model."):
            continue
        for pname, p in list(m.named_parameters(recurse=False)):
            rel = f"{full[len('diffusion_model.'):]}.{pname}"
            if not rx.match(rel) or p.numel() > max_k * 1000 or (pname == "weight" and p.dim() >= 2):
                continue
            via_fn = pname in ("weight", "bias") and hasattr(m, f"{pname}_function")
            sd_ = SmallDelta(m, pname, full, via_fn)
            if via_fn:
                mp.add_weight_wrapper(f"{full}.{pname}", sd_)
            else:  # plain parameter: the hooks move the delta with the module's own device at forward time
                pass
            out[f"{full}.{pname}"] = sd_
            params.append(sd_.delta)
    if not out:
        raise RuntimeError(f"G56CT_SMALL matched no parameter: {regex}")
    return out, params


def small_deltas(c):
    return {k: a for k, a in c["adapters"].items() if isinstance(a, SmallDelta)}


def init_adapter(ad, mode):
    """comfy: exactly LoRAAdapter.create_train (up: kaiming_uniform a=sqrt(5) over the rank fan-in = U(-1/sqrt(r), 1/sqrt(r)),
    down: zeros). gauss: down ~ N(0, 1/in_dim) (unit-norm rows), up: zeros - the convention of the R1 leader's LoRAs; with Adam
    at the same lr every step moves the delta ~4-5x less than the comfy convention (the fixed random factor is that much smaller)."""
    import torch
    with torch.no_grad():
        if mode == "gauss":
            torch.nn.init.zeros_(ad.lora_up.weight)
            torch.nn.init.normal_(ad.lora_down.weight, std=ad.lora_down.weight.shape[1] ** -0.5)
        elif mode == "comfy":
            torch.nn.init.kaiming_uniform_(ad.lora_up.weight, a=5 ** 0.5)
            torch.nn.init.zeros_(ad.lora_down.weight)
        else:
            raise ValueError(f"unknown G56CT_INIT {mode!r}")


def attach_lora(mp, include_re, rank, alpha):
    import torch
    from comfy.weight_adapter import adapter_maps
    rx = re.compile(include_re)
    direct_rx = re.compile(CFG["direct"]) if CFG["direct"] else None
    adapters, params, names = {}, [], []
    for full, m in mp.model.named_modules():
        if not full.startswith("diffusion_model."):
            continue
        name = full[len("diffusion_model."):]
        if not rx.match(name):
            continue
        if not hasattr(m, "weight_function") or getattr(m, "weight", None) is None or len(m.weight.shape) < 2:
            continue
        if direct_rx is not None and direct_rx.search(name) and m.weight.numel() <= CFG["direct_max_m"] * 1e6:
            ad = DirectDelta(m.weight)
            mp.add_weight_wrapper(f"{full}.weight", ad)
            adapters[full] = ad
            params.append(ad.delta)
            names.append(full)
            continue
        ad = adapter_maps["LoRA"].create_train(m.weight, rank=rank, alpha=alpha).to(torch.float32)
        if CFG["init"] != "comfy":
            init_adapter(ad, CFG["init"])
        ad.train().requires_grad_(True)
        ad.alpha.requires_grad_(False)
        mp.add_weight_wrapper(f"{full}.weight", ad)
        if CFG["fast_lora"]:
            fast_lora_install(mp, m, full, ad)
        adapters[full] = ad
        params += [ad.lora_up.weight, ad.lora_down.weight]
        names.append(full)
    if not adapters:
        raise RuntimeError(f"no module matched {include_re}")
    return adapters, params


# ----------------------------------------------------------------------------------------------------------------------
# G56CT_FAST_LORA (ours): factored backward for the DiT LoRA Linears. On the weight-function path autograd keeps the merged
# weight W' = W + s*U@D (out x in, compute dtype) of every LoRA Linear for backward (+-23-25 GiB on krea2 / qwen-image ->
# gradient checkpointing on every block) and runs the full grad_W' = gy^T x GEMM. Here the forward is unchanged: W' comes
# from comfy.ops.cast_bias_weight under no_grad (the same bytes as the old path) and y = F.linear(x, W', b); autograd keeps
# x, U and D only. Backward (U = lora_up out x r, D = lora_down r x in, s = alpha / rank as in LoraDiff.__call__):
#   grad_x = gy @ W + (gy @ U) @ (s D)   W = the base weight as Comfy feeds it to the adapter (the resident weight, fp8 /
#                                        QuantizedTensor dequantized per layer on the fly and dropped again); gy @ U (fp32)
#                                        and s D are rounded to the compute dtype for the low-rank GEMM, whose compute-dtype
#                                        result is added to gy @ W in that GEMM's epilogue (the low-rank term is rounded
#                                        once on its own, then the sum)
#   grad_U = s gy^T (x D^T),  grad_D = s (gy U)^T x   fp32 factor gradients: the tokens x r intermediates x D^T and gy U
#                                        are fp32 and enter the token contraction as hi + lo compute-dtype parts (_mm32_hl),
#                                        so a token-shared (DC) input component costs no precision (bf16 GEMMs with fp32
#                                        accumulation and fp32 results on CUDA: mm out_dtype; exact fp32 on CPU)
# grad_x differs from the old path's gy @ W' by the bf16 rounding of W' (<= 1/2 ulp per weight): rounding noise that
# propagates through the model depth. So fast_lora_guard judges each module LOCALLY (the old path's formulas on the same
# x and gy, strict) and the whole-model gradients of the two paths only loosely (wiring). Any exception on the fast path
# switches it off for the rest of the run and the training step is redone on the old path (FastLoRAFallback).
# ----------------------------------------------------------------------------------------------------------------------
class FastLoRAState(object):
    """Per model patcher (mp._g56_fast): the modules with the fast forward {name: (module, adapter)}, the switch (off:
    every wrapper takes Comfy's own forward), op classes left on the old path, per-module fallbacks, the guard verdict,
    the guard's local statistics while its fast pass runs (probe) and the steps redone on the old path after an exception."""

    def __init__(self):
        self.mods, self.skipped, self.bad = {}, [], []
        self.on, self.checked, self.verdict, self.calls = False, False, None, 0
        self.probe, self.live, self.off_reason, self.retries, self.stats = None, None, None, 0, None
        # G56CT_FAST_MERGE: knob, armed (after the first optimizer step: _run_one), per module [state pending|ok|off,
        # trivial checks, scale tensor or None (s == 1), class, why off, verified compute dtype], merges by _fm_merge
        self.fm_on, self.fm_armed, self.fm, self.fm_calls = False, False, {}, 0

    def fail(self, why):
        if self.on:
            log(f"WARN fast lora: old path for the rest of the run: {why}")
        self.on = False
        self.off_reason = self.off_reason or why
        self.verdict = self.verdict or f"off: {why}"

    def attached(self, adapters):
        """The fast-path modules whose adapter is still attached (drop_unloadable detaches the unloadable ones)."""
        return sorted(k for k, (m, ad) in self.mods.items() if adapters.get(k) is ad)

    def report(self):
        return {"modules": len(self.mods) if self.live is None else self.live, "installed": len(self.mods),
                "skipped": self.skipped[:10], "module_fallbacks": self.bad[:10], "on": self.on, "guard": self.verdict,
                "guard_stats": self.stats, "off_reason": self.off_reason, "fast_forwards": self.calls,
                "step_retries": self.retries, "mm32": _mm32_status(),
                "fast_merge": self.fm_report() if self.fm_on else None}

    def fm_report(self):
        """G56CT_FAST_MERGE counts: modules ok / off / pending (overall and per weight class), merges, why off."""
        n = {"ok": 0, "off": 0, "pending": 0}
        cls = {}
        for rs in self.fm.values():
            n[rs[0]] += 1
            cls.setdefault(rs[3], {"ok": 0, "off": 0, "pending": 0})[rs[0]] += 1
        return dict(n, armed=self.fm_armed, merges=self.fm_calls, classes=cls,
                    off_modules=[(k, rs[4]) for k, rs in sorted(self.fm.items()) if rs[0] == "off"][:5])

    def fm_line(self):
        r = self.fm_report()
        return (f"fast merge: {r['ok']} ok / {r['off']} off" + (f" / {r['pending']} pending" if r["pending"] else "")
                + f" ({r['merges']} merges; classes {r['classes']})" + (f"; off {r['off_modules']}" if r["off"] else ""))


class FastLoRAFallback(RuntimeError):
    """The fast path raised in a forward or backward (the switch is off from then on): _run_one redoes the training step
    on the old path, as after an OOM. A step cannot finish on a mix of the two paths once blocks are checkpointed: the
    recomputation of a block would take the old path where its original forward took the fast one."""


_FAST_IMPLS = ("disable_weight_init.Linear.forward_comfy_cast_weights", "fp8_ops.Linear.forward_comfy_cast_weights",
               "mixed_precision_ops.<locals>.MixedPrecisionOps.Linear.forward_comfy_cast_weights")
_FAST_FORWARDS = ("disable_weight_init.Linear.forward", "mixed_precision_ops.<locals>.MixedPrecisionOps.Linear.forward")
_FAST_FN = []
_MM32 = {}
# guard verdict: LOCAL (per module group and gradient, fast vs the old path's formulas on the same x / gy) and END-TO-END
# (the whole fast pass vs the whole old pass: pooled over every LoRA factor gradient, plus a per-group norm-ratio band)
_FAST_TOL = dict(local_cos=0.999, local_ratio=0.01, e2e_cos=0.99, e2e_ratio=0.05, e2e_band=(0.5, 2.0))


def _fast_fn():
    """The autograd.Function of the fast path (built on first use: this module imports torch lazily)."""
    if not _FAST_FN:
        import torch

        class FastLoRALinear(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w_merged, bias, up, down, rec):
                ctx.save_for_backward(x, up, down)
                ctx.rec = rec
                return torch.nn.functional.linear(x, w_merged, bias)

            @staticmethod
            def backward(ctx, gy):
                x, up, down = ctx.saved_tensors[:3]
                st, name = ctx.rec[2], ctx.rec[3]
                try:
                    gx, gu, gd = _fast_backward(ctx.rec, ctx.needs_input_grad, gy, x, up, down)
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as e:  # noqa: BLE001
                    st.fail(f"{name} backward {type(e).__name__}: {str(e)[:160]}")
                    raise FastLoRAFallback(f"fast lora backward failed in {name}: {type(e).__name__}: {str(e)[:120]}") from e
                return gx, None, None, gu, gd, None

        _FAST_FN.append(FastLoRALinear)
    return _FAST_FN[0]


def _fast_backward(rec, need, gy, x, up, down):
    """Gradients of y = x (W + s U D)^T + b w.r.t. x, U, D (the bias is frozen: _fast_eligible)."""
    import torch
    m, ad, st, name = rec
    dt = gy.dtype                                   # = x.dtype = W'.dtype (cast_bias_weight casts to the input dtype)
    gy2 = gy.reshape(-1, gy.shape[-1])
    x2 = x.reshape(-1, x.shape[-1])
    s = (ad.alpha / ad.rank).detach()               # LoraDiff.__call__'s scale (fp32)
    u, d = up.detach(), down.detach()
    gyu = _mm32(gy2, u.to(dt)) if (need[0] or need[4]) else None                  # tokens x r, fp32
    gx = gu = gd = None
    if need[0]:
        gx = torch.mm(gyu.to(dt), (d * s).to(dt)).addmm_(gy2, _fast_base_weight(m, dt)).reshape(x.shape)
    if need[3]:
        gu = _mm32_hl(gy2.t(), _mm32(x2, d.to(dt).t())).mul_(s)                   # out x r
    if need[4]:
        gd = _mm32_hl(gyu.t(), x2).mul_(s)                                        # r x in
    if st.probe is not None:
        _fast_probe(st, name, m, ad, gy2, x2, u, d, gx, gu, gd)
    return gx, gu, gd


def _mm32(a, b):
    """a @ b with an fp32 result. Compute-dtype (bf16 / fp16) operands on CUDA: one tensor-core GEMM with fp32
    accumulation and an fp32 result (mm out_dtype, probed once per dtype on the backward's operand layouts; unavailable
    or failing -> the compute-dtype result upcast, reported by _mm32_status); on CPU the operands are upcast (exact) and
    multiplied in fp32; fp32 operands: a plain fp32 GEMM."""
    import torch
    if a.dtype in (torch.bfloat16, torch.float16):
        if a.is_cuda:
            if _MM32.get(a.dtype) is None:
                _MM32[a.dtype] = _mm32_probe(a.dtype, a.device)
            if _MM32[a.dtype] is True:
                try:
                    return torch.mm(a, b, out_dtype=torch.float32)
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as e:  # noqa: BLE001
                    _MM32[a.dtype] = f"failed: {type(e).__name__}: {str(e)[:80]}"
            return torch.mm(a, b).float()
        return torch.mm(a.float(), b.float())
    return torch.mm(a, b).float()


def _mm32_probe(dtype, device):
    """True when torch.mm(.., out_dtype=float32) works for `dtype` on `device` with the backward's operand layouts
    (transposed view x contiguous, contiguous x transposed view, contiguous x contiguous) and equals the fp32 product of
    the upcast operands; else a short reason (the backward then upcasts compute-dtype GEMM results)."""
    import torch
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        A, B, C, D = (torch.randn(*sh, generator=g).to(device=device, dtype=dtype)
                      for sh in ((24, 40), (24, 8), (40, 8), (8, 40)))
        # gy^T @ xD / gyU^T @ x (transposed view x contiguous), gy @ U (contiguous x contiguous), x @ D^T (x view)
        for a, b in ((A.t(), B), (B.t(), A), (A, C), (A, D.t())):
            r = torch.mm(a, b, out_dtype=torch.float32)
            if r.dtype != torch.float32 or not bool(torch.allclose(r, torch.mm(a.float(), b.float()), rtol=1e-5, atol=1e-5)):
                return "mismatch"
        return True
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {type(e).__name__}: {str(e)[:80]}"


def _mm32_status():
    return {str(k).replace("torch.", ""): v for k, v in _MM32.items()} or "unused (CPU or fp32 compute)"


def _mm32_hl(a, b):
    """a @ b with an fp32 result, one operand an fp32 tokens x r intermediate (x D^T or gy U) and the other a compute-dtype
    activation (gy^T or x). The fp32 operand enters as hi + lo compute-dtype parts (hi its rounding, lo the rounding of
    the remainder: <= 2^-17 relative) through two _mm32 GEMMs. A token-shared (DC) component of the input, which the token
    contraction cancels (attention keys: softmax shift invariance), then costs no precision: a compute-dtype intermediate
    lost ~2^-9 of that component before the cancellation (29 Sep review: 12x the old path's error at |DC| = 10 x the rest).
    fp32 compute: one fp32 GEMM."""
    import torch
    f32 = torch.float32
    if a.dtype == f32 and b.dtype == f32:
        return torch.mm(a, b)
    if a.dtype == f32:
        hi = a.to(b.dtype)
        return _mm32(hi, b).add_(_mm32((a - hi.float()).to(b.dtype), b))
    hi = b.to(a.dtype)
    return _mm32(a, hi).add_(_mm32(a, (b - hi.float()).to(a.dtype)))


def _fast_probe(st, name, m, ad, gy2, x2, u, d, gx, gu, gd):
    """fast_lora_guard's fast pass: this call's fast-path gradients vs the OLD path's formulas on the SAME (x, gy):
    grad_x = gy @ W' (W' re-merged by the adapter = the forward's bytes, checked by _fast_verify), grad_W' = gy^T x in the
    compute dtype (F.linear's backward) -> fp32 -> x scale (LoraDiff's own alpha / rank) -> grad_U = G D^T, grad_D = U^T G
    (the backward of LoraDiff's fp32 merge). Local: independent of how far the two passes' output gradients drifted apart
    through the depth."""
    import torch
    grp = re.sub(r"\.\d+\.", ".#.", name, count=1)
    dt = gy2.dtype
    with torch.no_grad():
        st.probe.setdefault("_modules", set()).add(name)
        if gx is not None:
            _probe_acc(st.probe, f"{grp}.gx", gx.reshape(gy2.shape[0], -1), torch.mm(gy2, ad(_fast_base_weight(m, dt))))
        if gu is not None or gd is not None:
            G = torch.mm(gy2.t(), x2).float().mul_(ad.alpha / ad.rank)
            if gu is not None:
                _probe_acc(st.probe, f"{grp}.up", gu, torch.mm(G, d.t()))
            if gd is not None:
                _probe_acc(st.probe, f"{grp}.down", gd, torch.mm(u.t(), G))


def _probe_acc(acc, key, a, b):
    """acc[key] += [a.b, |a|^2, |b|^2] (fp64), finite flag, calls."""
    import torch
    s = acc.setdefault(key, [0.0, 0.0, 0.0, True, 0])
    a64, b64 = a.double(), b.double()
    s[0] += float((a64 * b64).sum())
    s[1] += float(a64.square().sum())
    s[2] += float(b64.square().sum())
    s[3] &= bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    s[4] += 1


def _fast_base_weight(m, dtype):
    """The base weight exactly as comfy.ops.cast_bias_weight hands it to the weight functions for a device-resident
    weight: .to(dtype), a QuantizedTensor dequantized. A weight already in `dtype` comes back as is (no copy)."""
    import comfy.ops
    w = m.weight.to(dtype=dtype)
    if isinstance(w, comfy.ops.QuantizedTensor):
        w = w.dequantize()
    return w


def _fast_cast_kw(args, kwargs):
    """mixed_precision_ops.Linear passes compute_dtype on to cast_bias_weight (the other classes call it without)."""
    if args:
        return {"compute_dtype": args[0]}
    return {"compute_dtype": kwargs["compute_dtype"]} if "compute_dtype" in kwargs else {}


def _fast_eligible(m, ad, x, args, kwargs):
    """The fast path applies to this call: training forward (grad on) of a plain tensor, the module's weight functions
    exactly [ad] and no bias function, frozen resident base weight / bias on the input's device, no lowvram / vbar
    machinery, and only the no-op extra arguments of mixed_precision_ops.Linear."""
    import torch
    if not torch.is_grad_enabled() or type(x) is not torch.Tensor:
        return False
    fns, w, b = m.weight_function, m.weight, getattr(m, "bias", None)
    if len(fns) != 1 or fns[0] is not ad or len(m.bias_function) > 0 or w is None or w.requires_grad:
        return False
    if (b is not None and b.requires_grad) or not (ad.lora_up.weight.requires_grad or ad.lora_down.weight.requires_grad):
        return False
    if w.device != x.device or hasattr(m, "_v") or getattr(m, "weight_lowvram_function", None) is not None \
            or getattr(m, "bias_lowvram_function", None) is not None:
        return False
    if len(args) > 1 or set(kwargs) - {"compute_dtype", "want_requant", "weight_only_quant"}:
        return False
    return not kwargs.get("want_requant") and not kwargs.get("weight_only_quant")


def _fast_verify(m, ad, x, args, kwargs):
    """First use of a module: the base weight rebuilt for backward must be exactly the tensor cast_bias_weight feeds the
    adapter, i.e. ad(_fast_base_weight) == W' bit for bit."""
    import torch
    import comfy.ops
    with torch.no_grad():
        w, b, ost = comfy.ops.cast_bias_weight(m, x, offloadable=True, **_fast_cast_kw(args, kwargs))
        try:
            return bool(torch.equal(ad(_fast_base_weight(m, w.dtype)), w))
        finally:
            comfy.ops.uncast_bias_weight(m, w, b, ost)


def _fast_forward(m, ad, x, args, kwargs, rec):
    import torch
    import comfy.ops
    st, name = rec[2], rec[3]
    rs = _fm_state(st, name, m, ad, x) if (st.fm_on and st.fm_armed) else None  # G56CT_FAST_MERGE (armed)
    w = None
    with torch.no_grad():  # W' merged by Comfy exactly as on the old path (same bytes), outside the autograd graph
        if rs is not None and rs[0] == "ok" and x.dtype == rs[5]:
            try:
                w, b, _ = _fm_merge(m, ad, x.dtype, rs[2])
                ost = None
                st.fm_calls += 1
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:  # noqa: BLE001  (this module back to Comfy's merge; the fast path stays)
                rs[0], rs[4] = "off", f"merge {type(e).__name__}: {str(e)[:80]}"
                log(f"WARN fast merge: {name}: {rs[4]} -> Comfy's merge for it")
                w = None
        if w is None:
            w, b, ost = comfy.ops.cast_bias_weight(m, x, offloadable=True, **_fast_cast_kw(args, kwargs))
            if rs is not None and rs[0] == "pending":
                _fm_verify(st, name, m, ad, x.dtype, w, b, rs)
    try:
        return _fast_fn().apply(x, w, b, ad.lora_up.weight, ad.lora_down.weight, rec)
    finally:
        if ost is not None:
            comfy.ops.uncast_bias_weight(m, w, b, ost)


_FM_TRIVIAL_MAX = 8  # verifications in which the delta changes no weight element before a module keeps Comfy's merge


def _fm_merge(m, ad, dt, s):
    """G56CT_FAST_MERGE: (W', bias, W) for a device-resident weight, the bytes comfy.ops.cast_bias_weight + LoraDiff.__call__
    produce (W' = (fp32(W) + s * (U @ D)) rounded to the compute dtype) with less memory traffic: W is the resident
    weight itself (no copy; fp8 / QuantizedTensor cast / dequantized as Comfy does, _fast_base_weight), U @ D is the same
    fp32 GEMM call, the scale is applied only when s != 1 (s = LoraDiff's 0-dim alpha / rank tensor, None when it is 1:
    x * 1 == x), and W + diff is computed in fp32 and rounded once into a compute-dtype buffer (torch.add(out=)). Never a
    fused add(alpha=s) / addmm: one rounding less gives other bytes. The bias exactly as cast_bias_weight casts it."""
    import torch
    W = _fast_base_weight(m, dt)
    diff = ad.lora_up.weight @ ad.lora_down.weight
    if s is not None:
        diff.mul_(s)
    w = torch.empty(W.shape, dtype=dt, device=W.device)
    torch.add(W, diff.reshape(W.shape), out=w)
    b = m.bias.to(dtype=dt) if getattr(m, "bias", None) is not None else None
    return w, b, W


def _fm_state(st, name, m, ad, x):
    """The module's G56CT_FAST_MERGE record, created on its first fast call after arming: pending (verified against
    Comfy's merge on the next calls), or off when the bias is not on the input's device."""
    rs = st.fm.get(name)
    if rs is None:
        scale = ad.alpha.detach() / ad.rank                 # LoraDiff.__call__'s alpha / rank (0-dim, alpha's device)
        cls = f"{type(m.weight).__name__}:{str(m.weight.dtype).replace('torch.', '')}->{str(x.dtype).replace('torch.', '')}"
        b = getattr(m, "bias", None)
        rs = st.fm[name] = ["pending", 0, None if bool(scale == 1.0) else scale, cls, None, None]
        if b is not None and b.device != x.device:
            rs[0], rs[4] = "off", "bias not on the input's device"
    return rs


def _fm_verify(st, name, m, ad, dt, w_ref, b_ref, rs):
    """One G56CT_FAST_MERGE check of a pending module against the (w, b) Comfy just merged for this call: bit-identical
    -> ok once the delta changed at least one weight element (with U D = 0 any merge returns W: nothing is proven; after
    _FM_TRIVIAL_MAX such calls the module keeps Comfy's merge); different, or any exception -> off (Comfy's merge)."""
    import torch
    try:
        w2, b2, W = _fm_merge(m, ad, dt, rs[2])
        same = torch.equal(w2, w_ref) and (b2 is None) == (b_ref is None) and (b2 is None or torch.equal(b2, b_ref))
        if not same:
            rs[0], rs[4] = "off", "differs from Comfy's merge"
            log(f"WARN fast merge: {name}: W' / bias differ from Comfy's merge -> Comfy's merge for it")
        elif torch.equal(w_ref, W):
            rs[1] += 1
            if rs[1] >= _FM_TRIVIAL_MAX:
                rs[0], rs[4] = "off", f"the delta changed no weight in {rs[1]} checks"
        else:
            rs[0], rs[5] = "ok", dt  # used for calls in this compute dtype only
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as e:  # noqa: BLE001
        rs[0], rs[4] = "off", f"check {type(e).__name__}: {str(e)[:80]}"
        log(f"WARN fast merge: {name}: {rs[4]} -> Comfy's merge for it")


def fast_lora_install(mp, m, name, ad):
    """G56CT_FAST_LORA for one DiT LoRA Linear: wrap its forward_comfy_cast_weights (Comfy's cast_bias_weight + F.linear).
    A call takes the fast path only while the switch is on and _fast_eligible holds; otherwise Comfy's own forward runs
    (exact-holdout weight swaps, dropped adapters, SMALL bias deltas, no_grad scoring, ...). Op classes with another
    forward keep the old path (returns False). An exception on the fast path switches it off and raises FastLoRAFallback
    (the step is redone on the old path); an OOM propagates as it is (the trainer's OOM handling)."""
    import torch
    st = getattr(mp, "_g56_fast", None)
    if st is None:
        st = mp._g56_fast = FastLoRAState()
    impl, fwd_cls, inner = (getattr(type(m), a, None) for a in ("forward_comfy_cast_weights", "forward", "_forward"))
    ok = (getattr(impl, "__module__", None) == "comfy.ops" and getattr(impl, "__qualname__", None) in _FAST_IMPLS
          and getattr(fwd_cls, "__qualname__", None) in _FAST_FORWARDS
          and (inner is None or getattr(inner, "__qualname__", "").endswith("MixedPrecisionOps.Linear._forward"))
          and getattr(m, "weight", None) is not None and m.weight.dim() == 2 and getattr(ad, "lora_mid", None) is None)
    if not ok:
        st.skipped.append(name)
        return False
    orig = m.forward_comfy_cast_weights
    first = {"ok": None}
    rec = (m, ad, st, name)

    def fwd(input, *args, **kwargs):
        if st.on and first["ok"] is not False and _fast_eligible(m, ad, input, args, kwargs):
            try:
                if first["ok"] is None:
                    first["ok"] = _fast_verify(m, ad, input, args, kwargs)
                    if not first["ok"]:
                        st.bad.append(name)
                        log(f"WARN fast lora: {name}: rebuilt base weight != the one Comfy merges -> old path for it")
                if first["ok"]:
                    st.calls += 1
                    if st.probe is not None:  # the guard's fast pass: modules that ran a fast forward
                        st.probe.setdefault("_forward", set()).add(name)
                    return _fast_forward(m, ad, input, args, kwargs, rec)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:  # noqa: BLE001
                st.fail(f"{name} forward {type(e).__name__}: {str(e)[:160]}")
                raise FastLoRAFallback(f"fast lora forward failed in {name}: {type(e).__name__}: {str(e)[:120]}") from e
        return orig(input, *args, **kwargs)

    m.forward_comfy_cast_weights = fwd
    st.mods[name] = (m, ad)
    st.on = True
    st.fm_on = bool(CFG.get("fast_merge"))
    return True


def _fast_stats(acc):
    """{key: (cos, norm ratio a/b, finite)} from accumulated [a.b, |a|^2, |b|^2, finite, ...] (keys starting with '_'
    skipped); both norms zero -> (1, 1)."""
    out = {}
    for k, v in acc.items():
        if k.startswith("_"):
            continue
        dot, na, nb, fin = v[:4]
        if na == 0.0 and nb == 0.0:
            out[k] = (1.0, 1.0, fin)
            continue
        cos = dot / math.sqrt(na * nb) if na > 0 and nb > 0 else 0.0
        out[k] = (cos, math.sqrt(na / nb) if nb > 0 else float("inf"), fin)
    return out


def _fast_verdict(local, ga, gb, exempt=(), tol=None):
    """Guard verdict.
    LOCAL (strict; the old path's formulas on the fast pass's own x / gy, _fast_probe): per module group (first index
    stripped: blocks.#.attn.wq) and gradient (gx, up, down): cos >= 0.999, |norm ratio - 1| <= 0.01, finite; every
    module whose factors got a gradient in the fast pass checked, except `exempt` (modules _fast_verify left on the old
    path: their gradients are the old path's in both passes).
    END-TO-END (loose, wiring; ga / gb = {name: (grad up, grad down)} of the whole fast / old pass): pooled over every
    LoRA factor gradient cos >= 0.99 and |norm ratio - 1| <= 0.05, per group and factor a norm ratio in [0.5, 2] (a
    gradient that does not arrive fails), all finite. Its differences include the bf16 rounding of grad_x propagated
    through the depth (<= 1.8% per group on a 28-block krea2, 29 Sep review), which is why it is not the strict test."""
    import torch
    tol = dict(_FAST_TOL, **(tol or {}))
    bad = []
    loc = _fast_stats(local)
    n_mod, n_fwd = len(local.get("_modules", ())), len(local.get("_forward", ()))
    lcos = min([v[0] for v in loc.values()] or [1.0])
    lrat = max([abs(v[1] - 1.0) for v in loc.values()] or [0.0])
    for k, (cos, ratio, fin) in sorted(loc.items()):
        if not fin or cos < tol["local_cos"] or abs(ratio - 1.0) > tol["local_ratio"]:
            bad.append(f"local {k} cos {cos:.6f} ratio {ratio:.4f}" + ("" if fin else " non-finite"))
    with_grad = {k for k, v in ga.items() if any(t is not None for t in v)}
    unchecked = sorted(with_grad - local.get("_modules", set()) - set(exempt))
    if unchecked:
        bad.append(f"local: {len(unchecked)} module(s) with a gradient not checked {unchecked[:2]}")
    elif not n_mod:
        bad.append("local: no module checked")
    e2e, pool = {}, [0.0, 0.0, 0.0, True]
    for k in ga:
        grp = re.sub(r"\.\d+\.", ".#.", k, count=1)
        for f, fac in ((0, "up"), (1, "down")):
            a, b = ga[k][f], gb[k][f]
            s = e2e.setdefault(f"{grp}.{fac}", [0.0, 0.0, 0.0, True])
            if a is None or b is None:
                s[3] &= a is None and b is None
                continue
            a64, b64 = a.double(), b.double()
            v = (float((a64 * b64).sum()), float(a64.square().sum()), float(b64.square().sum()))
            fin = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            for acc in (s, pool):
                acc[0] += v[0]
                acc[1] += v[1]
                acc[2] += v[2]
                acc[3] &= fin
    est = _fast_stats(e2e)
    pc, pr, pf = _fast_stats({"pool": pool})["pool"]
    lo, hi = tol["e2e_band"]
    if not pf or pc < tol["e2e_cos"] or abs(pr - 1.0) > tol["e2e_ratio"]:
        bad.append(f"end-to-end pooled cos {pc:.5f} ratio {pr:.4f}" + ("" if pf else " non-finite"))
    for k, (cos, ratio, fin) in sorted(est.items()):
        if not fin or not lo <= ratio <= hi:
            bad.append(f"end-to-end {k} ratio {ratio:.4f}" + ("" if fin else " non-finite"))
    ecos = min([v[0] for v in est.values()] or [1.0])
    erat = max([abs(v[1] - 1.0) for v in est.values()] or [0.0])
    info = (f"local {n_mod} modules (of {n_fwd} fast, {len(with_grad)} with a gradient) / {len(loc)} groups: min cos "
            f"{lcos:.6f}, max |ratio-1| {lrat:.2e}; end-to-end {len(est)} groups: pooled cos {pc:.6f} ratio {pr:.4f}, per "
            f"group min cos {ecos:.5f} max |ratio-1| {erat:.2e}")
    stats = dict(local_modules=n_mod, fast_modules=n_fwd, grad_modules=len(with_grad), local_groups=len(loc),
                 local_min_cos=lcos, local_max_ratio_dev=lrat, e2e_groups=len(est), e2e_pooled_cos=pc,
                 e2e_pooled_ratio=pr, e2e_min_cos=ecos, e2e_max_ratio_dev=erat)
    return not bad, info + (f"; bad {bad[:4]}" if bad else ""), stats


def lora_state_dict(adapters, use_ema=None, fp32=True):
    import torch
    sd = {}
    for full, ad in adapters.items():
        if isinstance(ad, (DirectDelta, SmallDelta)):
            d = use_ema[id(ad.delta)] if use_ema is not None else ad.delta.detach()
            key = ad.key if isinstance(ad, SmallDelta) else f"{full}.diff"
            sd[key] = d.to("cpu", torch.float32 if fp32 else torch.bfloat16).contiguous().clone()
            continue
        up, down = ad.lora_up.weight.detach(), ad.lora_down.weight.detach()
        if use_ema is not None:
            up, down = use_ema[id(ad.lora_up.weight)], use_ema[id(ad.lora_down.weight)]
        dt = torch.float32 if fp32 else torch.bfloat16
        sd[f"{full}.lora_up.weight"] = up.to("cpu", dt).contiguous().clone()
        sd[f"{full}.lora_down.weight"] = down.to("cpu", dt).contiguous().clone()
        sd[f"{full}.alpha"] = torch.tensor(float(ad.alpha.item()), dtype=torch.float32)
    return sd


def verify_lora_loadable(mp, sd):
    """Replicates the evaluator's strictness: every key must be consumed by comfy.lora.load_lora."""
    import logging
    import comfy.lora
    key_map = comfy.lora.model_lora_keys_unet(mp.model, {})
    missing = []

    class H(logging.Handler):
        def emit(self, r):
            m = r.getMessage()
            if m.startswith(("lora key not loaded:", "NOT LOADED ")):
                missing.append(m)
    h = H()
    logging.getLogger().addHandler(h)
    try:
        patches = comfy.lora.load_lora(sd, key_map)
    finally:
        logging.getLogger().removeHandler(h)
    return len(patches), missing


class _NonFiniteStep(Exception):
    """G56CT_SAFE: the loss or the total gradient norm of a training step is not finite. _run_one skips the step: no
    optimizer / EMA update, the step counter stays, the case is not repeated."""


# G56CT_SAFE: worst measured seconds per byte of a save_lora write (finite check + save_file + os.replace) and of an
# ensemble average + its write; the all-data member stop margin is derived from it (_save_margin)
_SAVE_STATS = {}


def _sd_finite(sd):
    """Every floating tensor of the state dict is finite."""
    import torch
    return all(bool(torch.isfinite(v).all()) for v in sd.values() if v.is_floating_point())


def _sd_nbytes(sd):
    return sum(v.numel() * v.element_size() for v in sd.values())


def _note_save_rate(seconds, nbytes):
    if nbytes > 0:
        _SAVE_STATS["spb"] = max(_SAVE_STATS.get("spb", 0.0), max(0.0, seconds) / nbytes)
        _SAVE_STATS["n"] = _SAVE_STATS.get("n", 0) + 1


def _save_margin(c):
    """G56CT_SAFE all-data stop margin (s): 2 x the measured time to average and write the running ensemble of
    k_done + 1 member-size files, + 10 s; 0 before the first measurement (the fallback identity write is the first)."""
    spb = _SAVE_STATS.get("spb")
    if not spb:
        return 0.0
    return 2.0 * (c.get("k_done", 0) + 1) * c.get("member_bytes", 0) * spb + 10.0


def _safe_score(fn, safe):
    """Holdout score fn(). The evaluator core raises on a non-finite MSE (evalcore.flow_prediction_mse): with SAFE off
    that ends the run (the crash path ships the last saved file). G56CT_SAFE: that eval scores NaN instead - never the
    best, counts as an eval without improvement - and the run goes on."""
    if not safe:
        return fn()
    try:
        return fn()
    except ValueError as e:
        if "Non-finite denoising MSE" not in str(e):
            raise
        log(f"WARN safe: non-finite holdout MSE ({e}) -> scored NaN", public=True)
        return {"score": float("nan"), "text": float("nan"), "no_text": float("nan")}


def _clip_nf(params, max_norm):
    """G56CT_SAFE clip_grad_norm_: a non-finite total gradient norm raises _NonFiniteStep (grads left untouched; the
    caller zeroes them). max_norm None: only the check (clipping off)."""
    import torch
    try:
        if max_norm is None:
            return torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None], error_if_nonfinite=True)
        return torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True)
    except torch.cuda.OutOfMemoryError:
        raise
    except RuntimeError as e:
        if "non-finite" in str(e):
            raise _NonFiniteStep("gradient norm not finite") from e
        raise


def save_lora(sd, out_dir, meta=None):
    """Atomic write of out_dir/last.safetensors. G56CT_SAFE: a state with a non-finite tensor is refused (returns False,
    the file on disk and the folder stay as they were); the write time is measured (_SAVE_STATS)."""
    from safetensors.torch import save_file
    safe = CFG["safe"]
    if safe:
        t0 = time.time()
        if not _sd_finite(sd):
            log("WARN safe: refused non-finite state; last.safetensors left as it was", public=True)
            return False
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / ".last.safetensors.tmp"
    save_file(sd, str(tmp), metadata=meta or {})
    os.replace(tmp, out_dir / "last.safetensors")
    if safe:
        _note_save_rate(time.time() - t0, _sd_nbytes(sd))
    try:
        os.chmod(out_dir / "last.safetensors", 0o644)
    except Exception:
        pass
    return True


# ----------------------------------------------------------------------------------------------------------------------
# Text-encoder LoRA (ours, G56CT_TE_LORA=1). Evaluator: LoraLoader().load_lora(model, clip, name, 1.0, 1.0); when the
# clip got patches, the caption AND "" are re-encoded with the patched TE (fp16 merge -> fp8 requantization with a
# recalculated scale + stochastic rounding seeded by the key, i.e. the same "patch tax" as the fp8 DiT layers).
# Keys: Comfy's generic clip format text_encoders.<clip state-dict key minus .weight> (comfy.lora.model_lora_keys_clip).
# ----------------------------------------------------------------------------------------------------------------------
TE_PREFIX = "text_encoders."


class TEBypass:
    """Forward of one adapted TE Linear: Comfy's own forward (fp8 weight dequantized to the fp32 TE activations) plus the
    LoRA bypass h(x) = up(down(x)) * alpha/rank (comfy.weight_adapter LoraDiff.h). Same function as the evaluator's merged
    weight W + alpha/rank * up @ down minus its fp8 re-quantization; no merged fp32 weight copy is kept for backward."""

    def __init__(self, module, adapter):
        self.orig = module.forward
        self.ad = adapter
        self.active = True

    def __call__(self, x, *args, **kwargs):
        y = self.orig(x, *args, **kwargs)
        if not self.active:
            return y
        return y + self.ad.h(x.to(self.ad.lora_down.weight.dtype), y).to(y.dtype)


def te_llama(clip):
    """(cond_stage_model, its SDClipModel, the Llama2_ language model) of a single-TE Comfy CLIP (Qwen3-VL families)."""
    cm = clip.cond_stage_model
    inner = getattr(cm, cm.clip)
    return cm, inner, inner.transformer.model


def te_layer_cut(fam, n_layers):
    """Comfy's Llama2_ captures the INPUT of layer t for every tap t (t == n_layers: the final output), so exactly the
    layers < max(tap) feed the conditioning: ideogram4 (taps 1..36) all 36, krea2 (taps 2..35) layers 0-34."""
    import importlib
    mod, attr = FAMILIES[fam]["te_taps"]
    taps = [int(t) for t in getattr(importlib.import_module(mod), attr) if int(t) <= n_layers]
    if not taps:
        raise RuntimeError(f"no {fam} TE tap within {n_layers} layers")
    return max(taps)


def attach_te_lora(clip, fam, cfg):
    """LoRA adapters (comfy LoRAAdapter.create_train, fp32) on the TE Linears selected by te_include / te_layers, applied
    as forward bypass. Returns ({lora key: adapter}, [bypass wrappers], info)."""
    import torch
    from comfy.weight_adapter import adapter_maps
    cm, inner, llama = te_llama(clip)
    root = f"{cm.clip}.transformer."
    n_layers = len(llama.layers)
    cut = te_layer_cut(fam, n_layers)
    lo, hi = 0, cut - 1
    if cfg["te_layers"]:
        a, _, b = str(cfg["te_layers"]).partition("-")
        lo, hi = max(lo, int(a)), min(hi, int(b or a))
    rx = re.compile(cfg["te_include"])
    rank = int(cfg["te_rank"])
    alpha = float(cfg["te_alpha"]) if cfg["te_alpha"] > 0 else float(rank)
    adapters, wrappers = {}, []
    for name, m in cm.named_modules():
        if not name.startswith(root) or not rx.match(name[len(root):]):
            continue
        lm = re.match(r"^model\.layers\.(\d+)\.", name[len(root):])
        if lm is None or not lo <= int(lm.group(1)) <= hi:
            continue
        w = getattr(m, "weight", None)
        if w is None or len(w.shape) != 2 or not hasattr(m, "weight_function"):
            continue
        ad = adapter_maps["LoRA"].create_train(w, rank=rank, alpha=alpha).to(torch.float32)
        ad._g56_init = cfg["te_init"]
        if cfg["te_init"] != "comfy":
            init_adapter(ad, cfg["te_init"])
        ad.train().requires_grad_(True)
        ad.alpha.requires_grad_(False)
        wr = TEBypass(m, ad)
        m.forward = wr
        adapters[TE_PREFIX + name] = ad
        wrappers.append(wr)
    if not adapters:
        raise RuntimeError(f"no TE module matched {cfg['te_include']} in layers {lo}-{hi}")
    return adapters, wrappers, {"layers": f"{lo}-{hi}", "n_layers": n_layers, "tap_cut": cut, "rank": rank, "alpha": alpha}


def set_te_checkpointing(llama, on):
    """Checkpoint every TE decoder block (non-reentrant, like the DiT blocks); a no-op under no_grad (evaluation)."""
    import torch
    n = 0
    for blk in llama.layers:
        if on and not hasattr(blk, "_g56_orig_forward"):
            orig = blk.forward
            blk._g56_orig_forward = orig

            def fwd(args, kwargs, _orig=orig):
                return _orig(*args, **kwargs)

            def ckpt_fwd(*args, _fwd=fwd, **kwargs):
                if not torch.is_grad_enabled():
                    return _fwd(args, kwargs)
                return torch.utils.checkpoint.checkpoint(_fwd, args, kwargs, use_reentrant=False)
            blk.forward = ckpt_fwd
            n += 1
        elif not on and hasattr(blk, "_g56_orig_forward"):
            blk.forward = blk._g56_orig_forward
            del blk._g56_orig_forward
    return n


def te_to_device(te, device):
    """The language model (embeddings, decoder layers, final norm) and the adapters go to the training device; the vision
    tower and the 8B lm_head (unused for text conditioning) stay where Comfy left them."""
    _, _, llama = te_llama(te["clip"])
    llama.embed_tokens.to(device)
    llama.layers.to(device)
    if getattr(llama, "norm", None) is not None:
        llama.norm.to(device)
    for ad in te["adapters"].values():
        ad.to(device)


def te_encode(te, prompt, device):
    """CLIPTextEncode(clip, prompt) without model management (CLIP.encode_from_tokens_scheduled, no hooks): same cond and
    pooled dict; autograd flows into the TE adapters when grad is enabled. Tokens are cached per prompt. The output stays on
    `device` (Comfy hops it to the intermediate device = CPU; values identical)."""
    import comfy.model_management as mm
    clip = te["clip"]
    tokens = te["tokens"].get(prompt)
    if tokens is None:
        tokens = te["tokens"][prompt] = clip.tokenize(prompt)
    cm = clip.cond_stage_model
    cm.reset_clip_options()
    if clip.layer_idx is not None:
        cm.set_clip_options({"layer": clip.layer_idx})
    cm.set_clip_options({"execution_device": device})
    inter = mm.intermediate_device
    mm.intermediate_device = lambda: device
    try:
        with mm.cuda_device_context(device):
            o = cm.encode_token_weights(tokens)
    finally:
        mm.intermediate_device = inter
    cond, pooled = o[:2]
    extra = {"pooled_output": pooled}
    if len(o) > 2:
        extra.update(o[2])
    out = [[cond, extra]]
    if te.get("guidance") is not None:
        import node_helpers
        out = node_helpers.conditioning_set_values(out, {"guidance": te["guidance"]})
    return out


def te_adapters(c):
    return {k: a for k, a in c["adapters"].items() if k.startswith(TE_PREFIX)}


def te_reset(c, seed=None):
    """TE adapters back to a zero delta (their init convention)."""
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        for ad in te_adapters(c).values():
            init_adapter(ad, getattr(ad, "_g56_init", "gauss"))


def te_exact_patch_apply(c, lora_sd, device):
    """The evaluator's patched TE weights (comfy.lora.load_lora on model_lora_keys_clip -> CLIP patcher ->
    patch_weight_to_device: fp16 merge, fp8 requantization with recalculated scale + stochastic rounding seeded by the key)
    swapped into our TE modules; the training bypass is switched off. Only the TE keys of lora_sd are used. Returns undo."""
    import torch
    import comfy.lora
    import comfy.utils
    te = c["te"]
    clip = te["clip"]
    key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
    patches = comfy.lora.load_lora({k: v for k, v in lora_sd.items() if k.startswith(TE_PREFIX)}, key_map, log_missing=False)
    tmp = clip.patcher.clone()
    keys = tmp.add_patches(patches, 1.0, 1.0)
    saved = []
    for wr in te["wrappers"]:
        wr.active = False
    try:
        for key in sorted(keys):
            new_w = tmp.patch_weight_to_device(key, device_to=device, return_weight=True)
            m = comfy.utils.get_attr(clip.cond_stage_model, key.rsplit(".", 1)[0])
            saved.append((m, m.weight))
            m.weight = torch.nn.Parameter(new_w, requires_grad=False)
    except Exception:
        te_exact_restore(te, saved)
        raise
    return saved


def te_exact_restore(te, saved):
    for m, w in reversed(saved):
        m.weight = w
    for wr in te["wrappers"]:
        wr.active = True


def te_holdout_conds(c, model_wrap, conds, device):
    """Copy of the processed conds with every holdout caption and "" re-encoded by the TE in its current state."""
    import torch
    hc = list(conds)
    done = {}
    with torch.no_grad():
        for item in c["holdout"]:
            lat = item["scaled"].to(device)
            for prompt, idx in ((item["caption"], item["cond_idx"]), ("", c["empty_idx"])):
                if prompt not in done:
                    done[prompt] = processed_cond(model_wrap, te_encode(c["te"], prompt, device), lat, device)
                hc[idx] = done[prompt]
    return hc


def verify_lora_loadable_te(mp, clip, sd):
    """The evaluator's apply_lora strictness on model AND clip: comfy.sd.load_lora_for_models(model, clip, lora, 1, 1), no
    'lora key not loaded:' / 'NOT LOADED ' record, model patches present, one clip patch per TE adapter."""
    import logging
    import comfy.sd
    missing = []

    class H(logging.Handler):
        def emit(self, r):
            m = r.getMessage()
            if m.startswith(("lora key not loaded:", "NOT LOADED ")):
                missing.append(m)
    h = H()
    logging.getLogger().addHandler(h)
    try:
        patched, patched_clip = comfy.sd.load_lora_for_models(mp, clip, sd, 1.0, 1.0)
    finally:
        logging.getLogger().removeHandler(h)
    n_te = sum(1 for k in sd if k.startswith(TE_PREFIX) and k.endswith(".lora_up.weight"))
    if len(patched_clip.patcher.patches) != n_te:
        missing.append(f"clip patches {len(patched_clip.patcher.patches)} != TE adapters in file {n_te}")
    return len(patched.patches), len(patched_clip.patcher.patches), missing


def adapter_params(ad):
    if isinstance(ad, (DirectDelta, SmallDelta)):
        return [ad.delta]
    return [ad.lora_up.weight, ad.lora_down.weight]


def detach_wrapper(mp, key, fn):
    """Undo mp.add_weight_wrapper(key, fn): the patcher's pending list and, when already applied, the module's own list."""
    import uuid
    import comfy.utils
    pend = getattr(mp, "weight_wrapper_patches", None)
    if pend is not None and key in pend:
        pend[key] = [f for f in pend[key] if f is not fn]
        if not pend[key]:
            del pend[key]
        mp.patches_uuid = uuid.uuid4()
    mod, pname = key.rsplit(".", 1)
    try:
        m = comfy.utils.get_attr(mp.model, mod)
    except AttributeError:
        return
    lst = getattr(m, f"{pname}_function", None)
    if isinstance(lst, list) and any(f is fn for f in lst):
        setattr(m, f"{pname}_function", [f for f in lst if f is not fn])


def drop_unloadable(mp, adapters, clip=None, te=None):
    """Pre-training key check. The evaluator rejects a LoRA when ANY key is not loaded, while a file in the upload folder
    counts as a usable adapter for the exit policy. So before training and before the identity LoRA is written, the
    evaluator's loader (comfy.lora.load_lora on model_lora_keys_unet; with a TE LoRA comfy.sd.load_lora_for_models) runs on
    the attached adapters' keys and every adapter it would not load is detached (WARN with the module names): every file
    that reaches the upload folder then loads completely. The final verify_lora_loadable stays as the safety net.
    Edits adapters (and te) in place; returns the dropped adapter names."""
    def check():
        probe = lora_state_dict(adapters, fp32=False)
        if te is not None:
            return verify_lora_loadable_te(mp, clip, probe)[2]
        return verify_lora_loadable(mp, probe)[1]
    missing = check()
    if not missing:
        return []
    owner = {}
    for name, ad in adapters.items():
        for k in lora_state_dict({name: ad}, fp32=False):
            owner[k] = name
    bad = set()
    for msg in missing:
        if msg.startswith("lora key not loaded:"):
            name = owner.get(msg.split(":", 1)[1].strip())
        elif msg.startswith("NOT LOADED "):
            k = msg[len("NOT LOADED "):].strip()
            stem = k[: -len(".weight")] if k.endswith(".weight") else k
            name = next((x for x in (stem, k, TE_PREFIX + stem) if x in adapters), None)
        else:
            name = None
        if name is None and te is not None:  # not attributable to one module (e.g. clip patch count): drop the TE LoRA
            bad.update(te["adapters"])
        elif name is None:
            raise RuntimeError(f"key check: cannot attribute {msg!r} to an adapter")
        else:
            bad.add(name)
    for name in sorted(bad):
        ad = adapters.pop(name)
        if te is not None and name in te["adapters"]:
            te["adapters"].pop(name)
            for wr in [w for w in te["wrappers"] if w.ad is ad]:
                wr.active = False
                wr.orig.__self__.__dict__.pop("forward", None)
                te["wrappers"].remove(wr)
        elif isinstance(ad, SmallDelta):
            ad.enabled = False
            if ad.via_fn:
                detach_wrapper(mp, name, ad)
        else:
            detach_wrapper(mp, f"{name}.weight", ad)
    log(f"WARN key check: {len(bad)} adapter(s) the evaluator would not load, dropped before training: {sorted(bad)[:20]}")
    log(f"key check: dropped {len(bad)} unloadable adapter(s)", public=True)
    if not any(not k.startswith(TE_PREFIX) and not isinstance(a, SmallDelta) for k, a in adapters.items()):
        raise RuntimeError("key check: no loadable LoRA target left")
    left = check()
    if left:
        raise RuntimeError(f"key check: still not loadable after the drop: {left[:3]}")
    return sorted(bad)


# ----------------------------------------------------------------------------------------------------------------------
# Training / evaluation inside a Comfy guider (identical conditioning + prediction path to the evaluator)
# ----------------------------------------------------------------------------------------------------------------------
class Loop:
    def __init__(self, ctx):
        self.ctx = ctx
        self.result = None

    # comfy.samplers.Sampler interface
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        import torch
        c = self.ctx
        conds = model_wrap.conds["positive"]
        with torch.inference_mode(False):
            self.result = train_loop(c, model_wrap, conds, extra_args, noise.device)
        return latent_image

    def max_denoise(self, model_wrap, sigmas):
        return False


def processed_cond(model_wrap, raw_cond, latent, device):
    """Turn a raw [tensor, meta] conditioning into the sampler's processed form (model_conds etc.), the way
    CFGGuider does before training. Used for the per-step TE LoRA conditioning and its holdout re-encoding."""
    import torch
    import comfy.samplers
    import comfy.sampler_helpers
    out = comfy.samplers.process_conds(model_wrap.inner_model, torch.zeros_like(latent),
                                       {"positive": comfy.sampler_helpers.convert_cond(raw_cond)},
                                       device, latent, None, 42)
    return out["positive"][0]


def predict(model_wrap, conds, cond_idx, x, sigma_value, extra_args):
    import torch
    model_wrap.conds["positive"] = [conds[cond_idx]]
    ts = torch.full((x.shape[0],), float(sigma_value), device=x.device, dtype=torch.float32)
    return model_wrap(x, ts, **extra_args)


class ExactMergeError(RuntimeError):
    """exact_patch_apply(strict=True): ComfyUI's merge swallowed an error that is not an out-of-memory (logged only; the
    weight kept no delta). do_eval with G56CT_SAFE then scores the holdout through the live wrappers, as after an OOM."""


def exact_patch_apply(c, lora_sd, device, strict=False):
    """Swap the evaluator's own merged weights into the model: comfy.lora.load_lora -> ModelPatcher.patch_weight_to_device
    (fp16 merge, fp8 re-quantization with stochastic rounding seeded by the key) and drop our training weight wrappers, so
    the forward runs exactly the evaluator's path (incl. fp8 activations on quantized layers). Returns the undo list.
    strict (do_eval with G56CT_SAFE; review B2, 1 Oct): ComfyUI's merge does not raise - LoRAAdapter.calculate_weight catches
    every exception (a CUDA OOM too), logs 'ERROR lora <key> <error>' and returns the weight WITHOUT the delta;
    comfy.lora.calculate_weight logs 'Calculate Weight Failed' / 'WEIGHT NOT MERGED' - so the holdout was scored on a
    partly merged LoRA (QMODs0 1 Oct: 3 such lines before an OOM that did propagate). Such a log record during the patch
    loop restores what was patched and raises torch.OutOfMemoryError (an out-of-memory text) or ExactMergeError."""
    import logging
    import torch
    import comfy.lora
    import comfy.utils
    mp = c["mp"]
    key_map = comfy.lora.model_lora_keys_unet(mp.model, {})
    patches = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
    tmp = mp.clone()
    tmp.add_patches(patches, 1.0, 1.0)
    saved, failed = [], []

    class MergeLog(logging.Handler):  # the root logger, as verify_lora_loadable (Comfy logs through logging.error / warning)
        def emit(self, r):
            m = r.getMessage()
            if m.startswith(("ERROR ", "Calculate Weight Failed")) or "WEIGHT NOT MERGED" in m:
                failed.append(m)
    h = MergeLog() if strict else None
    if h is not None:
        logging.getLogger().addHandler(h)
    try:
        for key in patches:  # weights, and (G56CT_SMALL) biases / norm gains / modulation offsets
            new_w = tmp.patch_weight_to_device(key, device_to=device, return_weight=True)
            if failed:  # this key's weight came back without its delta: nothing of it is swapped in
                oom = any("out of memory" in m.lower() for m in failed)
                log(f"exact merge: ComfyUI logged {len(failed)} merge failure(s) at {key}: {failed[0][:300]}")
                raise (torch.OutOfMemoryError if oom else ExactMergeError)(
                    f"ComfyUI's LoRA merge failed without raising ({'out of memory' if oom else 'not an OOM'}; "
                    f"{len(failed)} log record(s))")
            mod_name, pname = key.rsplit(".", 1)
            m = comfy.utils.get_attr(mp.model, mod_name)
            fn_attr = f"{pname}_function" if pname in ("weight", "bias") and hasattr(m, f"{pname}_function") else None
            saved.append((m, pname, m._parameters[pname], fn_attr, list(getattr(m, fn_attr)) if fn_attr else None))
            m._parameters[pname] = torch.nn.Parameter(new_w, requires_grad=False)
            if fn_attr:
                setattr(m, fn_attr, [])
        for sdl in small_deltas(c).values():  # plain-parameter deltas: the patched value already holds them
            if not sdl.via_fn:
                sdl.enabled = False
                saved.append((sdl, None, None, None, None))
    except Exception:
        exact_patch_restore(saved)
        raise
    finally:
        if h is not None:
            logging.getLogger().removeHandler(h)
    return saved


def exact_patch_restore(saved):
    for m, pname, w, fn_attr, fns in reversed(saved):
        if isinstance(m, SmallDelta):
            m.enabled = True
            continue
        m._parameters[pname] = w
        if fn_attr:
            setattr(m, fn_attr, fns)


def small_disable(c):
    """Base-model scoring with G56CT_SMALL: detach every small delta (function wrappers and forward hooks); undo list."""
    undo = []
    for sdl in small_deltas(c).values():
        if sdl.via_fn:
            fns = getattr(sdl.module, f"{sdl.pname}_function")
            undo.append((sdl, list(fns)))
            setattr(sdl.module, f"{sdl.pname}_function", [f for f in fns if f is not sdl])
        else:
            sdl.enabled = False
            undo.append((sdl, None))
    return undo


def small_restore(undo):
    for sdl, fns in reversed(undo):
        if fns is not None:
            setattr(sdl.module, f"{sdl.pname}_function", fns)
        else:
            sdl.enabled = True


def holdout_score(c, model_wrap, conds, extra_args, device):
    """Evaluator metric on the held-out images: 16 strata x holdout_noises, text and empty prompt, 0.5/0.5 weighting."""
    import torch
    from evalcore import flow_cases, flow_prediction_mse
    if not c["holdout"]:
        return None
    text, notext = [], []
    with torch.no_grad():
        for item in c["holdout"]:
            cases = flow_cases(item["scaled"], item["sha"], c["cfg"]["holdout_strata"], c["cfg"]["holdout_noises"])
            for mode, idx in (("text", item["cond_idx"]), ("no_text", c["empty_idx"])):
                losses = []
                for case in cases:
                    for s in range(0, len(case["seeds"]), c["cfg"]["eval_batch"]):
                        noisy = case["noisy"][s:s + c["cfg"]["eval_batch"]].to(device)
                        target = case["target"][s:s + c["cfg"]["eval_batch"]].to(device)
                        den = predict(model_wrap, conds, idx, noisy, case["sigma"], extra_args)
                        losses += flow_prediction_mse(noisy, den, target, case["sigma"]).cpu().tolist()
                (text if mode == "text" else notext).append(sum(losses) / len(losses))
    t, n = sum(text) / len(text), sum(notext) / len(notext)
    return {"score": 0.5 * t + 0.5 * n, "text": t, "no_text": n}


def holdout_score_te(c, model_wrap, conds, extra_args, device, sd):
    """holdout_score; with a TE LoRA the holdout captions and "" are first re-encoded by the TE patched with the TE keys of
    sd exactly like the evaluator (a file without TE keys -> the base TE, as the evaluator's cached conditioning)."""
    if not c.get("te"):
        return holdout_score(c, model_wrap, conds, extra_args, device)
    if not c["holdout"]:
        return None
    saved = te_exact_patch_apply(c, sd, device)
    try:
        return holdout_score(c, model_wrap, te_holdout_conds(c, model_wrap, conds, device), extra_args, device)
    finally:
        te_exact_restore(c["te"], saved)


def set_checkpointing(mp, on):
    """Gradient checkpointing per transformer block, same wrapping as comfy_extras.nodes_train.patch()."""
    import torch
    blocks = [m for full, m in mp.model.named_modules()
              if re.match(r"^diffusion_model\.(layers|blocks|transformer_blocks|double_blocks|single_blocks|noise_refiner|context_refiner)\.\d+$", full)]
    for m in blocks:
        if on and not hasattr(m, "_g56_orig_forward"):
            orig = m.forward
            m._g56_orig_forward = orig

            def fwd(args, kwargs, _orig=orig):
                return _orig(*args, **kwargs)

            def ckpt_fwd(*args, _fwd=fwd, **kwargs):
                return torch.utils.checkpoint.checkpoint(_fwd, args, kwargs, use_reentrant=False)
            m.forward = ckpt_fwd
        elif not on and hasattr(m, "_g56_orig_forward"):
            m.forward = m._g56_orig_forward
            del m._g56_orig_forward
    return len(blocks)


def scaled_sd(sd, s):
    """G56CT_FINAL_SCALE: every delta scaled by s (LoRA up, full .diff/.diff_b); s = 1 returns sd unchanged."""
    if s == 1.0:
        return sd
    return {k: (v * s if k.endswith((".lora_up.weight", ".diff", ".diff_b")) else v) for k, v in sd.items()}


def patch_flux_inplace():
    """Ours: ComfyUI's flux DoubleStreamBlock/SingleStreamBlock.forward add their residual branches IN PLACE (img += ...,
    txt += ..., x += ...): fine for inference, but training fails ('modified by an inplace operation', 24 Sep CF1). Recompile
    both forwards from ComfyUI's OWN source with exactly those augmented assignments rewritten out of place (X = X + ...):
    identical math and dtype, always in step with the evaluator's ComfyUI version. Returns the number of rewrites."""
    import inspect
    import textwrap
    import comfy.ldm.flux.layers as L
    total = 0
    for cls in (L.DoubleStreamBlock, L.SingleStreamBlock):
        src = textwrap.dedent(inspect.getsource(cls.forward))
        new, n = re.subn(r"^(\s*)(img|txt|x) \+= ", r"\1\2 = \2 + ", src, flags=re.M)
        if n:
            ns = {}
            exec(compile(new, f"<g56 out-of-place {cls.__name__}.forward>", "exec"), L.__dict__, ns)
            cls.forward = ns["forward"]
            total += n
    return total


def side_dir(c):
    """Research side outputs: G56CT_RESEARCH_DIR (persistent, per queue label) or the task's out_dir."""
    d = Path(c["cfg"]["research_dir"]) if c["cfg"]["research_dir"] else c["out_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def accum_groups(n_images, band_pool, m, same_band, rng):
    """G56CT_ACCUM: one pass over every (image, band) case, chunked into optimizer steps of <= m cases. same_band: each
    step holds different images at ONE sigma band (the AND-mask then compares images, not noise levels); the images of a
    band are split into ceil(n/m) near-equal chunks (no lone single-image step)."""
    if same_band:
        groups = []
        for b in band_pool:
            imgs = list(range(n_images))
            rng.shuffle(imgs)
            nch = -(-n_images // m)
            groups += [[(i, b) for i in imgs[k::nch]] for k in range(nch)]
    else:
        cases = [(i, b) for i in range(n_images) for b in band_pool]
        rng.shuffle(cases)
        groups = [cases[k:k + m] for k in range(0, len(cases), m)]
    rng.shuffle(groups)
    return groups


def fill_member_steps(durs, steps, remaining, full_cap, frac, margin=30.0, safety=1.10):
    """G56CT_SEEDS_FILL: steps a last, shortened ensemble member can run in `remaining` seconds, from the measured wall time
    per step of the finished members (x safety, minus margin); 0 when that is below frac x full_cap (or fill is off)."""
    if frac <= 0 or full_cap <= 0 or not durs:
        return 0
    per_step = max(d / max(1, n) for d, n in zip(durs, steps))
    n = int((remaining - margin) / (safety * per_step))
    return min(n, full_cap) if n >= frac * full_cap else 0


def and_mask(gsum, ssum, n, tau):
    """AND-mask (Parascandolo et al., ICLR 2021): gsum = summed per-case gradients, ssum = summed per-case signs over n
    cases; an element survives only where |ssum| / n >= tau (the cases agree on its direction). Returns the masked
    gradients and the surviving fraction."""
    out, kept, tot = [], 0.0, 0
    for gs, ss in zip(gsum, ssum):
        msk = (ss.abs() >= tau * n - 1e-6).to(gs.dtype)
        out.append(gs * msk)
        kept += float(msk.sum())
        tot += msk.numel()
    return out, kept / max(1, tot)


def case_noise(shape, seed, ci, band, visit):
    """G56CT_ANTITHETIC: noise of the visit-th training visit of (image ci, band): visits 2k and 2k+1 share one draw with
    opposite signs (antithetic pair, zero-mean first-order gradient noise cancels across the pair); pairs are independent."""
    import torch
    g = torch.Generator(device="cpu").manual_seed((seed * 1000003 + ci * 7919 + band * 131 + (visit // 2) * 104729) & 0x7FFFFFFFFFFF)
    eps = torch.randn(shape, generator=g, dtype=torch.float32)
    return -eps if visit % 2 == 1 else eps


def _run_one(c, model_wrap, conds, extra_args, device, seed, t_end, run_idx):
    import torch
    from evalcore import flow_prediction_mse  # noqa: F401  (same target definition)
    cfg = c["cfg"]
    params = c["params"]
    te = c.get("te")
    small_params = c.get("small_params") or []
    up_ids, down_ids = set(), set()
    if cfg["lora_plus"] != 1.0:  # LoRA+: the DiT LoRA up matrices get their own lr multiplier
        up_ids = {id(a.lora_up.weight) for k, a in c["adapters"].items()
                  if not k.startswith(TE_PREFIX) and hasattr(a, "lora_up")}
    if cfg["lora_down_mult"] != 1.0:  # ours: the DiT LoRA down matrices get their own lr multiplier
        down_ids = {id(a.lora_down.weight) for k, a in c["adapters"].items()
                    if not k.startswith(TE_PREFIX) and hasattr(a, "lora_down")}
    if te is not None or small_params or up_ids or down_ids:  # own groups: TE / small deltas / LoRA+ ups / downs (own lr)
        other = {id(p) for p in c.get("te_params", [])} | {id(p) for p in small_params}
        dit_params = [p for p in params if id(p) not in other]
        groups = [{"params": [p for p in dit_params if id(p) not in up_ids and id(p) not in down_ids]}]
        if up_ids:
            groups.append({"params": [p for p in dit_params if id(p) in up_ids], "lr": cfg["lr"] * cfg["lora_plus"],
                           "g56_lr_scale": cfg["lora_plus"]})
        if down_ids:
            groups.append({"params": [p for p in dit_params if id(p) in down_ids], "lr": cfg["lr"] * cfg["lora_down_mult"],
                           "g56_lr_scale": cfg["lora_down_mult"]})
        if te is not None:
            groups.append({"params": c["te_params"], "lr": cfg["te_lr"], "g56_lr_scale": cfg["te_lr"] / cfg["lr"]})
        if small_params:
            groups.append({"params": small_params, "lr": cfg["small_lr"], "g56_lr_scale": cfg["small_lr"] / cfg["lr"],
                           "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, cfg["beta2"]), weight_decay=cfg["wd"], eps=1e-8)
    else:
        opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, cfg["beta2"]), weight_decay=cfg["wd"], eps=1e-8)
    ema = {id(p): p.detach().clone() for p in params}
    ema_x = {d: {id(p): p.detach().clone() for p in params} for d in cfg["ema_extra"] if d > 0}
    ema_p0 = {id(p): p.detach().clone() for p in params} if cfg["ema_debias"] else None

    def ema_src(e=None, d=None):  # the EMA state to evaluate / ship (debiased when G56CT_EMA_DEBIAS)
        e = ema if e is None else e
        d = cfg["ema"] if d is None else d
        if ema_p0 is None or step == 0:
            return e
        bt = d ** step
        return {k: (v - bt * ema_p0[k]) / (1.0 - bt) for k, v in e.items()}
    # anchor copy lives in host RAM (krea2 runs sit at ~75 GB of 80) and is applied in batches of anchor_every steps
    anchor = [p.detach().to("cpu", copy=True) for p in params] if cfg["anchor"] > 0 else None
    rng = random.Random(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    train = c["train"]
    order, band_order = [], []
    t_start = time.time()
    step, evals, best, bad = 0, [], None, 0
    hist = []
    cases, band_c = [], 0
    visits, ci = {}, -1
    acc_groups, agree_frac, acc_case = [], [], 0
    late_on = False
    safe = cfg.get("safe", False)
    nf_run, nf_total, abandoned = 0, 0, False  # G56CT_SAFE: non-finite steps in a row / in this run
    fst = getattr(c["mp"], "_g56_fast", None)
    fm_arm = bool(cfg.get("fast_merge")) and fst is not None and fst.fm_on and not fst.fm_armed
    if cfg["late_frac"] > 0 and not cfg["case_cycle"]:
        raise RuntimeError("G56CT_LATE_FRAC needs G56CT_CASE_CYCLE=1")
    if cfg["accum"] > 1 and (te is not None or cfg["crop_frac"] > 0 or cfg["antithetic"]
                             or cfg["pair_anti"] or not cfg["case_cycle"]):
        raise RuntimeError("G56CT_ACCUM > 1 needs G56CT_CASE_CYCLE=1 and no TE LoRA / crop / antithetic / pair_anti")
    band_pool = list(range(cfg["band_min"], cfg["band_max"] or cfg["bands"]))
    if cfg["edge_bands"] and cfg["edge_repeat"] > 1:
        _edge = set(range(cfg["edge_bands"])) | set(range(cfg["bands"] - cfg["edge_bands"], cfg["bands"]))
        band_pool = [b for b in band_pool for _ in range(cfg["edge_repeat"] if b in _edge else 1)]
    if cfg["low_bands"] and cfg["low_repeat"] > 1:
        band_pool = [b for b in band_pool for _ in range(cfg["low_repeat"] if b < cfg["low_bands"] else 1)]
    step_times = []
    ckpt_state = c["grad_ckpt_on"]

    def lr_at(now):
        if cfg["max_steps"] and not c["holdout"]:  # fixed step budget on all data: schedule by steps, unless the run's
            # wall budget runs out first (slower GPU / shorter task): then decay by time so the LR still reaches its floor
            p = min(1.0, max(step / max(1, cfg["max_steps"]), (now - t_start) / max(1.0, t_end - t_start)))
        else:
            p = min(1.0, max(0.0, (now - t_start) / max(1.0, t_end - t_start)))
        w = min(1.0, (step + 1) / max(1, cfg["warmup"]))
        return cfg["lr"] * w * (cfg["lr_final_frac"] + (1 - cfg["lr_final_frac"]) * 0.5 * (1 + math.cos(math.pi * p)))

    def swap_in(src):  # src: dict id->tensor ; returns backup
        bak = {}
        for p in params:
            bak[id(p)] = p.detach().clone()
            p.data.copy_(src[id(p)])
        return bak

    def do_eval(tag_step):
        nonlocal best, bad
        res = {}
        for tag in cfg["eval_tags"]:
            exact_done = False
            if c["exact"]:
                src = ema_src() if tag == "ema" else {id(p): p.detach() for p in params}
                sd_x = lora_state_dict(c["adapters"], use_ema=src, fp32=True)
                try:  # restores what it patched itself when it raises; strict: a merge failure Comfy only logs raises too
                    saved = exact_patch_apply(c, sd_x, device, strict=safe)
                    try:
                        r = _safe_score(lambda: holdout_score_te(c, model_wrap, conds, extra_args, device, sd_x), safe)
                    finally:
                        exact_patch_restore(saved)
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    exact_done = True
                except (torch.OutOfMemoryError, ExactMergeError) as e:
                    # G56CT_SAFE (1 Oct, QMODs0 crash): the evaluator-exact merge of big modules can run out of memory at the
                    # first eval; without SAFE that ends the run and the identity fallback is shipped. With SAFE: free the
                    # memory and score the holdout through the live LoRA wrappers from now on (no merged weight copies).
                    # Also when ComfyUI swallowed the error and left a weight without its delta (review B2, 1 Oct).
                    if not safe:
                        raise
                    sd_x = None
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    c["exact"] = False
                    what = "out of memory" if isinstance(e, torch.OutOfMemoryError) else "a failed LoRA merge"
                    log(f"WARN safe: {what} in the evaluator-exact holdout eval ({str(e)[:80]}); the holdout is scored "
                        f"through the live LoRA wrappers from now on", public=True)
            if not exact_done:
                bak = swap_in(ema_src()) if tag == "ema" else None
                try:  # TE part (fp8 in both TE families) always on the evaluator's merged weights
                    r = _safe_score(lambda: holdout_score_te(
                        c, model_wrap, conds, extra_args, device,
                        lora_state_dict(te_adapters(c), fp32=True) if te is not None else None), safe)
                finally:
                    if bak is not None:
                        swap_in(bak)
            if r is None:
                continue
            r.update(step=tag_step, tag=tag, run=run_idx, t=round(time.time() - T0, 1))
            if cfg["rebalance"] > 0 and tag == cfg["eval_tags"][-1] and r["text"] > 0 and r["no_text"] > 0:
                gap = (r["no_text"] - r["text"]) / (0.5 * (r["no_text"] + r["text"]))  # relative, signed
                w = min(0.75, max(0.25, 0.5 + cfg["rebalance"] * gap))                 # gain knob: 10 -> +0.06 at a 0.6% gap
                if abs(w - cfg["empty_weight"]) > 0.01:
                    log(f"rebalance: empty weight {cfg['empty_weight']:.3f} -> {w:.3f} (text {r['text']:.6f} no_text {r['no_text']:.6f})")
                    cfg["empty_weight"] = w
            evals.append(r)
            res[tag] = r
            log(f"eval run {run_idx} step {tag_step} {tag}: {r['score']:.6f} (text {r['text']:.6f} no_text {r['no_text']:.6f})"
                + (f"  base {c['base_score']['score']:.6f}" if c.get("base_score") else ""))
            if cfg["delta_avg"] > 1 and tag == "ema":
                hist.append(lora_state_dict(c["adapters"], use_ema=ema_src(), fp32=cfg["save_fp32"]))
                del hist[:-cfg["delta_avg"]]
            if safe and not math.isfinite(r["score"]):  # G56CT_SAFE: never the best (NaN would lock in: x < NaN is False)
                log(f"WARN safe: non-finite holdout score ignored (run {run_idx} step {tag_step} {tag})", public=True)
            elif best is None or r["score"] < best["score"]:
                best = dict(r)
                src = ema_src() if tag == "ema" else {id(p): p.detach() for p in params}
                best["sd"] = lora_state_dict(c["adapters"], use_ema=src, fp32=cfg["save_fp32"])
                if c.get("global_best") is None or r["score"] < c["global_best"]:
                    c["global_best"] = r["score"]
                    save_lora(best["sd"], c["out_dir"])
                    log(f"  new best run{run_idx} {tag}@{tag_step} saved")
                bad = 0
        if safe:  # a non-finite score counts as an eval without improvement
            if res and all(not math.isfinite(v["score"]) or (best is not None and v["score"] > best["score"])
                           for v in res.values()):
                bad += 1
        elif res and best is not None and all(v["score"] > best["score"] for v in res.values()):
            bad += 1
        return res

    next_eval = cfg["eval_first"]
    while True:
        now = time.time()
        avg = sum(step_times[-20:]) / len(step_times[-20:]) if step_times else 5.0
        eval_cost = c.get("eval_cost", 60.0)
        margin = eval_cost * 1.2
        if safe and not c["holdout"]:  # G56CT_SAFE: room for the measured running-average save after this member
            margin = max(margin, _save_margin(c))
        if now + avg + margin > t_end or (cfg["max_steps"] and step >= cfg["max_steps"]):
            break
        limit = cfg["stop_after_bad"] or (cfg["run_stop_bad"] if c.get("multi") else 0)
        if limit and bad >= limit:
            log(f"stop run {run_idx}: {bad} evals without improvement")
            break
        ts0 = time.time()
        if cfg["late_frac"] > 0 and not late_on:  # ours: late low-sigma phase (G56CT_LATE_FRAC / LATE_BANDS)
            prog = step / cfg["max_steps"] if (cfg["max_steps"] and not c["holdout"]) else \
                (ts0 - t_start) / max(1.0, t_end - t_start)
            if prog >= 1.0 - cfg["late_frac"]:
                late_on = True
                band_pool = [b for b in band_pool if b < cfg["late_bands"]] or band_pool
                cases, acc_groups = [], []
                log(f"late phase from step {step}: only sigma bands < {cfg['late_bands']} ({len(set(band_pool))} bands)")
        if cfg["accum"] > 1:  # ours: several cases per optimizer step, optionally AND-masked (G56CT_ACCUM / _BAND / AGREE)
            if not acc_groups:
                acc_groups = accum_groups(len(train), band_pool, cfg["accum"], cfg["accum_band"], rng)
            group = acc_groups.pop()
            for g in opt.param_groups:
                g["lr"] = lr_at(ts0) if "g56_lr_scale" not in g else lr_at(ts0) * g["g56_lr_scale"]
            masking = cfg["agree"] > 0 and len(group) > 1
            try:
                tot = 0.0
                if masking:
                    gsum = [torch.zeros_like(p) for p in params]
                    ssum = [torch.zeros_like(p) for p in params]
                for ci, band_c in group:
                    item, cond_idx = train[ci], train[ci]["cond_idx"]
                    u = 0.5 if cfg["sigma_mode"] == "mid" else rng.random()
                    sigma = min(max((band_c + u) / cfg["bands"], 1e-4), 1 - 1e-4)
                    if cfg["late_frac"] > 0:
                        c.setdefault("band_trace", []).append((step, band_c))
                    x0 = item["scaled"].to(device)
                    eps = torch.randn(x0.shape, generator=gen, dtype=torch.float32).to(device)
                    xt = (1 - sigma) * x0 + sigma * eps
                    target = eps - x0
                    if cfg["prompt_mode"] == "paired":
                        plan = [(cond_idx, 1.0 - cfg["empty_weight"]), (c["empty_idx"], cfg["empty_weight"])]
                    elif cfg["prompt_mode"] == "balanced":
                        plan = [(cond_idx if acc_case % 2 == 0 else c["empty_idx"], 1.0)]
                    else:
                        plan = [(c["empty_idx"] if rng.random() < cfg["empty_frac"] else cond_idx, 1.0)]
                    acc_case += 1
                    for idx, w in plan:
                        if w <= 0:
                            continue
                        den = predict(model_wrap, conds, idx, xt, sigma, extra_args)
                        v = (xt - den.float()) / sigma
                        loss = (v - target).square().mean() * (w / len(group))
                        loss.backward()
                        tot += float(loss.detach())
                    if masking:  # keep this case's gradient apart: its sign votes, the sum is what survives
                        with torch.no_grad():
                            for p, gs, ss in zip(params, gsum, ssum):
                                if p.grad is not None:
                                    gs.add_(p.grad)
                                    ss.add_(torch.sign(p.grad))
                                    p.grad = None
                if safe and not math.isfinite(tot):
                    raise _NonFiniteStep(f"loss {tot}")
                if masking:
                    with torch.no_grad():
                        grads, kept = and_mask(gsum, ssum, len(group), cfg["agree"])
                        for p, g in zip(params, grads):
                            p.grad = g
                        agree_frac.append(kept)
                if cfg["clip"] > 0:
                    (_clip_nf if safe else torch.nn.utils.clip_grad_norm_)(params, cfg["clip"])
                elif safe:
                    _clip_nf(params, None)
                opt.step()
                opt.zero_grad(set_to_none=True)
            except _NonFiniteStep as e:  # G56CT_SAFE (first: nothing else may take it): skip the group
                opt.zero_grad(set_to_none=True)
                nf_run, nf_total = nf_run + 1, nf_total + 1
                log(f"WARN safe: non-finite step skipped ({e}; run {run_idx} after step {step}, {nf_run} in a row, "
                    f"{nf_total} in this run)", public=nf_total == 1)
                if nf_run > cfg["safe_nf_max"]:
                    log(f"safe: member abandoned after {nf_run} non-finite steps in a row (run {run_idx}, step {step})",
                        public=True)
                    abandoned = True
                    break
                continue
            except FastLoRAFallback as e:  # G56CT_FAST_LORA raised (now off): redo the group on the old path
                # a raise inside backward leaves the failed group's graph (saved activations) alive through these locals
                # until the redo's forward returns: drop it first (review B1, 1 Oct); the accumulators are rebuilt
                den = v = loss = gsum = ssum = None
                opt.zero_grad(set_to_none=True)
                _fast_retry(c, e)
                gc.collect()
                acc_groups.append(group)
                continue
            except torch.cuda.OutOfMemoryError:
                den = v = loss = gsum = ssum = None  # the failed group's graph, as above (review B1): not alive in the retry
                opt.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                if not ckpt_state:
                    n = set_checkpointing(c["mp"], True)
                    ckpt_state = True
                    log(f"OOM -> gradient checkpointing enabled on {n} blocks; retrying")
                    acc_groups.append(group)
                    continue
                raise
        else:
            if cfg["case_cycle"]:
                if not cases:
                    cases = [(i, b) for i in range(len(train)) for b in band_pool]
                    rng.shuffle(cases)
                ci, band_c = cases.pop()
                item, cond_idx = train[ci], train[ci]["cond_idx"]
            if not order:
                order = list(range(len(train)))
                rng.shuffle(order)
            if not band_order:
                band_order = list(range(cfg["band_min"], cfg["band_max"] or cfg["bands"]))
                if cfg["edge_bands"] and cfg["edge_repeat"] > 1:  # oversample the lowest/highest sigma bands (leader's krea2 skill)
                    edge = set(range(cfg["edge_bands"])) | set(range(cfg["bands"] - cfg["edge_bands"], cfg["bands"]))
                    band_order = [b for b in band_order for _ in range(cfg["edge_repeat"] if b in edge else 1)]
                if cfg["low_bands"] and cfg["low_repeat"] > 1:  # ours: lowest bands only
                    band_order = [b for b in band_order for _ in range(cfg["low_repeat"] if b < cfg["low_bands"] else 1)]
                rng.shuffle(band_order)
            if not cfg["case_cycle"]:
                item = train[order.pop()]
                cond_idx = item["cond_idx"]
            band = band_c if cfg["case_cycle"] else band_order.pop()
            if cfg["late_frac"] > 0:
                c.setdefault("band_trace", []).append((step, band))
            u = 0.5 if cfg["sigma_mode"] == "mid" else rng.random()
            sigma = (band + u) / cfg["bands"]
            sigma = min(max(sigma, 1e-4), 1 - 1e-4)
            x0 = item["scaled"].to(device)
            crop_empty = False
            if cfg["crop_frac"] > 0 and sigma <= cfg["crop_maxsig"] and rng.random() < cfg["crop_frac"]:
                hh, ww = x0.shape[-2], x0.shape[-1]
                sc = cfg["crop_min"] + (1.0 - cfg["crop_min"]) * rng.random()
                ch, cw = max(2, int(hh * sc) // 2 * 2), max(2, int(ww * sc) // 2 * 2)  # even: krea2 patchifies 2x2
                top, left = 2 * rng.randrange(0, (hh - ch) // 2 + 1), 2 * rng.randrange(0, (ww - cw) // 2 + 1)
                x0 = x0[..., top:top + ch, left:left + cw].contiguous()
                crop_empty = cfg["crop_prompt"] == "empty"
            if cfg["antithetic"] and cfg["case_cycle"] and x0.shape == item["scaled"].shape:
                v = visits.get((ci, band_c), 0)
                visits[(ci, band_c)] = v + 1
                eps = case_noise(x0.shape, seed, ci, band_c, v).to(device)
            else:
                eps = torch.randn(x0.shape, generator=gen, dtype=torch.float32).to(device)
            xt = (1 - sigma) * x0 + sigma * eps
            target = eps - x0
            if cfg["prompt_mode"] == "paired":
                plan = [(cond_idx, 1.0 - cfg["empty_weight"]), (c["empty_idx"], cfg["empty_weight"])]
            elif cfg["prompt_mode"] == "balanced":  # caption / empty alternate step by step: one forward per step
                plan = [(cond_idx if step % 2 == 0 else c["empty_idx"], 1.0)]
            else:
                plan = [(c["empty_idx"] if rng.random() < cfg["empty_frac"] else cond_idx, 1.0)]
            if crop_empty:
                plan = [(c["empty_idx"], 1.0)]
            for g in opt.param_groups:
                g["lr"] = lr_at(ts0) if "g56_lr_scale" not in g else lr_at(ts0) * g["g56_lr_scale"]
            try:
                tot = 0.0
                for idx, w in plan:
                    if w <= 0:
                        continue
                    if te is not None:  # ours: condition through the trainable TE every step (graph into the TE adapters)
                        empty = idx == c["empty_idx"]
                        slot = c["te_slots"][1 if empty else 0]
                        with torch.set_grad_enabled(not c.get("te_frozen")):
                            raw = te_encode(te, "" if empty else item["caption"], device)
                        conds[slot] = processed_cond(model_wrap, raw, x0, device)
                        idx = slot
                    xt_i, target_i = xt, target
                    if cfg["pair_anti"] and cfg["prompt_mode"] == "paired" and idx == c["empty_idx"]:
                        xt_i, target_i = (1 - sigma) * x0 - sigma * eps, -eps - x0  # antithetic partner of the caption half
                    den = predict(model_wrap, conds, idx, xt_i, sigma, extra_args)
                    v = (xt_i - den.float()) / sigma
                    loss = (v - target_i).square().mean() * w
                    loss.backward()
                    tot += float(loss.detach())
                if safe and not math.isfinite(tot):
                    raise _NonFiniteStep(f"loss {tot}")
                if cfg["clip"] > 0:
                    clip_ = _clip_nf if safe else torch.nn.utils.clip_grad_norm_
                    if te is not None or small_params:
                        clip_(dit_params, cfg["clip"])
                        if te is not None:
                            clip_(c["te_params"], cfg["clip"])
                        if small_params:
                            clip_(small_params, cfg["clip"])
                    else:
                        clip_(params, cfg["clip"])
                elif safe:
                    _clip_nf(params, None)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if anchor is not None and (step + 1) % cfg["anchor_every"] == 0:  # stay near the starting delta
                    a = min(1.0, cfg["anchor"] * cfg["anchor_every"])
                    with torch.no_grad():
                        for p, p0 in zip(params, anchor):
                            p.mul_(1 - a).add_(p0.to(p.device, non_blocking=True), alpha=a)
            except _NonFiniteStep as e:  # G56CT_SAFE (first: nothing else may take it): skip the step
                opt.zero_grad(set_to_none=True)
                if te is not None:  # drop the step's TE graphs
                    raw = None
                    for s in c["te_slots"]:
                        conds[s] = c["te_placeholder"]
                nf_run, nf_total = nf_run + 1, nf_total + 1
                log(f"WARN safe: non-finite step skipped ({e}; run {run_idx} after step {step}, {nf_run} in a row, "
                    f"{nf_total} in this run)", public=nf_total == 1)
                if nf_run > cfg["safe_nf_max"]:
                    log(f"safe: member abandoned after {nf_run} non-finite steps in a row (run {run_idx}, step {step})",
                        public=True)
                    abandoned = True
                    break
                continue
            except FastLoRAFallback as e:  # G56CT_FAST_LORA raised (now off): redo the step on the old path
                # a raise inside backward leaves the failed step's graph (saved activations) alive through these locals
                # until the redo's forward returns: drop it first (review B1, 1 Oct)
                den = v = loss = None
                opt.zero_grad(set_to_none=True)
                if te is not None:  # drop the step's TE graphs
                    raw = None
                    for s in c["te_slots"]:
                        conds[s] = c["te_placeholder"]
                _fast_retry(c, e)
                gc.collect()
                continue
            except torch.cuda.OutOfMemoryError:
                den = v = loss = None  # the failed step's graph, as above (review B1): not alive in the retry's forward
                opt.zero_grad(set_to_none=True)
                if te is not None:  # drop the step's TE graphs before emptying the cache
                    raw = None
                    for s in c["te_slots"]:
                        conds[s] = c["te_placeholder"]
                gc.collect()
                torch.cuda.empty_cache()
                if not ckpt_state:
                    n = set_checkpointing(c["mp"], True)
                    ckpt_state = True
                    log(f"OOM -> gradient checkpointing enabled on {n} blocks; retrying")
                    continue
                if te is not None and not c.get("te_frozen"):
                    c["te_frozen"] = True
                    log("OOM with checkpointing on -> TE adapters frozen at their current delta (still used and exported); retrying")
                    continue
                raise
        with torch.no_grad():
            d = cfg["ema"]
            for p in params:
                e = ema[id(p)]
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            for dx, ex in ema_x.items():
                for p in params:
                    ex[id(p)].mul_(dx).add_(p.detach(), alpha=1 - dx)
        nf_run = 0
        if fm_arm:  # G56CT_FAST_MERGE: armed after the first optimizer step (the guard and step 1 used Comfy's merge; from
            fst.fm_armed, fm_arm = True, False  # here on the LoRA delta is nonzero, so the per-module checks prove something)
        step += 1
        step_times.append(time.time() - ts0)
        if step % 200 == 0:
            log("training advancing", public=True)
        if cfg["save_every"] and not c["holdout"] and step % cfg["save_every"] == 0 and (run_idx == 0 or not c.get("multi")):
            # crash-safe interim file (members k>0 of an ensemble leave the running average of earlier members in place)
            save_lora(scaled_sd(lora_state_dict(c["adapters"], use_ema=ema_src(), fp32=cfg["save_fp32"]),
                                cfg["final_scale"] if not c.get("multi") else 1.0), c["out_dir"])
        if step <= 3 or step % 50 == 0:
            log(f"step {step} loss {tot:.5f} sigma {sigma:.3f} lr {lr_at(ts0):.2e} {step_times[-1]:.2f}s/step"
                + (f" agree-kept {sum(agree_frac[-50:]) / len(agree_frac[-50:]):.3f}" if agree_frac else "")
                + (f" mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB" if torch.cuda.is_available() else ""))
        if step >= next_eval and c["holdout"]:
            te0 = time.time()
            do_eval(step)
            c["eval_cost"] = time.time() - te0
            next_eval = step + (cfg["eval_every"] if step < cfg["eval_late_from"] else cfg["eval_every_late"])
    # final evaluation of the last state
    if c["holdout"] and (not evals or evals[-1]["step"] != step):
        do_eval(step)
    if c["holdout"] and cfg["delta_avg"] > 1 and len(hist) >= cfg["delta_avg"] and best is not None \
            and time.time() + 1.3 * c.get("eval_cost", 60.0) < c["deadline"]:
        sd_avg = average_loras(hist[-cfg["delta_avg"]:])
        r = score_sd_exact(c, model_wrap, conds, extra_args, device, sd_avg)
        log(f"delta-average of last {cfg['delta_avg']} evals: {r['score']:.6f} (best {best['score']:.6f})")
        if r["score"] < best["score"]:
            best = {"tag": f"davg{cfg['delta_avg']}", "step": step, "score": r["score"], "sd": sd_avg}
            save_lora(sd_avg, c["out_dir"])
    if not c["holdout"]:  # no holdout: keep EMA of the final state
        best = {"tag": "ema", "step": step, "score": None,
                "sd": lora_state_dict(c["adapters"], use_ema=ema_src(), fp32=cfg["save_fp32"])}
        if not (safe and c.get("multi")):  # G56CT_SAFE: an ensemble member leaves the running average on disk (the
            # caller writes the new average; the member alone was on disk for the seconds until then)
            save_lora(scaled_sd(best["sd"], cfg["final_scale"] if not c.get("multi") else 1.0), c["out_dir"])
        if cfg["ema_extra"]:  # research side outputs (E2-lite): other EMA decays of the same run, 0 = raw final weights
            from safetensors.torch import save_file
            rd = side_dir(c)
            for dx in cfg["ema_extra"]:
                src = ema_src(ema_x[dx], dx) if dx > 0 else {id(p): p.detach() for p in params}
                save_file(lora_state_dict(c["adapters"], use_ema=src, fp32=cfg["save_fp32"]),
                          str(rd / f"r{run_idx}_ema{dx:g}.safetensors"))
            log(f"EMA_EXTRA: wrote {len(cfg['ema_extra'])} side outputs to {rd}")
    c["grad_ckpt_on"] = ckpt_state
    if cfg.get("fast_merge") and fst is not None and fst.fm_on and fst.fm_armed:
        log(fst.fm_line())
    out = {"steps": step, "evals": evals, "best": {k: v for k, v in (best or {}).items() if k != "sd"},
           "best_sd": (best or {}).get("sd"), "base": c.get("base_score"),
           "step_time": sum(step_times) / max(1, len(step_times))}
    if safe:
        out.update(abandoned=abandoned, nonfinite_steps=nf_total)
    return out




def reinit_adapters(c, seed, member=0):
    """Fresh LoRA init for the next run (convention G56CT_INIT, see init_adapter); G56CT_SPEC_INIT: spectral down directions
    (block `member` with SPEC_BLOCKS, else the top block for every member)."""
    import torch
    torch.manual_seed(seed)
    with torch.no_grad():
        for ad in c["adapters"].values():
            if isinstance(ad, (DirectDelta, SmallDelta)):
                ad.delta.zero_()
            else:
                init_adapter(ad, getattr(ad, "_g56_init", CFG["init"]))
    if c.get("spec_ranked") is not None:
        apply_spectral(c, member if CFG["spec_blocks"] else 0)


def average_loras(sds):
    """Exact average of K LoRA deltas as one rank-(sum r_k) LoRA: up=[s_k U_k]/K, down=[D_k], alpha=rank (scale 1)."""
    import torch
    mods = sorted({k[: -len(".lora_up.weight")] for k in sds[0] if k.endswith(".lora_up.weight")})
    out = {}
    for m in mods:
        ups, downs = [], []
        for sd in sds:
            u, d = sd[f"{m}.lora_up.weight"].float(), sd[f"{m}.lora_down.weight"].float()
            a = float(sd[f"{m}.alpha"]) if f"{m}.alpha" in sd else float(d.shape[0])
            ups.append(u * (a / d.shape[0]))
            downs.append(d)
        up = torch.cat(ups, dim=1) / len(sds)
        down = torch.cat(downs, dim=0)
        out[f"{m}.lora_up.weight"] = up.contiguous()
        out[f"{m}.lora_down.weight"] = down.contiguous()
        out[f"{m}.alpha"] = torch.tensor(float(down.shape[0]), dtype=torch.float32)
    for k in sds[0]:  # full-rank direct / small deltas average elementwise
        if k.endswith((".diff", ".diff_b")):
            out[k] = sum(sd[k].float() for sd in sds).div_(len(sds)).contiguous()
    return out


class _SafeEnsemble(object):
    """G56CT_SAFE running ensemble of the all-data members (seeds-auto, fixed-K all-data). A member joins when its state is
    finite and it was not abandoned (non-finite steps in a row, _run_one); the first abandoned member is kept aside and
    used only while no complete member exists. The running average (x scale) is written after every member that changes
    it (crash safety; the average + write time feeds _save_margin), and the final average IS the last running one when
    that write succeeded: no recomputation, no second write (was: member EMA + running average + final average)."""

    def __init__(self, c, scale):
        self.c, self.scale = c, scale
        self.sds, self.spare, self.last = [], None, None

    def members(self):
        return self.sds if self.sds else ([self.spare] if self.spare is not None else [])

    def key(self):
        return len(self.sds), self.spare is not None and not self.sds

    def average(self):
        m = self.members()
        return scaled_sd(average_loras(m) if len(m) > 1 else m[0], self.scale) if m else None

    def add(self, r, k):
        sd = r.get("best_sd")
        if sd is None:
            return
        if not _sd_finite(sd):
            log(f"WARN safe: member {k + 1} ended with a non-finite state: not averaged", public=True)
            return
        if r.get("abandoned"):
            if self.sds or self.spare is not None:
                log(f"safe: abandoned member {k + 1} not averaged", public=True)
                return
            self.spare = sd
            log(f"safe: abandoned member {k + 1} kept aside (shipped only if no member completes)", public=True)
        else:
            self.sds.append(sd)
        t0 = time.time()
        sd = self.average()
        ok = save_lora(sd, self.c["out_dir"])
        if ok:
            _note_save_rate(time.time() - t0, _sd_nbytes(sd))
        self.last = (self.key(), sd) if ok else None

    def final(self):
        if self.last is not None and self.last[0] == self.key():
            log(f"safe: final reused (the running average of {len(self.members())} member(s) is on disk)")
            return self.last[1]
        sd = self.average()
        if sd is None:
            raise RuntimeError("safe: no finite member")
        if not save_lora(sd, self.c["out_dir"]):
            raise RuntimeError("safe: the final average is not finite")
        return sd


def score_sd_exact(c, model_wrap, conds, extra_args, device, sd):
    import torch
    saved = exact_patch_apply(c, sd, device)
    try:
        return _safe_score(lambda: holdout_score_te(c, model_wrap, conds, extra_args, device, sd),
                           c["cfg"].get("safe", False))
    finally:
        exact_patch_restore(saved)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def te_drop_check(c, model_wrap, conds, extra_args, device, best):
    """Ours: every TE key costs a requantization (+ stochastic rounding) of that fp8 TE layer at scoring time. Score the pick
    with and without its TE keys on the evaluator path and keep the better file. best=(score, tag, sd)."""
    if not c["cfg"]["te_drop_check"] or not c["holdout"] or best[2] is None \
            or not any(k.startswith(TE_PREFIX) for k in best[2]):
        return best
    if time.time() + (1 if c["exact"] else 2) * 1.2 * c.get("eval_cost", 60.0) > c["deadline"]:
        log("skip TE drop check: no time")
        return best
    full = best[0] if (c["exact"] and best[0] is not None) else \
        score_sd_exact(c, model_wrap, conds, extra_args, device, best[2])["score"]
    sd_dit = {k: v for k, v in best[2].items() if not k.startswith(TE_PREFIX)}
    r = score_sd_exact(c, model_wrap, conds, extra_args, device, sd_dit)
    # G56CT_SAFE: a non-finite evaluator score with the TE keys and a finite one without -> drop them (NaN < x is False)
    drop = r["score"] < full or (c["cfg"].get("safe", False) and not math.isfinite(full) and math.isfinite(r["score"]))
    log(f"TE drop check: with TE {full:.6f}, DiT only {r['score']:.6f} -> {'drop' if drop else 'keep'} TE keys")
    if drop:
        return (r["score"], f"{best[1]}-te", sd_dit)
    return (full, best[1], best[2])


def scale_search(c, model_wrap, conds, extra_args, device, best):
    """Ours: the winning delta is not necessarily best at strength 1.0.
    Score a few rescaled copies on the holdout with the evaluator numerics and keep the best. best=(score, tag, sd)."""
    cfg = c["cfg"]
    scales = [float(x) for x in str(cfg["scale_search"]).split(",") if x.strip()]
    if not scales or not c["holdout"] or best[2] is None:
        return best
    tried = [(best[0], 1.0, best[2])]
    for sc in scales:
        if abs(sc - 1.0) < 1e-6:
            continue
        if time.time() + 1.2 * c.get("eval_cost", 60.0) > c["deadline"]:
            log("skip scale search: no time")
            break
        sd_s = {k: (v * sc if k.endswith(".lora_up.weight") else v) for k, v in best[2].items()}
        r = score_sd_exact(c, model_wrap, conds, extra_args, device, sd_s)
        log(f"scale {sc:g}: {r['score']:.6f}")
        tried.append((r["score"], sc, sd_s))
    sc_best = min(tried, key=lambda x: x[0])
    if sc_best[1] != 1.0:
        log(f"scale search picked x{sc_best[1]:g} ({sc_best[0]:.6f} vs {best[0]:.6f})")
        best = (sc_best[0], f"{best[1]}x{sc_best[1]:g}", sc_best[2])
    groups = [g for g in str(cfg["scale_groups"]).split(",") if g.strip()]
    for grp in groups:  # greedy per-group strength (adaLN / attention / feed_forward may want different strengths)
        for sc in scales:
            if abs(sc - 1.0) < 1e-6 or time.time() + 1.2 * c.get("eval_cost", 60.0) > c["deadline"]:
                continue
            # DiT groups never touch TE keys (TE LoRA: group "text_encoders" scales them)
            sd_g = {k: (v * sc if (k.endswith((".lora_up.weight", ".diff")) and grp in k
                                   and k.startswith(TE_PREFIX) == grp.startswith(TE_PREFIX[:-1])) else v) for k, v in best[2].items()}
            r = score_sd_exact(c, model_wrap, conds, extra_args, device, sd_g)
            if r["score"] < best[0]:
                log(f"group scale {grp} x{sc:g}: {r['score']:.6f} (kept)")
                best = (r["score"], f"{best[1]}+{grp}x{sc:g}", sd_g)
    return best


def _lora_ads(c):
    return [a for k, a in c["adapters"].items() if not k.startswith(TE_PREFIX) and hasattr(a, "lora_up")]


def spectral_down_init(c, model_wrap, conds, extra_args, device):
    """G56CT_SPEC_INIT: rank the input directions of every DiT LoRA module by the mean full-weight gradient G at the base
    (paired-prompt training loss over N probe cases: every image once, bands stratified, fixed noise). Pass 1: up = Psi
    (random), down = 0 -> grad_down = Psi^T G (row-space sketch, P blocks); QR -> Q (in x P*r). Pass 2: up = 0, down = Q_b^T
    -> grad_up = G Q_b; SVD of G Q ranks the directions. Returns {id(adapter): in x P*r ranked orthonormal directions}.
    Deltas stay exactly 0 throughout (forward = base); no optimizer step; W untouched -> export stays a standard LoRA."""
    import torch
    cfg = c["cfg"]
    ads = _lora_ads(c)
    items = c["train"]
    rng = random.Random(4242)
    bands = list(range(cfg["bands"]))
    rng.shuffle(bands)
    cases = [(it, bands[i % len(bands)]) for i, it in enumerate(items)]
    others = [p for p in c["params"] if not any(p is a.lora_up.weight or p is a.lora_down.weight for a in ads)]

    def probe(which):
        for p in c["params"]:
            p.grad = None
        g = torch.Generator(device="cpu").manual_seed(777)
        for it, band in cases:
            sigma = (band + 0.5) / cfg["bands"]
            x0 = it["scaled"].to(device)
            eps = torch.randn(x0.shape, generator=g, dtype=torch.float32).to(device)
            xt, target = (1 - sigma) * x0 + sigma * eps, eps - x0
            for idx, w in ((it["cond_idx"], 1.0 - cfg["empty_weight"]), (c["empty_idx"], cfg["empty_weight"])):
                den = predict(model_wrap, conds, idx, xt, sigma, extra_args)
                ((((xt - den.float()) / sigma - target).square().mean()) * (w / len(cases))).backward()
        out = {id(a): (a.lora_down.weight.grad if which == "down" else a.lora_up.weight.grad).detach().float().clone()
               for a in ads}
        for p in c["params"]:
            p.grad = None
        return out

    t0 = time.time()
    gP = torch.Generator(device="cpu").manual_seed(12345)
    P, Z = max(1, cfg["spec_probes"]), {id(a): [] for a in ads}
    with torch.no_grad():
        saved = {id(a): (a.lora_up.weight.detach().clone(), a.lora_down.weight.detach().clone()) for a in ads}
    for _ in range(P):
        with torch.no_grad():
            for a in ads:
                o, r = a.lora_up.weight.shape
                a.lora_up.weight.copy_(torch.randn(o, r, generator=gP).div_(o ** 0.5).to(a.lora_up.weight.device))
                a.lora_down.weight.zero_()
        for k, v in probe("down").items():
            Z[k].append(v)
    Q = {k: torch.linalg.qr(torch.cat(v, 0).T.double())[0] for k, v in Z.items()}  # in x min(in, P*r)
    r0 = ads[0].lora_down.weight.shape[0]
    Q = {k: torch.cat([_block(q, r0, b) for b in range(P)], 1) for k, q in Q.items()}  # in x P*r (zero-padded if tiny)
    M = {id(a): [] for a in ads}
    for b in range(P):
        with torch.no_grad():
            for a in ads:
                r = a.lora_down.weight.shape[0]
                a.lora_up.weight.zero_()
                a.lora_down.weight.copy_(_block(Q[id(a)], r, b).T.to(a.lora_down.weight.dtype).to(a.lora_down.weight.device))
        for k, v in probe("up").items():
            M[k].append(v)
    ranked = {}
    with torch.no_grad():
        for a in ads:
            Mm = torch.cat(M[id(a)], 1).double()                       # out x P*r = G Q
            _, _, Vh = torch.linalg.svd(Mm, full_matrices=False)
            ranked[id(a)] = (Q[id(a)] @ Vh.T).float()                  # in x P*r, columns ranked by gradient energy
            up0, dn0 = saved[id(a)]
            a.lora_up.weight.copy_(up0)
            a.lora_down.weight.copy_(dn0)
    del others
    log(f"spectral init: ranked {P * ads[0].lora_down.weight.shape[0]} input directions for {len(ads)} modules from "
        f"{len(cases)} probe cases in {time.time() - t0:.0f}s")
    return ranked


def _block(cols, r, b):
    """Columns [r*b, r*(b+1)) of `cols` (in x m); zero-padded when m is short (tiny layers: in < probes x r)."""
    import torch
    blk = cols[:, r * b:r * (b + 1)]
    if blk.shape[1] < r:
        blk = torch.cat([blk, torch.zeros(cols.shape[0], r - blk.shape[1], dtype=cols.dtype, device=cols.device)], 1)
    return blk


def apply_spectral(c, block):
    """Set every DiT LoRA module to down = ranked directions block `block` (rows orthonormal, gauss-init scale), up = 0."""
    import torch
    ranked = c["spec_ranked"]
    with torch.no_grad():
        for a in _lora_ads(c):
            r = a.lora_down.weight.shape[0]
            cols = ranked[id(a)]
            b = block % max(1, -(-cols.shape[1] // r))
            a.lora_down.weight.copy_(_block(cols, r, b).T.to(a.lora_down.weight.device, a.lora_down.weight.dtype))
            a.lora_up.weight.zero_()


def fast_lora_guard(c, model_wrap, conds, extra_args, device):
    """G56CT_FAST_LORA runtime guard, once before the first training step, on one training case (first training image,
    sigma band bands//2, fixed noise, both prompts with their training weights) with probe factors instead of the init
    (up = 0 would leave down's gradient and the low-rank part of grad_x zero on both paths): down ~ N(0, 1/in), up random
    and scaled per module so that |s U D| = 0.1 |W|. Two passes: the fast path (every fast-path backward also evaluates
    the old path's formulas on its own x / gy: _fast_probe) and the old weight-function path. Verdict _fast_verdict:
    strict per module group LOCAL agreement (grad_x, grad_U, grad_D: cos >= 0.999, norm ratio within 1%) and a loose
    END-TO-END comparison of the two passes' factor gradients (wiring: pooled cos >= 0.99 / ratio within 5%, per group a
    ratio in [0.5, 2]) - the bf16 rounding of grad_x propagates through the depth, so whole-model gradients of the two
    paths are no strict test (29 Sep review: false failures on a 28-block krea2). A failure or any exception switches the
    fast path off for the rest of the process. Both passes run with block checkpointing (the old path must fit); factors,
    gradients, RNG streams and the checkpointing state are restored, so a failed guard leaves training byte-identical to
    G56CT_FAST_LORA=0."""
    import torch
    st = getattr(c["mp"], "_g56_fast", None)
    if st is None or st.checked:
        return None
    st.checked = True
    names = st.attached(c["adapters"])
    st.live = len(names)
    if not st.on or not names or not c["train"]:
        st.fail("guard: no attached fast-path module" if not names else "guard: switched off before the check")
        return False
    cfg, t0 = c["cfg"], time.time()
    g = torch.Generator(device="cpu").manual_seed(5656)
    item = c["train"][0]
    sigma = (cfg["bands"] // 2 + 0.5) / cfg["bands"]
    x0 = item["scaled"].to(device)
    eps = torch.randn(x0.shape, generator=g, dtype=torch.float32).to(device)
    xt, target = (1 - sigma) * x0 + sigma * eps, eps - x0
    plan = [(item["cond_idx"], 1.0 - cfg["empty_weight"]), (c["empty_idx"], cfg["empty_weight"])]
    ckpt0, saved, grads, outs, local = c["grad_ckpt_on"], {}, {}, {"fast": [], "old": []}, {}
    try:
        with torch.no_grad():
            for k in names:
                m, ad = st.mods[k]
                up, dn = ad.lora_up.weight, ad.lora_down.weight
                saved[k] = (up.detach().clone(), dn.detach().clone())
                d = torch.randn(dn.shape, generator=g, dtype=torch.float32).mul_(dn.shape[1] ** -0.5)
                u = torch.randn(up.shape, generator=g, dtype=torch.float32)
                nd = float(((u.t() @ u) * (d @ d.t())).sum().clamp_min(0.0).sqrt()) * float(ad.alpha) / ad.rank  # |sUD|
                nw = float(_fast_base_weight(m, torch.float32).norm())
                up.copy_(u.mul_(0.1 * nw / nd if nd > 0 and nw > 0 else 0.01).to(up.device, up.dtype))
                dn.copy_(d.to(dn.device, dn.dtype))
        if not ckpt0:
            set_checkpointing(c["mp"], True)
        for tag, on in (("fast", True), ("old", False)):
            st.on, n0 = on, st.calls
            st.probe = local if on else None
            for p in c["params"]:
                p.grad = None
            for idx, w in plan:
                if w > 0:
                    den = predict(model_wrap, conds, idx, xt, sigma, extra_args)
                    outs[tag].append(den.detach().float().clone())
                    (((xt - den.float()) / sigma - target).square().mean() * w).backward()
            st.probe = None
            if on and not st.on:  # a fast forward / backward raised (st.fail): the check cannot pass
                raise RuntimeError(f"fast path failed during the check ({st.off_reason})")
            if on and st.calls == n0:  # e.g. trainable base weights: nothing to compare, the knob would do nothing
                raise RuntimeError("no module took the fast path (trainable base weight / bias, extra weight functions?)")
            grads[tag] = {k: tuple(None if p.grad is None else p.grad.detach().clone()
                                   for p in (st.mods[k][1].lora_up.weight, st.mods[k][1].lora_down.weight)) for k in names}
        ok, info, st.stats = _fast_verdict(local, grads["fast"], grads["old"], exempt=st.bad)
        # diagnostics only (a GPU kernel may be non-deterministic): the fast forward should be the old one bit for bit
        same = all(torch.equal(a, b) for a, b in zip(outs["fast"], outs["old"]))
        drel = max(float((a - b).norm() / b.norm().clamp_min(1e-30)) for a, b in zip(outs["fast"], outs["old"]))
        info += f"; model output fast vs old bit-identical {same} (rel {drel:.1e})"
    except Exception as e:  # noqa: BLE001  (incl. OOM: training then starts on the old path and its OOM fallback)
        ok, info = False, f"{type(e).__name__}: {str(e)[:200]}"
        try:  # both passes run checkpointed: a raise inside a block must not leave its hooks behind
            _fast_release(e)
        except Exception as e2:  # noqa: BLE001  (the guard's contract: any exception -> FAST off; review A F3, 1 Oct)
            log(f"WARN fast lora: releasing the failed check's checkpoint state raised {type(e2).__name__}: {str(e2)[:160]}")
    finally:
        st.probe = None
        grads.clear()
        outs.clear()
        local.clear()
        for p in c["params"]:
            p.grad = None
        with torch.no_grad():
            for k, (u0, d0) in saved.items():
                st.mods[k][1].lora_up.weight.copy_(u0)
                st.mods[k][1].lora_down.weight.copy_(d0)
        if not ckpt0:
            set_checkpointing(c["mp"], False)
        if torch.cuda.is_available():  # the step log's "mem" (max allocated) then covers the training steps only
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    st.on = ok
    st.verdict = ("pass: " if ok else "FAIL: ") + info
    log(f"fast lora guard: {'PASS' if ok else 'FAIL -> old path for the whole run'} on {len(names)} modules "
        f"({st.calls} fast forwards): {info} ({time.time() - t0:.1f}s; mm out_dtype fp32 {_mm32_status()}"
        + (f"; per-module fallbacks {st.bad[:3]}" if st.bad else "") + ")")
    return ok


def _fast_release(e):
    """Release what an exception from a checkpointed forward still holds, before the step is redone. torch's
    non-reentrant checkpoint() leaves its generator suspended INSIDE the saved-tensor-hooks context when the checkpointed
    function raises; the hooks stay pushed until that frame is garbage collected, and ComfyUI's calc_cond_batch keeps the
    exception in a list its own frame holds (a reference cycle: freed only by the cyclic GC, at an arbitrary later point).
    Meanwhile every forward packs its saved tensors into the dead checkpoint frame and the next backward raises
    CheckpointError. So: close those generators now, innermost first (pops the hooks), then clear the dead frames of the
    tracebacks (drops the failed step's activations and the cycle). Returns the number of generators closed."""
    import inspect
    import traceback as tb_mod
    import torch.utils.checkpoint as tuc
    fname = getattr(tuc, "__file__", None)
    chain, gens, x = [], [], e
    while x is not None and all(x is not y for y in chain):
        chain.append(x)
        x = x.__cause__ or x.__context__
    for x in chain:
        tb = x.__traceback__
        while tb is not None:
            f = tb.tb_frame
            if f.f_code.co_filename == fname and f.f_code.co_name == "checkpoint":
                g = f.f_locals.get("gen")
                if inspect.isgenerator(g) and inspect.getgeneratorstate(g) == inspect.GEN_SUSPENDED \
                        and all(g is not h for h in gens):
                    gens.append(g)
            tb = tb.tb_next
    for g in reversed(gens):
        g.close()
    for x in chain:
        tb_mod.clear_frames(x.__traceback__)
    return len(gens)


def _fast_retry(c, e):
    """_run_one after a FastLoRAFallback: the fast path is off (st.fail); the step is redone on the old path once the
    failed step's checkpoint state is released (_fast_release). At most a few per run (the switch stays off), more means
    the old path itself raises -> re-raise."""
    st = getattr(c["mp"], "_g56_fast", None)
    if st is None or st.on:
        raise e
    try:
        n = _fast_release(e)
    except Exception as e2:  # noqa: BLE001  (must not replace the redo by a crash: review A F3, 1 Oct)
        n = "?"
        log(f"WARN fast lora: releasing the failed step's checkpoint state raised {type(e2).__name__}: {str(e2)[:160]}")
    st.retries += 1
    log(f"WARN fast lora: training step redone on the old path ({st.retries}; {n} checkpoint frame(s) released): {e}")
    if st.retries > 3:
        raise e


def train_loop(c, model_wrap, conds, extra_args, device):
    """K independent seeds (G56CT_SEEDS) share the wall budget; K=1 is a single run. With K>1 the per-run best
    checkpoints and their exact delta averages are scored on the evaluator path and the best candidate is kept."""
    import torch
    cfg = c["cfg"]
    K = max(1, int(cfg["seeds"]))
    if c.get("te"):
        c.setdefault("te_placeholder", conds[c["te_slots"][0]])
    if cfg["fast_lora"]:  # ours: the factored LoRA backward is checked against the old path before the first step
        fast_lora_guard(c, model_wrap, conds, extra_args, device)
    if cfg["spec_init"] and "spec_ranked" not in c:
        c["spec_ranked"] = spectral_down_init(c, model_wrap, conds, extra_args, device)
        apply_spectral(c, 0)
    if cfg.get("safe"):  # bytes of one member's state dict (the unit of the save margin, _save_margin)
        c["member_bytes"] = sum(p.numel() for p in c["params"]) * (4 if cfg["save_fp32"] else 2)
    if not c["holdout"] and cfg["seeds_auto"] > 1:  # adaptive all-data ensemble: members while time allows
        c["multi"] = True
        sds, runs, dur = [], [], []
        ens = _SafeEnsemble(c, cfg["final_scale"]) if cfg.get("safe") else None
        for k in range(cfg["seeds_auto"]):
            fill = 0
            extra = 0.0
            if ens is not None:  # G56CT_SAFE: members in the average before this one (its save margin, _save_margin)
                c["k_done"] = len(ens.sds)
                # the member stops max(72 s, save margin) before the deadline: plan the next member with the part of
                # the margin above the default 72 s (0 unless the disk is slow), so it is not started only to be cut
                extra = max(0.0, _save_margin(c) - 1.2 * c.get("eval_cost", 60.0))
            if k > 0:
                need = 1.10 * max(dur) + 30.0 + extra
                if time.time() + need > c["deadline"]:
                    fill = fill_member_steps(dur, [r_["steps"] for r_ in runs], c["deadline"] - time.time(),
                                             cfg["max_steps"], cfg["seeds_fill"], margin=30.0 + extra)
                    if not fill:
                        log(f"seeds-auto: stop after {k} member(s) (next needs ~{need:.0f}s, {c['deadline'] - time.time():.0f}s left)")
                        break
                    log(f"seeds-auto: last member {k + 1} on a shortened schedule: {fill} of {cfg['max_steps']} steps "
                        f"({c['deadline'] - time.time():.0f}s left)")
                reinit_adapters(c, cfg["seed"] + 1000 * k, k)
            t0m = time.time()
            full_cap = cfg["max_steps"]
            if fill:
                cfg["max_steps"] = fill
            try:
                r = _run_one(c, model_wrap, conds, extra_args, device, cfg["seed"] + 1000 * k, c["deadline"], k)
            finally:
                cfg["max_steps"] = full_cap
            dur.append(time.time() - t0m)
            runs.append(r)
            if ens is not None:  # G56CT_SAFE: finite, complete members only; running average on disk
                ens.add(r, k)
            elif r.get("best_sd") is not None:
                sds.append(r["best_sd"])
                save_lora(scaled_sd(average_loras(sds) if len(sds) > 1 else sds[0], cfg["final_scale"]), c["out_dir"])
            if cfg["save_runs"] and r.get("best_sd") is not None:
                from safetensors.torch import save_file
                save_file(r["best_sd"], str(side_dir(c) / f"run{k}.safetensors"))
            log(f"seeds-auto: member {k + 1} done in {dur[-1]:.0f}s ({r['steps']} steps)")
            if fill:
                break
        if ens is not None:
            sd, n_avg = ens.final(), len(ens.members())
        else:
            sd = scaled_sd(average_loras(sds) if len(sds) > 1 else sds[0], cfg["final_scale"])
            save_lora(sd, c["out_dir"])
            n_avg = len(sds)
        log(f"seeds-auto: shipped the exact average of {n_avg} member(s), scale {cfg['final_scale']:g}")
        return {"steps": sum(r["steps"] for r in runs), "evals": [], "best": {"score": None, "tag": f"auto_avg{n_avg}"},
                "best_sd": sd, "base": None, "step_time": sum(r["step_time"] for r in runs) / len(runs), "runs": len(runs)}
    if not c["holdout"] and K > 1:
        # all-data ensemble: K independent runs on every image (fixed step budget each, step-based schedule), final EMA of
        # each run averaged exactly (no holdout -> no selection; the average itself is the variance reduction)
        t0 = time.time()
        c["multi"] = True
        sds, runs = [], []
        ens = _SafeEnsemble(c, 1.0) if cfg.get("safe") else None
        for k in range(K):
            if k > 0:
                reinit_adapters(c, cfg["seed"] + 1000 * k, k)
            t_end = time.time() + max(60.0, (c["deadline"] - time.time()) / (K - k))
            log(f"all-data run {k + 1}/{K} until +{t_end - time.time():.0f}s")
            if ens is not None:
                c["k_done"] = len(ens.sds)
            r = _run_one(c, model_wrap, conds, extra_args, device, cfg["seed"] + 1000 * k, t_end, k)
            runs.append(r)
            if ens is not None:  # G56CT_SAFE: finite, complete members only; running average on disk
                ens.add(r, k)
            elif r.get("best_sd") is not None:
                sds.append(r["best_sd"])
                save_lora(average_loras(sds) if len(sds) > 1 else sds[0], c["out_dir"])  # crash-safe running average
            if cfg["save_runs"] and r.get("best_sd") is not None:  # research only: score each member against the
                from safetensors.torch import save_file            # average (paired ensemble gain)
                save_file(r["best_sd"], str(side_dir(c) / f"run{k}.safetensors"))
        if ens is not None:
            sd, n_avg = ens.final(), len(ens.members())
        else:
            sd = average_loras(sds) if len(sds) > 1 else sds[0]
            save_lora(sd, c["out_dir"])
            n_avg = len(sds)
        log(f"all-data ensemble: averaged {n_avg} runs")
        return {"steps": sum(r["steps"] for r in runs), "evals": [], "best": {"score": None, "tag": f"all_avg{n_avg}"},
                "best_sd": sd, "base": None, "step_time": sum(r["step_time"] for r in runs) / len(runs), "runs": len(runs)}
    if c["holdout"]:
        if c["exact"]:  # evaluator path for the base too: no weight wrappers, no patches
            import comfy.utils
            mods = [comfy.utils.get_attr(c["mp"].model, full) for full, a in c["adapters"].items()
                    if not full.startswith(TE_PREFIX) and not isinstance(a, SmallDelta)]
            wf = [list(m.weight_function) for m in mods]
            for m in mods:
                m.weight_function = []
            undo = small_disable(c)
            try:
                c["base_score"] = holdout_score(c, model_wrap, conds, extra_args, device)
            finally:
                small_restore(undo)
                for m, f in zip(mods, wf):
                    m.weight_function = f
        else:
            c["base_score"] = holdout_score(c, model_wrap, conds, extra_args, device)
        log(f"base holdout score {c['base_score']}")
    t0 = time.time()
    reserve = 0.0 if K == 1 else 3.0 * max(60.0, c.get("eval_cost", 60.0))
    c["multi"] = K > 1
    folds = bool(cfg["folds"]) and K > 1
    all_items = c["train"] + c["holdout"]
    h = len(c["holdout"])
    runs, fold_sets = [], []
    for k in range(K):
        if k > 0:
            reinit_adapters(c, cfg["seed"] + 1000 * k, k)
        if folds:  # fold k = a different block of the deterministic order (wraps around)
            idx = [(k * h + j) % len(all_items) for j in range(h)]
            c["holdout"] = [all_items[i] for i in idx]
            c["train"] = [it for i, it in enumerate(all_items) if i not in set(idx)]
            c["global_best"] = None if k == 0 else c.get("global_best")
            fold_sets.append(idx)
        remaining = c["deadline"] - reserve - time.time()
        t_end = time.time() + max(60.0, remaining / (K - k))  # early-stopped runs hand their time to later runs
        log(f"run {k + 1}/{K} until +{t_end - time.time():.0f}s" + (f" fold {idx}" if folds else ""))
        runs.append(_run_one(c, model_wrap, conds, extra_args, device, cfg["seed"] + 1000 * k, t_end, k))
    if K == 1:
        r0 = runs[0]
        if r0.get("best_sd") is not None and c["holdout"]:
            sc, tag, sd = scale_search(c, model_wrap, conds, extra_args, device,
                                       (r0["best"].get("score"), r0["best"].get("tag", "best"), r0["best_sd"]))
            if sd is not r0["best_sd"]:
                r0["best"]["score"], r0["best"]["tag"], r0["best_sd"] = sc, tag, sd
                save_lora(sd, c["out_dir"])
            if c.get("te"):
                sc, tag, sd = te_drop_check(c, model_wrap, conds, extra_args, device, (sc, tag, sd))
                if sd is not r0["best_sd"]:
                    r0["best"]["score"], r0["best"]["tag"], r0["best_sd"] = sc, tag, sd
                    save_lora(sd, c["out_dir"])
        return r0
    if folds:  # per-run scores come from different holdouts -> not comparable; submit the CV ensemble average
        sds = [r["best_sd"] for r in runs if r.get("best_sd") is not None]
        sd = average_loras(sds) if len(sds) > 1 else sds[0]
        save_lora(sd, c["out_dir"])
        log(f"folds: submitted average of {len(sds)} fold-bests (own-fold scores {[round(r['best']['score'], 6) for r in runs if r.get('best')]})")
        return {"steps": sum(r["steps"] for r in runs), "evals": sum((r["evals"] for r in runs), []),
                "best": {"score": None, "tag": f"cv_avg{len(sds)}"}, "best_sd": sd, "base": c.get("base_score"),
                "step_time": sum(r["step_time"] for r in runs) / len(runs), "runs": len(runs), "folds": fold_sets}
    cands = [(r["best"]["score"], f"run{k}:{r['best']['tag']}@{r['best']['step']}", r["best_sd"])
             for k, r in enumerate(runs) if r.get("best_sd") is not None]
    good = sorted(cands, key=lambda x: x[0])
    combos = []
    if len(good) >= 2:
        combos.append(("avg_all", [x[2] for x in good]))
    if len(good) >= 3:
        combos.append(("avg_top2", [x[2] for x in good[:2]]))
    combo_scores = []
    for name, sds in combos:
        if time.time() + 1.2 * c.get("eval_cost", 60.0) > c["deadline"] + reserve:
            log(f"skip combo {name}: no time")
            continue
        sd = average_loras(sds)
        r = score_sd_exact(c, model_wrap, conds, extra_args, device, sd)
        log(f"combo {name}: {r['score']:.6f} (text {r['text']:.6f} no_text {r['no_text']:.6f})")
        combo_scores.append((r["score"], name, sd))
    allc = good + combo_scores
    best = min(allc, key=lambda x: x[0])
    best = scale_search(c, model_wrap, conds, extra_args, device, best)
    if c.get("te"):
        best = te_drop_check(c, model_wrap, conds, extra_args, device, best)
    save_lora(best[2], c["out_dir"])
    log(f"picked {best[1]} score {best[0]:.6f} (single runs {[round(x[0], 6) for x in good]}, combos {[(n, round(s, 6)) for s, n, _ in combo_scores]})")
    return {"steps": sum(r["steps"] for r in runs), "evals": sum((r["evals"] for r in runs), []),
            "best": {"score": best[0], "tag": best[1]}, "best_sd": best[2], "base": c.get("base_score"),
            "step_time": sum(r["step_time"] for r in runs) / len(runs), "runs": len(runs),
            "combos": [(n, s) for s, n, _ in combo_scores]}


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def _anchor_start(raw, t0, safe):
    """Wall-clock anchor of the time budget -> (t_start, warning or None). raw = G56CT_START_TS (container start, when an
    entrypoint exports it), t0 = the trainer's start. Default: float(raw), else t0 (a malformed value raises).
    G56CT_SAFE: raw only when it parses and lies in [t0 - 600 s, t0], so the deadline can only move earlier; a future,
    stale or malformed value -> t0 with a warning (it used to crash main() before the fallback file was written)."""
    if not safe:
        return float(raw if raw is not None else t0), None
    if raw is None or raw.strip() == "":
        return t0, None
    try:
        v = float(raw)
    except ValueError:
        return t0, f"G56CT_START_TS {raw[:40]!r} is not a number"
    if not (math.isfinite(v) and t0 - 600.0 <= v <= t0):
        return t0, f"G56CT_START_TS {raw[:40]!r} is not within the 600 s before the trainer start"
    return min(t0, v), None


def main():
    args = parse_args()
    fam = args.model_type
    if CFG["te_lora"] and not FAMILIES[fam].get("te_taps"):
        raise RuntimeError(f"G56CT_TE_LORA=1 is not supported for {fam} (text-encoder LoRA: ideogram4, krea2)")
    hours = float(args.hours_to_complete)
    out_dir = Path(args.out_dir) if args.out_dir else CKPT_ROOT / args.task_id / args.expected_repo_name
    global _OUT_DIR
    _OUT_DIR = out_dir
    log("comfy trainer start", public=True)
    # the checkpoints volume is shared across attempts of the same task: never let a stale artifact from a crashed
    # earlier attempt be uploaded (scope: this task's output dir only)
    try:
        if out_dir.exists():
            for f in out_dir.iterdir():
                if f.is_file():
                    f.unlink()
    except Exception as e:  # noqa: BLE001
        log(f"stale-output clean skipped: {e}")
    fam_applied = apply_family_defaults(fam)  # production recipe per family (env vars still override every key)
    # the deadline after the family defaults (wall_frac / reserve_s / safe may be among them)
    t_start, why = _anchor_start(os.environ.get("G56CT_START_TS"), T0, CFG["safe"])
    if why:
        log(f"WARN safe: {why}; the budget is anchored at the trainer start", public=True)
    deadline = t_start + hours * 3600 * CFG["wall_frac"] - CFG["reserve_s"]
    try:  # the insurance must never end the run before the fallback adapter is written (review 30 Sep)
        _start_watchdog(t_start, hours, out_dir)  # G56CT_WATCHDOG_S (0: nothing)
    except Exception as e:  # noqa: BLE001
        log(f"WARN watchdog: not armed ({e})", public=True)
    if fam_applied:
        log(f"family defaults ({fam}): {json.dumps(fam_applied, default=str)}")
    log(f"start fam={fam} model={args.model} hours={hours} deadline_in={deadline - time.time():.0f}s out={out_dir}")
    log("cfg " + json.dumps(CFG))

    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np  # noqa: F401
    import torch
    import comfy.model_management as mm
    import comfy.samplers
    import comfy_extras.nodes_custom_sampler as ncs
    import nodes
    import node_helpers
    from PIL import Image
    from evalcore import adjust_image

    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    device = mm.get_torch_device()
    # Production scoring runs on A100s (no fp8 compute): fp8 checkpoints are evaluated with bf16 matmuls and no fp8
    # activation quantization. Mirror that on any GPU unless G56CT_FP8_COMPUTE=1.
    if os.environ.get("G56CT_FP8_COMPUTE", "0") != "1":
        mm.supports_fp8_compute = lambda device=None: False
        log("fp8 compute disabled (A100 evaluator numerics)")
    spec = FAMILIES[fam]
    if fam == "flux":
        log(f"flux: {patch_flux_inplace()} in-place residual adds rewritten out of place for autograd")

    # ---- data (evaluator preprocessing)
    rows = read_rows(dataset_zip_path(args))
    items = decode_items(rows)
    n = len(items)
    big_n = apply_big_n_defaults(fam, n)  # task-size conditional family defaults (before the step cap, the LoRA and the optimizer)
    if big_n:
        log(f"task-size defaults (N={n} >= {FAMILY_DEFAULTS_BIG_N[fam][0]}): {big_n}", public=True)
    n_hold = 0 if (n < 6 or CFG["holdout_frac"] <= 0) else max(CFG["holdout_min"], int(round(n * CFG["holdout_frac"])))
    order = sorted(range(n), key=lambda i: hashlib.sha256(f"{CFG['seed']}:{items[i]['sha']}".encode()).hexdigest())
    hold_ids = set(order[:n_hold])
    log(f"dataset: {n} images, holdout {n_hold}, sizes {sorted(set(it['image'].size for it in items))[:6]}")
    if args.trigger_word:
        log(f"trigger word given ({args.trigger_word!r}); captions kept verbatim like the evaluator")

    # ---- base model (CPU until training) -> latents need its latent_format
    mp = load_base_model(fam, args.model, device)

    # ---- VAE latents
    vae = load_vae(fam, args.model)
    for it in items:
        lat = encode_latent(vae, it["image"], device)
        it["scaled"] = mp.model.process_latent_in(lat.clone()).float().cpu()
        it["image"] = None
    del vae
    gc.collect()
    mm.soft_empty_cache()
    log(f"latents encoded, e.g. {tuple(items[0]['scaled'].shape)}")

    # ---- conditioning (evaluator TE, CLIPTextEncode)
    clip, te_rep = load_clip(fam, device, args.model)
    prompts = sorted(set(it["caption"] for it in items))
    conds, pidx = [], {}
    for p in prompts + [""]:
        cnd = nodes.CLIPTextEncode().encode(clip, p)[0]
        if len(cnd) != 1:
            raise RuntimeError(f"expected one conditioning entry per prompt, got {len(cnd)}")
        if spec["guidance"] is not None:
            cnd = node_helpers.conditioning_set_values(cnd, {"guidance": spec["guidance"]})
        pidx[p] = len(conds)
        conds.extend(cnd)
    te, te_slots, te_params = None, None, []
    if CFG["te_lora"]:  # ours: the TE stays (trained); two scratch slots for the per-step TE conditioning (caption, "")
        te_slots = [len(conds), len(conds) + 1]
        conds.extend([conds[pidx[""]], conds[pidx[""]]])
        te = {"clip": clip, "tokens": {}, "guidance": spec["guidance"]}
    else:
        del clip
    gc.collect()
    mm.unload_all_models()
    mm.soft_empty_cache()
    log(f"conditioning: {len(prompts)} captions + empty")

    for it in items:
        it["cond_idx"] = pidx[it["caption"]]
    train = [it for i, it in enumerate(items) if i not in hold_ids]
    hold = [it for i, it in enumerate(items) if i in hold_ids]
    if CFG["steps_per_image"] > 0 and not CFG["max_steps"]:  # task-size step cap (holdout optimum ~ k x N, flat around it)
        CFG["max_steps"] = max(50, int(round(CFG["steps_per_image"] * len(train))))
        log(f"step cap {CFG['max_steps']} = {CFG['steps_per_image']:g} x {len(train)} training images")

    # ---- LoRA
    mp.model.requires_grad_(False)
    include = CFG["include"] or spec["include"]
    if CFG["include_extra"]:
        include = f"(?:{include})|(?:{CFG['include_extra']})"
    if CFG["include"]:
        log(f"LoRA module regex overridden: {include}")
    adapters, params = attach_lora(mp, include, CFG["rank"], CFG["alpha"])
    n_dit = len(adapters)
    small_params = []
    if CFG["small"]:  # ours: DiffFit/BitFit-style small deltas (after the LoRA: its init draws stay identical)
        rx_small = SMALL_DEFAULT.get(fam, "") if CFG["small"] == "auto" else CFG["small"]
        smalls, small_params = attach_small(mp, rx_small, CFG["small_max_k"])
        adapters.update(smalls)
        params = params + small_params
        CFG["small_lr"] = CFG["small_lr"] if CFG["small_lr"] > 0 else CFG["lr"]
        log(f"small deltas: {len(smalls)} params ({sum(p.numel() for p in small_params) / 1e3:.1f}K values), lr "
            f"{CFG['small_lr']:.2e}, regex {rx_small}")
    if te is not None:  # after the DiT adapters: their init draws stay identical to a TE-off run
        clip.cond_stage_model.requires_grad_(False)
        te["adapters"], te["wrappers"], te_info = attach_te_lora(clip, fam, CFG)
        adapters.update(te["adapters"])
        log(f"TE LoRA r{te_info['rank']} a{te_info['alpha']:g} init {CFG['te_init']} lr {CFG['te_lr']:.2e}: "
            f"{len(te['adapters'])} modules, TE layers {te_info['layers']} of {te_info['n_layers']} (taps < {te_info['tap_cut']}), "
            f"{sum(a.lora_up.weight.numel() + a.lora_down.weight.numel() for a in te['adapters'].values()) / 1e6:.1f}M params")
        te_rep = dict(te_rep, lora=te_info)
    dropped = drop_unloadable(mp, adapters, clip if te is not None else None, te)  # before the identity LoRA is written
    if dropped:
        live = {id(p) for a in adapters.values() for p in adapter_params(a)}
        params = [p for p in params if id(p) in live]
        small_params = [p for p in small_params if id(p) in live]
        n_dit = sum(1 for k, a in adapters.items() if not k.startswith(TE_PREFIX) and not isinstance(a, SmallDelta))
    log(f"key check: {len(adapters)} adapters load in the evaluator (0 missing), dropped {len(dropped)}")
    log(f"LoRA r{CFG['rank']} a{CFG['alpha']}: {n_dit} modules, {sum(p.numel() for p in params) / 1e6:.1f}M params")
    if CFG["fast_lora"]:
        fst = getattr(mp, "_g56_fast", None)
        n_fast = len(fst.attached(adapters)) if fst else 0  # adapters dropped by the key check no longer count
        n_lora = sum(1 for k, a in adapters.items()
                     if not k.startswith(TE_PREFIX) and not isinstance(a, (SmallDelta, DirectDelta)))
        log(f"fast lora: factored backward on {n_fast} of {n_lora} DiT LoRA modules (W' not kept for backward; guard "
            f"before step 1)" + (f", old path for op classes of {fst.skipped[:3]}" if fst and fst.skipped else ""))
    if CFG["fallback"]:  # identity delta (up or down is zero at init): loadable, scores as the base model - never a forfeit
        save_lora(lora_state_dict(adapters, fp32=CFG["save_fp32"]), out_dir)
        log("fallback: identity LoRA written before training")
    gc_on = CFG["grad_ckpt"] == "on"
    if gc_on:
        log(f"gradient checkpointing on {set_checkpointing(mp, True)} blocks")

    mm.load_models_gpu([mp], memory_required=1e20, force_full_load=True)
    for a in adapters.values():  # plain-parameter small deltas are not weight functions: follow their module's device
        if isinstance(a, SmallDelta) and not a.via_fn:
            a.move_to(a.module._parameters[a.pname].device)
    if te is not None:  # outside model management (nothing may unload or re-patch it while training)
        te_to_device(te, device)
        te_params = [p for a in te["adapters"].values() for p in (a.lora_up.weight, a.lora_down.weight)]
        n_ck = set_te_checkpointing(te_llama(clip)[2], True) if CFG["te_grad_ckpt"] else 0
        log(f"TE language model on {device}, {n_ck} checkpointed blocks"
            + (f", allocated {torch.cuda.memory_allocated() / 2**30:.1f}GiB" if torch.cuda.is_available() else ""))
    exact = CFG["exact_holdout"] == "on" or (CFG["exact_holdout"] == "auto" and fam in ("ideogram4", "qwen-image"))
    log(f"exact holdout (evaluator-merged weights): {exact}" + (" (TE part: always)" if te is not None else ""))
    ctx = {"exact": exact, "cfg": CFG, "params": params, "adapters": adapters, "train": train, "holdout": hold,
           "small_params": small_params,
           "conds": conds,
           "empty_idx": pidx[""], "deadline": deadline, "out_dir": out_dir, "mp": mp, "grad_ckpt_on": gc_on}
    if te is not None:
        ctx.update(te=te, te_params=te_params, te_slots=te_slots, params=params + te_params)

    from comfy_extras.nodes_train import TrainGuider
    guider = TrainGuider(mp, offloading=False)
    guider.set_conds(conds)
    loop = Loop(ctx)
    lat0 = train[0]["scaled"]
    mm.in_training = True
    try:
        guider.sample(torch.zeros_like(lat0), torch.zeros_like(lat0), loop, torch.tensor([1.0, 0.0]),
                      disable_pbar=True, seed=CFG["seed"])
    finally:
        mm.in_training = False
    res = loop.result
    if res is None or res.get("best_sd") is None:
        raise RuntimeError("training produced no checkpoint")
    if te is not None:  # the evaluator's apply_lora on model AND clip
        n_patch, n_clip, missing = verify_lora_loadable_te(mp, clip, res["best_sd"])
        log(f"TE LoRA in the file: {n_clip} clip patches" + (" (TE keys dropped by the drop check)" if not n_clip else ""))
    else:
        n_patch, missing = verify_lora_loadable(mp, res["best_sd"])
    log(f"final: best {res['best']} base {res['base']} steps {res['steps']} avg step {res['step_time']:.2f}s; "
        f"loadable patches {n_patch} missing {len(missing)}")
    if missing:
        raise RuntimeError(f"LoRA would be rejected by the evaluator: {missing[:3]}")
    summary = {k: v for k, v in res.items() if k != "best_sd"}
    summary.update(te=te_rep, cfg=CFG, fam=fam)
    if CFG["fast_lora"]:
        summary["fast_lora"] = mp._g56_fast.report() if getattr(mp, "_g56_fast", None) else None
    sp = Path(os.environ.get("G56CT_SUMMARY", "/tmp/g56ct_summary.json"))
    try:
        sp.write_text(json.dumps(summary, default=str, indent=1))
    except Exception:
        pass
    log(f"done in {time.time() - T0:.0f}s -> {out_dir / 'last.safetensors'}")
    log("comfy trainer finished", public=True)


def usable_adapter(out_dir):
    """True when out_dir/last.safetensors is a complete safetensors file with at least one tensor, all floating tensors
    finite (saves are atomic: fallback identity LoRA, interim EMA saves, ensemble running averages, new holdout bests)."""
    try:
        f = Path(out_dir) / "last.safetensors"
        if not f.is_file() or f.stat().st_size == 0:
            return False
        import torch
        from safetensors import safe_open
        with safe_open(str(f), "pt") as sf:
            keys = list(sf.keys())
            if not keys:
                return False
            for k in keys:
                t = sf.get_tensor(k)
                if t.is_floating_point() and not bool(torch.isfinite(t).all()):
                    return False
        return True
    except Exception:  # noqa: BLE001
        return False


_MAIN_DONE = []  # G56CT_WATCHDOG_S: [threading.Event] set by cli() once main() returned or raised


def _main_done():
    import threading
    if not _MAIN_DONE:
        _MAIN_DONE.append(threading.Event())
    return _MAIN_DONE[0]


def _start_watchdog(t_start, hours, out_dir):
    """G56CT_WATCHDOG_S > 0: a daemon thread ends the process with exit 0 WATCHDOG_S seconds before the validator's kill
    (t_start + int(hours x 3600): its container timeout) when main() is still running and out_dir holds a usable adapter
    (the partial .last.safetensors.tmp of an interrupted save is removed first). The validator uploads the folder after
    a clean exit as it does after its kill; the stop only makes the end clean when a run overruns (a stuck step, a slow
    save) and never touches the training state (saves are atomic: the file on disk is the last complete one). No usable
    adapter at that time: it logs and leaves the end to the validator (a save may still complete). A normal run ends
    >= 0.03 x the budget + reserve_s before the kill, i.e. long before the stop. Returns the stop time (None: off)."""
    ws = CFG["watchdog_s"]
    if ws <= 0:
        return None
    import threading
    t_fire = t_start + max(1, int(hours * 3600)) - ws
    done = _main_done()

    def watch():
        while not done.is_set() and time.time() < t_fire:
            done.wait(min(5.0, max(0.0, t_fire - time.time())))
        if done.is_set():
            return
        if usable_adapter(out_dir):
            try:
                (Path(out_dir) / ".last.safetensors.tmp").unlink()
            except FileNotFoundError:
                pass
            except Exception as e:  # noqa: BLE001
                log(f"watchdog: could not remove the partial save: {e}")
            log(f"watchdog: hard stop before the validator's kill ({ws:g}s left): the saved adapter is shipped, exit 0",
                public=True)
            os._exit(0)
        log("watchdog: no usable adapter at the hard stop; the validator's kill decides", public=True)

    threading.Thread(target=watch, name="g56-watchdog", daemon=True).start()
    log(f"watchdog: armed, stop at +{t_fire - time.time():.0f}s ({ws:g}s before the validator's kill)")
    return t_fire


def cli(main_fn=None):
    """Exit-code policy. The validator treats a container exit != 0 as a failed task WITHOUT upload, but uploads the
    checkpoint folder when it kills the container at the deadline. So: 0 after a normal run; after a crash 0 if a usable
    adapter is already in the upload folder (it scores; a failed task ranks behind every finished miner), else 1."""
    try:
        (main_fn or main)()
        return 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code != 0 and _OUT_DIR is not None and usable_adapter(_OUT_DIR):
            log(f"exit {code} after a usable adapter was saved -> exit 0", public=True)
            return 0
        return code
    except BaseException as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        print(f"comfy trainer failed ({type(e).__name__})" if not VERBOSE else tb, flush=True)
        if _OUT_DIR is not None and usable_adapter(_OUT_DIR):
            log("crash after a usable adapter was saved -> exit 0 (the validator uploads the checkpoint folder)", public=True)
            return 0
        return 1
    finally:
        if _MAIN_DONE:  # G56CT_WATCHDOG_S: main() is over; the watchdog stays quiet
            _MAIN_DONE[0].set()


if __name__ == "__main__":
    sys.exit(cli())
