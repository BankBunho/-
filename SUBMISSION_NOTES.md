# Submission notes (for reviewers)

**Live training path:** both validator Dockerfiles (`ops/docker/standalone-image-trainer.dockerfile` for flux,
`ops/docker/standalone-image-toolkit-trainer.dockerfile` for every other image type) build one lean image whose
ENTRYPOINT is `comfytrain/g56_comfy_train.py`. Only `comfytrain/` and `ops/docker/bake_comfy_assets.py` are copied into
the image; the repository holds nothing else besides the two Dockerfiles and these notes.

## What the trainer does (all at training time, in `comfytrain/g56_comfy_train.py`)

1. **Trains on the evaluator's own stack.** Same Python/torch/diffusers/transformers pins and the same ComfyUI commit
   (694815f4) as the validator's diffusion evaluator. The base model is loaded through ComfyUI and trained through a
   ComfyUI guider with ComfyUI LoRA weight adapters, so the prediction path, the text conditioning (`CLIPTextEncode`)
   and the saved LoRA key format are exactly the ones the evaluator loads.
2. **Objective = the evaluator's metric.** Rectified-flow velocity MSE at the evaluator's 16 sigma-stratum midpoints,
   walked (image, stratum) pair by pair; each step trains the caption half and the empty-prompt half with the evaluator's
   50/50 weights (z-image alternates the two halves step by step).
3. **Per-family recipe** (`FAMILY_DEFAULTS`, measured on real tournament tasks with the evaluator's quantity):
   - ideogram4, krea2, flux, z-image: all training images, a step cap per image, and an ensemble of as many independent
     members (different seeds) as the budget fits, exactly averaged into one LoRA (`average_loras`: rank = sum of member
     ranks, delta = mean of member deltas); rank 64 (flux: rank 16, 55 steps per image; z-image: 100 steps per image).
     krea2 also adapts the embedding / text-projection modules (the module set of the publicly released champion trainer).
     When a member's step cap does not fit the budget, its LR schedule decays by time and still reaches its floor.
   - qwen-image: LoRA on the full module set (attention incl. the text-stream projections + image/text MLPs), rank 32;
     a held-out split carved from the provided training images picks the best step (raw or EMA weights)
     with the same quantity, scored on the evaluator-merged weights (the merge ComfyUI performs when the evaluator
     applies the LoRA).
   - LoRA init: down ~ N(0, 1/in), up = 0.
4. **Evaluator numerics.** fp8 compute is disabled during training (production scoring runs on A100-class GPUs without
   fp8 compute).
5. **Robustness of the container contract.**
   - The trainer stops at 97% of the budget minus 120 s.
   - Before training, a key check runs the evaluator's LoRA loader on the attached adapters and drops any module it would
     not load.
   - Then an identity LoRA is written, so there is never an empty upload.
   - Every later save is atomic (interim EMA saves, ensemble running averages, new held-out bests).
   - The exit code is 0 whenever a usable adapter is in the upload folder.
   - There is exactly one `last.safetensors`.

## Baked assets (build time, public files, sha256-pinned)

- The evaluator's own VAE files: `flux2-vae` (ideogram4), `qwen_image_vae` (krea2, qwen-image) and `ae` (flux). The
  training container is offline, and these files are identical to what the evaluator loads. The evaluator loads the
  latest revision of these files, so if upstream replaces one, the build uses the new file only when it is a complete
  safetensors file with the pinned tensor structure (names, shapes, dtypes), with a WARNING in the build log; anything
  else still fails the build.
- Flux image only: the evaluator's own flux text encoders (`comfyanonymous/flux_text_encoders` `clip_l` +
  `t5xxl_fp16`, pinned revision). Flux conditioning is therefore the evaluator's `DualCLIPLoader` output.
- `comfytrain/assets/*.json`: per-tensor fp8 scale tables read from public Comfy-Org fp8_scaled files
  (`comfytrain/tools/extract_quant_tables.py`). They contain scales only, no weights, and let the trainer reproduce
  the evaluator's exact fp8 weights offline.

## Integrity declaration

- Training reads only the validator-provided dataset zip and base model from `/cache`, plus the evaluator's public
  VAE / text-encoder files listed above. There are no external datasets, no pretrained or bundled adapters, no warm
  starts from earlier LoRAs, and no images from other tasks. The research-only options for those were removed from this
  tree.
- The held-out split is carved from the training images only. No evaluator test data or score is read or manipulated.
- The container is offline at runtime (`HF_HUB_OFFLINE=1`). `comfytrain/tools/` holds CPU self-checks that the trainer
  never runs. Host-side benchmarking tooling is kept out of this tree.
