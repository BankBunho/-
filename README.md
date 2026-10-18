# g56 image trainer (Bittensor SN56 / Gradients image tournament)

LoRA trainer for `ImageTask` tournaments (flux, ideogram4, z-image, krea2, qwen-image). It trains on the validator's own
evaluation stack (same ComfyUI commit and library pins as the diffusion evaluator) and optimizes the evaluator's metric,
the rectified-flow velocity MSE. `SUBMISSION_NOTES.md` has the details for reviewers.

## Layout

- `comfytrain/g56_comfy_train.py` — the trainer (container ENTRYPOINT).
- `comfytrain/evalcore.py` — the evaluator's dataset reader / image preprocessing.
- `comfytrain/assets/` — public fp8 scale tables (scales only).
- `comfytrain/tools/` — CPU self-checks (`exit_check`, `keycheck_check`, `modeldir_check`, `flux_layout_check`,
  `small_check`, `knobs_check`, ...). The trainer never runs them.
- `ops/docker/standalone-image-trainer.dockerfile` (flux) and `ops/docker/standalone-image-toolkit-trainer.dockerfile`
  (every other image type) build the same lean image; the flux one also bakes the evaluator's flux text encoders.
- `ops/docker/bake_comfy_assets.py` bakes the evaluator's public VAE (and flux text-encoder) files, sha256-pinned.

## Container contract

```
python comfytrain/g56_comfy_train.py --task-id <id> --model <repo> --dataset-zip <zip> --model-type <family> \
    --expected-repo-name <name> --hours-to-complete <h> [--trigger-word <w>]
```

The base model and dataset are read from the validator-staged `/cache`. The container runs offline.

Output: `/app/checkpoints/<task-id>/<expected-repo-name>/last.safetensors` (exactly one file). It is written atomically
and early (an identity LoRA first), so the folder always holds a loadable LoRA. The exit code is 0 whenever a usable
adapter exists.

## License

Apache-2.0 (`LICENSE.md`, `NOTICE`).
