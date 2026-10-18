#!/usr/bin/env python3
"""Is our offline-rebuilt text encoder identical to the evaluator's? Loads the evaluator's Comfy-Org TE file exactly like
CLIPLoader does, and ours (rebuilt from the /cache bf16 snapshot + baked scale table, and optionally the bf16 snapshot as-is),
encodes the dataset captions + the empty prompt with CLIPTextEncode and compares every conditioning tensor bit-for-bit.

Run inside the trainer image's comfy venv:  python te_check.py --family ideogram4 --comfy-te /x/qwen3vl_8b_fp8_scaled.safetensors --dataset /cache/datasets/<t>_tourn.zip [--n 8]
"""
import argparse, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))


def flat(c):
    out = {"cond": c[0][0]}
    for k, v in c[0][1].items():
        if hasattr(v, "shape"):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--comfy-te", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    import torch
    import comfy.sd, comfy.model_management as mm, nodes
    import g56_comfy_train as T
    from evalcore import read_dataset
    rows = read_dataset(a.dataset, Path("/tmp/_tc.zip"))
    prompts = [cap for _, _, cap in rows if cap and cap.strip()][: a.n] + [""]
    spec = T.FAMILIES[a.family]
    ct = getattr(comfy.sd.CLIPType, spec["clip_type"])

    def encode_all(clip):
        return [flat(nodes.CLIPTextEncode().encode(clip, p)[0]) for p in prompts]

    t = time.time(); ev = comfy.sd.load_clip(ckpt_paths=[a.comfy_te], embedding_directory=None, clip_type=ct, model_options={})
    E = encode_all(ev); del ev; mm.unload_all_models(); print(f"[te_check] evaluator TE encoded {len(E)} prompts ({time.time() - t:.0f}s)", flush=True)
    results = {}
    for src in ("rebuild", "bf16"):
        T.CFG["te_source"] = src
        t = time.time(); clip, rep = T.load_clip(a.family, mm.get_torch_device())
        O = encode_all(clip); del clip; mm.unload_all_models()
        same = sum(all(torch.equal(e[k].cpu(), o[k].cpu()) for k in e) and set(e) == set(o) for e, o in zip(E, O))
        md = max(float((e["cond"].float().cpu() - o["cond"].float().cpu()).abs().max()) for e, o in zip(E, O))
        rel = max(float((e["cond"].float().cpu() - o["cond"].float().cpu()).norm() / (e["cond"].float().cpu().norm() + 1e-12)) for e, o in zip(E, O))
        results[src] = (same, md, rel)
        print(f"[te_check] {a.family} TE={src}: bit-identical {same}/{len(E)}  max|diff| {md:.3e}  max rel L2 {rel:.3e}  "
              f"fingerprint_ok={rep.get('fingerprint_ok')} fp8_layers={rep.get('fp8_layers')} ({time.time() - t:.0f}s)", flush=True)
    ok = results["rebuild"][0] == len(E)
    print(f"[te_check] VERDICT {a.family}: {'EXACT' if ok else 'NOT EXACT'}", flush=True)


if __name__ == "__main__":
    main()
