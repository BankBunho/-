#!/usr/bin/env python3
"""Extract the public per-tensor fp8 scale table (+ comfy_quant configs and full key/dtype/shape list) of a Comfy-Org
fp8_scaled safetensors file via HTTP range requests (no full download). Output JSON is baked into the trainer image so that,
offline at training time, the evaluator's exact fp8 weights can be rebuilt from the bf16 weights in /cache
(q = RTNE(W / scale), verified byte-identical on Qwen3-VL-8B layer 0 q_proj).

Usage: extract_quant_tables.py <repo> <file> <out.json>
"""
import http.client, json, struct, sys, time, urllib.parse, urllib.request

H = {"User-Agent": "g56-research"}


def main():
    repo, fn, out = sys.argv[1:4]
    u = f"https://huggingface.co/{repo}/resolve/main/{fn}"
    r = urllib.request.urlopen(urllib.request.Request(u, headers={**H, "Range": "bytes=0-7"}), timeout=60)
    n = struct.unpack("<Q", r.read(8))[0]
    final = r.geturl()
    hdr = json.loads(urllib.request.urlopen(urllib.request.Request(final, headers={**H, "Range": f"bytes=8-{7 + n}"}), timeout=120).read())
    meta = hdr.pop("__metadata__", {})
    base = 8 + n
    pu = urllib.parse.urlsplit(final)
    conn = http.client.HTTPSConnection(pu.netloc, timeout=60)
    path = pu.path + ("?" + pu.query if pu.query else "")

    def rng(a, b):
        nonlocal conn
        for attempt in range(5):
            try:
                conn.request("GET", path, headers={**H, "Range": f"bytes={base + a}-{base + b - 1}"})
                resp = conn.getresponse(); data = resp.read()
                if resp.status in (200, 206) and len(data) == b - a:
                    return data
                raise IOError(f"status {resp.status} len {len(data)}")
            except Exception as e:
                time.sleep(1 + attempt); conn.close(); conn = http.client.HTTPSConnection(pu.netloc, timeout=60)
        raise SystemExit(f"range {a}-{b} failed")

    small = sorted((v["data_offsets"][0], v["data_offsets"][1], k, v["dtype"]) for k, v in hdr.items()
                   if k.endswith((".weight_scale", ".input_scale", ".comfy_quant", ".scale_weight", ".scale_input")))
    scales, quant, other = {}, {}, {}
    for i, (a, b, k, d) in enumerate(small):
        raw = rng(a, b)
        if k.endswith(".comfy_quant"):
            quant[k[: -len(".comfy_quant")]] = raw.decode()
        elif d == "F32" and b - a == 4:
            (scales if k.endswith(("weight_scale", "scale_weight")) else other)[k] = struct.unpack("<f", raw)[0]
        else:
            other[k] = {"dtype": d, "bytes": raw.hex()[:64], "n": b - a}
        if i % 50 == 0:
            print(f"{i}/{len(small)}", flush=True)
    keys = {k: [v["dtype"], v["shape"]] for k, v in hdr.items()}
    json.dump({"repo": repo, "file": fn, "metadata": meta, "keys": keys, "weight_scale": scales,
               "comfy_quant": quant, "other_small": other}, open(out, "w"))
    print(f"wrote {out}: {len(keys)} keys, {len(scales)} weight scales, {len(quant)} comfy_quant, {len(other)} other")


if __name__ == "__main__":
    main()
