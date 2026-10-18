#!/usr/bin/env python3
"""Docker-build step: download the evaluator's own VAE files (public Comfy-Org mirrors) with retries and sha256
verification. The training container is offline, so every file the ComfyUI-native trainer needs beyond /cache is baked here.
Fails the build loudly on any mismatch (a silent wrong VAE would poison every latent) - except a VAE that upstream replaced:
the evaluator loads the latest resolve/main revision of the same file, so a complete download with the pinned structure
(tensor names, shapes, dtypes) is used with a loud WARNING instead of failing the build of every task. The flux text
encoders are pinned by revision in the URL: sha256 only."""
import hashlib, json, os, sys, time, urllib.request

# (url, name, pinned sha256, structure digest of that file - see structure_digest; None = sha256 only)
FILES = [
    ("https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/vae/flux2-vae.safetensors", "flux2-vae.safetensors",
     "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
     "81ac8ce2b18daf20e8f61c781521b8c5f9c3dc7ae3a74dbc34ee9678fd32e12d"),
    ("https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors", "qwen_image_vae.safetensors",
     "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
     "238f0fb58279311cce78fc22c7d88cdd63847c4427825da4246562e7f7ec1e51"),
    ("https://huggingface.co/rayonlabs/FLUX.1-dev/resolve/main/ae.safetensors", "flux_ae.safetensors",
     "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
     "56c1fd9cbaa193c7b8cb5622efcad2956e51b7919ae62f686465685b403b8c82"),
]
FLUX_TE_FILES = [  # the evaluator's own flux text encoders (comfyanonymous/flux_text_encoders, sha256 == main)
    ("https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/clip_l.safetensors",
     "@flux/clip_l.safetensors", "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd", None),
    ("https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp16.safetensors",
     "@flux/t5xxl_fp16.safetensors", "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635", None),
]
DTYPE_BYTES = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
               "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def header_tensors(path):
    """{name: (shape, dtype)} of a complete safetensors file. Raises on what safetensors itself refuses to load: a header
    that does not parse, data offsets that do not tile the data section at each tensor's exact byte size, and a file size
    != 8 + header length + data end (a truncated download, trailing bytes)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if size < 8 or n > min(size - 8, 100_000_000):
            raise ValueError(f"header length {n} in a {size}-byte file")
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    out, end = {}, 0
    for k, v in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"]):
        shape, dtype, (b, e) = v["shape"], v["dtype"], v["data_offsets"]
        if not all(type(x) is int and x >= 0 for x in [*shape, b, e]):
            raise ValueError(f"{k}: shape {shape} / data_offsets {b}..{e} not non-negative ints")
        numel = 1
        for d in shape:
            numel *= d
        if b != end or e - b != numel * DTYPE_BYTES[dtype]:
            raise ValueError(f"{k}: data_offsets {b}..{e} do not tile the data ({numel} x {dtype} from {end})")
        out[k], end = (tuple(shape), dtype), e
    if size != 8 + n + end:
        raise ValueError(f"file size {size} != 8 + header {n} + data {end}")
    return out


def structure_digest(tensors):
    """sha256 of the sorted "name|d0,d1,...|dtype" lines (no values, no order): the digests pinned in FILES."""
    lines = sorted(f"{k}|{','.join(map(str, s))}|{t}" for k, (s, t) in tensors.items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def structure_problem(path, digest):
    """None if path is a complete safetensors file with exactly the pinned structure, else the reason."""
    try:
        got = structure_digest(header_tensors(path))
    except Exception as e:  # truncated download, broken header: never accepted
        return f"not a complete safetensors file ({type(e).__name__}: {str(e)[:100]})"
    return None if got == digest else f"tensor names/shapes/dtypes differ from the pinned file (structure {got[:16]})"


def fetch(url, name, want, path, digest=None, tries=6, wait=5):
    """Download url to path via path.part: True once the sha256 is the pinned one or, for a file with a structure digest,
    once a complete download has the pinned structure (upstream replaced it: accepted at once with a WARNING). Network
    errors and other mismatches retry; False after `tries` failed attempts."""
    for attempt in range(tries):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "g56-build"}), timeout=120) as r, open(path + ".part", "wb") as f:
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    h.update(b); f.write(b)
            got = h.hexdigest()
            if got != want:
                why = structure_problem(path + ".part", digest) if digest else "sha256 only"
                if why:
                    if digest:
                        print(f"[bake] {name}: sha256 not pinned and structure not verified: {why}", flush=True)
                    raise IOError(f"sha256 mismatch for {name}: {got}")
                os.replace(path + ".part", path)
                print(f"[bake] WARNING {name}: upstream changed the file - sha256 {got} != pinned {want}; structure verified "
                      f"(same tensor names, shapes, dtypes): using the new file, as the evaluator does", flush=True)
                return True
            os.replace(path + ".part", path)
            print(f"[bake] {name} ok", flush=True)
            return True
        except Exception as e:  # transient network errors retry; an unverified hash mismatch also retries then fails the build
            print(f"[bake] {name} attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(wait * (attempt + 1))
    return False


def main(argv=None):
    argv = sys.argv if argv is None else argv
    dest = argv[1] if len(argv) > 1 else "/opt/g56/baked"
    flux_te_dest = argv[argv.index("--flux-te") + 1] if "--flux-te" in argv else None  # flux image only
    files = FILES + (FLUX_TE_FILES if flux_te_dest else [])
    os.makedirs(dest, exist_ok=True)
    if flux_te_dest:
        os.makedirs(flux_te_dest, exist_ok=True)
    for url, name, want, digest in files:
        path = os.path.join(flux_te_dest, name[len("@flux/"):]) if name.startswith("@flux/") else os.path.join(dest, name)
        if not fetch(url, name, want, path, digest):
            sys.exit(f"[bake] giving up on {name}")


if __name__ == "__main__":
    main()
