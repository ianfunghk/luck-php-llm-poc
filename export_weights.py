#!/usr/bin/env python3
"""
export_weights.py
=================

Download the real open-weight tiny LLM `shibatch/tiny1m` from HuggingFace
and convert its safetensors checkpoint into a PHP-friendly on-disk layout
that can be streamed tensor-by-tensor by `infer.php`.

The layout produced under `weights/`:

    weights/
        manifest.json     # one entry per tensor (name, shape, dtype, filename, nbytes, byte_offset)
        config.json       # model hyper-parameters PHP needs to run the forward pass
        tokens.json       # vocab list, indexed by id: [{id, piece, score, type}, ...]
        <safe_name>.bin   # one little-endian float32 file per tensor

No giant base64 blob, no combined container — each tensor is its own file so
PHP can `fread` exactly the bytes it needs, when it needs them, and `unset()`
the result before reading the next layer's weights.

Usage:
    pip install -r requirements.txt
    python export_weights.py                # default model = shibatch/tiny1m
    python export_weights.py --model roneneldan/TinyStories-8M
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file as safetensors_load_file
from huggingface_hub import hf_hub_download


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Characters that are awkward on case-sensitive shared-hosting filesystems.
# We convert HF dotted tensor names like `model.layers.0.self_attn.q_proj.weight`
# into snake_case filenames `model_layers_0_self_attn_q_proj_weight.bin`.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_]+")


def safe_filename(tensor_name: str) -> str:
    """Map a HuggingFace tensor name to a filesystem-safe `.bin` filename."""
    slug = _UNSAFE_CHARS.sub("_", tensor_name)
    return f"{slug}.bin"


def to_le_float32_c_contig(arr: np.ndarray) -> np.ndarray:
    """Return the tensor as little-endian float32, C-contiguous (row-major).

    PHP will unpack with `unpack('f*', ...)` which assumes the host native
    byte order — every platform we care about (Linux x86-64 / arm64 shared
    hosting) is little-endian, so we force LE here for portability.
    """
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if sys.byteorder == "big":
        arr = arr.byteswap().newbyteorder()
    return np.ascontiguousarray(arr, dtype="<f4")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_repo_files(repo_id: str, out_dir: Path) -> dict:
    """Pull the safetensors checkpoint + config + tokenizer from HF Hub.

    Returns a dict with local paths: {safetensors, config, tokenizer}.
    `shibatch/tiny1m` keeps these under `hf/` rather than the repo root,
    so we try the common subpath first and fall back to the root.
    """
    candidates_safetensors = ["hf/model.safetensors", "model.safetensors"]
    candidates_config = ["hf/config.json", "config.json"]
    candidates_tokenizer = ["tokenizer.model"]

    paths: dict = {}
    for label, candidates in [
        ("safetensors", candidates_safetensors),
        ("config", candidates_config),
        ("tokenizer", candidates_tokenizer),
    ]:
        last_err: Exception | None = None
        for cand in candidates:
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=cand,
                    local_dir=None,  # cache only, return path
                )
                paths[label] = local
                print(f"  [hf] {label}: {repo_id}/{cand} -> {local}")
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        else:
            raise RuntimeError(
                f"Could not find {label} in {repo_id}. Tried {candidates}. Last error: {last_err}"
            )
    return paths


# --------------------------------------------------------------------------- #
# Export safetensors -> per-tensor .bin + manifest.json
# --------------------------------------------------------------------------- #

def export_tensors(safetensors_path: str, weights_dir: Path) -> tuple[list[dict], int]:
    """Write one `.bin` per tensor and return (manifest_entries, total_bytes)."""
    tensors = safetensors_load_file(safetensors_path)
    manifest: list[dict] = []
    total_bytes = 0

    for name in sorted(tensors.keys()):
        raw = tensors[name]
        arr = to_le_float32_c_contig(raw)
        fname = safe_filename(name)
        out_path = weights_dir / fname
        arr.tofile(out_path)  # write raw LE float32 bytes

        nbytes = arr.nbytes
        total_bytes += nbytes
        manifest.append(
            {
                "name": name,
                "shape": [int(d) for d in arr.shape],
                "dtype": "float32",
                "filename": fname,
                "nbytes": int(nbytes),
                "byte_offset": 0,  # one tensor per file
            }
        )

    return manifest, total_bytes


# --------------------------------------------------------------------------- #
# Tokenizer export
# --------------------------------------------------------------------------- #

def export_tokenizer(tokenizer_model_path: str, weights_dir: Path) -> dict:
    """Dump the SentencePiece vocab to `tokens.json` indexed by id.

    PHP consumes this directly to build id<->piece maps. The pieces are kept
    in their raw SentencePiece representation (with the U+2581 '▁' space marker
    and the `<0xHH>` byte-fallback tokens unchanged).
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model_path)

    vocab_size = sp.GetPieceSize()
    tokens = []
    for tid in range(vocab_size):
        tokens.append(
            {
                "id": int(tid),
                "piece": sp.IdToPiece(tid),
                "score": float(sp.GetScore(tid)),
                "type": int(sp.IsUnknown(tid) * 1 + sp.IsControl(tid) * 2),
            }
        )

    (weights_dir / "tokens.json").write_text(
        json.dumps(tokens, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "vocab_size": vocab_size,
        "bos_id": int(sp.PieceToId("<s>")) if sp.PieceToId("<s>") >= 0 else 1,
        "eos_id": int(sp.PieceToId("</s>")) if sp.PieceToId("</s>") >= 0 else 2,
        "pad_id": int(sp.PieceToId("</s>")) if sp.PieceToId("</s>") >= 0 else 2,
        "unk_id": int(sp.unk_id()),
    }


# --------------------------------------------------------------------------- #
# Config export
# --------------------------------------------------------------------------- #

def export_config(hf_config_path: str, weights_dir: Path, tokenizer_meta: dict,
                  context_limit: int) -> dict:
    """Build the minimal config.json that PHP needs to run the forward pass."""
    hf_cfg = json.loads(Path(hf_config_path).read_text(encoding="utf-8"))

    hidden = int(hf_cfg["hidden_size"])
    num_heads = int(hf_cfg["num_attention_heads"])
    num_kv_heads = int(hf_cfg.get("num_key_value_heads", num_heads))
    head_dim = int(hf_cfg.get("head_dim", hidden // num_heads))

    # RoPE: Llama2-style models use rope_theta. tiny1m sets rope_theta=10000.
    rope_theta = float(hf_cfg.get("rope_theta")
                       or hf_cfg.get("rope_scaling", {}).get("rope_theta", 10000.0)
                       or 10000.0)

    cfg = {
        "architecture": "LlamaForCausalLM",
        "hidden_size": hidden,
        "num_layers": int(hf_cfg["num_hidden_layers"]),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": int(hf_cfg["intermediate_size"]),
        "vocab_size": int(hf_cfg["vocab_size"]),
        "max_position": int(hf_cfg["max_position_embeddings"]),
        "rms_norm_eps": float(hf_cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": rope_theta,
        "rope_enabled": True,
        "tie_embeddings": bool(hf_cfg.get("tie_word_embeddings", False)),
        "bos_token_id": int(hf_cfg.get("bos_token_id", tokenizer_meta["bos_id"])),
        "eos_token_id": int(hf_cfg.get("eos_token_id", tokenizer_meta["eos_id"])),
        "pad_token_id": int(hf_cfg.get("pad_token_id", tokenizer_meta["pad_id"])),
        # Hard cap we promise the PHP runtime to respect (<<max_position).
        "context_limit": min(int(context_limit), int(hf_cfg["max_position_embeddings"])),
    }

    (weights_dir / "config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    return cfg


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="shibatch/tiny1m",
        help="HuggingFace repo id (default: shibatch/tiny1m)",
    )
    parser.add_argument(
        "--out", default="weights",
        help="Output directory for `.bin` files + manifest (default: weights)",
    )
    parser.add_argument(
        "--context-limit", type=int, default=128,
        help="Hard context cap the PHP runtime should respect (default: 128)",
    )
    args = parser.parse_args()

    repo_id = args.model
    weights_dir = Path(args.out)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== php-llm export ===")
    print(f"model    : {repo_id}")
    print(f"out dir  : {weights_dir.resolve()}")
    print()

    # 1. Download ----------------------------------------------------------------
    print("[1/4] Downloading checkpoint / config / tokenizer from HuggingFace...")
    paths = download_repo_files(repo_id, weights_dir)
    print()

    # 2. Tensors -----------------------------------------------------------------
    print("[2/4] Converting safetensors -> per-tensor .bin files...")
    manifest, total_bytes = export_tensors(paths["safetensors"], weights_dir)
    (weights_dir / "manifest.json").write_text(
        json.dumps({"tensors": manifest}, indent=2), encoding="utf-8"
    )
    print(f"        wrote {len(manifest)} tensors, total {total_bytes} bytes "
          f"({total_bytes / 1024 / 1024:.2f} MB)")
    print()

    # 3. Tokenizer ---------------------------------------------------------------
    print("[3/4] Exporting SentencePiece vocab -> tokens.json ...")
    tokenizer_meta = export_tokenizer(paths["tokenizer"], weights_dir)
    print(f"        vocab size: {tokenizer_meta['vocab_size']} "
          f"(bos={tokenizer_meta['bos_id']}, eos={tokenizer_meta['eos_id']})")
    print()

    # 4. Config ------------------------------------------------------------------
    print("[4/4] Writing config.json ...")
    cfg = export_config(paths["config"], weights_dir, tokenizer_meta,
                        context_limit=args.context_limit)
    print(f"        {cfg['num_layers']} layers, hidden={cfg['hidden_size']}, "
          f"heads={cfg['num_heads']}, kv_heads={cfg['num_kv_heads']}, "
          f"head_dim={cfg['head_dim']}")
    print()

    # Summary --------------------------------------------------------------------
    print("=== Summary ===")
    print(f"Total weight bytes : {total_bytes} ({total_bytes / 1024 / 1024:.2f} MB)")
    print(f"Tensors            : {len(manifest)}")
    print(f"Vocab size         : {tokenizer_meta['vocab_size']}")
    print(f"Context limit      : {cfg['context_limit']} tokens")
    print()
    print("Files PHP will need to upload (under weights/):")
    needed = sorted(p.name for p in weights_dir.iterdir())
    for n in needed:
        print(f"  - weights/{n}")
    print()
    print("Done. Next step:")
    print("  php infer.php \"Once upon a time\"    # CLI")
    print("  docker compose up --build -d          # HTTP via Nginx + PHP-FPM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
