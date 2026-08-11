#!/usr/bin/env python3
"""
export_weights.py
=================

Download a real open-weight tiny LLM from HuggingFace and convert its
safetensors checkpoint into a PHP-friendly on-disk layout that can be
streamed tensor-by-tensor by `infer.php`.

Supported models (any Llama2-architecture LlamaForCausalLM with RoPE +
SwiGLU + RMSNorm):

  * shibatch/tiny1m                       (default, 0.9M params, SentencePiece)
  * shibatch/stories-converted            (subpath: hf_stories260K / hf_stories15M, GPT2 BPE)
  * Qwen/Qwen3-0.6B                       (Llama2-like; ties embeddings)

Layout produced under `weights/`:

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
    python export_weights.py                                  # shibatch/tiny1m
    python export_weights.py --model shibatch/stories-converted \\
                             --subpath hf_stories15M \\
                             --context-limit 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file as safetensors_load_file
from huggingface_hub import hf_hub_download


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

import re
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_]+")


def safe_filename(tensor_name: str) -> str:
    """Map a HuggingFace tensor name to a filesystem-safe `.bin` filename."""
    slug = _UNSAFE_CHARS.sub("_", tensor_name)
    return f"{slug}.bin"


def to_le_float32_c_contig(arr: np.ndarray) -> np.ndarray:
    """Return the tensor as little-endian float32, C-contiguous (row-major)."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if sys.byteorder == "big":
        arr = arr.byteswap().newbyteorder()
    return np.ascontiguousarray(arr, dtype="<f4")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_repo_files(repo_id: str, subpath: str | None, out_dir: Path) -> dict:
    """Pull safetensors + config + tokenizer from HF Hub.

    `subpath` lets us point at e.g. `hf_stories15M/` inside a multi-model repo
    like `shibatch/stories-converted`. Pass None for single-model repos.
    """
    prefix = f"{subpath}/" if subpath else ""

    # Try common locations for each artifact.
    candidates_safetensors = [
        f"{prefix}model.safetensors",
        f"{prefix.replace('hf/', '')}model.safetensors",  # tiny1m uses hf/
        "model.safetensors",
    ]
    candidates_config = [f"{prefix}config.json", "config.json"]
    # Tokenizer artifacts we know how to parse, in priority order.
    candidates_tokenizer = [
        f"{prefix}tokenizer.json",      # GPT2 BPE (stories15M, Qwen)
        f"{prefix.replace('hf/', '')}tokenizer.model",  # SentencePiece (tiny1m)
        "tokenizer.json",
        "tokenizer.model",
    ]

    paths: dict = {}
    for label, candidates in [
        ("safetensors", candidates_safetensors),
        ("config", candidates_config),
        ("tokenizer", candidates_tokenizer),
    ]:
        last_err: Exception | None = None
        for cand in candidates:
            try:
                local = hf_hub_download(repo_id=repo_id, filename=cand)
                paths[label] = local
                print(f"  [hf] {label}: {repo_id}/{cand} -> {local}")
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        else:
            raise RuntimeError(
                f"Could not find {label} in {repo_id} (subpath={subpath!r}). "
                f"Tried {candidates}. Last error: {last_err}"
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
        arr.tofile(out_path)

        nbytes = arr.nbytes
        total_bytes += nbytes
        manifest.append(
            {
                "name": name,
                "shape": [int(d) for d in arr.shape],
                "dtype": "float32",
                "filename": fname,
                "nbytes": int(nbytes),
                "byte_offset": 0,
            }
        )

    return manifest, total_bytes


# --------------------------------------------------------------------------- #
# Tokenizer export
# --------------------------------------------------------------------------- #

def export_tokenizer_gpt2(tokenizer_json_path: str, weights_dir: Path) -> dict:
    """Parse a HuggingFace fast tokenizer.json (GPT2 BPE) into tokens.json.

    tokens.json is a flat array indexed by id:
        [{id, piece, score, type}, ...]
    PHP's greedy longest-match encoder consumes this directly. For GPT2 BPE
    we use the raw piece strings (including the leading 'Ġ' for space-marked
    tokens — PHP's Tokenizer mirrors SentencePiece's U+2581 handling for that
    marker but treats Ġ as a literal character to match against).
    """
    data = json.loads(Path(tokenizer_json_path).read_text(encoding="utf-8"))
    model = data.get("model", {})
    vocab = model.get("vocab", {})   # {piece: id}
    merged = [(piece, int(tid)) for piece, tid in vocab.items()]
    merged.sort(key=lambda x: x[1])

    # Detect special tokens from the top-level added_tokens (if any).
    added = data.get("added_tokens", [])
    added_ids = {int(t["id"]): t["content"] for t in added}

    tokens = []
    for piece, tid in merged:
        tokens.append({
            "id": int(tid),
            "piece": piece,
            "score": 0.0,
            "type": 1 if (tid in added_ids) else 0,
        })
    seen_ids = {t["id"] for t in tokens}
    for tid, content in added_ids.items():
        if tid not in seen_ids:
            tokens.append({"id": int(tid), "piece": content, "score": 0.0, "type": 1})
    tokens.sort(key=lambda t: t["id"])

    # Auto-detect which space marker the vocab actually uses. HF "GPT2 BPE"
    # tokenizers converted from llama2.c (e.g. shibatch/stories-converted) end
    # up using SentencePiece's U+2581 marker instead of GPT2's U+0120 — so we
    # look at the data instead of guessing.
    space_marker = "sentencepiece"  # default: U+2581
    has_g_dot = any(t["piece"].startswith("\u0120") for t in tokens[260:])
    has_sp_marker = any(t["piece"].startswith("\u2581") for t in tokens[260:])
    if has_g_dot and not has_sp_marker:
        space_marker = "gpt2_bpe"   # U+0120

    # Pick up bos/eos from added_tokens or fall back to GPT2 conventions.
    bos_id = next((int(t["id"]) for t in added if t["content"] == "<|endoftext|>"), None)
    eos_id = bos_id
    eos_im_end = next((int(t["id"]) for t in added if t["content"] == "<|im_end|>"), None)
    if eos_im_end is not None:
        eos_id = eos_im_end

    if bos_id is None:
        bos_id = 50256
        eos_id = 50256

    (weights_dir / "tokens.json").write_text(
        json.dumps(tokens, ensure_ascii=False), encoding="utf-8"
    )
    return {"vocab_size": len(tokens), "bos_id": int(bos_id), "eos_id": int(eos_id),
            "pad_id": int(eos_id), "unk_id": int(bos_id),
            "tokenizer_kind": space_marker}


def export_tokenizer_sentencepiece(tokenizer_model_path: str, weights_dir: Path) -> dict:
    """Dump a SentencePiece vocab to tokens.json. Used for shibatch/tiny1m."""
    try:
        import sentencepiece as spm
    except ImportError as e:
        raise RuntimeError(
            "This model uses SentencePiece. Install sentencepiece: "
            "pip install sentencepiece"
        ) from e

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model_path)

    vocab_size = sp.GetPieceSize()
    tokens = []
    for tid in range(vocab_size):
        tokens.append({
            "id": int(tid),
            "piece": sp.IdToPiece(tid),
            "score": float(sp.GetScore(tid)),
            "type": int(sp.IsUnknown(tid) * 1 + sp.IsControl(tid) * 2),
        })

    (weights_dir / "tokens.json").write_text(
        json.dumps(tokens, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "vocab_size": vocab_size,
        "bos_id": int(sp.PieceToId("<s>")) if sp.PieceToId("<s>") >= 0 else 1,
        "eos_id": int(sp.PieceToId("</s>")) if sp.PieceToId("</s>") >= 0 else 2,
        "pad_id": int(sp.PieceToId("</s>")) if sp.PieceToId("</s>") >= 0 else 2,
        "unk_id": int(sp.unk_id()),
        "tokenizer_kind": "sentencepiece",
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

    rope_theta = float(hf_cfg.get("rope_theta")
                       or hf_cfg.get("rope_scaling", {}).get("rope_theta", 10000.0)
                       or 10000.0)

    cfg = {
        "architecture": hf_cfg.get("architectures", ["LlamaForCausalLM"])[0],
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
        "pad_token_id": int(hf_cfg.get("pad_token_id") or tokenizer_meta["pad_id"]),
        "tokenizer_kind": tokenizer_meta["tokenizer_kind"],
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
        "--subpath", default=None,
        help="Subpath inside the repo (e.g. hf_stories15M for shibatch/stories-converted)",
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
    print(f"model    : {repo_id}" + (f" ({args.subpath})" if args.subpath else ""))
    print(f"out dir  : {weights_dir.resolve()}")
    print()

    # 1. Download ----------------------------------------------------------------
    print("[1/4] Downloading checkpoint / config / tokenizer from HuggingFace...")
    paths = download_repo_files(repo_id, args.subpath, weights_dir)
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
    print("[3/4] Exporting vocab -> tokens.json ...")
    tok_path = paths["tokenizer"]
    if tok_path.endswith("tokenizer.json"):
        tokenizer_meta = export_tokenizer_gpt2(tok_path, weights_dir)
    else:
        tokenizer_meta = export_tokenizer_sentencepiece(tok_path, weights_dir)
    print(f"        {tokenizer_meta['tokenizer_kind']}: vocab size "
          f"{tokenizer_meta['vocab_size']} (bos={tokenizer_meta['bos_id']}, "
          f"eos={tokenizer_meta['eos_id']})")
    print()

    # 4. Config ------------------------------------------------------------------
    print("[4/4] Writing config.json ...")
    cfg = export_config(paths["config"], weights_dir, tokenizer_meta,
                        context_limit=args.context_limit)
    print(f"        {cfg['architecture']}: {cfg['num_layers']} layers, "
          f"hidden={cfg['hidden_size']}, heads={cfg['num_heads']}/"
          f"kv={cfg['num_kv_heads']}, head_dim={cfg['head_dim']}, "
          f"vocab={cfg['vocab_size']}, tie={cfg['tie_embeddings']}")
    print()

    # Summary --------------------------------------------------------------------
    print("=== Summary ===")
    print(f"Total weight bytes : {total_bytes} ({total_bytes / 1024 / 1024:.2f} MB)")
    print(f"Tensors            : {len(manifest)}")
    print(f"Vocab size         : {tokenizer_meta['vocab_size']}")
    print(f"Tokenizer          : {tokenizer_meta['tokenizer_kind']}")
    print(f"Context limit      : {cfg['context_limit']} tokens")
    print()
    print("Done. Next step:")
    print("  php infer.php \"Once upon a time\"    # CLI")
    print("  docker compose up --build -d          # HTTP via Nginx + PHP-FPM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
