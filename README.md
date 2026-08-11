# php-llm

A proof of concept that a **real, trained, open-weight LLM** can run a full
transformer forward pass in **100% pure PHP** — no FFI, no `exec`, no C
extensions — and stay well under a shared-hosting **1 GB memory_limit**.

It targets [`shibatch/tiny1m`](https://huggingface.co/shibatch/tiny1m), a
real Llama2-architecture model trained on TinyStories (~896k params, 4 layers,
hidden=128, vocab=512).

```
┌─────────────────────────┐      ┌────────────────────────────────────┐
│  Host (Python, once)    │      │  Docker container (or shared host) │
│                         │      │                                    │
│  export_weights.py      │  ├──>│  infer.php                         │
│   shibatch/tiny1m  ─────┼─┐    │   ├─ src/Math.php     (RMSNorm,    │
│   (HF safetensors)      │ │    │   │                     matvec,     │
│        │                │ │    │   │                     RoPE,       │
│        ▼                │ │    │   │                     softmax)    │
│   weights/              │─┴──┤   ├─ src/Loader.php     (per-tensor   │
│     *.bin   (39 files)  │    │   │                     streamed)    │
│     manifest.json       │    │   ├─ src/Tokenizer.php  (greedy +     │
│     config.json         │    │   │                     byte fallback)│
│     tokens.json         │    │   ├─ src/Sampler.php    (temperature) │
└─────────────────────────┘    │   ├─ src/Forward.php    (Llama fwd)  │
                                │   └─ web/index.php     (HTTP/JSON)  │
                                │                                    │
                                │   PHP 8.2 + Nginx, 1 GB mem cap    │
                                └────────────────────────────────────┘
```

---

## 1. Quick start

```bash
# 1. Clone & enter
git clone <your-fork-url> php-llm && cd php-llm

# 2. Export real model weights (one-time, on the host)
pip install -r requirements.txt
python export_weights.py          # downloads shibatch/tiny1m (~3.6 MB)

# 3. Run via Docker (PHP 8.2-FPM + Nginx)
docker compose up --build -d
curl 'http://localhost:8080/?prompt=Once%20upon%20a%20time&max_tokens=60&format=text'

# Or, without Docker, just use any PHP 8.2+ CLI:
php infer.php "Once upon a time"
```

Expected output (real run on a Mac, your text will vary because of temperature sampling):

```
Prompt   : Once upon a time
Generated: time, there was a little girl named Lily. She had a delicate bandage,
            but she was very proud of her long bandage. One day, her mommy gave
            her a character to her teacher and suffering her regular bandage.
            Lily felt happy too, so she used her pride and practiced her
------------------------------------------------------------
Tokens generated : 124
Decode elapsed   : 3.995 s
Total wall time  : 4.133 s
Peak memory      : 16,777,216 bytes (16.0 MB)
Memory limit     : 1024M
```

The model produces coherent TinyStories-style English. It is **not** random
noise — the Llama2 weights are real, loaded straight from HuggingFace.

---

## 2. What's in this repo

| File | Role |
|---|---|
| `export_weights.py` | Python: downloads `shibatch/tiny1m` from HuggingFace, converts each safetensors tensor to a little-endian float32 `.bin`, emits `manifest.json`, `config.json`, `tokens.json`. |
| `requirements.txt` | Python deps: `numpy`, `safetensors`, `huggingface_hub`, `sentencepiece`. |
| `infer.php` | Pure-PHP CLI entry point + shared `run_inference()` function. Top-of-file tunables (`WEIGHTS_DIR`, `CONTEXT_LIMIT`, `MAX_TOKENS`, `TEMPERATURE`). |
| `src/Math.php` | Pure-PHP math: `matvec`, `rmsNorm`, `silu`, `softmax`, `applyRope`. Heavily commented. |
| `src/Loader.php` | Streams each tensor from disk via `fopen` + `fread` + `unpack('f*', ...)`; never loads the whole model. |
| `src/Tokenizer.php` | Greedy longest-match encoder + byte-fallback + SentencePiece marker-aware decoder. |
| `src/Sampler.php` | Temperature sampling (softmax + cumulative distribution). |
| `src/Forward.php` | Llama2 forward pass with layer-by-layer weight streaming and a growing KV cache. |
| `web/index.php` | HTTP entry point — HTML form, `?format=text`, `?format=json`. |
| `Dockerfile` | PHP 8.2-FPM + Nginx + supervisord, single image, no Python. |
| `docker-compose.yml` | Builds & exposes the image on `:8080`, bind-mounts `./weights`. |
| `nginx.conf`, `php.ini`, `.dockerignore` | Supporting configs. |

---

## 3. Local Python export (in detail)

`export_weights.py` downloads the safetensors checkpoint, tokenizer, and
config from `shibatch/tiny1m`, then writes one `.bin` per tensor:

```
weights/
  config.json                                       # hyperparameters PHP needs
  manifest.json                                     # one entry per tensor
  tokens.json                                       # [{id, piece, score, type}, ...]
  lm_head_weight.bin                                # [vocab, hidden] = [512, 128]
  model_embed_tokens_weight.bin                     # [vocab, hidden]
  model_layers_0_input_layernorm_weight.bin         # [hidden]
  model_layers_0_mlp_down_proj_weight.bin           # [hidden, intermediate]
  model_layers_0_mlp_gate_proj_weight.bin           # [intermediate, hidden]
  model_layers_0_mlp_up_proj_weight.bin             # [intermediate, hidden]
  model_layers_0_post_attention_layernorm_weight.bin
  model_layers_0_self_attn_{k,o,q,v}_proj_weight.bin
  model_layers_{1,2,3}_*                            # same pattern per layer
  model_norm_weight.bin                             # [hidden]
```

39 tensors total, 3.57 MB on disk. Each `.bin` is little-endian float32,
row-major (C-contiguous) — exactly what PHP's `unpack('f*', ...)` expects on
any little-endian host.

To switch to a different model:

```bash
python export_weights.py --model roneneldan/TinyStories-8M --context-limit 128
```

> The PHP runtime currently assumes a Llama2-style architecture (RMSNorm,
> SwiGLU MLP, RoPE attention). Other model families will not run as-is.

---

## 4. Running it

### 4a. Docker (recommended)

The Dockerfile builds a single image that runs `php-fpm` + `nginx` under
`supervisord`. The `weights/` directory is bind-mounted in, so re-running
`export_weights.py` refreshes the model with no image rebuild.

```bash
docker compose up --build -d
docker compose ps                                   # confirm healthy
curl 'http://localhost:8080/?prompt=hello&format=text'
curl 'http://localhost:8080/?prompt=hello&format=json'
curl 'http://localhost:8080/'                        # HTML form
```

CLI inside the container:

```bash
docker compose exec app php /var/www/html/infer.php "hello"
```

### 4b. Bare PHP CLI (no Docker)

If you have PHP 8.2+ locally:

```bash
php -d memory_limit=1024M -d max_execution_time=120 infer.php "Once upon a time"
```

### 4c. Real shared hosting (CGI / php-fpm + Apache or Nginx)

Upload these files (everything else is dev-only):

```
infer.php
src/
web/index.php        (or move its contents to your docroot)
weights/             (the whole directory produced by export_weights.py)
```

In your host's `php.ini` (or `.user.ini` for cPanel-style hosts):

```ini
memory_limit = 1024M
max_execution_time = 120
```

Point a browser at `web/index.php?prompt=hello&format=text`.

---

## 5. How the forward pass works

The PHP side implements a textbook Llama2 transformer. See the comments in
[`src/Forward.php`](src/Forward.php) for the line-by-line math. At a high
level, for each generated token:

1. **Embedding lookup** — `model.embed_tokens.weight[token_id]` → `[hidden]` vector.
2. **For each of 4 layers** (weights streamed from disk, freed after use):
   - **Pre-attention RMSNorm** — normalise by root-mean-square, scale by learned weights.
   - **QKV projections** — three `matvec`s producing Q, K, V.
   - **RoPE** — rotate Q/K pairs by a position-dependent angle.
   - **Causal self-attention** — for each of 2 heads: scaled dot-product
     `softmax(Q·Kᵀ / √d) · V`, accumulating the new K/V into the per-layer
     cache for future tokens.
   - **Output projection + residual** — `h += W_o · attn_out`.
   - **Pre-MLP RMSNorm**.
   - **SwiGLU MLP** — `h += W_down · (SiLU(W_gate · x) ⊙ W_up · x)`.
3. **Final RMSNorm**.
4. **LM head** — `matvec` to produce `[vocab]` logits.
5. **Temperature sampling** — `softmax(logits/T)`, draw from the CDF.

The KV cache grows by `[hidden]` floats per token per layer; at 128 tokens
that's `128 × 4 × 2 × 128 × 4 bytes = 512 KB` — negligible.

---

## 6. Performance and memory

On a 2023 MacBook Pro (M2) running PHP 8.2 in the Alpine container:

| Metric | Value |
|---|---|
| Total weight bytes on disk | 3.57 MB |
| Peak RSS during generation | 14–16 MB |
| Throughput (decode only) | ~30 tokens/sec |
| Time to generate 128 tokens | ~4 seconds |
| Time to prefill a 4-token prompt | <50 ms |

PHP itself caps at 16 MB here because:
- Weights are streamed one tensor at a time, `unset()` before loading the
  next layer (`src/Forward.php::forwardLayer`).
- Tensors are kept as flat `float[]` arrays (no nested-array overhead).
- The KV cache is small (see above).

For the 1 GB shared-hosting ceiling, that's a **~60× safety margin**.

---

## 7. Limitations (intentional)

This is a **proof of concept**, not a production runtime. Specifically:

- **Tokenizer is a greedy approximation.** [`src/Tokenizer.php`](src/Tokenizer.php)
  does longest-match against the SentencePiece vocab plus byte fallback, which
  is exact for ASCII English and adequate for non-ASCII. It is **not** a full
  SentencePiece BPE/Unigram implementation — that would dwarf the rest of the
  code. The encoder is verified to produce byte-identical ids to
  `sentencepiece.EncodeAsIds` for ordinary English prompts.

- **Pure-PHP speed is slow vs. native.** ~30 tokens/sec is fine for a demo
  but orders of magnitude off llama.cpp. The point is feasibility, not
  throughput.

- **No batching / no parallelism.** Prefill runs one token at a time.

- **Architecture is Llama2-only.** The math handles `num_kv_heads !=
  num_heads` (grouped-query attention), but does not implement, e.g.,
  Mixture-of-Experts or sliding-window attention.

- **1B+ models are out of scope for v1.** The PHP array overhead
  (≈80 bytes per float) means even a 350 MB weight set balloons past 1 GB.
  Quantised loading would be needed for larger models.

- **BCMath is not used.** Native `float` is fast and precise enough at
  `head_dim=64`; BCMath would be many orders of magnitude slower.

---

## 8. Troubleshooting

**`Fatal error: Allowed memory size of X bytes exhausted`**
Reduce `CONTEXT_LIMIT` and `MAX_TOKENS` in `infer.php`. With tiny1m the
defaults (128 / 128) peak at ~16 MB, so hitting the ceiling means something
else is wrong — check that `memory_limit` is actually 1024M.

**`Maximum execution time of 60 seconds exceeded`**
Lower `MAX_TOKENS`, or raise `max_execution_time` in `php.ini` /
`set_time_limit()` in `infer.php`.

**`404` from the HTTP container**
The Nginx root is `/var/www/html/web`, not `/var/www/html`. Hit
`http://localhost:8080/?prompt=hello` (no `/web/` prefix).

**Garbled or repetitive output**
Try `temperature=0.5` (sharper) or `temperature=1.2` (more varied). For
deterministic runs, set `temperature=0` (greedy argmax).

**`Unknown tensor: ...`**
The PHP loader received an unexpected tensor name. Re-run
`python export_weights.py` to regenerate `weights/manifest.json` matching
the current PHP code.

---

## 9. Acceptance checklist

- [x] `export_weights.py` runs locally, downloads real open-weight tiny model,
      produces `manifest.json` + per-tensor `.bin` files + `tokens.json` +
      `config.json`.
- [x] No single giant base64 JSON; tensors are streamable `.bin` files.
- [x] `infer.php` is pure PHP — no FFI, no exec, no external binaries.
- [x] `infer.php` runs end-to-end and prints completion + token count,
      elapsed time, peak memory.
- [x] Peak memory on a 128-token generation is **16 MB**, far under 1 GB.
- [x] Generated text is from the real trained model (coherent TinyStories
      English), not random weights.
- [x] Code is heavily commented; the transformer math (RMSNorm / attention /
      SiLU MLP / temperature sampling) is easy to follow.
- [x] Bonus: HTTP/JSON entry point via Docker (PHP-FPM + Nginx).

---

## 10. License

The PHP code in this repo is MIT. The `shibatch/tiny1m` model weights are
also MIT (per the upstream HuggingFace model card). Third-party dependencies
(`numpy`, `safetensors`, `huggingface_hub`, `sentencepiece`, PHP itself,
Nginx, supervisord) retain their respective licenses.
