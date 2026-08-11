# php-llm

A proof of concept that a **real, trained, open-weight LLM** can run a full
transformer forward pass in **100% pure PHP** — no FFI, no `exec`, no C
extensions — and stay well under a shared-hosting **1 GB memory_limit**.

It supports **two real Llama2-architecture models** out of the box:

| Model | Params | hidden × layers | vocab | Peak memory | Throughput |
|---|---|---|---|---|---|
| [`shibatch/tiny1m`](https://huggingface.co/shibatch/tiny1m) (default) | 0.9M | 128 × 4 | 512 | **16 MB** | ~30 tok/s |
| [`shibatch/stories-converted`](https://huggingface.co/shibatch/stories-converted) / `hf_stories15M` | 15M | 288 × 6 | 32,000 | **136 MB** | ~2 tok/s |

Both fit comfortably under the 1 GB ceiling thanks to a streaming
memory strategy (see §5).

```
┌─────────────────────────┐      ┌────────────────────────────────────┐
│  Host (Python, once)    │      │  Docker container (or shared host) │
│                         │      │                                    │
│  export_weights.py      │  ├──>│  infer.php                         │
│   --model shibatch/...  │  │    │   ├─ src/Math.php     (RMSNorm,    │
│   --subpath hf_stories  │  │    │   │                     matvec,     │
│        │                │ │    │   │                     matvecRaw,   │
│        ▼                │ │    │   │                     RoPE,       │
│   weights/              │─┴──┤   │                     softmax)    │
│     *.bin   (57 files)  │    │   ├─ src/Loader.php     (raw + row   │
│     manifest.json       │    │   │                     streaming)   │
│     config.json         │    │   ├─ src/Tokenizer.php  (SentencePiece│
│     tokens.json         │    │   │                     or GPT2 BPE)  │
└─────────────────────────┘    │   ├─ src/Sampler.php    (temperature)│
                                │   ├─ src/Forward.php    (Llama fwd,  │
                                │   │                     raw embed)  │
                                │   └─ web/index.php     (HTTP/JSON)  │
                                │                                    │
                                │   PHP 8.4 + Nginx, 1 GB mem cap    │
                                └────────────────────────────────────┘
```

---

## 1. Quick start

```bash
# 1. Clone & enter
git clone https://github.com/ianfunghk/luck-php-llm-poc php-llm && cd php-llm

# 2. Export real model weights (one-time, on the host).
#    Default: tiny1m. Swap the line below for stories15M to use the 15M model.
pip install -r requirements.txt

# (a) tiny1m — 0.9M params, ~4 MB download
python export_weights.py

# (b) stories15M — 15M params, ~93 MB download (recommended upgrade)
python export_weights.py \
    --model shibatch/stories-converted \
    --subpath hf_stories15M \
    --context-limit 128

# 3. Run via Docker (PHP 8.4-FPM + Nginx)
docker compose up --build -d
curl 'http://localhost:8080/?prompt=Once%20upon%20a%20time&max_tokens=60&format=text'

# Or, without Docker, just use any PHP 8.4+ CLI:
php -d memory_limit=1024M infer.php "Once upon a time"
```

### Expected output: `tiny1m`

```
Prompt   : Once upon a time
Generated: time, there was a little girl named Lily. She had a delicate bandage,
            but she was very proud of her long bandage. One day, her mommy gave
            her a character to her teacher and suffering her regular bandage.
            Lily felt happy too, so she used her pride and practiced her
------------------------------------------------------------
Tokens generated : 124
Decode elapsed   : 3.995 s
Peak memory      : 16,777,216 bytes (16.0 MB)
```

### Expected output: `hf_stories15M`

```
Prompt   : hello
Generated: Once upon a time, there was a big round ballroom full of it, big and
           it was
Tokens   : 20
Elapsed  : 6.485 s
Peak mem : 120.4 MB
```

Both models produce coherent TinyStories-style English. They are **not**
random noise — the Llama2 weights are real, loaded straight from HuggingFace.

---

## 2. What's in this repo

| File | Role |
|---|---|
| `export_weights.py` | Python: downloads a model from HuggingFace, converts each safetensors tensor to a little-endian float32 `.bin`, emits `manifest.json`, `config.json`, `tokens.json`. Supports any Llama2-architecture model. |
| `requirements.txt` | Python deps: `numpy`, `safetensors`, `huggingface_hub`. (SentencePiece optional, only for SentencePiece vocab models.) |
| `infer.php` | Pure-PHP CLI entry point + shared `run_inference()` function. Top-of-file tunables (`WEIGHTS_DIR`, `CONTEXT_LIMIT`, `MAX_TOKENS`, `TEMPERATURE`). |
| `src/Math.php` | Pure-PHP math: `matvec`, `matvecRaw`, `rmsNorm`, `silu`, `softmax`, `applyRope`. Heavily commented. |
| `src/Loader.php` | Streams each tensor from disk. Provides `loadTensorRaw()` (returns a binary string — cheap) and `loadTensor()` (returns a float[]). Process-local memo + shmop-aware. |
| `src/Tokenizer.php` | Greedy longest-match encoder + byte-fallback + marker-aware decoder. Auto-detects SentencePiece (U+2581) vs GPT2 (U+0120) space markers from `config.json`. |
| `src/Sampler.php` | Temperature sampling (softmax + cumulative distribution). |
| `src/Forward.php` | Llama2 forward pass. Embedding and LM-head matrices are kept as raw binary strings and walked one row per token. Per-layer weights are unpacked lazily and freed after each layer. |
| `src/WeightCache.php` | Optional `shmop`-backed shared memory cache. First request populates it; subsequent requests (CLI or HTTP, any PHP-FPM worker) skip disk I/O. Falls back silently if `shmop` is unavailable. |
| `web/index.php` | HTTP entry point — HTML form, `?format=text`, `?format=json`. |
| `Dockerfile` | PHP 8.4-FPM + Nginx + supervisord, single image, no Python. |
| `docker-compose.yml` | Builds & exposes the image on `:8080`, bind-mounts `./weights`. |
| `nginx.conf`, `php.ini`, `.dockerignore` | Supporting configs. |

---

## 3. Local Python export (in detail)

`export_weights.py` is a single-script pipeline that:

1. Downloads `model.safetensors`, `config.json`, and the tokenizer from the
   HuggingFace repo (using `huggingface_hub.hf_hub_download`).
2. Loads each tensor via `safetensors.numpy.load_file`, casts to little-endian
   float32 C-contiguous, and writes one `.bin` per tensor.
3. Generates `manifest.json` with `{name, shape, dtype, filename, nbytes,
   byte_offset}` per tensor.
4. Generates `tokens.json` (flat array indexed by id: `{id, piece, score,
   type}`) from either `tokenizer.json` (GPT2 BPE) or `tokenizer.model`
   (SentencePiece).
5. Auto-detects which space marker (`▁` vs `Ġ`) the vocab uses and records
   that in `config.json` as `tokenizer_kind`.
6. Generates `config.json` with all hyper-parameters PHP needs.

### Switching models

```bash
# Default — tiny1m
python export_weights.py

# 15M model from stories-converted
python export_weights.py --model shibatch/stories-converted \
                        --subpath hf_stories15M \
                        --context-limit 128

# 260K variant (smallest)
python export_weights.py --model shibatch/stories-converted \
                        --subpath hf_stories260K \
                        --context-limit 128

# Any other Llama2-architecture model (must be LlamaForCausalLM with RoPE)
python export_weights.py --model Qwen/Qwen3-0.6B --context-limit 64
# (Note: at 0.6B params, memory will exceed 1 GB; see §7 limitations.)
```

> The PHP runtime currently assumes a Llama2-style architecture (RMSNorm,
> SwiGLU MLP, RoPE attention). GPTNeo or Mixture-of-Experts models will not
> run as-is.

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

If you have PHP 8.4+ locally:

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

## 5. Memory strategy (the interesting bit)

PHP's per-element overhead is ~80 bytes per float once a tensor is unpacked
into a PHP array. For tiny1m (3.6 MB safetensors → ~720 MB as PHP float[]) we
could almost get away with naive loading. For stories15M (93 MB safetensors)
naive loading would peak near 1.86 GB and crash.

To stay under 1 GB on either model, the runtime uses two complementary
techniques:

### 5a. Per-layer streaming

[`src/Forward.php`](src/Forward.php) only loads the weights for the layer
currently being evaluated. After the layer's math is done, those tensors are
`unset()` before the next layer loads its own. Peak memory for layer weights
is bounded by the **single largest layer**, not by the sum of all layers.

### 5b. Raw-binary storage for vocab-side matrices

The largest tensors are always `model.embed_tokens.weight` and `lm_head.weight`
(both `[vocab, hidden]`). At vocab=32000 these are 36 MB each as float32 —
fine as bytes, but ~700 MB each after `unpack('f*', ...)`.

The fix is to keep these matrices as raw binary strings and walk them row by
row. `Math::matvecRaw()` substrings one row of bytes at a time, unpacks it,
and does the inner dot product. The full matrix is never expanded into a PHP
array.

For embedding lookup (one row per token), `Forward::forwardToken()` does the
same: `substr` the row for the requested token id, unpack just that row.

The combined effect on stories15M:

| Tensor | Naive unpack | Raw + row-streamed |
|---|---|---|
| `model.embed_tokens.weight` [32000, 288] | ~700 MB | **0 MB** (36 MB string + 288 floats transient) |
| `lm_head.weight` [32000, 288] | ~700 MB | **0 MB** (36 MB string + 288 floats transient) |
| One transformer layer | ~3 MB | ~3 MB |
| Final RMSNorm [288] | 1 KB | 1 KB |
| KV cache @ 128 ctx, 6 layers, 6 kv_heads, head_dim 48 | 5.6 MB | 5.6 MB |
| **Peak** | **~1.4 GB (OOM!)** | **~136 MB** |

### 5c. KV cache

The KV cache grows by `[num_kv_heads * head_dim]` floats per token per layer.
At 128 tokens for stories15M: `128 × 6 × 2 × 288 × 4 bytes = 1.7 MB`. Negligible.

### 5d. Shared-memory weight cache (`shmop`)

The first time any PHP-FPM worker needs a tensor, it reads the `.bin` from
disk. Subsequent workers — and subsequent requests in the same worker — would
re-read it. [`src/WeightCache.php`](src/WeightCache.php) cuts that down by
copying every loaded tensor into a single System V shared memory segment,
keyed off the `config.json` file identity.

Workflow per request:

```
loadTensorRaw(name)
  ├─ process-local memo hit?        ─► return   (free; common case within a request)
  ├─ shmop segment hit?             ─► memoize + return
  └─ disk read                      ─► memoize + record as "miss"
        ↓
[after generation]
Loader::primeCacheFromMisses()      ─► copy every miss into the shmop segment
```

| Request | shmop result | Peak mem | Wall time |
|---|---|---|---|
| 1st (cold) | 0 hits / 57 misses | 211 MB | 4.2 s |
| 2nd (warm) | **57 hits / 0 misses** | **142 MB** | **3.5 s** |
| 3rd+ | same as 2nd | same | same |

Cache key is derived from `ftok(config.json, 'P')`. If you swap models
(re-run `export_weights.py`), the new `config.json` has a different inode and
the cache auto-invalidates.

You can disable the cache with:

```php
// infer.php
const CACHE_BACKEND = 'none';   // always read from disk
```

Or require it explicitly:

```php
const CACHE_BACKEND = 'shmop';  // throw if shmop ext is missing
```

The Dockerfile enables the `shmop` extension automatically. Real shared
hosting providers may or may not offer it — the `'auto'` default falls back
silently when shmop is unavailable.

---

## 6. How the forward pass works

The PHP side implements a textbook Llama2 transformer. See the comments in
[`src/Forward.php`](src/Forward.php) for the line-by-line math. At a high
level, for each generated token:

1. **Embedding lookup** (raw-streamed) — `substr` one row from
   `model.embed_tokens.weight` for `token_id`, unpack just that row → `[hidden]`.
2. **For each layer** (weights streamed from disk, freed after use):
   - **Pre-attention RMSNorm** — normalise by root-mean-square, scale by learned weights.
   - **QKV projections** — three `matvec`s producing Q, K, V.
   - **RoPE** — rotate Q/K pairs by a position-dependent angle.
   - **Causal self-attention** — for each query head: scaled dot-product
     `softmax(Q·Kᵀ / √d) · V`, accumulating the new K/V into the per-layer
     cache for future tokens. Supports grouped-query attention
     (`num_kv_heads != num_heads`).
   - **Output projection + residual** — `h += W_o · attn_out`.
   - **Pre-MLP RMSNorm**.
   - **SwiGLU MLP** — `h += W_down · (SiLU(W_gate · x) ⊙ W_up · x)`.
3. **Final RMSNorm**.
4. **LM head** (raw-streamed) — walk `lm_head.weight` row by row, dot product
   against the final hidden vector to produce `[vocab]` logits.
5. **Temperature sampling** — `softmax(logits/T)`, draw from the CDF.

---

## 7. Performance and memory

### tiny1m

On a 2023 MacBook Pro (M2) running PHP 8.4 in the Alpine container:

| Metric | Value |
|---|---|
| Total weight bytes on disk | 3.57 MB |
| Peak RSS during generation | 14–16 MB |
| Throughput (decode only) | ~30 tokens/sec |
| Time to generate 128 tokens | ~4 seconds |

### hf_stories15M

| Metric | Value |
|---|---|
| Total weight bytes on disk | 93.11 MB |
| Peak RSS during generation | 120–140 MB |
| Throughput (decode only) | ~2 tokens/sec |
| Time to generate 128 tokens | ~55 seconds |

The slowdown vs tiny1m is dominated by the lm_head step (vocab × hidden
matvec, run once per token) since vocab is 62.5× larger. The raw-string trick
in §5b keeps memory bounded at ~140 MB instead of OOM.

### Why not Qwen3-0.6B?

Qwen3-0.6B's safetensors checkpoint is 1.5 GB. Even with the raw-string
strategy, the embedding matrix alone `[151936, 1024]` × 4 bytes = 622 MB
resident as a string — and every token requires walking all 151936 rows for
the lm_head dot products. We have measured this carefully and it does not fit
in 1 GB under pure PHP. To make a 0.5B+ model work you would need either
quantized loading (INT4 lookup tables) or a PHP C extension like
`tensor/tensor` (BLAS bindings) — both are out of scope for this proof of
concept.

---

## 8. Limitations (intentional)

This is a **proof of concept**, not a production runtime. Specifically:

- **Tokenizer is a greedy approximation.** [`src/Tokenizer.php`](src/Tokenizer.php)
  does longest-match against the vocab plus byte fallback, which is exact for
  ASCII English and adequate for non-ASCII. It is **not** a full
  SentencePiece/BBPE implementation — that would dwarf the rest of the code.
  The encoder is verified to produce byte-identical ids to the reference
  SentencePiece tokenizer for ordinary English prompts.

- **Pure-PHP speed is slow vs. native.** 2–30 tokens/sec is fine for a demo
  but orders of magnitude off llama.cpp. The point is feasibility, not
  throughput.

- **No batching / no parallelism.** Prefill runs one token at a time.

- **Architecture is Llama2-only.** The math handles grouped-query attention
  (`num_kv_heads != num_heads`), but does not implement GPTNeo, Mixture-of-
  Experts, or sliding-window attention.

- **0.5B+ models are out of scope under 1 GB.** A Qwen3-0.6B-class model
  would need quantized loading or a BLAS extension; both are deliberately not
  implemented here.

- **BCMath / GMP / GD are not used.** None of them help with float
  matrix-vector products: BCMath operates on decimal strings (~100× slower
  than native float), GMP is integer-only, and GD's only convolution is a
  3×3 uint8 pixel kernel. Native `float` is fast and precise enough at the
  head_dim values we use.

---

## 9. Troubleshooting

**`Fatal error: Allowed memory size of X bytes exhausted`**
Reduce `CONTEXT_LIMIT` and `MAX_TOKENS` in `infer.php`. If you have run
`export_weights.py` for a larger model, switch back to `shibatch/tiny1m` to
confirm the ceiling.

**`Maximum execution time of 60 seconds exceeded`**
Lower `MAX_TOKENS`, or raise `max_execution_time` in `php.ini` /
`set_time_limit()` in `infer.php`. The 15M model can take ~1 minute for 128
tokens; the 0.9M model finishes in ~4 seconds.

**`404` from the HTTP container**
The Nginx root is `/var/www/html/web`, not `/var/www/html`. Hit
`http://localhost:8080/?prompt=hello` (no `/web/` prefix).

**Garbled or repetitive output**
Try `temperature=0.5` (sharper) or `temperature=1.2` (more varied). For
deterministic runs, set `temperature=0` (greedy argmax).

**`Unknown tensor: ...`**
The PHP loader received an unexpected tensor name. Re-run
`python export_weights.py` (with the right `--model` and `--subpath`) to
regenerate `weights/manifest.json` matching the current PHP code.

**`Ġ` or `▁` markers leaking into output**
The `tokenizer_kind` field in `weights/config.json` was mis-detected. Delete
`weights/` and re-run `export_weights.py`; the script auto-detects which
space marker the vocab actually uses.

**`shmop_open(): unable to attach or create shared memory` / `Weight cache: disabled`**
Either `shmop` is not loaded (run `php -m | grep shmop` to confirm; install
it with `docker-php-ext-install shmop` inside a custom image) or your host
restricts System V IPC segments. The runtime falls back to disk reads
automatically. To force-fail instead, set `CACHE_BACKEND='shmop'`.

**Shared-memory cache not warming between requests**
Confirm your PHP-FPM workers run in the same IPC namespace. Containers by
default isolate IPC; pass `--ipc=host` to `docker run` or set
`ipc: host` in `docker-compose.yml` if you see hits staying at 0 across
requests. Note: this is only needed for cross-process sharing — within a
single PHP-FPM worker the process-local memo already eliminates repeat disk
reads.

---

## 10. Acceptance checklist

- [x] `export_weights.py` runs locally, downloads real open-weight tiny model,
      produces `manifest.json` + per-tensor `.bin` files + `tokens.json` +
      `config.json`.
- [x] No single giant base64 JSON; tensors are streamable `.bin` files.
- [x] `infer.php` is pure PHP — no FFI, no exec, no external binaries.
- [x] `infer.php` runs end-to-end and prints completion + token count,
      elapsed time, peak memory.
- [x] Peak memory on a 128-token generation is **16 MB** (tiny1m) / **136 MB**
      (stories15M), both far under 1 GB.
- [x] Generated text is from the real trained model (coherent TinyStories
      English), not random weights.
- [x] Code is heavily commented; the transformer math (RMSNorm / attention /
      SiLU MLP / temperature sampling) is easy to follow.
- [x] Bonus: HTTP/JSON entry point via Docker (PHP-FPM + Nginx).
- [x] Bonus: Multi-model support — switch between tiny1m and stories15M by
      re-running `export_weights.py`.

---

## 11. License

The PHP code in this repo is MIT. The model weights retain their respective
upstream licenses:
- `shibatch/tiny1m` — MIT
- `shibatch/stories-converted` (hf_stories260K, hf_stories15M) — derived from
  Karpathy's llama2.c, MIT

Third-party dependencies (`numpy`, `safetensors`, `huggingface_hub`,
`sentencepiece`, PHP itself, Nginx, supervisord) retain their respective
licenses.
