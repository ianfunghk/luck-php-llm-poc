<?php
declare(strict_types=1);

/**
 * src/Forward.php
 * ===============
 * Llama2-style transformer forward pass in pure PHP.
 *
 * Memory strategy
 * ---------------
 * Per-layer weight tensors are loaded from disk ONLY when that layer is being
 * evaluated and `unset()` immediately after, so peak memory stays close to
 * (one layer's worth of weights) + (KV cache so far) + (the hidden vector).
 * For tiny1m (~3.5 MB total) this is trivial, but the design carries over to
 * larger models without PHP blowing the 1 GB limit.
 *
 * Tensor name conventions (HuggingFace Llama):
 *   model.embed_tokens.weight                          [vocab, hidden]
 *   model.layers.<i>.input_layernorm.weight            [hidden]
 *   model.layers.<i>.self_attn.{q,k,v,o}_proj.weight   [hidden, hidden] (MHA)
 *   model.layers.<i>.post_attention_layernorm.weight   [hidden]
 *   model.layers.<i>.mlp.{gate,up}_proj.weight         [intermediate, hidden]
 *   model.layers.<i>.mlp.down_proj.weight              [hidden, intermediate]
 *   model.norm.weight                                  [hidden]
 *   lm_head.weight                                     [vocab, hidden]
 *
 * KV cache layout (per layer):
 *   $kvCache[$layer]['k'][$t] = flat float[hidden]      // K vector at time t
 *   $kvCache[$layer]['v'][$t] = flat float[hidden]
 * where $t is the absolute position in the decoded sequence.
 */

namespace PhpLlm;

class Forward
{
    private Loader $loader;
    private array $cfg;

    private int $hidden;
    private int $numLayers;
    private int $numHeads;
    private int $numKvHeads;
    private int $headDim;
    private int $intermediate;
    private int $vocab;
    private float $rmsEps;
    private float $ropeTheta;
    private bool $ropeEnabled;

    /** @var array|null Embedding matrix [vocab, hidden] (small for tiny1m). */
    private ?array $embed = null;

    /** @var array|null Final-norm scale [hidden]. */
    private ?array $finalNorm = null;

    /** @var array|null LM head [vocab, hidden]. */
    private ?array $lmHead = null;

    public function __construct(Loader $loader)
    {
        $this->loader = $loader;
        $this->cfg = $loader->config;

        $this->hidden       = (int)$this->cfg['hidden_size'];
        $this->numLayers    = (int)$this->cfg['num_layers'];
        $this->numHeads     = (int)$this->cfg['num_heads'];
        $this->numKvHeads   = (int)$this->cfg['num_kv_heads'];
        $this->headDim      = (int)$this->cfg['head_dim'];
        $this->intermediate = (int)$this->cfg['intermediate_size'];
        $this->vocab        = (int)$this->cfg['vocab_size'];
        $this->rmsEps       = (float)$this->cfg['rms_norm_eps'];
        $this->ropeTheta    = (float)$this->cfg['rope_theta'];
        $this->ropeEnabled  = (bool)$this->cfg['rope_enabled'];
    }

    /**
     * Allocate a fresh KV cache. Call once per generation.
     */
    public function newKvCache(): array
    {
        $cache = [];
        for ($l = 0; $l < $this->numLayers; $l++) {
            $cache[$l] = ['k' => [], 'v' => []];
        }
        return $cache;
    }

    /**
     * Run the forward pass for a single token at a single position.
     *
     * Returns the vocab-size logits vector.
     *
     * @param array $kvCache modified in place: appends this step's K and V.
     */
    public function forwardToken(int $tokenId, int $position, array &$kvCache): array
    {
        // ----- 1. Embedding lookup ---------------------------------------------
        // embed shape [vocab, hidden]; row $tokenId is the token's vector.
        // Loaded once and cached (it's only vocab*hidden*4 bytes; tiny1m = 256KB).
        if ($this->embed === null) {
            $this->embed = $this->loader->loadData('model.embed_tokens.weight');
        }
        $base = $tokenId * $this->hidden;
        $h = array_fill(0, $this->hidden, 0.0);
        for ($j = 0; $j < $this->hidden; $j++) {
            $h[$j] = $this->embed[$base + $j];
        }

        // ----- 2. Transformer layers (streamed) -------------------------------
        for ($l = 0; $l < $this->numLayers; $l++) {
            $h = $this->forwardLayer($l, $h, $position, $kvCache[$l]);
        }

        // ----- 3. Final RMSNorm -----------------------------------------------
        if ($this->finalNorm === null) {
            $this->finalNorm = $this->loader->loadData('model.norm.weight');
        }
        $h = Math::rmsNorm($h, $this->finalNorm, $this->rmsEps);

        // ----- 4. LM head -> logits -------------------------------------------
        // lm_head shape [vocab, hidden]: logits[i] = dot(lm_head[i,:], h).
        if ($this->lmHead === null) {
            $lmHeadName = !empty($this->cfg['tie_embeddings']) ? 'model.embed_tokens.weight' : 'lm_head.weight';
            $this->lmHead = $this->loader->loadData($lmHeadName);
        }
        $logits = Math::matvec($this->lmHead, $h, $this->vocab, $this->hidden);

        return $logits;
    }

    /**
     * Run one transformer layer; returns the updated hidden vector.
     *
     * @param array $layerKv  this layer's KV cache (modified in place)
     */
    private function forwardLayer(int $l, array $h, int $position, array &$layerKv): array
    {
        $hidden = $this->hidden;

        // ----- Load this layer's weights --------------------------------------
        $ln1 = $this->loader->loadData("model.layers.{$l}.input_layernorm.weight");
        $wq  = $this->loader->loadData("model.layers.{$l}.self_attn.q_proj.weight");
        $wk  = $this->loader->loadData("model.layers.{$l}.self_attn.k_proj.weight");
        $wv  = $this->loader->loadData("model.layers.{$l}.self_attn.v_proj.weight");
        $wo  = $this->loader->loadData("model.layers.{$l}.self_attn.o_proj.weight");
        $ln2 = $this->loader->loadData("model.layers.{$l}.post_attention_layernorm.weight");
        $wg  = $this->loader->loadData("model.layers.{$l}.mlp.gate_proj.weight");
        $wu  = $this->loader->loadData("model.layers.{$l}.mlp.up_proj.weight");
        $wd  = $this->loader->loadData("model.layers.{$l}.mlp.down_proj.weight");

        // ===== Self-attention block ===========================================
        // Pre-attention RMSNorm.
        $normed = Math::rmsNorm($h, $ln1, $this->rmsEps);

        // QKV projections. Each is matvec(W, x).
        // q has length num_heads * head_dim = hidden.
        // k, v have length num_kv_heads * head_dim.
        $qProj = $this->numHeads * $this->headDim;
        $kvProj = $this->numKvHeads * $this->headDim;

        $q = Math::matvec($wq, $normed, $qProj, $hidden);
        $k = Math::matvec($wk, $normed, $kvProj, $hidden);
        $v = Math::matvec($wv, $normed, $kvProj, $hidden);

        // Apply RoPE (rotary position embedding) in place.
        if ($this->ropeEnabled) {
            Math::applyRope(
                $q, $k,
                $this->headDim,
                $this->numHeads,
                $this->numKvHeads,
                $position,
                $this->ropeTheta
            );
        }

        // Save K, V into this layer's cache for future positions.
        $layerKv['k'][] = $k;
        $layerKv['v'][] = $v;

        // For each query head, compute scaled dot-product attention against
        // all cached K/V up to and including the current position.
        $scale = 1.0 / sqrt((float)$this->headDim);

        // If GQA: each query head h maps to kv head h % num_kv_heads.
        $kvPerQuery = ($this->numKvHeads > 0)
            ? intdiv($this->numHeads, $this->numKvHeads)
            : 1;

        $attnOut = array_fill(0, $this->numHeads * $this->headDim, 0.0);
        $tCount = count($layerKv['k']);   // number of cached positions

        for ($qh = 0; $qh < $this->numHeads; $qh++) {
            $kvh = ($this->numKvHeads === $this->numHeads)
                ? $qh
                : intdiv($qh, $kvPerQuery);  // GQA: shared KV head

            $qBase = $qh * $this->headDim;

            // ----- Scores for this query head against every cached key --------
            $scores = array_fill(0, $tCount, 0.0);
            for ($t = 0; $t < $tCount; $t++) {
                $kVec = $layerKv['k'][$t];
                $kBase = $kvh * $this->headDim;
                $dot = 0.0;
                for ($d = 0; $d < $this->headDim; $d++) {
                    $dot += $q[$qBase + $d] * $kVec[$kBase + $d];
                }
                $scores[$t] = $dot * $scale;
            }

            // Causal mask: positions > current are masked out (-INF).
            // (We never cache future positions in this streaming design,
            // so the mask is implicitly enforced by only walking $t <= position;
            // but we keep the explicit guard for clarity.)
            for ($t = $position + 1; $t < $tCount; $t++) {
                $scores[$t] = -INF;
            }

            $weights = Math::softmax($scores);

            // ----- Weighted sum of V ------------------------------------------
            $outBase = $qBase;
            for ($d = 0; $d < $this->headDim; $d++) {
                $acc = 0.0;
                for ($t = 0; $t < $tCount; $t++) {
                    $vVec = $layerKv['v'][$t];
                    $vBase = $kvh * $this->headDim;
                    $acc += $weights[$t] * $vVec[$vBase + $d];
                }
                $attnOut[$outBase + $d] = $acc;
            }
        }

        // Output projection: o = Wo @ concatHeads(attnOut).
        $o = Math::matvec($wo, $attnOut, $hidden, $hidden);

        // Residual connection: h = h + o.
        for ($j = 0; $j < $hidden; $j++) {
            $h[$j] += $o[$j];
        }

        // ===== MLP block (SwiGLU) ============================================
        // Pre-MLP RMSNorm.
        $normed2 = Math::rmsNorm($h, $ln2, $this->rmsEps);

        // gate & up projections -> [intermediate]
        $g = Math::matvec($wg, $normed2, $this->intermediate, $hidden);
        $u = Math::matvec($wu, $normed2, $this->intermediate, $hidden);

        // SwiGLU activation: silu(gate) * up.
        $act = array_fill(0, $this->intermediate, 0.0);
        for ($i = 0; $i < $this->intermediate; $i++) {
            $act[$i] = Math::silu($g[$i]) * $u[$i];
        }

        // Down projection -> [hidden].
        $d = Math::matvec($wd, $act, $hidden, $this->intermediate);

        // Residual: h = h + d.
        for ($j = 0; $j < $hidden; $j++) {
            $h[$j] += $d[$j];
        }

        // ----- Free this layer's weights before returning --------------------
        unset($ln1, $wq, $wk, $wv, $wo, $ln2, $wg, $wu, $wd,
              $normed, $q, $k, $v, $attnOut, $o, $normed2, $g, $u, $act, $d);

        return $h;
    }
}
