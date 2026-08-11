<?php
declare(strict_types=1);

/**
 * src/Forward.php
 * ===============
 * Llama2-style transformer forward pass in pure PHP, with a memory strategy
 * tuned for the 1 GB shared-hosting ceiling.
 *
 * Memory strategy (updated for stories15M)
 * ----------------------------------------
 * Weight matrices come in two flavours:
 *
 *   - SMALL per-layer weights (q/k/v/o proj, gate/up/down, norms): unpack
 *     into float[] once when the layer is being evaluated, unset() before
 *     loading the next layer. Even at hidden=288 / intermediate=768 this
 *     is only a few MB per layer.
 *
 *   - LARGE vocab-side weights (embed_tokens, lm_head): kept as RAW binary
 *     strings (4 bytes per element) for the whole generation. We never
 *     unpack the whole thing; per-token we substr() the one row we need
 *     and unpack just that. For stories15M (vocab=32000, hidden=288) each
 *     matrix is 36 MB as a string vs ~700 MB unpacked — a 20× saving and
 *     the difference between fitting and OOM.
 *
 * KV cache layout (per layer):
 *   $kvCache[$layer]['k'][$t] = flat float[hidden]      // K vector at time t
 *   $kvCache[$layer]['v'][$t] = flat float[hidden]
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
    private bool $tieEmbeddings;

    /** @var string|null Raw bytes of the embedding matrix [vocab, hidden]. */
    private ?string $embedRaw = null;

    /** @var string|null Raw bytes of the LM head [vocab, hidden]. */
    private ?string $lmHeadRaw = null;

    /** @var array|null Final-norm scale [hidden]. */
    private ?array $finalNorm = null;

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
        $this->tieEmbeddings = (bool)$this->cfg['tie_embeddings'];
    }

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
     * @param array $kvCache modified in place: appends this step's K and V.
     */
    public function forwardToken(int $tokenId, int $position, array &$kvCache): array
    {
        // ----- 1. Embedding lookup (row-streamed) ----------------------------
        if ($this->embedRaw === null) {
            $this->embedRaw = $this->loader->loadTensorRaw('model.embed_tokens.weight');
        }
        // One row = $hidden float32 bytes. Substr + unpack only that row.
        $rowBlob = substr($this->embedRaw, $tokenId * $this->hidden * 4, $this->hidden * 4);
        $h = array_values(unpack('f*', $rowBlob));

        // ----- 2. Transformer layers (streamed) ------------------------------
        for ($l = 0; $l < $this->numLayers; $l++) {
            $h = $this->forwardLayer($l, $h, $position, $kvCache[$l]);
        }

        // ----- 3. Final RMSNorm ----------------------------------------------
        if ($this->finalNorm === null) {
            $this->finalNorm = $this->loader->loadData('model.norm.weight');
        }
        $h = Math::rmsNorm($h, $this->finalNorm, $this->rmsEps);

        // ----- 4. LM head -> logits (row-streamed) ---------------------------
        // For each vocab id i: logits[i] = dot(lm_head_row[i], h).
        // We walk the matrix one row at a time so the full unpack never happens.
        $lmHeadName = $this->tieEmbeddings ? 'model.embed_tokens.weight' : 'lm_head.weight';
        if ($this->lmHeadRaw === null) {
            $this->lmHeadRaw = $this->loader->loadTensorRaw($lmHeadName);
        }
        $logits = Math::matvecRaw($this->lmHeadRaw, $h, $this->vocab, $this->hidden);

        return $logits;
    }

    /**
     * Run one transformer layer; returns the updated hidden vector.
     *
     * @param array $layerKv this layer's KV cache (modified in place)
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
        $normed = Math::rmsNorm($h, $ln1, $this->rmsEps);

        $qProj = $this->numHeads * $this->headDim;
        $kvProj = $this->numKvHeads * $this->headDim;

        $q = Math::matvec($wq, $normed, $qProj, $hidden);
        $k = Math::matvec($wk, $normed, $kvProj, $hidden);
        $v = Math::matvec($wv, $normed, $kvProj, $hidden);

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

        $layerKv['k'][] = $k;
        $layerKv['v'][] = $v;

        $scale = 1.0 / sqrt((float)$this->headDim);

        $kvPerQuery = ($this->numKvHeads > 0)
            ? intdiv($this->numHeads, $this->numKvHeads)
            : 1;

        $attnOut = array_fill(0, $this->numHeads * $this->headDim, 0.0);
        $tCount = count($layerKv['k']);

        for ($qh = 0; $qh < $this->numHeads; $qh++) {
            $kvh = ($this->numKvHeads === $this->numHeads)
                ? $qh
                : intdiv($qh, $kvPerQuery);

            $qBase = $qh * $this->headDim;

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

            // Causal mask (defensive; we never cache future positions).
            for ($t = $position + 1; $t < $tCount; $t++) {
                $scores[$t] = -INF;
            }

            $weights = Math::softmax($scores);

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

        $o = Math::matvec($wo, $attnOut, $hidden, $hidden);

        for ($j = 0; $j < $hidden; $j++) {
            $h[$j] += $o[$j];
        }

        // ===== MLP block (SwiGLU) ============================================
        $normed2 = Math::rmsNorm($h, $ln2, $this->rmsEps);

        $g = Math::matvec($wg, $normed2, $this->intermediate, $hidden);
        $u = Math::matvec($wu, $normed2, $this->intermediate, $hidden);

        $act = array_fill(0, $this->intermediate, 0.0);
        for ($i = 0; $i < $this->intermediate; $i++) {
            $act[$i] = Math::silu($g[$i]) * $u[$i];
        }

        $d = Math::matvec($wd, $act, $hidden, $this->intermediate);

        for ($j = 0; $j < $hidden; $j++) {
            $h[$j] += $d[$j];
        }

        // ----- Free this layer's weights before returning --------------------
        unset($ln1, $wq, $wk, $wv, $wo, $ln2, $wg, $wu, $wd,
              $normed, $q, $k, $v, $attnOut, $o, $normed2, $g, $u, $act, $d);

        return $h;
    }

    /**
     * Optional: release the embed/lm_head strings between independent
     * generations. Not normally needed; PHP cleans up at end of request.
     */
    public function releasePersistentWeights(): void
    {
        $this->embedRaw = null;
        $this->lmHeadRaw = null;
        $this->finalNorm = null;
    }
}
