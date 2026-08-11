<?php
declare(strict_types=1);

/**
 * src/Math.php
 * ===========
 * Pure-PHP math primitives used by the Llama2-style forward pass.
 *
 * All tensors are stored as **flat 1-D PHP arrays** (the most memory-efficient
 * representation; nested arrays waste ~80 bytes per element on PHP's zval
 * overhead). Matrices follow **row-major (C-contiguous)** layout, so a tensor
 * with logical shape [R, C] lives in a flat array of length R*C where the
 * element at row r, column c is at index `r*C + c`.
 *
 * Index convention note:
 *   PHP arrays are 1-indexed by default, but here we *force* 0-indexing by
 *   always packing with `unpack` and trimming/normalising so that
 *   `$a[$i]` matches the math (`$i` starts at 0). This keeps the formulas
 *   in the comments honest.
 */

namespace PhpLlm;

final class Math
{
    /**
     * Matrix-vector multiply: y = W @ x
     *
     * W is a flat row-major array with logical shape [out_dim, in_dim].
     * x is a flat array of length in_dim.
     * Returns a flat array of length out_dim.
     *
     * Formula (per output row i):
     *     y[i] = sum_{j=0..in_dim-1}  W[i*in_dim + j] * x[j]
     *
     * This is the hottest loop in the forward pass — every projection layer
     * (Q/K/V/O, gate/up/down, embedding lookup, LM head) calls it.
     */
    public static function matvec(array $W, array $x, int $outDim, int $inDim): array
    {
        $y = array_fill(0, $outDim, 0.0);
        for ($i = 0; $i < $outDim; $i++) {
            $base = $i * $inDim;
            $sum = 0.0;
            // Hoist the inner loop as a tight float accumulator.
            for ($j = 0; $j < $inDim; $j++) {
                $sum += $W[$base + $j] * $x[$j];
            }
            $y[$i] = $sum;
        }
        return $y;
    }

    /**
     * Root Mean Square Layer Normalisation (RMSNorm).
     *
     * Llama2 uses RMSNorm (Zhang & Sennrich 2019) instead of LayerNorm.
     * Unlike LayerNorm, RMSNorm has no mean-subtraction; it only scales by
     * the root-mean-square of the inputs:
     *
     *     ms = mean(x^2) = (1/D) * sum_{i=1..D} x[i]^2
     *     x_norm[i] = x[i] / sqrt(ms + eps)
     *     y[i] = w[i] * x_norm[i]           (w is a learned per-element scale)
     *
     * The eps inside the sqrt prevents division by zero for all-zero inputs.
     */
    public static function rmsNorm(array $x, array $w, float $eps): array
    {
        $d = count($x);
        $ss = 0.0;
        for ($i = 0; $i < $d; $i++) {
            $v = $x[$i];
            $ss += $v * $v;
        }
        // ms = mean of squares; the normaliser is 1 / sqrt(ms + eps).
        $ms = $ss / $d;
        $inv = 1.0 / sqrt($ms + $eps);

        $y = array_fill(0, $d, 0.0);
        for ($i = 0; $i < $d; $i++) {
            $y[$i] = $w[$i] * ($x[$i] * $inv);
        }
        return $y;
    }

    /**
     * SiLU (Sigmoid Linear Unit, aka "swish") activation.
     *
     *     silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
     *
     * Used inside the Llama2 SwiGLU MLP: down(silu(gate(x)) * up(x)).
     */
    public static function silu(float $x): float
    {
        return $x / (1.0 + exp(-$x));
    }

    /**
     * Numerically stable softmax over a flat array of logits.
     *
     * Subtract the max before exp() to prevent overflow when logits are large.
     *     p[i] = exp(z[i] - max(z)) / sum_j exp(z[j] - max(z))
     */
    public static function softmax(array $logits): array
    {
        $n = count($logits);
        if ($n === 0) {
            return [];
        }
        $max = $logits[0];
        for ($i = 1; $i < $n; $i++) {
            if ($logits[$i] > $max) {
                $max = $logits[$i];
            }
        }
        $exp = array_fill(0, $n, 0.0);
        $sum = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $e = exp($logits[$i] - $max);
            $exp[$i] = $e;
            $sum += $e;
        }
        $inv = 1.0 / $sum;
        for ($i = 0; $i < $n; $i++) {
            $exp[$i] *= $inv;
        }
        return $exp;
    }

    /**
     * Apply Rotary Position Embedding (RoPE) to query (q) and key (k) vectors.
     *
     * RoPE (Su et al. 2021) is how Llama2 injects position information. It
     * rotates consecutive pairs of channels within each head by an angle that
     * grows with position, so that the dot product q·k naturally depends on
     * their relative position.
     *
     * For head dimension d and position p, with frequency base theta:
     *     freq_k      = 1 / (theta ^ (2k/d))           for k = 0,1,...,d/2-1
     *     angle_{p,k} = p * freq_k
     *
     * The (2k, 2k+1) pair inside a head is rotated by angle_{p,k}:
     *     q'[2k]   = q[2k]   * cos(angle) - q[2k+1] * sin(angle)
     *     q'[2k+1] = q[2k]   * sin(angle) + q[2k+1] * cos(angle)
     *   (same for k)
     *
     * The q and k arrays are flat and contain all heads laid out as
     * [num_heads, head_dim]. We modify them in place.
     *
     * NOTE: num_kv_heads may differ from num_heads (grouped-query attention).
     * k has length num_kv_heads*head_dim while q has num_heads*head_dim.
     */
    public static function applyRope(
        array &$q,
        array &$k,
        int $headDim,
        int $numHeads,
        int $numKvHeads,
        int $position,
        float $theta
    ): void {
        $half = intdiv($headDim, 2);

        // Pre-compute cos/sin per frequency pair (shared across heads & q/k).
        $cos = array_fill(0, $half, 0.0);
        $sin = array_fill(0, $half, 0.0);
        for ($kk = 0; $kk < $half; $kk++) {
            $freq = 1.0 / pow($theta, (2.0 * $kk) / $headDim);
            $angle = $position * $freq;
            $cos[$kk] = cos($angle);
            $sin[$kk] = sin($angle);
        }

        // Rotate q: layout [num_heads, head_dim].
        for ($h = 0; $h < $numHeads; $h++) {
            $base = $h * $headDim;
            for ($kk = 0; $kk < $half; $kk++) {
                $i0 = $base + $kk;            // even-indexed channel
                $i1 = $base + $kk + $half;    // odd-indexed channel
                $q0 = $q[$i0];
                $q1 = $q[$i1];
                $q[$i0] = $q0 * $cos[$kk] - $q1 * $sin[$kk];
                $q[$i1] = $q0 * $sin[$kk] + $q1 * $cos[$kk];
            }
        }

        // Rotate k: layout [num_kv_heads, head_dim].
        for ($h = 0; $h < $numKvHeads; $h++) {
            $base = $h * $headDim;
            for ($kk = 0; $kk < $half; $kk++) {
                $i0 = $base + $kk;
                $i1 = $base + $kk + $half;
                $k0 = $k[$i0];
                $k1 = $k[$i1];
                $k[$i0] = $k0 * $cos[$kk] - $k1 * $sin[$kk];
                $k[$i1] = $k0 * $sin[$kk] + $k1 * $cos[$kk];
            }
        }
    }
}
