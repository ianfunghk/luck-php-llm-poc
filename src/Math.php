<?php
declare(strict_types=1);

/**
 * src/Math.php
 * ===========
 * Pure-PHP math primitives used by the Llama2-style forward pass.
 *
 * Two representations of a 2-D weight matrix W with logical shape [R, C]:
 *
 *   - "flat array"  : 0-indexed float[] of length R*C. Logical element
 *                     W[r][c] lives at index r*C + c. This is what callers
 *                     used in v1; element overhead is ~80 bytes/float.
 *
 *   - "raw binary"  : PHP string holding R*C*4 little-endian float32 bytes
 *                     (exactly the on-disk layout). Overhead is essentially
 *                     the byte count. We unpack rows lazily as needed.
 *
 * The stories15M upgrade added the raw-binary variants (matvecRaw, dotRaw)
 * because the lm_head/embedding matrices at vocab=32000 would consume
 * ~1.5 GB after unpack, blowing the 1 GB ceiling. Keeping them as strings
 * keeps peak memory near the on-disk size.
 *
 * Index convention: PHP arrays here are forced to 0-indexed (we use
 * array_values(unpack(...)) to normalise).
 */

namespace PhpLlm;

final class Math
{
    // --------------------------------------------------------------------- //
    // Flat-array variants (used for small tensors and per-layer weights).
    // --------------------------------------------------------------------- //

    /**
     * Matrix-vector multiply: y = W @ x with W as a flat float[].
     *
     * W is row-major [outDim, inDim]. Element W[i][j] is at index i*inDim + j.
     *     y[i] = sum_j W[i*inDim + j] * x[j]
     */
    public static function matvec(array $W, array $x, int $outDim, int $inDim): array
    {
        $y = array_fill(0, $outDim, 0.0);
        for ($i = 0; $i < $outDim; $i++) {
            $base = $i * $inDim;
            $sum = 0.0;
            for ($j = 0; $j < $inDim; $j++) {
                $sum += $W[$base + $j] * $x[$j];
            }
            $y[$i] = $sum;
        }
        return $y;
    }

    // --------------------------------------------------------------------- //
    // Raw-binary variants (used for large vocab-side matrices).
    // --------------------------------------------------------------------- //

    /**
     * Matrix-vector multiply where W is a raw little-endian float32 string.
     *
     * We walk W one row at a time: substr the row's bytes, unpack once,
     * then do the inner dot product. This bounds peak memory to roughly
     * (one row's worth of floats) regardless of W's total size.
     *
     *     y[i] = sum_j W_row[i][j] * x[j]      for i = 0..outDim-1
     *
     * @param string $Wraw   raw bytes, length outDim*inDim*4
     * @param array  $x      float[] of length inDim
     */
    public static function matvecRaw(string $Wraw, array $x, int $outDim, int $inDim): array
    {
        $rowBytes = $inDim * 4;
        $y = array_fill(0, $outDim, 0.0);
        for ($i = 0; $i < $outDim; $i++) {
            $rowBlob = substr($Wraw, $i * $rowBytes, $rowBytes);
            // unpack('f*', ...) is 1-indexed; iterate $j from 1..inDim.
            $wRow = unpack('f*', $rowBlob);
            $sum = 0.0;
            for ($j = 0; $j < $inDim; $j++) {
                $sum += $wRow[$j + 1] * $x[$j];
            }
            $y[$i] = $sum;
        }
        return $y;
    }

    /**
     * Vector-vector dot product where one vector is a raw float32 string.
     *
     * Used when x is a small float[] (the hidden vector) and W is a single
     * row of a large matrix unpacked on the fly.
     */
    public static function dotRaw(string $rowBlob, array $x, int $inDim): float
    {
        $wRow = unpack('f*', $rowBlob);
        $sum = 0.0;
        for ($j = 0; $j < $inDim; $j++) {
            $sum += $wRow[$j + 1] * $x[$j];
        }
        return $sum;
    }

    // --------------------------------------------------------------------- //
    // Normalisation + activation (small tensors only; flat-array form).
    // --------------------------------------------------------------------- //

    /**
     * Root Mean Square Layer Normalisation (RMSNorm).
     *
     *     ms = (1/D) * sum_i x[i]^2
     *     y[i] = w[i] * x[i] / sqrt(ms + eps)
     */
    public static function rmsNorm(array $x, array $w, float $eps): array
    {
        $d = count($x);
        $ss = 0.0;
        for ($i = 0; $i < $d; $i++) {
            $v = $x[$i];
            $ss += $v * $v;
        }
        $ms = $ss / $d;
        $inv = 1.0 / sqrt($ms + $eps);

        $y = array_fill(0, $d, 0.0);
        for ($i = 0; $i < $d; $i++) {
            $y[$i] = $w[$i] * ($x[$i] * $inv);
        }
        return $y;
    }

    /**
     * SiLU (Sigmoid Linear Unit / "swish"):
     *     silu(x) = x / (1 + exp(-x))
     */
    public static function silu(float $x): float
    {
        return $x / (1.0 + exp(-$x));
    }

    /**
     * Numerically stable softmax over a flat array of logits.
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
     * Softmax over a raw float32 string of logits.
     *
     * Used by the LM-head path when logits come straight from matvecRaw.
     */
    public static function softmaxRaw(string $raw): array
    {
        $vals = unpack('f*', $raw);
        // unpack is 1-indexed; normalise.
        $logits = array_values($vals);
        return self::softmax($logits);
    }

    /**
     * Apply Rotary Position Embedding (RoPE) to query (q) and key (k) vectors.
     *
     * For head dimension d and position p with frequency base theta:
     *     freq_k      = 1 / (theta ^ (2k/d))           k = 0..d/2-1
     *     angle_{p,k} = p * freq_k
     * Pair (2k, 2k+1) inside a head is rotated by angle_{p,k}:
     *     q'[2k]   = q[2k]*cos - q[2k+1]*sin
     *     q'[2k+1] = q[2k]*sin + q[2k+1]*cos
     *   (same for k)
     *
     * q has shape [num_heads, head_dim]; k has shape [num_kv_heads, head_dim].
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

        $cos = array_fill(0, $half, 0.0);
        $sin = array_fill(0, $half, 0.0);
        for ($kk = 0; $kk < $half; $kk++) {
            $freq = 1.0 / pow($theta, (2.0 * $kk) / $headDim);
            $angle = $position * $freq;
            $cos[$kk] = cos($angle);
            $sin[$kk] = sin($angle);
        }

        for ($h = 0; $h < $numHeads; $h++) {
            $base = $h * $headDim;
            for ($kk = 0; $kk < $half; $kk++) {
                $i0 = $base + $kk;
                $i1 = $base + $kk + $half;
                $q0 = $q[$i0];
                $q1 = $q[$i1];
                $q[$i0] = $q0 * $cos[$kk] - $q1 * $sin[$kk];
                $q[$i1] = $q0 * $sin[$kk] + $q1 * $cos[$kk];
            }
        }

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
