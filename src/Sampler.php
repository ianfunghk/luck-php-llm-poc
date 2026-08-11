<?php
declare(strict_types=1);

/**
 * src/Sampler.php
 * ===============
 * Temperature sampling over the LM-head logits.
 *
 *     scaled[i] = logits[i] / temperature
 *     probs     = softmax(scaled)
 *     next_id   = sample(probs)    (cumulative-distribution inverse sample)
 *
 * For temperature <= 0 (or extremely small) we fall back to greedy argmax.
 */

namespace PhpLlm;

class Sampler
{
    /**
     * Pick the next token id from a vocab-size vector of logits.
     */
    public static function sample(array $logits, float $temperature): int
    {
        $n = count($logits);

        // Greedy fallback for near-zero temperature.
        if ($temperature <= 1e-3) {
            $best = 0;
            $bestScore = $logits[0];
            for ($i = 1; $i < $n; $i++) {
                if ($logits[$i] > $bestScore) {
                    $bestScore = $logits[$i];
                    $best = $i;
                }
            }
            return $best;
        }

        // Scale + softmax.
        $scaled = array_fill(0, $n, 0.0);
        for ($i = 0; $i < $n; $i++) {
            $scaled[$i] = $logits[$i] / $temperature;
        }
        $probs = Math::softmax($scaled);

        // Sample from the cumulative distribution.
        // mt_rand()/mt_getrandmax() gives a uniform [0,1) float.
        $r = (float)(mt_rand() / mt_getrandmax());
        $cum = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $cum += $probs[$i];
            if ($r < $cum) {
                return $i;
            }
        }
        // Floating-point rounding safety net.
        return $n - 1;
    }
}
