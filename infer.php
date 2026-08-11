<?php
declare(strict_types=1);

/**
 * infer.php
 * =========
 * CLI entry point for the php-llm proof of concept.
 *
 * Also exposes run_inference() so that web/index.php (or any other caller)
 * can reuse the exact same code path with a different output formatter.
 *
 * Usage (CLI):
 *     php infer.php "Once upon a time"
 *     php infer.php                  # uses default PROMPT below
 *     PROMPT="hello" php infer.php   # env var also accepted
 *
 * Constraints (shared-hosting friendly):
 *     - 1 GB memory_limit, 120 s time limit.
 *     - All weights streamed from disk; nothing preloaded.
 *
 * See src/*.php for the actual transformer math.
 */

// ----------------------------------------------------------------------------
// Tunables (override via environment variables when needed)
// ----------------------------------------------------------------------------

/** Directory containing config.json, manifest.json, tokens.json, *.bin */
const WEIGHTS_DIR = __DIR__ . '/weights';

/** Hard cap on total tokens (prompt + generated). Reduce first if you hit memory limits. */
const CONTEXT_LIMIT = 128;

/** How many NEW tokens to generate. */
const MAX_TOKENS = 128;

/** Sampling temperature. 0.x < 1.0 sharpens; > 1.0 flattens. */
const TEMPERATURE = 0.8;

/** Default prompt when none is provided. */
const DEFAULT_PROMPT = 'Once upon a time';

// ----------------------------------------------------------------------------
// Runtime safety
// ----------------------------------------------------------------------------
set_time_limit(120);
error_reporting(E_ALL);
ini_set('display_errors', '1');

// ----------------------------------------------------------------------------
// PSR-4-ish autoloader for the PhpLlm\ namespace under src/
// ----------------------------------------------------------------------------
spl_autoload_register(function (string $class): void {
    $prefix = 'PhpLlm\\';
    if (strncmp($class, $prefix, strlen($prefix)) !== 0) {
        return;
    }
    $relative = substr($class, strlen($prefix));
    $file = __DIR__ . '/src/' . $relative . '.php';
    if (is_file($file)) {
        require $file;
    }
});

// ----------------------------------------------------------------------------
// Shared inference routine (used by both CLI and web/index.php)
// ----------------------------------------------------------------------------

/**
 * Run prompt -> generated text.
 *
 * @return array{prompt:string, generated_ids:int[], generated_text:string,
 *               tokens_generated:int, elapsed_sec:float,
 *               peak_memory_bytes:int, mem_per_token_bytes:int[]}
 */
function run_inference(string $prompt, int $maxTokens = MAX_TOKENS,
                       float $temperature = TEMPERATURE,
                       ?string $weightsDir = null): array
{
    $weightsDir = $weightsDir ?? WEIGHTS_DIR;

    $loader   = new PhpLlm\Loader($weightsDir);
    $tokenizer = new PhpLlm\Tokenizer(
        $loader->tokens,
        $loader->config['bos_token_id'] ?? 1,
        $loader->config['eos_token_id'] ?? 2,
        0,
        $loader->config['tokenizer_kind'] ?? 'sentencepiece'
    );
    $engine   = new PhpLlm\Forward($loader);
    $kvCache  = $engine->newKvCache();

    $eosId   = $loader->config['eos_token_id'] ?? 2;
    $ctxCap  = min($loader->config['context_limit'] ?? CONTEXT_LIMIT,
                   $loader->config['max_position'] ?? CONTEXT_LIMIT);

    // ---------------- Encode prompt -----------------------------------------
    $promptIds = $tokenizer->encode($prompt);

    // Cap prompt length so generation still has room within ctxCap.
    if (count($promptIds) > $ctxCap - 1) {
        $promptIds = array_slice($promptIds, 0, $ctxCap - 1);
    }

    // ---------------- Prefill: feed every prompt token but the last ---------
    // The last prompt token is fed inside the decode loop so its logits get sampled.
    $generated = [];
    $memPerToken = [];

    $nPrompt = count($promptIds);
    // We will run forward on tokens [0 .. nPrompt-2] for prefill, then sample
    // from the logits of token [nPrompt-2] to get the first generated token.
    $position = 0;
    $lastLogits = null;
    for ($i = 0; $i < $nPrompt - 1; $i++) {
        $lastLogits = $engine->forwardToken($promptIds[$i], $position++, $kvCache);
    }
    // If prompt was just [BOS] (single token), lastLogits stays null; feed it
    // here so we always have logits to sample from.
    if ($lastLogits === null) {
        $lastLogits = $engine->forwardToken($promptIds[0], $position++, $kvCache);
    }

    // ---------------- Decode loop -------------------------------------------
    $start = microtime(true);
    for ($step = 0; $step < $maxTokens; $step++) {
        // Respect the absolute context cap (prompt + generated).
        if ($position >= $ctxCap) {
            break;
        }
        $nextId = PhpLlm\Sampler::sample($lastLogits, $temperature);
        if ($nextId === $eosId) {
            break;
        }
        $generated[] = $nextId;
        $memPerToken[] = memory_get_usage(true);

        // Feed the freshly sampled token to get logits for the next step.
        $lastLogits = $engine->forwardToken($nextId, $position++, $kvCache);
    }
    $elapsed = microtime(true) - $start;

    return [
        'prompt'             => $prompt,
        'prompt_ids'         => $promptIds,
        'generated_ids'      => $generated,
        'generated_text'     => $tokenizer->decode($generated),
        'tokens_generated'   => count($generated),
        'elapsed_sec'        => $elapsed,
        'peak_memory_bytes'  => memory_get_peak_usage(true),
        'mem_per_token_bytes'=> $memPerToken,
    ];
}

// ----------------------------------------------------------------------------
// CLI entry point (skipped when this file is included by web/index.php)
// ----------------------------------------------------------------------------

if (PHP_SAPI === 'cli' && !getenv('PHPLLM_NO_AUTORUN')) {

    // Prompt resolution priority: argv[1] -> $PROMPT env -> DEFAULT_PROMPT.
    $prompt = DEFAULT_PROMPT;
    if (isset($argv[1]) && $argv[1] !== '') {
        $prompt = $argv[1];
    } elseif (($envPrompt = getenv('PROMPT')) !== false && $envPrompt !== '') {
        $prompt = $envPrompt;
    }

    fwrite(STDERR, "php-llm: loading weights from " . WEIGHTS_DIR . " ...\n");
    $t0 = microtime(true);
    $result = run_inference($prompt);
    $loadTime = microtime(true) - $t0;

    // ---- Output -----------------------------------------------------------
    echo "Prompt   : " . $result['prompt'] . "\n";
    echo "Generated: " . $result['generated_text'] . "\n";
    echo str_repeat('-', 60) . "\n";
    echo "Tokens generated : " . $result['tokens_generated'] . "\n";
    echo "Decode elapsed   : " . sprintf("%.3f s", $result['elapsed_sec']) . "\n";
    echo "Total wall time  : " . sprintf("%.3f s", $loadTime) . "\n";
    echo "Peak memory      : " . number_format($result['peak_memory_bytes']) . " bytes ("
        . sprintf("%.1f MB", $result['peak_memory_bytes'] / 1024 / 1024) . ")\n";
    echo "Memory limit     : " . ini_get('memory_limit') . "\n";

    if (count($result['mem_per_token_bytes']) > 0) {
        $first = $result['mem_per_token_bytes'][0];
        $last  = $result['mem_per_token_bytes'][count($result['mem_per_token_bytes']) - 1];
        echo "Mem at token #1   : " . sprintf("%.1f MB", $first / 1024 / 1024) . "\n";
        echo "Mem at last token : " . sprintf("%.1f MB", $last / 1024 / 1024) . "\n";
    }

    exit(0);
}
