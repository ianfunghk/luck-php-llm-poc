<?php
declare(strict_types=1);

/**
 * web/index.php
 * =============
 * HTTP entry point for the php-llm proof of concept.
 *
 * Reuses run_inference() from infer.php so CLI and HTTP produce identical
 * results. Output format is chosen by `?format=` (or the Accept header):
 *
 *     http://host/?prompt=hello&max_tokens=64&temperature=0.8&format=text
 *     http://host/?prompt=hello&format=json
 *     http://host/                            -> HTML form
 *
 * The `PHPLLM_NO_AUTORUN` env var is set before including infer.php so the
 * CLI block at the bottom of infer.php is skipped under HTTP.
 */

putenv('PHPLLM_NO_AUTORUN=1');
require __DIR__ . '/../infer.php';

// ----------------------------------------------------------------------------
// Parse request
// ----------------------------------------------------------------------------
$prompt       = isset($_GET['prompt']) ? (string)$_GET['prompt'] : 'Once upon a time';
$maxTokens    = isset($_GET['max_tokens']) ? max(1, min(256, (int)$_GET['max_tokens'])) : 64;
$temperature  = isset($_GET['temperature']) ? (float)$_GET['temperature'] : 0.8;
$format       = $_GET['format'] ?? null;
$acceptHtml   = (isset($_SERVER['HTTP_ACCEPT']) && strpos($_SERVER['HTTP_ACCEPT'], 'text/html') !== false);

if ($format === null) {
    // Heuristic: prefer HTML for browsers, plain text otherwise.
    $format = $acceptHtml ? 'html' : 'text';
}
$format = strtolower($format);

// Only actually run the model when a prompt was explicitly provided OR when
// a non-HTML format was requested. The bare HTML page shows the form first.
$shouldRun = isset($_GET['prompt']) || $format !== 'html';

$result = null;
$error  = null;
if ($shouldRun) {
    try {
        set_time_limit(120);
        $result = run_inference($prompt, $maxTokens, $temperature);
    } catch (\Throwable $e) {
        $error = $e->getMessage() . "\n" . $e->getTraceAsString();
    }
}

// ----------------------------------------------------------------------------
// Output
// ----------------------------------------------------------------------------
if ($format === 'json') {
    header('Content-Type: application/json; charset=utf-8');
    if ($error !== null) {
        echo json_encode(['error' => $error]);
    } else {
        echo json_encode([
            'prompt'             => $result['prompt'],
            'generated'          => $result['generated_text'],
            'tokens'             => $result['tokens_generated'],
            'elapsed_sec'        => round($result['elapsed_sec'], 4),
            'peak_memory_mb'     => round($result['peak_memory_bytes'] / 1024 / 1024, 2),
        ], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    }
    exit;
}

if ($format === 'text' || $error !== null) {
    header('Content-Type: text/plain; charset=utf-8');
    if ($error !== null) {
        echo "ERROR: " . $error;
    } else {
        echo "Prompt   : " . $result['prompt'] . "\n";
        echo "Generated: " . $result['generated_text'] . "\n";
        echo "Tokens   : " . $result['tokens_generated'] . "\n";
        echo "Elapsed  : " . sprintf("%.3f s", $result['elapsed_sec']) . "\n";
        echo "Peak mem : " . sprintf("%.1f MB", $result['peak_memory_bytes'] / 1024 / 1024) . "\n";
    }
    exit;
}

// HTML -----------------------------------------------------------------------
header('Content-Type: text/html; charset=utf-8');
?><!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>php-llm</title>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    max-width: 760px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55;
  }
  h1 { font-size: 1.3rem; margin-bottom: 0.25rem; }
  .sub { color: #888; font-size: .85rem; margin-bottom: 1.5rem; }
  form { display: flex; flex-direction: column; gap: .5rem; margin-bottom: 1.5rem; }
  textarea { width: 100%; font: inherit; padding: .5rem; box-sizing: border-box; }
  .row { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; }
  label { font-size: .85rem; color: #888; }
  input[type=number] { width: 5rem; }
  button { padding: .4rem 1rem; font: inherit; cursor: pointer; }
  pre {
    background: rgba(127,127,127,.12); padding: .75rem 1rem; border-radius: 6px;
    white-space: pre-wrap; word-wrap: break-word;
  }
  .meta { color: #888; font-size: .85rem; margin-top: .5rem; }
  .err  { color: #c33; }
</style>
</head>
<body>
<h1>php-llm &mdash; pure-PHP Llama2 forward pass</h1>
<div class="sub">
  Model: <code>shibatch/tiny1m</code> (896k params, 4 layers, hidden=128, vocab=512)
  &middot; streamable per-tensor float32 weights &middot; no FFI, no exec.
</div>

<form method="get" action="">
  <label for="prompt">Prompt</label>
  <textarea id="prompt" name="prompt" rows="2" placeholder="Once upon a time"><?= htmlspecialchars($prompt, ENT_QUOTES) ?></textarea>
  <div class="row">
    <label>max_tokens <input type="number" name="max_tokens" value="<?= htmlspecialchars((string)$maxTokens) ?>" min="1" max="256"></label>
    <label>temperature <input type="number" name="temperature" value="<?= htmlspecialchars((string)$temperature) ?>" min="0" max="2" step="0.1"></label>
    <label>format
      <select name="format">
        <option value="html" <?= $format==='html'?'selected':'' ?>>html</option>
        <option value="text" <?= $format==='text'?'selected':'' ?>>text</option>
        <option value="json" <?= $format==='json'?'selected':'' ?>>json</option>
      </select>
    </label>
    <button type="submit">Generate</button>
  </div>
</form>

<?php if ($error !== null): ?>
  <h2>Error</h2>
  <pre class="err"><?= htmlspecialchars($error) ?></pre>
<?php elseif ($result !== null): ?>
  <h2>Generated</h2>
  <pre><?= htmlspecialchars($result['generated_text']) ?></pre>
  <div class="meta">
    <?= $result['tokens_generated'] ?> tokens in <?= sprintf("%.3f", $result['elapsed_sec']) ?> s
    &middot; peak memory <?= sprintf("%.1f MB", $result['peak_memory_bytes']/1024/1024) ?>
    &middot; PHP <?= htmlspecialchars(PHP_VERSION) ?>
  </div>
<?php else: ?>
  <p class="sub">Press <b>Generate</b> to run inference.</p>
<?php endif; ?>

</body>
</html>
