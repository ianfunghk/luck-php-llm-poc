<?php
declare(strict_types=1);

/**
 * src/Loader.php
 * ==============
 * Streams weights from disk one tensor at a time, exactly as a PHP script on
 * a 1-GB shared host needs: never load the whole model into PHP arrays at once.
 *
 * The on-disk layout (produced by export_weights.py) is:
 *
 *     weights/
 *         manifest.json   # {tensors: [{name, shape, dtype, filename, nbytes, byte_offset}, ...]}
 *         config.json     # hyperparameters (hidden_size, num_layers, ...)
 *         tokens.json     # [{id, piece, score, type}, ...]
 *         <safe_name>.bin # one little-endian float32 file per tensor
 *
 * Workflow:
 *     $loader = new Loader('/path/to/weights');
 *     $tensor = $loader->loadTensor('model.layers.0.self_attn.q_proj.weight');
 *     //   $tensor['data']  -> 0-indexed flat float[] (length = prod(shape))
 *     //   $tensor['shape'] -> [out, in]
 *     unset($tensor);   // free before loading the next layer
 */

namespace PhpLlm;

class Loader
{
    /** @var string Absolute path to the weights/ directory. */
    public string $dir;

    /** @var array Parsed config.json. */
    public array $config;

    /** @var array<string,array{name:string,shape:int[],dtype:string,filename:string,nbytes:int,byte_offset:int}> */
    private array $byName = [];

    /** @var array Parsed tokens.json: [{id, piece, score, type}, ...]. */
    public array $tokens;

    public function __construct(string $dir)
    {
        $this->dir = rtrim($dir, '/');

        $this->config = json_decode(file_get_contents($this->dir . '/config.json'), true);
        if (!is_array($this->config)) {
            throw new \RuntimeException("Failed to read {$this->dir}/config.json");
        }

        $manifest = json_decode(file_get_contents($this->dir . '/manifest.json'), true);
        if (!is_array($manifest) || !isset($manifest['tensors'])) {
            throw new \RuntimeException("Failed to read {$this->dir}/manifest.json");
        }
        foreach ($manifest['tensors'] as $entry) {
            $this->byName[$entry['name']] = $entry;
        }

        $this->tokens = json_decode(file_get_contents($this->dir . '/tokens.json'), true);
        if (!is_array($this->tokens)) {
            throw new \RuntimeException("Failed to read {$this->dir}/tokens.json");
        }
    }

    /**
     * Read one tensor from disk and return a 0-indexed flat float[] plus shape.
     *
     * Internally:
     *   1. fopen the .bin (no whole-file slurp).
     *   2. fread exactly nbytes.
     *   3. unpack('f*', $blob) — PHP returns a 1-indexed array; we slice [1..N]
     *      via array_values to give callers a clean 0-indexed array so the
     *      indexing math in Math.php stays honest.
     *
     * `unpack('f*', ...)` parses in host byte order, which is little-endian on
     * every shared-hosting platform we care about (x86-64 / arm64 Linux). The
     * Python export forces little-endian on disk, so the bytes line up.
     */
    public function loadTensor(string $name): array
    {
        if (!isset($this->byName[$name])) {
            throw new \RuntimeException("Unknown tensor: {$name}");
        }
        $entry = $this->byName[$name];
        $path = $this->dir . '/' . $entry['filename'];

        $fp = fopen($path, 'rb');
        if ($fp === false) {
            throw new \RuntimeException("Failed to open {$path}");
        }
        if ($entry['byte_offset'] > 0) {
            fseek($fp, $entry['byte_offset']);
        }
        $blob = fread($fp, $entry['nbytes']);
        fclose($fp);
        if (strlen($blob) !== $entry['nbytes']) {
            throw new \RuntimeException(
                "Short read for {$name}: expected {$entry['nbytes']}, got " . strlen($blob)
            );
        }

        $unpacked = unpack('f*', $blob);
        // PHP unpack returns 1-indexed; normalise to 0-indexed.
        return [
            'data'  => array_values($unpacked),
            'shape' => $entry['shape'],
        ];
    }

    /**
     * Convenience: load only the `data` array. Use when shape is known.
     */
    public function loadData(string $name): array
    {
        return $this->loadTensor($name)['data'];
    }

    /**
     * Helper: how many logical elements (floats) a tensor holds.
     */
    public static function numElements(array $shape): int
    {
        $n = 1;
        foreach ($shape as $d) {
            $n *= $d;
        }
        return $n;
    }
}
