<?php
declare(strict_types=1);

/**
 * src/Loader.php
 * ==============
 * Streams weights from disk in TWO representations:
 *
 *   1. loadTensorRaw($name) -> string
 *      Returns the raw little-endian float32 bytes untouched. This is the
 *      memory-cheap representation: a [32000, 288] matrix is exactly 36 MB
 *      as a PHP string (vs ~700 MB after unpack into a float[] array).
 *
 *   2. loadTensor($name) -> ['data' => float[], 'shape' => int[]]
 *      The original unpack-into-array form, for small tensors (RMSNorm
 *      scales, etc.) where the array overhead doesn't matter.
 *
 * For matrix-vector multiplies against large weight matrices we provide
 * helper accessors that:
 *   - return one row at a time as a float[] (loadTensorRow)
 *   - expose the raw string + shape (loadTensorRaw) so callers can drive
 *     their own streaming loop
 *
 * PHP's `unpack('f*', ...)` parses in host byte order (little-endian on every
 * platform we care about), and the Python export forces little-endian on
 * disk, so bytes line up.
 */

namespace PhpLlm;

class Loader
{
    public string $dir;
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
     * Read one tensor's raw little-endian float32 bytes (no unpack).
     *
     * Use this for large matrices that you want to keep in memory as a string
     * and walk row-by-row with `substr` + `unpack` on demand. Memory cost is
     * essentially the on-disk size (1 byte per byte), with very small PHP
     * string overhead.
     */
    public function loadTensorRaw(string $name): string
    {
        if (!isset($this->byName[$name])) {
            throw new \RuntimeException("Unknown tensor: {$name}");
        }
        $entry = $this->byName[$name];
        $path = $this->dir . '/' . $entry['filename'];
        $blob = file_get_contents($path);
        if ($blob === false || strlen($blob) !== $entry['nbytes']) {
            throw new \RuntimeException(
                "Failed to read {$name} ({$entry['nbytes']} bytes) from {$path}"
            );
        }
        return $blob;
    }

    /**
     * Read one tensor fully into a 0-indexed float[] + shape.
     *
     * Use for SMALL tensors only (norms, biases). For weight matrices prefer
     * loadTensorRaw() + loadTensorRow() to keep memory bounded.
     */
    public function loadTensor(string $name): array
    {
        $entry = $this->byName[$name];
        $blob = $this->loadTensorRaw($name);
        $unpacked = unpack('f*', $blob);
        return [
            'data'  => array_values($unpacked),
            'shape' => $entry['shape'],
        ];
    }

    public function loadData(string $name): array
    {
        return $this->loadTensor($name)['data'];
    }

    /**
     * Get the shape for a tensor without touching disk.
     */
    public function shapeOf(string $name): array
    {
        if (!isset($this->byName[$name])) {
            throw new \RuntimeException("Unknown tensor: {$name}");
        }
        return $this->byName[$name]['shape'];
    }

    /**
     * Read ONE row from a [R, C] matrix stored as raw float32 bytes.
     *
     * Returns a 0-indexed float[] of length $cols. This is the workhorse for
     * the embedding lookup and the LM-head projection: both touch a single
     * row per token, so we never need the whole matrix unpacked at once.
     *
     * @param string $blob  raw bytes from loadTensorRaw()
     * @param int    $row   0-indexed row
     * @param int    $cols  number of columns (row length)
     */
    public static function rowOf(string $blob, int $row, int $cols): array
    {
        $bytesPerRow = $cols * 4;
        $offset = $row * $bytesPerRow;
        $rowBlob = substr($blob, $offset, $bytesPerRow);
        $unpacked = unpack('f*', $rowBlob);
        return array_values($unpacked);
    }

    public static function numElements(array $shape): int
    {
        $n = 1;
        foreach ($shape as $d) {
            $n *= $d;
        }
        return $n;
    }
}
