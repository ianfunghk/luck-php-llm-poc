<?php
declare(strict_types=1);

/**
 * src/WeightCache.php
 * ==================
 * Cross-process shared-memory cache for weight tensors, built on the shmop
 * extension (no APCu dependency, no external libs).
 *
 * WHY
 * ---
 * On the HTTP path every PHP-FPM worker would otherwise re-read 93 MB of
 * `.bin` files from disk per request. shmop lets the first worker slurp
 * them once into a single shared segment; subsequent workers (CLI or FPM)
 * attach to the same segment and `sho_read()` slices directly — no disk
 * I/O, no per-process unpack.
 *
 * LAYOUT (single System V shared memory segment)
 * ---------------------------------------------
 *   Header (fixed size, 64 bytes):
 *     [0..3]    magic = 0x454C4C50  ("PLLE" little-endian, "php-llm")
 *     [4..7]    model_hash (32 bits from config.json inode + mtime + size)
 *     [8..11]   tensor_count (uint32 LE)
 *     [12..15]  reserved (0)
 *     [16..63]  reserved (0)
 *
 *   Offset table (tensor_count * 64 bytes per entry):
 *     [0..63]   tensor name (null-padded ASCII, max 64 chars)
 *     [64..71]  byte offset within segment (uint64 LE)
 *     [72..79]  byte length (uint64 LE)
 *
 *   Tensor payloads:
 *     concatenated raw bytes (the same bytes that live in the .bin files)
 *
 * The whole segment is allocated up-front with the exact size needed plus a
 * small safety margin. If a model's weights change on disk, the model_hash
 * check rejects the existing segment; the caller falls back to disk and a
 * new segment is created (deleting the old one first).
 *
 * FALLBACK
 * --------
 * If shmop is not available, or if attaching/creating fails for any reason
 * (permissions, segment size limit, etc.), every method degrades silently
 * to "cache miss" — callers always get a correct answer, just from disk.
 */

namespace PhpLlm;

class WeightCache
{
    /** Header is fixed at 64 bytes; each offset entry is 64 bytes (name) + 16 (offset+len). */
    private const HEADER_BYTES     = 64;
    private const ENTRY_NAME_BYTES = 64;
    private const ENTRY_META_BYTES = 16;  // 8 offset + 8 length
    private const ENTRY_BYTES      = self::ENTRY_NAME_BYTES + self::ENTRY_META_BYTES;
    private const MAGIC            = 0x454C4C50;  // 'PLLE' LE = "php-llm" tag
    private const SAFETY_MARGIN    = 4096;

    private string $dir;
    private int $modelHash;
    private int $shmKey;

    /** @var array<string,array{offset:int,length:int}>|null null until first lookup */
    private ?array $index = null;

    /** Whether shmop is available AND a usable segment is attached. */
    private bool $active = false;

    /** @var \Shmop|null */
    private $shm = null;

    public bool $debug = false;
    public int $hits = 0;
    public int $misses = 0;

    /**
     * Call shmop_close() if it exists (deprecated in 8.2+, removed in 8.4+).
     * Suppresses the deprecation warning.
     */
    private static function closeShm($shm): void
    {
        if (function_exists('shmop_close')) {
            @shmop_close($shm);
        }
    }

    public function __construct(string $dir, array $config)
    {
        $this->dir = $dir;

        // Hash the config.json file's identity (inode + size + mtime) into a
        // 32-bit model signature. If anything about the export changes, this
        // changes and invalidates the cache.
        $cfgPath = $dir . '/config.json';
        if (is_file($cfgPath)) {
            $s = stat($cfgPath);
            if ($s !== false) {
                $this->modelHash = (int)(
                    ($s['ino'] ?? 0)
                    ^ ($s['size'] & 0xFFFFFFFF)
                    ^ ((int)$s['mtime'] & 0xFFFFFFFF)
                );
                // Force into positive 32-bit range (PHP int is signed 64-bit).
                $this->modelHash &= 0x7FFFFFFF;
            } else {
                $this->modelHash = 0;
            }
        } else {
            $this->modelHash = 0;
        }

        // Derive a System V IPC key from the directory path.
        // ftok needs a real file — pin it on config.json so the key is stable.
        $this->shmKey = $this->deriveKey($cfgPath);
    }

    /**
     * Attempt to attach to an existing segment matching our model hash.
     * Returns true on success. On any failure (no shmop ext, missing segment,
     * stale contents), returns false and the caller should go to disk + call
     * `prime()` to rebuild.
     */
    public function attach(): bool
    {
        if ($this->active) {
            return true;
        }
        if (!function_exists('shmop_open')) {
            return false;
        }

        // Probe for an existing segment at our key (mode 'a' = attach-only,
        // but we need a non-zero size — use 1 since we won't write).
        // Suppress warnings; missing segment is the common cold-start case.
        set_error_handler(function () { /* swallow */ });
        try {
            $shm = @shmop_open($this->shmKey, 'a', 0, 0);
        } finally {
            restore_error_handler();
        }
        if ($shm === false) {
            return false;
        }

        // Read header and validate.
        $header = shmop_read($shm, 0, self::HEADER_BYTES);
        if ($header === false || strlen($header) !== self::HEADER_BYTES) {
            self::closeShm($shm);
            return false;
        }
        $unpacked = unpack('V4', $header);
        // V4 returns [1=>magic, 2=>hash, 3=>count, 4=>reserved]
        if ($unpacked[1] !== self::MAGIC || $unpacked[2] !== $this->modelHash) {
            // Stale segment for a different model. Close + signal miss.
            self::closeShm($shm);
            return false;
        }

        $this->shm = $shm;
        $this->active = true;
        $this->buildIndex((int)$unpacked[3]);
        return true;
    }

    /**
     * Look up a tensor by name. Returns the raw bytes or null on miss.
     */
    public function get(string $name): ?string
    {
        if (!$this->active && !$this->attach()) {
            $this->misses++;
            return null;
        }
        if (!isset($this->index[$name])) {
            $this->misses++;
            return null;
        }
        $entry = $this->index[$name];
        $bytes = shmop_read($this->shm, $entry['offset'], $entry['length']);
        if ($bytes === false || strlen($bytes) !== $entry['length']) {
            $this->misses++;
            return null;
        }
        $this->hits++;
        return $bytes;
    }

    /**
     * Populate the cache from a {name => rawBytes} map. Replaces any
     * existing segment at this key. Returns true on success.
     *
     * @param array<string,string> $tensors name -> raw bytes
     */
    public function prime(array $tensors): bool
    {
        if (!function_exists('shmop_open')) {
            return false;
        }

        // If a stale segment exists, delete it first.
        $this->deleteStale();

        // Layout: header + offset table + payloads.
        $count = count($tensors);
        $tableBytes = $count * self::ENTRY_BYTES;
        $payloadBytes = 0;
        foreach ($tensors as $blob) {
            $payloadBytes += strlen($blob);
        }
        $total = self::HEADER_BYTES + $tableBytes + $payloadBytes + self::SAFETY_MARGIN;

        // Create new segment (mode 'c' = create if missing). Octal 0644 perms.
        // Suppress warnings on size-too-large / permission failures.
        set_error_handler(function () { /* swallow */ });
        try {
            $shm = @shmop_open($this->shmKey, 'c', 0644, $total);
        } finally {
            restore_error_handler();
        }
        if ($shm === false) {
            return false;
        }

        // ----- Build the buffer in memory, write in one sho ------------------
        $buf = str_repeat("\x00", $total);

        // Header: magic, hash, count, reserved — four little-endian uint32s.
        // PHP's pack() doesn't treat '/' as a separator; we build a single
        // 16-byte string and pad the rest of the 64-byte header with zeros.
        $header = pack('V4', self::MAGIC, $this->modelHash & 0xFFFFFFFF, $count, 0);
        $header = str_pad($header, self::HEADER_BYTES, "\x00");
        for ($i = 0; $i < self::HEADER_BYTES; $i++) {
            $buf[$i] = $header[$i];
        }

        // Offset table + payloads.
        $cursor = self::HEADER_BYTES + $tableBytes;  // payloads start after table
        $entryIdx = 0;
        foreach ($tensors as $name => $blob) {
            $entryOff = self::HEADER_BYTES + $entryIdx * self::ENTRY_BYTES;

            // Name (64 bytes, null-padded).
            $nameBytes = substr($name, 0, self::ENTRY_NAME_BYTES);
            $namePadded = str_pad($nameBytes, self::ENTRY_NAME_BYTES, "\x00");
            for ($i = 0; $i < self::ENTRY_NAME_BYTES; $i++) {
                $buf[$entryOff + $i] = $namePadded[$i];
            }

            // Meta: 8 bytes offset LE + 8 bytes length LE (P = uint64 LE).
            $meta = pack('P2', $cursor, strlen($blob));
            for ($i = 0; $i < self::ENTRY_META_BYTES; $i++) {
                $buf[$entryOff + self::ENTRY_NAME_BYTES + $i] = $meta[$i];
            }

            // Payload.
            for ($i = 0, $n = strlen($blob); $i < $n; $i++) {
                $buf[$cursor + $i] = $blob[$i];
            }
            $cursor += strlen($blob);
            $entryIdx++;
        }

        // One big write.
        $written = shmop_write($shm, $buf, 0);
        if ($written === false || $written !== strlen($buf)) {
            // Best-effort cleanup.
            shmop_delete($shm);
            self::closeShm($shm);
            return false;
        }

        // Attach for further use.
        if ($this->shm !== null) {
            self::closeShm($this->shm);
        }
        $this->shm = $shm;
        $this->active = true;
        $this->buildIndex($count);
        return true;
    }

    /**
     * Build the in-process name => {offset,length} index by reading the
     * offset table once on attach/prime.
     */
    private function buildIndex(int $count): void
    {
        $this->index = [];
        if ($count === 0) {
            return;
        }
        $tableBlob = shmop_read(
            $this->shm,
            self::HEADER_BYTES,
            $count * self::ENTRY_BYTES
        );
        for ($i = 0; $i < $count; $i++) {
            $entryBlob = substr($tableBlob, $i * self::ENTRY_BYTES, self::ENTRY_BYTES);
            // First 64 bytes = name; trim at first null.
            $nameBin = substr($entryBlob, 0, self::ENTRY_NAME_BYTES);
            $nullPos = strpos($nameBin, "\x00");
            $name = ($nullPos === false) ? $nameBin : substr($nameBin, 0, $nullPos);
            // Next 16 bytes = offset + length (P2 = two uint64 LE).
            $meta = unpack('P2', substr($entryBlob, self::ENTRY_NAME_BYTES, self::ENTRY_META_BYTES));
            $this->index[$name] = [
                'offset' => (int)$meta[1],
                'length' => (int)$meta[2],
            ];
        }
    }

    /**
     * Delete an existing segment at our key if present.
     */
    private function deleteStale(): void
    {
        set_error_handler(function () { /* swallow */ });
        try {
            $existing = @shmop_open($this->shmKey, 'a', 0, 0);
            if ($existing !== false) {
                @shmop_delete($existing);
                self::closeShm($existing);
            }
        } finally {
            restore_error_handler();
        }
    }

    /**
     * Derive a System V IPC key. ftok() needs an existing file; we pin it
     * on config.json. The project-id byte is fixed at 0x50 ('P').
     */
    private function deriveKey(string $path): int
    {
        if (function_exists('ftok') && is_file($path)) {
            $k = ftok($path, 'P');
            if ($k !== -1) {
                return $k;
            }
        }
        // Fallback: hash the path into a 32-bit int.
        $h = crc32(__CLASS__ . ':' . $path);
        return $h & 0x7FFFFFFF;
    }

    public function __destruct()
    {
        if ($this->shm !== null) {
            self::closeShm($this->shm);
        }
    }
}
