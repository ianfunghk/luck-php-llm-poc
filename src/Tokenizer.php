<?php
declare(strict_types=1);

/**
 * src/Tokenizer.php
 * =================
 * Minimal text<->token id conversion for a SentencePiece + byte_fallback vocab.
 *
 * IMPORTANT APPROXIMATION
 * -----------------------
 * Implementing full SentencePiece (BPE/Unigram with byte fallback and the
 * U+2581 word-boundary marker) in pure PHP would dwarf the rest of this
 * project. For the tiny1m target model we use a deliberately simpler scheme
 * that is good enough for the TinyStories vocab:
 *
 *   encode() = greedy longest-match on raw piece strings + byte fallback
 *
 *   At each cursor position we try the longest piece in the vocab that matches
 *   the start of the remaining text. If nothing matches, we fall back to the
 *   `<0xHH>` byte token for that byte. This is exact for ASCII English and
 *   degrades gracefully to byte-level coverage for anything else.
 *
 * For decode() we just concatenate pieces, post-processing:
 *   - `<0xHH>` byte tokens -> the raw byte
 *   - U+2581 ("▁") -> a space (SentencePiece's word-boundary marker)
 *   - control tokens (`<s>`, `</s>`, `<unk>`) -> skipped
 *
 * This is the approach the spec explicitly approves.
 */

namespace PhpLlm;

class Tokenizer
{
    /** @var array<int,string> id -> piece */
    private array $idToPiece;

    /** @var array<string,int> piece -> id (longest piece wins on collision) */
    private array $pieceToId;

    /** @var int length of the longest piece in the vocab, in BYTES. */
    private int $maxPieceBytes;

    /** @var array<int,int> byte value (0..255) -> token id */
    private array $byteFallbackId;

    /**
     * Marker that the model uses to represent a leading/inter-word space.
     *  - SentencePiece (tiny1m): U+2581 ('▁')
     *  - GPT2 BPE (stories15M, Qwen): U+0120 ('Ġ')
     */
    private string $spaceMarker;

    /** Which byte the space marker is. Used by encode/decode fast paths. */
    private string $spaceMarkerChar;

    public int $bosId;
    public int $eosId;
    public int $unkId;

    public function __construct(array $tokens, int $bosId = 1, int $eosId = 2, int $unkId = 0,
                                string $kind = 'sentencepiece')
    {
        $this->bosId = $bosId;
        $this->eosId = $eosId;
        $this->unkId = $unkId;

        // GPT2 BPE uses U+0120 ( LATIN CAPITAL LETTER G WITH BREVE, "Ġ" ),
        // which the original GPT2 byte-level BPE inserts in place of spaces.
        // SentencePiece uses U+2581 ( LOWER HALF BLOCK, "▁" ).
        if ($kind === 'gpt2_bpe') {
            $this->spaceMarker     = "\xC4\xA0";   // U+0120
            $this->spaceMarkerChar = "\xC4\xA0";
        } else {
            $this->spaceMarker     = "\xE2\x96\x81"; // U+2581
            $this->spaceMarkerChar = "\xE2\x96\x81";
        }

        $this->idToPiece = [];
        $this->pieceToId = [];
        $this->byteFallbackId = [];
        $this->maxPieceBytes = 0;

        foreach ($tokens as $entry) {
            $id = (int)$entry['id'];
            $piece = (string)$entry['piece'];

            $this->idToPiece[$id] = $piece;

            if ($this->isByteFallbackPiece($piece)) {
                $byteVal = hexdec(substr($piece, 3, 2));
                $this->byteFallbackId[(int)$byteVal] = $id;
            } elseif (!$this->isControlPiece($piece)) {
                if (!isset($this->pieceToId[$piece])) {
                    $this->pieceToId[$piece] = $id;
                }
                $lenBytes = strlen($piece);
                if ($lenBytes > $this->maxPieceBytes) {
                    $this->maxPieceBytes = $lenBytes;
                }
            }
        }
    }

    /**
     * Tokenise a UTF-8 string into ids, with BOS prepended.
     *
     * Mirrors the relevant piece of SentencePiece/GPT2 normalisation:
     *   - prepend the appropriate space marker at the very start
     *   - replace ASCII spaces with the same marker
     */
    public function encode(string $text): array
    {
        $marker = $this->spaceMarker;
        $text = $marker . $text;
        $text = str_replace(' ', $marker, $text);

        $bytes = strlen($text);
        $ids = [$this->bosId];

        $i = 0;
        while ($i < $bytes) {
            // Greedy: try the longest candidate that exists in the vocab,
            // shrinking from maxPieceBytes down to 1.
            $matched = false;
            $maxTry = min($this->maxPieceBytes, $bytes - $i);
            for ($len = $maxTry; $len >= 1; $len--) {
                $candidate = substr($text, $i, $len);
                if (isset($this->pieceToId[$candidate])) {
                    $ids[] = $this->pieceToId[$candidate];
                    $i += $len;
                    $matched = true;
                    break;
                }
            }

            if (!$matched) {
                // Fallback: emit one byte token for the byte at position $i.
                $byteVal = ord($text[$i]);
                if (isset($this->byteFallbackId[$byteVal])) {
                    $ids[] = $this->byteFallbackId[$byteVal];
                } else {
                    // No byte-fallback token available — emit UNK.
                    // (GPT2 BPE models do not have <0xHH> tokens; their byte
                    // coverage is achieved via dedicated pieces for common
                    // characters, so we generally do not hit this path for
                    // ordinary ASCII input.)
                    $ids[] = $this->unkId;
                }
                $i += 1;
            }
        }

        return $ids;
    }

    /**
     * Decode a list of ids back to a printable string.
     *
     * Rules:
     *   - byte-fallback tokens (`<0xHH>`) -> raw byte
     *   - control tokens (`<s>`, `</s>`, `<unk>`, `<|endoftext|>`...) -> skipped
     *   - the chosen space marker -> ASCII space
     */
    public function decode(array $ids): string
    {
        $out = '';
        foreach ($ids as $id) {
            if (!isset($this->idToPiece[$id])) {
                continue;
            }
            $piece = $this->idToPiece[$id];

            if ($this->isControlPiece($piece)) {
                continue;
            }
            if ($this->isByteFallbackPiece($piece)) {
                $byteVal = (int)hexdec(substr($piece, 3, 2));
                $out .= chr($byteVal);
                continue;
            }
            // Replace the space marker with an ASCII space.
            $out .= str_replace($this->spaceMarkerChar, ' ', $piece);
        }
        // Collapse leading space that comes from the prepended marker.
        return ltrim($out, ' ');
    }

    private function isControlPiece(string $piece): bool
    {
        // SentencePiece-style: <s>, </s>, <unk>, <pad> ...
        // GPT2-style: <|endoftext|>, <|im_start|>, <|im_end|> ...
        // All of these start with '<' and end with '>'. Byte-fallback tokens
        // (<0xHH>) also match that pattern but are handled separately.
        return strlen($piece) >= 2 && $piece[0] === '<' && substr($piece, -1) === '>'
            && !$this->isByteFallbackPiece($piece);
    }

    private function isByteFallbackPiece(string $piece): bool
    {
        // Exactly `<0xHH>` where HH are hex digits.
        return strlen($piece) === 6
            && $piece[0] === '<'
            && $piece[1] === '0'
            && $piece[2] === 'x'
            && $piece[5] === '>'
            && ctype_xdigit(substr($piece, 3, 2));
    }
}
