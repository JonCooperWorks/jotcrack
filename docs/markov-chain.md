# Markov Chain Implementation

## Overview

jotcrack uses an **order-1 (bigram), position-aware Markov model** to generate
password candidates on the GPU. Instead of sending a wordlist to the GPU, it
learns statistical patterns from a wordlist — which characters tend to follow
which, at each position in a word — and uses those patterns to generate
plausible candidates directly in GPU registers with zero memory traffic.

## Data Structures

### `FrequencyCounts` — Training Artifact

Defined in `src/markov.rs`:

```rust
pub(crate) struct FrequencyCounts {
    freq: Vec<u32>,        // Flat 3D array of bigram counts
    max_positions: usize,  // Maximum word length tracked
    word_count: u64,       // Total words seen during training
}
```

The `freq` array is indexed as `freq[(pos * 256 + prev) * 256 + next]` — a 3D
table of `[position][previous_byte][next_byte] → count`. For every position in a
word, it records how many times each byte followed each other byte.

**Size:** `max_positions × 256 × 256 × 4` bytes (4 MB for `max_positions=16`).

### `MarkovModel` — GPU-Ready Compact Table

```rust
pub(crate) struct MarkovModel {
    table: Vec<u8>,        // Ranked character lookup
    threshold: usize,      // Top-T characters per context
    max_positions: usize,
}
```

Indexed as `table[(pos * 256 + prev) * T + rank]`. For each `(position,
previous character)` context, stores only the **top-T most frequent** successor
characters, sorted by frequency.

**Size:** `max_positions × 256 × T` bytes (~122 KB for `max_positions=16, T=30`).

## Training

Training (`FrequencyCounts::train` in `src/markov.rs`) is a single O(n) pass
over the wordlist:

1. Split input on newlines, strip `\r` for CRLF compatibility.
2. For each word, walk character-by-character, using `prev=0` as a
   start-of-word sentinel.
3. For each `(position, prev_char, current_char)` triple, increment the
   corresponding frequency counter with saturating arithmetic (preventing
   overflow on pathological inputs).

## Ranked Table Generation

`FrequencyCounts::ranked_table()` converts raw counts into the compact GPU
table:

1. For each `(position, prev_char)` context, collect all successor bytes that
   appeared at least once.
2. Sort by descending frequency (stable sort preserves byte order for ties).
3. Take the top-T characters and write them into the output table.
4. If fewer than T distinct successors exist, pad remaining slots with the
   rank-0 (most frequent) character — producing harmless duplicate candidates.

## GPU Candidate Generation

Each GPU thread gets a unique global index and deterministically decodes it into
a password candidate. The index is treated as a **base-T number**, where each
digit selects a rank at that position:

```metal
uint8_t prev = 0;
for (uint32_t pos = 0; pos < len; ++pos) {
    uint32_t rank = uint32_t(idx % uint64_t(T));
    idx /= uint64_t(T);
    uint32_t table_idx = (pos * 256u + uint32_t(prev)) * T + rank;
    candidate[pos] = markov_table[table_idx];
    prev = candidate[pos];
}
```

The chain walks forward: each character depends on the previous one via `prev`,
so output follows learned bigram patterns. The candidate is generated entirely
in registers and immediately hashed — no intermediate buffer is written to GPU
memory.

## Keyspace

Total candidates = `sum(T^L for L in min_len..=max_len)`.

With the default `T=30, max_len=8`: ~6.56 trillion candidates. Far fewer than
brute-force (`96^8 ≈ 7.2 quadrillion`) but covering the most statistically
likely passwords.

## Caching

Raw `FrequencyCounts` are saved to `<wordlist>.markov` with a binary format
(magic `JCMARKOV`, version, header, flat `u32` array). Changing `--threshold`
or `--max-len` does not require re-scanning the wordlist.

## Size Estimates

### Model size is independent of wordlist size

| `max_len` | FrequencyCounts | GPU Table (T=30) | GPU Table (T=50) |
|-----------|-----------------|-------------------|-------------------|
| 8         | 2 MB            | 61 KB             | 102 KB            |
| 16        | 4 MB            | 122 KB            | 204 KB            |
| 32        | 8 MB            | 245 KB            | 409 KB            |
| 64        | 16 MB           | 491 KB            | 819 KB            |

A 4 GB wordlist produces the **exact same sized model** as a 4 KB wordlist with
the same `max_len`. The wordlist size only affects training time and the quality
of frequency counts.
