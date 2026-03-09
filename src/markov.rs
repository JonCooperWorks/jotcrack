//! Order-1 Markov model for GPU-accelerated password candidate generation.
//!
//! Trains a bigram frequency model from a wordlist: for each (position, prev_char)
//! context, ranks successor characters by frequency. The ranked table is uploaded
//! to the GPU where each thread decodes a global index into per-position rank
//! selections, walks the Markov chain, and immediately hashes the generated
//! candidate — no intermediate candidate buffer needed.
//!
//! The raw frequency counts are cached to `<wordlist>.markov` so subsequent runs
//! with different `--threshold` or `--max-len` values skip the wordlist scan.

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

// ---------------------------------------------------------------------------
// Cache file format
// ---------------------------------------------------------------------------

const CACHE_MAGIC: [u8; 8] = *b"JCMARKOV";
const CACHE_VERSION: u32 = 1;

/// Size of the cache file header in bytes.
const CACHE_HEADER_BYTES: usize = 8 + 4 + 4 + 8; // magic + version + max_positions + word_count

// ---------------------------------------------------------------------------
// FrequencyCounts — raw bigram counts, cacheable
// ---------------------------------------------------------------------------

/// Raw bigram frequency counts: `freq[pos][prev_char][next_char]` as `u32`.
///
/// This is the heavyweight training artifact. The flat `u32` array is saved
/// to disk as the `.markov` cache so that different thresholds can be derived
/// without re-scanning the wordlist.
///
/// Layout: `freq[(pos * 256 + prev) * 256 + next]`
/// Size: `max_positions * 256 * 256 * 4` bytes (4 MB for max_positions=16).
pub(crate) struct FrequencyCounts {
    freq: Vec<u32>,
    max_positions: usize,
    word_count: u64,
}

impl FrequencyCounts {
    /// Train from raw wordlist bytes (newline-delimited).
    pub(crate) fn train(data: &[u8], max_positions: usize) -> Self {
        let table_len = max_positions * 256 * 256;
        let mut freq = vec![0u32; table_len];
        let mut word_count: u64 = 0;

        for line in data.split(|&b| b == b'\n') {
            let line = if line.last() == Some(&b'\r') {
                &line[..line.len() - 1]
            } else {
                line
            };
            if line.is_empty() {
                continue;
            }
            word_count += 1;

            let len = line.len().min(max_positions);
            let mut prev: u8 = 0; // start-of-word sentinel
            for pos in 0..len {
                let ch = line[pos];
                let idx = (pos * 256 + prev as usize) * 256 + ch as usize;
                // Saturating add prevents overflow on pathological wordlists.
                freq[idx] = freq[idx].saturating_add(1);
                prev = ch;
            }
        }

        Self {
            freq,
            max_positions,
            word_count,
        }
    }

    /// Save frequency counts to a binary cache file.
    pub(crate) fn save_cache(&self, path: &Path) -> anyhow::Result<()> {
        let mut buf = Vec::with_capacity(CACHE_HEADER_BYTES + self.freq.len() * 4);
        buf.extend_from_slice(&CACHE_MAGIC);
        buf.extend_from_slice(&CACHE_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.max_positions as u32).to_le_bytes());
        buf.extend_from_slice(&self.word_count.to_le_bytes());
        // Write freq array as little-endian u32s.
        for &v in &self.freq {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, &buf)
            .with_context(|| format!("failed to write Markov cache: {path:?}"))?;
        Ok(())
    }

    /// Load frequency counts from a binary cache file.
    pub(crate) fn load_cache(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read(path)
            .with_context(|| format!("failed to read Markov cache: {path:?}"))?;
        if data.len() < CACHE_HEADER_BYTES {
            bail!("Markov cache too small: {path:?}");
        }
        if &data[0..8] != &CACHE_MAGIC {
            bail!("invalid Markov cache magic: {path:?}");
        }
        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        if version != CACHE_VERSION {
            bail!("unsupported Markov cache version {version} (expected {CACHE_VERSION}): {path:?}");
        }
        let max_positions = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let word_count = u64::from_le_bytes(data[16..24].try_into().unwrap());

        let expected_freq_bytes = max_positions * 256 * 256 * 4;
        if data.len() != CACHE_HEADER_BYTES + expected_freq_bytes {
            bail!(
                "Markov cache size mismatch: expected {} bytes, got {}: {path:?}",
                CACHE_HEADER_BYTES + expected_freq_bytes,
                data.len()
            );
        }

        let freq_data = &data[CACHE_HEADER_BYTES..];
        let freq: Vec<u32> = freq_data
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(Self {
            freq,
            max_positions,
            word_count,
        })
    }

    /// Derive a ranked GPU lookup table for the given threshold.
    ///
    /// Returns a flat `u8` array: `table[(pos * 256 + prev) * threshold + rank]`
    /// where rank 0 is the most frequent successor.
    ///
    /// Slots beyond the number of distinct successors for a context are filled
    /// with rank-0 (the most frequent character), producing harmless duplicate
    /// candidates on the GPU.
    pub(crate) fn ranked_table(&self, threshold: usize, max_len: usize) -> Vec<u8> {
        let effective_positions = max_len.min(self.max_positions);
        let table_len = effective_positions * 256 * threshold;
        let mut table = vec![0u8; table_len];

        // Temporary buffer for sorting one context bucket.
        let mut bucket: Vec<(u32, u8)> = Vec::with_capacity(256);

        for pos in 0..effective_positions {
            for prev in 0..256usize {
                bucket.clear();
                for ch in 0..256u16 {
                    let freq_idx = (pos * 256 + prev) * 256 + ch as usize;
                    let count = self.freq[freq_idx];
                    if count > 0 {
                        bucket.push((count, ch as u8));
                    }
                }
                // Sort descending by frequency, stable (preserves byte order for ties).
                bucket.sort_by(|a, b| b.0.cmp(&a.0));

                let table_base = (pos * 256 + prev) * threshold;
                let fill_char = bucket.first().map(|&(_, c)| c).unwrap_or(0);
                for rank in 0..threshold {
                    table[table_base + rank] = if rank < bucket.len() {
                        bucket[rank].1
                    } else {
                        fill_char
                    };
                }
            }
        }

        table
    }

    pub(crate) fn max_positions(&self) -> usize {
        self.max_positions
    }

    pub(crate) fn word_count(&self) -> u64 {
        self.word_count
    }
}

// ---------------------------------------------------------------------------
// MarkovModel — GPU-ready ranked table + metadata
// ---------------------------------------------------------------------------

/// GPU-ready Markov model with a ranked character table.
///
/// Created from `FrequencyCounts` for a specific threshold. The `table` field
/// is uploaded directly to the GPU as a constant buffer.
pub(crate) struct MarkovModel {
    table: Vec<u8>,
    threshold: usize,
    max_positions: usize,
}

impl MarkovModel {
    /// Build a GPU-ready model from frequency counts and a threshold.
    pub(crate) fn from_counts(counts: &FrequencyCounts, threshold: usize, max_len: usize) -> Self {
        let effective_positions = max_len.min(counts.max_positions());
        let table = counts.ranked_table(threshold, max_len);
        Self {
            table,
            threshold,
            max_positions: effective_positions,
        }
    }

    /// Raw table bytes for GPU upload.
    pub(crate) fn table_bytes(&self) -> &[u8] {
        &self.table
    }

    pub(crate) fn threshold(&self) -> usize {
        self.threshold
    }

    pub(crate) fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Total keyspace for a range of candidate lengths.
    pub(crate) fn keyspace(&self, min_len: usize, max_len: usize) -> u128 {
        let t = self.threshold as u128;
        (min_len..=max_len.min(self.max_positions))
            .map(|l| t.checked_pow(l as u32).unwrap_or(u128::MAX))
            .fold(0u128, |acc, x| acc.saturating_add(x))
    }

    /// Reconstruct a candidate password from a global index (CPU-side).
    ///
    /// Used to recover the matched password after the GPU reports a hit.
    pub(crate) fn candidate_from_index(&self, length: usize, mut index: u64) -> Vec<u8> {
        let t = self.threshold as u64;
        let mut candidate = Vec::with_capacity(length);
        let mut prev: u8 = 0;
        for pos in 0..length {
            let rank = (index % t) as usize;
            index /= t;
            let table_idx = (pos * 256 + prev as usize) * self.threshold + rank;
            let ch = self.table[table_idx];
            candidate.push(ch);
            prev = ch;
        }
        candidate
    }
}

// ---------------------------------------------------------------------------
// High-level load-or-train API
// ---------------------------------------------------------------------------

/// Cache file path for a given wordlist.
pub(crate) fn cache_path_for(wordlist: &Path) -> PathBuf {
    let mut p = wordlist.as_os_str().to_owned();
    p.push(".markov");
    PathBuf::from(p)
}

/// Load cached frequency counts or train from the wordlist and save.
///
/// If `retrain` is true, always re-scan the wordlist even if a cache exists.
pub(crate) fn load_or_train(
    wordlist: &Path,
    max_len: usize,
    retrain: bool,
) -> anyhow::Result<FrequencyCounts> {
    let cache = cache_path_for(wordlist);

    if !retrain {
        if let Ok(counts) = FrequencyCounts::load_cache(&cache) {
            if counts.max_positions() >= max_len {
                eprintln!(
                    "MARKOV loaded cache: {} ({} words, max_positions={})",
                    cache.display(),
                    counts.word_count(),
                    counts.max_positions()
                );
                return Ok(counts);
            }
            eprintln!(
                "MARKOV cache max_positions={} < requested max_len={}, retraining",
                counts.max_positions(),
                max_len
            );
        }
    }

    eprintln!("MARKOV training from: {} (max_len={})", wordlist.display(), max_len);
    let data = std::fs::read(wordlist)
        .with_context(|| format!("failed to read wordlist: {wordlist:?}"))?;
    let counts = FrequencyCounts::train(&data, max_len);
    eprintln!(
        "MARKOV trained: {} words, saving cache to {}",
        counts.word_count(),
        cache.display()
    );
    if let Err(e) = counts.save_cache(&cache) {
        eprintln!("MARKOV warning: failed to save cache: {e:#}");
    }
    Ok(counts)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL_WORDLIST: &[u8] = b"password\npassword\npassword\nabc\nabc\nxyz\n";

    #[test]
    fn train_basic_counts() {
        let counts = FrequencyCounts::train(SMALL_WORDLIST, 8);
        assert_eq!(counts.word_count(), 6);
        assert_eq!(counts.max_positions(), 8);

        // Position 0, prev=0 (sentinel): 'p' appears 3 times, 'a' 2, 'x' 1.
        let freq_p = counts.freq[(0 * 256 + 0) * 256 + b'p' as usize];
        let freq_a = counts.freq[(0 * 256 + 0) * 256 + b'a' as usize];
        let freq_x = counts.freq[(0 * 256 + 0) * 256 + b'x' as usize];
        assert_eq!(freq_p, 3);
        assert_eq!(freq_a, 2);
        assert_eq!(freq_x, 1);
    }

    #[test]
    fn ranked_table_order() {
        let counts = FrequencyCounts::train(SMALL_WORDLIST, 8);
        let table = counts.ranked_table(3, 8);

        // Position 0, prev=0: rank 0 = 'p' (3), rank 1 = 'a' (2), rank 2 = 'x' (1).
        let base = (0 * 256 + 0) * 3;
        assert_eq!(table[base], b'p');
        assert_eq!(table[base + 1], b'a');
        assert_eq!(table[base + 2], b'x');
    }

    #[test]
    fn candidate_from_index_roundtrip() {
        let counts = FrequencyCounts::train(SMALL_WORDLIST, 8);
        let model = MarkovModel::from_counts(&counts, 3, 8);

        // Index 0, length 1 → most frequent first char = 'p'.
        let c = model.candidate_from_index(1, 0);
        assert_eq!(c, b"p");

        // Index 1, length 1 → second most frequent = 'a'.
        let c = model.candidate_from_index(1, 1);
        assert_eq!(c, b"a");

        // Index 2, length 1 → third = 'x'.
        let c = model.candidate_from_index(1, 2);
        assert_eq!(c, b"x");
    }

    #[test]
    fn candidate_length_2() {
        let counts = FrequencyCounts::train(SMALL_WORDLIST, 8);
        let model = MarkovModel::from_counts(&counts, 3, 8);

        // Index 0, length 2:
        //   pos 0: rank = 0%3 = 0 → 'p', prev = 'p'
        //   pos 1: rank = 0/3%3 = 0 → most freq after 'p' in the wordlist
        // After 'p', 'a' appears 3 times (in "password" × 3).
        let c = model.candidate_from_index(2, 0);
        assert_eq!(c[0], b'p');
        assert_eq!(c[1], b'a'); // 'a' is most common after 'p' in "password"
    }

    #[test]
    fn keyspace_calculation() {
        let counts = FrequencyCounts::train(SMALL_WORDLIST, 8);
        let model = MarkovModel::from_counts(&counts, 3, 8);

        // T=3: lengths 1..3 = 3 + 9 + 27 = 39.
        assert_eq!(model.keyspace(1, 3), 39);
        // Single length 2: 3^2 = 9.
        assert_eq!(model.keyspace(2, 2), 9);
    }

    #[test]
    fn cache_roundtrip() {
        let counts = FrequencyCounts::train(SMALL_WORDLIST, 4);
        let dir = std::env::temp_dir().join("jotcrack_markov_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.markov");

        counts.save_cache(&path).unwrap();
        let loaded = FrequencyCounts::load_cache(&path).unwrap();

        assert_eq!(loaded.max_positions(), counts.max_positions());
        assert_eq!(loaded.word_count(), counts.word_count());
        assert_eq!(loaded.freq, counts.freq);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn empty_wordlist() {
        let counts = FrequencyCounts::train(b"", 8);
        assert_eq!(counts.word_count(), 0);
    }

    #[test]
    fn crlf_handling() {
        let counts = FrequencyCounts::train(b"abc\r\ndef\r\n", 4);
        assert_eq!(counts.word_count(), 2);
        // 'a' at pos 0.
        let freq = counts.freq[(0 * 256 + 0) * 256 + b'a' as usize];
        assert_eq!(freq, 1);
        // '\r' should NOT appear in training data.
        let freq_cr = counts.freq[(0 * 256 + 0) * 256 + b'\r' as usize];
        assert_eq!(freq_cr, 0);
    }

    #[test]
    fn threshold_exceeds_alphabet() {
        // Only 3 distinct first chars: p, a, x. Threshold 10 should fill extras
        // with the most frequent char.
        let counts = FrequencyCounts::train(SMALL_WORDLIST, 8);
        let table = counts.ranked_table(10, 8);
        let base = (0 * 256 + 0) * 10;
        assert_eq!(table[base], b'p');     // rank 0
        assert_eq!(table[base + 1], b'a'); // rank 1
        assert_eq!(table[base + 2], b'x'); // rank 2
        assert_eq!(table[base + 3], b'p'); // rank 3+ filled with rank-0 char
        assert_eq!(table[base + 9], b'p');
    }
}
