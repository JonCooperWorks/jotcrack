#![cfg(test)]

use std::fs::{self, File as StdFile};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

use memmap2::MmapOptions;
use metal::Device;

use super::args::ParserConfig;
use super::parser::{MmapWordlistBatchReader, ParallelMmapWordlistBatchReader};

static TEMP_WORDLIST_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) fn test_device() -> Device {
    Device::system_default().expect("Metal device is required for hs256wordlist tests")
}

pub(super) fn write_temp_wordlist(bytes: &[u8]) -> std::path::PathBuf {
    let unique = TEMP_WORDLIST_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "jotcrack-hs256wordlist-test-{}-{nanos}-{unique}.txt",
        std::process::id()
    ));
    fs::write(&path, bytes).expect("failed to write temp wordlist");
    path
}

pub(super) fn mmap_reader_from_temp_file(
    bytes: &[u8],
) -> (MmapWordlistBatchReader, std::path::PathBuf) {
    let path = write_temp_wordlist(bytes);
    let file = StdFile::open(&path).expect("failed to open temp wordlist");
    let mmap = unsafe { MmapOptions::new().map(&file) }.expect("failed to mmap temp wordlist");
    (MmapWordlistBatchReader::new(test_device(), mmap), path)
}

pub(super) fn test_parser_config(parser_threads: usize, chunk_bytes: usize) -> ParserConfig {
    ParserConfig {
        parser_threads,
        chunk_bytes,
        queue_capacity: parser_threads.saturating_mul(4).max(1),
    }
}

pub(super) fn parallel_mmap_reader_from_temp_file(
    bytes: &[u8],
    parser_threads: usize,
    chunk_bytes: usize,
) -> (ParallelMmapWordlistBatchReader, std::path::PathBuf) {
    let path = write_temp_wordlist(bytes);
    let file = StdFile::open(&path).expect("failed to open temp wordlist");
    let mmap = unsafe { MmapOptions::new().map(&file) }.expect("failed to mmap temp wordlist");
    let reader = ParallelMmapWordlistBatchReader::new(
        test_device(),
        mmap,
        test_parser_config(parser_threads, chunk_bytes),
    )
    .expect("failed to build parallel mmap reader");
    (reader, path)
}

pub(super) fn collect_all_words_from_mmap_reader(
    reader: &mut MmapWordlistBatchReader,
) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while let Some(batch) = reader
        .next_batch_reusing(None)
        .expect("sequential mmap batch")
    {
        for i in 0..batch.candidate_count() {
            out.push(batch.word(i).expect("candidate").to_vec());
        }
    }
    out
}

pub(super) fn collect_all_words_from_parallel_reader(
    reader: &mut ParallelMmapWordlistBatchReader,
) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while let Some(batch) = reader
        .next_batch_reusing(None)
        .expect("parallel mmap batch")
    {
        for i in 0..batch.candidate_count() {
            out.push(batch.word(i).expect("candidate").to_vec());
        }
    }
    out
}
