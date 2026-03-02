#![cfg(test)]

use std::fs::{self, File as StdFile};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

use memmap2::MmapOptions;
use metal::Device;

use super::args::ParserConfig;
use super::parser::{MmapWordlistBatchReader, ParallelMmapWordlistBatchReader};

static TEMP_WORDLIST_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn test_device() -> Device {
    Device::system_default().expect("Metal device is required for hs256wordlist tests")
}

pub(crate) fn write_temp_wordlist(bytes: &[u8]) -> std::path::PathBuf {
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

pub(crate) fn mmap_reader_from_temp_file(
    bytes: &[u8],
) -> (MmapWordlistBatchReader, std::path::PathBuf) {
    let path = write_temp_wordlist(bytes);
    let file = StdFile::open(&path).expect("failed to open temp wordlist");
    let mmap = unsafe { MmapOptions::new().map(&file) }.expect("failed to mmap temp wordlist");
    (MmapWordlistBatchReader::new(test_device(), mmap), path)
}

pub(crate) fn test_parser_config(parser_threads: usize, chunk_bytes: usize) -> ParserConfig {
    ParserConfig {
        parser_threads,
        chunk_bytes,
        queue_capacity: parser_threads.saturating_mul(4).max(1),
    }
}

pub(crate) fn parallel_mmap_reader_from_temp_file(
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

pub(crate) fn collect_all_words_from_mmap_reader(
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

pub(crate) fn make_test_jwt(alg: &str, payload_json: &str, secret: &[u8]) -> String {
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use hmac::{Hmac, Mac};

    let header = format!(r#"{{"alg":"{}","typ":"JWT"}}"#, alg);
    let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json.as_bytes());
    let signing_input = format!("{}.{}", header_b64, payload_b64);

    let signature = match alg {
        "HS256" => {
            let mut mac = Hmac::<sha2::Sha256>::new_from_slice(secret).unwrap();
            mac.update(signing_input.as_bytes());
            mac.finalize().into_bytes().to_vec()
        }
        "HS384" => {
            let mut mac = Hmac::<sha2::Sha384>::new_from_slice(secret).unwrap();
            mac.update(signing_input.as_bytes());
            mac.finalize().into_bytes().to_vec()
        }
        "HS512" => {
            let mut mac = Hmac::<sha2::Sha512>::new_from_slice(secret).unwrap();
            mac.update(signing_input.as_bytes());
            mac.finalize().into_bytes().to_vec()
        }
        _ => panic!("unsupported algorithm: {}", alg),
    };

    format!("{}.{}", signing_input, URL_SAFE_NO_PAD.encode(&signature))
}

pub(crate) fn collect_all_words_from_parallel_reader(
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
