use std::any::{Any, type_name};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use serde::Deserialize;

/// Typed attack-kernel interface.
///
/// `P` is the host-side params type that mirrors the Metal struct layout for a specific kernel.
pub trait AttackKernel<P> {
    fn attack_id(&self) -> &'static str;
    fn metal_function_name(&self) -> &'static str;
    fn metal_source_path(&self) -> &'static str;
    fn encode_params(&self, params: &P) -> Vec<u8>;
}

/// Runtime-safe error for dynamic parameter encoding.
#[derive(Debug)]
pub enum AttackParamError {
    WrongType {
        expected: &'static str,
        got: &'static str,
    },
}

impl fmt::Display for AttackParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongType { expected, got } => {
                write!(f, "wrong attack params type: expected {expected}, got {got}")
            }
        }
    }
}

impl Error for AttackParamError {}

/// Erased interface for runtime registry lookup (`HashMap<&str, Box<dyn AttackKernelDyn>>`).
pub trait AttackKernelDyn {
    fn attack_id(&self) -> &'static str;
    fn metal_function_name(&self) -> &'static str;
    fn metal_source_path(&self) -> &'static str;
    fn encode_params_dyn(&self, params_any: &dyn Any) -> Result<Vec<u8>, AttackParamError>;
    fn expected_param_type(&self) -> &'static str;
}

/// Adapter that bridges a typed attack kernel into the dynamic registry interface.
pub struct AttackKernelAdapter<K, P> {
    inner: K,
    _marker: PhantomData<P>,
}

impl<K, P> AttackKernelAdapter<K, P> {
    pub fn new(inner: K) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

impl<K, P> AttackKernelDyn for AttackKernelAdapter<K, P>
where
    K: AttackKernel<P> + Send + Sync + 'static,
    P: 'static + Send + Sync,
{
    fn attack_id(&self) -> &'static str {
        self.inner.attack_id()
    }

    fn metal_function_name(&self) -> &'static str {
        self.inner.metal_function_name()
    }

    fn metal_source_path(&self) -> &'static str {
        self.inner.metal_source_path()
    }

    fn encode_params_dyn(&self, params_any: &dyn Any) -> Result<Vec<u8>, AttackParamError> {
        let Some(params) = params_any.downcast_ref::<P>() else {
            return Err(AttackParamError::WrongType {
                expected: type_name::<P>(),
                got: "<unknown dynamic type>",
            });
        };

        Ok(self.inner.encode_params(params))
    }

    fn expected_param_type(&self) -> &'static str {
        type_name::<P>()
    }
}

pub type AttackRegistry = HashMap<&'static str, Box<dyn AttackKernelDyn>>;

/// HS256-specific host-side params type.
///
/// The name intentionally matches the Metal struct name so the mapping is obvious.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Hs256BruteForceParams {
    pub target_signature: [u8; 32],
    pub message_length: u32,
    pub candidate_count: u32,
    pub candidate_index_base: u32,
}

impl Default for Hs256BruteForceParams {
    fn default() -> Self {
        Self {
            target_signature: [0u8; 32],
            message_length: 0,
            candidate_count: 0,
            candidate_index_base: 0,
        }
    }
}

/// Example plugin object for the HS256 brute-force kernel.
pub struct Hs256BruteForceKernel;

impl AttackKernel<Hs256BruteForceParams> for Hs256BruteForceKernel {
    fn attack_id(&self) -> &'static str {
        "hs256_brute_force"
    }

    fn metal_function_name(&self) -> &'static str {
        "hs256_brute_force"
    }

    fn metal_source_path(&self) -> &'static str {
        "src/kernels/hs256_brute_force.metal"
    }

    fn encode_params(&self, params: &Hs256BruteForceParams) -> Vec<u8> {
        let byte_len = std::mem::size_of::<Hs256BruteForceParams>();
        let ptr = (params as *const Hs256BruteForceParams).cast::<u8>();

        // SAFETY: `params` is valid for reads of its own size and `repr(C)` guarantees stable layout.
        unsafe { std::slice::from_raw_parts(ptr, byte_len) }.to_vec()
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn build_attack_registry() -> AttackRegistry {
    let mut registry: AttackRegistry = HashMap::new();

    registry.insert(
        "hs256_brute_force",
        Box::new(AttackKernelAdapter::<Hs256BruteForceKernel, Hs256BruteForceParams>::new(
            Hs256BruteForceKernel,
        )),
    );

    registry
}

#[derive(Debug, Clone)]
struct CliArgs {
    jwt: String,
    wordlist: PathBuf,
}

#[derive(Debug, Clone)]
struct ParsedJwtTarget {
    signing_input: Vec<u8>,
    target_signature: [u8; 32],
}

#[derive(Debug, Deserialize)]
struct JwtHeader {
    alg: String,
}

#[derive(Debug, Clone)]
struct WordBatch {
    candidate_index_base: u32,
    words: Vec<String>,
    word_bytes: Vec<u8>,
    word_offsets: Vec<u32>,
    word_lengths: Vec<u16>,
}

impl WordBatch {
    fn candidate_count_u32(&self) -> u32 {
        self.word_offsets.len() as u32
    }

    fn is_empty(&self) -> bool {
        self.word_offsets.is_empty()
    }
}

const DEFAULT_WORDLIST_PATH: &str = "breach.txt";
const RESULT_NOT_FOUND_SENTINEL: u32 = u32::MAX;
const MAX_CANDIDATES_PER_BATCH: usize = 65_536;
const MAX_WORD_BYTES_PER_BATCH: usize = 16 * 1024 * 1024;

fn parse_cli_args<I>(mut args: I) -> anyhow::Result<CliArgs>
where
    I: Iterator<Item = String>,
{
    let _program = args.next();
    let mut jwt: Option<String> = None;
    let mut wordlist = PathBuf::from(DEFAULT_WORDLIST_PATH);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wordlist" => {
                let path = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value after --wordlist"))?;
                wordlist = PathBuf::from(path);
            }
            _ if arg.starts_with("--") => bail!("unknown flag: {arg}"),
            _ => {
                if jwt.is_some() {
                    bail!("expected exactly one JWT positional argument");
                }
                jwt = Some(arg);
            }
        }
    }

    let jwt = jwt.ok_or_else(|| anyhow!("usage: jotcrack <jwt> [--wordlist <path>]") )?;
    Ok(CliArgs { jwt, wordlist })
}

fn parse_hs256_jwt(jwt: &str) -> anyhow::Result<ParsedJwtTarget> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        bail!("malformed JWT: expected 3 dot-separated segments");
    }

    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .context("invalid base64url JWT header")?;
    let header: JwtHeader = serde_json::from_slice(&header_bytes).context("invalid JWT header JSON")?;
    if header.alg != "HS256" {
        bail!("unsupported JWT alg: expected HS256, got {}", header.alg);
    }

    let signature_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .context("invalid base64url JWT signature")?;
    if signature_bytes.len() != 32 {
        bail!(
            "invalid HS256 signature length: expected 32 bytes, got {}",
            signature_bytes.len()
        );
    }
    let mut target_signature = [0u8; 32];
    target_signature.copy_from_slice(&signature_bytes);

    let signing_input = format!("{}.{}", parts[0], parts[1]).into_bytes();

    Ok(ParsedJwtTarget {
        signing_input,
        target_signature,
    })
}

struct WordlistBatchReader<R: BufRead> {
    reader: R,
    line_buf: String,
    pending_line: Option<String>,
    next_index: u32,
    done: bool,
}

impl<R: BufRead> WordlistBatchReader<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            line_buf: String::new(),
            pending_line: None,
            next_index: 0,
            done: false,
        }
    }

    fn next_batch(&mut self) -> anyhow::Result<Option<WordBatch>> {
        if self.done {
            return Ok(None);
        }

        let candidate_index_base = self.next_index;
        let mut batch = WordBatch {
            candidate_index_base,
            words: Vec::new(),
            word_bytes: Vec::new(),
            word_offsets: Vec::new(),
            word_lengths: Vec::new(),
        };

        loop {
            let line = if let Some(line) = self.pending_line.take() {
                line
            } else {
                self.line_buf.clear();
                let bytes_read = self
                    .reader
                    .read_line(&mut self.line_buf)
                    .context("failed to read wordlist")?;
                if bytes_read == 0 {
                    self.done = true;
                    break;
                }

                while self.line_buf.ends_with('\n') || self.line_buf.ends_with('\r') {
                    self.line_buf.pop();
                }
                if self.line_buf.is_empty() {
                    continue;
                }
                self.line_buf.clone()
            };

            let word_bytes = line.as_bytes();
            if word_bytes.len() > u16::MAX as usize {
                bail!(
                    "wordlist entry at candidate index {} exceeds {} bytes",
                    self.next_index,
                    u16::MAX
                );
            }

            let would_exceed_count = batch.word_offsets.len() >= MAX_CANDIDATES_PER_BATCH;
            let would_exceed_bytes = !batch.word_offsets.is_empty()
                && (batch.word_bytes.len() + word_bytes.len() > MAX_WORD_BYTES_PER_BATCH);
            if would_exceed_count || would_exceed_bytes {
                self.pending_line = Some(line);
                break;
            }

            let offset = u32::try_from(batch.word_bytes.len()).context("packed word bytes exceed u32")?;
            batch.word_offsets.push(offset);
            batch.word_lengths.push(word_bytes.len() as u16);
            batch.word_bytes.extend_from_slice(word_bytes);
            batch.words.push(line);
            self.next_index = self
                .next_index
                .checked_add(1)
                .ok_or_else(|| anyhow!("candidate index overflow"))?;
        }

        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }
}

struct GpuHs256BruteForcer {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
}

impl GpuHs256BruteForcer {
    fn new(kernel: &Hs256BruteForceKernel) -> anyhow::Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow!("no Metal device available"))?;
        let source = std::fs::read_to_string(kernel.metal_source_path())
            .with_context(|| format!("failed to read Metal source {}", kernel.metal_source_path()))?;
        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(&source, &compile_options)
            .map_err(|e| anyhow!("failed to compile Metal kernel: {e}"))?;
        let function = library
            .get_function(kernel.metal_function_name(), None)
            .map_err(|e| anyhow!("failed to get Metal function {}: {e}", kernel.metal_function_name()))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow!("failed to create compute pipeline: {e}"))?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    fn try_batch(
        &self,
        signing_input: &[u8],
        target_signature: [u8; 32],
        batch: &WordBatch,
    ) -> anyhow::Result<Option<u32>> {
        let candidate_count = batch.candidate_count_u32();
        if candidate_count == 0 {
            return Ok(None);
        }
        let message_length = u32::try_from(signing_input.len()).context("JWT signing input too long")?;
        let _max_index = batch
            .candidate_index_base
            .checked_add(candidate_count - 1)
            .ok_or_else(|| anyhow!("candidate index overflow in batch"))?;

        let params = Hs256BruteForceParams {
            target_signature,
            message_length,
            candidate_count,
            candidate_index_base: batch.candidate_index_base,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let params_bytes = bytes_of(&params);
        let params_buf = self.device.new_buffer_with_data(
            params_bytes.as_ptr().cast(),
            params_bytes.len() as u64,
            options,
        );
        let msg_buf = self.device.new_buffer_with_data(
            signing_input.as_ptr().cast(),
            signing_input.len() as u64,
            options,
        );
        let word_bytes_buf = self.device.new_buffer_with_data(
            batch.word_bytes.as_ptr().cast(),
            batch.word_bytes.len() as u64,
            options,
        );
        let offsets_buf = self.device.new_buffer_with_data(
            batch.word_offsets.as_ptr().cast(),
            std::mem::size_of_val(batch.word_offsets.as_slice()) as u64,
            options,
        );
        let lengths_buf = self.device.new_buffer_with_data(
            batch.word_lengths.as_ptr().cast(),
            std::mem::size_of_val(batch.word_lengths.as_slice()) as u64,
            options,
        );
        let result_init = RESULT_NOT_FOUND_SENTINEL;
        let result_buf = self.device.new_buffer_with_data(
            (&result_init as *const u32).cast(),
            std::mem::size_of::<u32>() as u64,
            options,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(&msg_buf), 0);
        encoder.set_buffer(2, Some(&word_bytes_buf), 0);
        encoder.set_buffer(3, Some(&offsets_buf), 0);
        encoder.set_buffer(4, Some(&lengths_buf), 0);
        encoder.set_buffer(5, Some(&result_buf), 0);

        let max_threads = self.pipeline.max_total_threads_per_threadgroup() as usize;
        let threadgroup_width = 256usize.min(max_threads.max(1));
        let threads_per_group = MTLSize::new(threadgroup_width as u64, 1, 1);
        let threads_per_grid = MTLSize::new(candidate_count as u64, 1, 1);
        encoder.dispatch_threads(threads_per_grid, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let result_ptr = result_buf.contents().cast::<u32>();
        // SAFETY: `result_buf` is a 4-byte shared buffer initialized above and still alive.
        let result = unsafe { *result_ptr };
        if result == RESULT_NOT_FOUND_SENTINEL {
            Ok(None)
        } else {
            Ok(Some(result))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttackStatus {
    Found,
    NotFound,
}

fn run() -> anyhow::Result<AttackStatus> {
    let cli = parse_cli_args(std::env::args())?;
    let target = parse_hs256_jwt(&cli.jwt)?;

    let kernel = Hs256BruteForceKernel;
    let gpu = GpuHs256BruteForcer::new(&kernel)?;

    let file = File::open(&cli.wordlist)
        .with_context(|| format!("failed to open wordlist {}", cli.wordlist.display()))?;
    let mut reader = WordlistBatchReader::new(BufReader::new(file));
    let started_at = Instant::now();
    let mut last_rate_report_at = started_at;
    let mut candidates_tested: u64 = 0;

    while let Some(batch) = reader.next_batch()? {
        let batch_candidate_count = u64::from(batch.candidate_count_u32());
        if let Some(global_index) = gpu.try_batch(&target.signing_input, target.target_signature, &batch)? {
            candidates_tested = candidates_tested.saturating_add(batch_candidate_count);
            let elapsed = started_at.elapsed();
            let rate = if elapsed.is_zero() {
                0.0
            } else {
                candidates_tested as f64 / elapsed.as_secs_f64()
            };
            print_final_stats(candidates_tested, elapsed, rate);

            let local_index = usize::try_from(global_index - batch.candidate_index_base)
                .context("GPU returned out-of-range result index")?;
            let secret = batch
                .words
                .get(local_index)
                .ok_or_else(|| anyhow!("GPU returned invalid local candidate index"))?;
            println!("HS256 key: {secret}");
            return Ok(AttackStatus::Found);
        }

        candidates_tested = candidates_tested.saturating_add(batch_candidate_count);
        let now = Instant::now();
        if now.duration_since(last_rate_report_at) >= Duration::from_secs(1) {
            let elapsed = now.duration_since(started_at);
            let rate = if elapsed.is_zero() {
                0.0
            } else {
                candidates_tested as f64 / elapsed.as_secs_f64()
            };
            eprintln!(
                "RATE {}/s ({} tested in {:.2}s)",
                format_human_count(rate),
                format_human_count(candidates_tested as f64),
                elapsed.as_secs_f64()
            );
            last_rate_report_at = now;
        }
    }

    let elapsed = started_at.elapsed();
    let rate = if elapsed.is_zero() {
        0.0
    } else {
        candidates_tested as f64 / elapsed.as_secs_f64()
    };
    print_final_stats(candidates_tested, elapsed, rate);
    println!("NOT FOUND");
    Ok(AttackStatus::NotFound)
}

fn format_human_count(value: f64) -> String {
    const UNITS: [(&str, f64); 4] = [
        ("trillion", 1_000_000_000_000.0),
        ("billion", 1_000_000_000.0),
        ("million", 1_000_000.0),
        ("thousand", 1_000.0),
    ];

    for (name, scale) in UNITS {
        if value >= scale {
            let scaled = value / scale;
            let precision = if scaled >= 100.0 {
                0
            } else if scaled >= 10.0 {
                1
            } else {
                2
            };
            return format!("{scaled:.precision$} {name}");
        }
    }

    format!("{value:.0}")
}

fn print_final_stats(candidates_tested: u64, elapsed: Duration, rate: f64) {
    eprintln!("STATS");
    eprintln!("  tested: {}", format_human_count(candidates_tested as f64));
    eprintln!("  elapsed: {:.2}s", elapsed.as_secs_f64());
    eprintln!("  rate: {}/s", format_human_count(rate));
}

fn bytes_of<T>(value: &T) -> &[u8] {
    // SAFETY: `value` is a valid pointer for `size_of::<T>()` bytes for the duration of the borrow.
    unsafe {
        std::slice::from_raw_parts(
            (value as *const T).cast::<u8>(),
            std::mem::size_of::<T>(),
        )
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(AttackStatus::Found) => ExitCode::from(0),
        Ok(AttackStatus::NotFound) => ExitCode::from(1),
        Err(err) => {
            eprintln!("ERROR: {err:#}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::path::Path;

    #[test]
    fn hs256_params_round_trip_size_matches() {
        let kernel = Hs256BruteForceKernel;
        let params = Hs256BruteForceParams::default();
        let bytes = kernel.encode_params(&params);

        assert_eq!(bytes.len(), std::mem::size_of::<Hs256BruteForceParams>());
    }

    #[test]
    fn registry_contains_hs256_kernel() {
        let registry = build_attack_registry();

        let kernel = registry.get("hs256_brute_force").expect("kernel must exist");
        assert_eq!(kernel.metal_function_name(), "hs256_brute_force");
        assert_eq!(kernel.metal_source_path(), "src/kernels/hs256_brute_force.metal");
        assert!(kernel.expected_param_type().contains("Hs256BruteForceParams"));
    }

    #[test]
    fn dynamic_encoding_rejects_wrong_param_type() {
        let kernel = AttackKernelAdapter::<Hs256BruteForceKernel, Hs256BruteForceParams>::new(
            Hs256BruteForceKernel,
        );
        let wrong_params: u32 = 42;

        let err = kernel
            .encode_params_dyn(&wrong_params)
            .expect_err("wrong type should be rejected");

        match err {
            AttackParamError::WrongType { expected, .. } => {
                assert!(expected.contains("Hs256BruteForceParams"));
            }
        }
    }

    #[test]
    fn parse_cli_defaults_wordlist_to_breach_txt() {
        let cli = parse_cli_args(
            ["jotcrack", "abc.def.ghi"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap();

        assert_eq!(cli.jwt, "abc.def.ghi");
        assert_eq!(cli.wordlist, PathBuf::from(DEFAULT_WORDLIST_PATH));
    }

    #[test]
    fn parse_hs256_jwt_success() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{"sub":"alice"}"#);
        let sig = URL_SAFE_NO_PAD.encode([7u8; 32]);
        let jwt = format!("{header}.{payload}.{sig}");
        let parsed = parse_hs256_jwt(&jwt).unwrap();

        assert!(parsed.signing_input.windows(1).count() > 0);
        assert_eq!(parsed.target_signature.len(), 32);
        assert!(String::from_utf8(parsed.signing_input).unwrap().contains('.'));
    }

    #[test]
    fn parse_hs256_jwt_rejects_wrong_segment_count() {
        let err = parse_hs256_jwt("abc.def").unwrap_err();
        assert!(format!("{err:#}").contains("3 dot-separated"));
    }

    #[test]
    fn parse_hs256_jwt_rejects_non_hs256() {
        let header_json = br#"{"alg":"HS384","typ":"JWT"}"#;
        let payload_json = br#"{"sub":"alice"}"#;
        let header = URL_SAFE_NO_PAD.encode(header_json);
        let payload = URL_SAFE_NO_PAD.encode(payload_json);
        let sig = URL_SAFE_NO_PAD.encode([0u8; 32]);
        let jwt = format!("{header}.{payload}.{sig}");

        let err = parse_hs256_jwt(&jwt).unwrap_err();
        assert!(format!("{err:#}").contains("HS256"));
    }

    #[test]
    fn wordlist_batch_reader_packs_offsets_and_lengths() {
        let data = b"alpha\r\n\nbe\ncharlie\n";
        let mut reader = WordlistBatchReader::new(Cursor::new(data.as_slice()));
        let batch = reader.next_batch().unwrap().unwrap();

        assert_eq!(batch.candidate_index_base, 0);
        assert_eq!(batch.words, vec!["alpha", "be", "charlie"]);
        assert_eq!(batch.word_offsets, vec![0, 5, 7]);
        assert_eq!(batch.word_lengths, vec![5, 2, 7]);
        assert_eq!(batch.word_bytes, b"alphabecharlie");
        assert!(reader.next_batch().unwrap().is_none());
    }

    #[test]
    fn wordlist_batch_reader_rejects_too_long_line() {
        let oversized = "a".repeat((u16::MAX as usize) + 1);
        let data = format!("{oversized}\n");
        let mut reader = WordlistBatchReader::new(Cursor::new(data.into_bytes()));
        let err = reader.next_batch().unwrap_err();
        assert!(format!("{err:#}").contains("exceeds"));
    }

    #[test]
    fn wordlist_batch_reader_preserves_absolute_indices_across_batches() {
        let lines = (0..(MAX_CANDIDATES_PER_BATCH + 2))
            .map(|i| format!("pw{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let data = format!("{lines}\n");
        let mut reader = WordlistBatchReader::new(Cursor::new(data.into_bytes()));

        let first = reader.next_batch().unwrap().unwrap();
        let second = reader.next_batch().unwrap().unwrap();

        assert_eq!(first.candidate_index_base, 0);
        assert_eq!(first.words.len(), MAX_CANDIDATES_PER_BATCH);
        assert_eq!(second.candidate_index_base, MAX_CANDIDATES_PER_BATCH as u32);
        assert_eq!(second.words.len(), 2);
    }

    #[test]
    fn wordlist_batch_reader_returns_none_for_all_empty_lines() {
        let mut reader = WordlistBatchReader::new(Cursor::new(b"\n\r\n".as_slice()));
        assert!(reader.next_batch().unwrap().is_none());
    }

    #[test]
    fn parse_cli_rejects_unknown_flag() {
        let err = parse_cli_args(
            ["jotcrack", "--bogus", "abc.def.ghi"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("unknown flag"));
    }

    #[test]
    fn parse_cli_accepts_custom_wordlist() {
        let cli = parse_cli_args(
            ["jotcrack", "abc.def.ghi", "--wordlist", "custom.txt"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap();
        assert_eq!(cli.wordlist, Path::new("custom.txt"));
    }

    #[test]
    fn parse_hs256_jwt_rejects_wrong_signature_length() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let sig = URL_SAFE_NO_PAD.encode([0u8; 31]);
        let err = parse_hs256_jwt(&format!("{header}.{payload}.{sig}")).unwrap_err();
        assert!(format!("{err:#}").contains("32 bytes"));
    }

    #[test]
    fn parse_hs256_jwt_rejects_bad_signature_base64() {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{}"#);
        let err = parse_hs256_jwt(&format!("{header}.{payload}.***")).unwrap_err();
        assert!(format!("{err:#}").contains("signature"));
    }
}
