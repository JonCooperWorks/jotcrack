pub const DEFAULT_WORDLIST_PATH: &str = "breach.txt";
pub(crate) const DEFAULT_PIPELINE_DEPTH: usize = 10;
pub(crate) const DEFAULT_PARSER_CHUNK_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ParserConfig {
    pub(crate) parser_threads: usize,
    pub(crate) chunk_bytes: usize,
    pub(crate) queue_capacity: usize,
}

impl ParserConfig {
    pub(crate) fn resolve(parser_threads: Option<usize>) -> Self {
        let auto_threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        let parser_threads = parser_threads.unwrap_or(auto_threads);
        let queue_capacity = parser_threads.saturating_mul(4).max(1);
        Self {
            parser_threads,
            chunk_bytes: DEFAULT_PARSER_CHUNK_BYTES,
            queue_capacity,
        }
    }
}

pub fn parse_positive_usize(input: &str) -> Result<usize, String> {
    let parsed = input
        .parse::<usize>()
        .map_err(|_| format!("invalid integer value: {input}"))?;
    if parsed == 0 {
        return Err("must be > 0".to_string());
    }
    Ok(parsed)
}
