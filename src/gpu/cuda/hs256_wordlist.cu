// ===========================================================================
// CUDA compute kernel for HMAC-SHA256 JWT cracking (wordlist mode).
//
// This is the CUDA port of the Metal shader `hs256_wordlist.metal`.
// Every educational comment from the Metal version is preserved and adapted
// to explain the CUDA equivalents.
//
// LEARNING OVERVIEW -- read this first if you're new to GPU programming.
//
// What this file does:
//   Each GPU thread receives one candidate secret from a wordlist and computes
//   HMAC-SHA256(secret, JWT_signing_input).  If the result matches the target
//   signature, the thread writes its index to an atomic result slot.
//
// Key concepts for understanding this kernel:
//
//   1. KERNEL DISPATCH MODEL
//      CUDA dispatches a "grid" of threads.  Each thread computes a unique
//      `gid` (global thread ID) as `blockIdx.x * blockDim.x + threadIdx.x`.
//      The host sets grid dimensions = ceil(num_candidates / block_size) and
//      block size = a tunable width (e.g. 256).  CUDA's scheduler maps blocks
//      onto Streaming Multiprocessors (SMs), and threads within a block
//      execute in lockstep groups of 32 called "warps" (analogous to Metal's
//      SIMD groups / "execution width" of 32 threads on Apple GPUs).
//
//   2. SHA-256 COMPRESSION
//      SHA-256 processes data in 512-bit (64-byte) blocks.  Each block goes
//      through 64 "rounds" of mixing using bitwise rotations, XORs, additions
//      mod 2^32, and a set of round constants (SHA256_K[64]).  The internal
//      state is eight 32-bit words (a..h) that accumulate the hash.
//
//   3. HMAC CONSTRUCTION (RFC 2104)
//      HMAC wraps a hash function with a key:
//        HMAC(K, M) = H((K ^ opad) || H((K ^ ipad) || M))
//      where ipad = 0x36 repeated, opad = 0x5c repeated, and K is zero-padded
//      to one block (64 bytes for SHA-256).  This requires two full SHA-256
//      computations: an "inner hash" and an "outer hash".
//
//   4. MEMORY SPACES
//      CUDA has distinct memory types analogous to Metal's address spaces:
//        - Local variables (registers/stack) = Metal's `thread` address space.
//          Per-thread private storage, fastest access.
//        - Global memory (`const T*` pointers) = Metal's `device` address space.
//          GPU VRAM, accessible by all threads, highest latency.
//        - `__constant__` memory = Metal's `constant` address space.
//          Read-only, hardware-cached, broadcast to all threads in a warp.
//          Ideal for data shared by all threads (like SHA-256 round constants).
//          On NVIDIA GPUs, constant memory uses a dedicated 64KB cache that
//          broadcasts a single read to all threads in a warp simultaneously.
//      The JWT message is passed via global memory pointers (const);
//      candidate secrets use global memory because each thread reads a
//      different slice.
//
//   5. ATOMIC RESULT REPORTING
//      When a thread finds a match, it uses `atomicMin` to write its `gid`
//      to the result slot.  This is the CUDA equivalent of Metal's
//      `atomic_fetch_min_explicit(..., memory_order_relaxed)`.  The `min`
//      ensures that if multiple threads match (shouldn't happen with correct
//      JWTs), the lowest-indexed match wins deterministically.
//
//   6. NVRTC AND extern "C"
//      When kernels are compiled at runtime with NVRTC (NVIDIA Runtime
//      Compilation), the `extern "C"` linkage specifier prevents C++ name
//      mangling.  This allows the host code to look up kernel functions by
//      their plain string names (e.g. "hs256_wordlist") without needing to
//      know the mangled symbol.
//
//   7. CUDA FUNCTION QUALIFIERS
//      - `__device__` marks functions callable only from GPU code (other
//        `__device__` or `__global__` functions).  Analogous to Metal's
//        regular functions that are called from kernel code.
//      - `__global__` marks kernel entry points that can be launched from
//        the host via <<<grid, block>>> syntax or the driver API.  Analogous
//        to Metal's `kernel` qualifier.
//      - `__forceinline__` is an NVIDIA-specific hint that aggressively
//        inlines the function, reducing call overhead.  Combined with
//        `__device__`, it replaces Metal's `static inline` pattern.
// ===========================================================================

// Portable fixed-width type aliases.
//
// NVRTC (NVIDIA Runtime Compilation) ships a minimal standard library
// that does NOT include system headers like <stdint.h>.  We define the
// exact-width types ourselves using CUDA's guaranteed primitive sizes:
//   unsigned char       = 8 bits
//   unsigned short      = 16 bits
//   unsigned int        = 32 bits
//   unsigned long long  = 64 bits
typedef unsigned long long uint64_t;
typedef unsigned int       uint32_t;
typedef unsigned short     uint16_t;
typedef unsigned char      uint8_t;

// Host-to-kernel parameter block.  The host (Rust code) uploads this struct
// as a GPU buffer with the exact same field order/layout.
// The struct must match the Rust-side `#[repr(C)]` struct field-for-field, byte-for-byte.
struct Hs256BruteForceParams {
    // Optimization: target stored as unsigned int[8] (precomputed on host) instead of
    // uint8_t[32].  Eliminates per-thread load_be_u32 calls in the comparison loop
    // -- the final check becomes a straight word-level `ostate[i] != target[i]`.
    unsigned int target_signature[8];
    unsigned int message_length;           // Actual byte length of `message_bytes` (the JWT `header.payload` string).
    unsigned int candidate_count;          // Number of candidate secrets in this GPU batch.
};

// Minimal SHA-256 streaming context used per GPU thread.
//
// Learning note: why is this so large?
// Each GPU thread needs its own copy of this struct (76+ bytes) because
// threads execute independently.  The "flat" HMAC path below avoids this
// struct entirely on the hot path, using bare unsigned int arrays instead,
// which dramatically reduces register pressure and improves GPU occupancy
// (= more threads can be in flight simultaneously).
//
// On NVIDIA GPUs, each SM has a finite register file (e.g. 65536 registers
// on many architectures).  Fewer registers per thread = more warps can be
// resident = higher occupancy = better latency hiding.
struct Sha256Ctx {
    unsigned int state[8];                 // Current SHA-256 state words (a..h / hash state).
    uint8_t block[64];                     // Current 512-bit block buffer being filled before compression.
    unsigned int block_len;                // Number of bytes currently stored in `block`.
    uint64_t total_len;                    // Total message length processed so far, in bytes (needed for final padding length).
};

// SHA-256 round constants from the specification (FIPS 180-4, section 4.2.2).
// These are the first 32 bits of the fractional parts of the cube roots of
// the first 64 prime numbers.  They are placed in `__constant__` memory
// so every thread shares one cached copy.
//
// CUDA note: `__constant__` memory resides in a dedicated 64KB cached region.
// When all threads in a warp read the same address (as they do for round
// constants), the hardware broadcasts one read to all 32 threads -- this is
// analogous to Metal's `constant` address space which is cached and shared
// by all threads in a SIMD group.
__constant__ unsigned int SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, // Rounds 0..3
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, // Rounds 4..7
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, // Rounds 8..11
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, // Rounds 12..15
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, // Rounds 16..19
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, // Rounds 20..23
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, // Rounds 24..27
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, // Rounds 28..31
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, // Rounds 32..35
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, // Rounds 36..39
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, // Rounds 40..43
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, // Rounds 44..47
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, // Rounds 48..51
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u, // Rounds 52..55
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, // Rounds 56..59
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u, // Rounds 60..63
};

// Rotate-right helper for 32-bit values. SHA-256 uses rotations heavily.
//
// CUDA note: `__device__ __forceinline__` replaces Metal's `static inline`.
// `__device__` means this function is callable only from GPU code (other
// __device__ or __global__ functions).  `__forceinline__` aggressively
// inlines the function body at every call site, eliminating function call
// overhead -- critical for tiny helpers called millions of times.
__device__ __forceinline__ unsigned int rotr32(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32u - n)); // Split + wrap bits from low end back to high end.
}

// SHA-256 choice function: chooses bits from y/z based on x.
__device__ __forceinline__ unsigned int ch32(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ ((~x) & z); // Equivalent to: x ? y : z, bitwise.
}

// SHA-256 majority function: bit is 1 if at least two inputs have bit 1.
__device__ __forceinline__ unsigned int maj32(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (x & z) ^ (y & z); // Standard SHA-256 boolean function.
}

// SHA-256 big sigma 0 for the compression round math.
__device__ __forceinline__ unsigned int bsig0(unsigned int x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); // XOR of three rotations.
}

// SHA-256 big sigma 1 for the compression round math.
__device__ __forceinline__ unsigned int bsig1(unsigned int x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); // XOR of three rotations.
}

// SHA-256 small sigma 0 for message schedule expansion.
__device__ __forceinline__ unsigned int ssig0(unsigned int x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); // Two rotates + one logical shift.
}

// SHA-256 small sigma 1 for message schedule expansion.
__device__ __forceinline__ unsigned int ssig1(unsigned int x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); // Two rotates + one logical shift.
}

// Read 4 bytes in big-endian order into one 32-bit word.
//
// CUDA note: unlike Metal, CUDA does not require separate overloads for
// different address spaces.  A single function accepting `const uint8_t*`
// works for pointers to global memory, local memory, or any other space.
// The CUDA compiler unifies address spaces behind a "generic" pointer;
// the hardware resolves the actual memory partition at runtime.
__device__ __forceinline__ unsigned int load_be_u32(const uint8_t* p) {
    return ((unsigned int)(p[0]) << 24) | // Most-significant byte.
           ((unsigned int)(p[1]) << 16) | // Next byte.
           ((unsigned int)(p[2]) << 8)  | // Next byte.
           (unsigned int)(p[3]);          // Least-significant byte.
}

// Write one 32-bit word into 4 bytes in big-endian order.
__device__ __forceinline__ void store_be_u32(uint8_t* p, unsigned int v) {
    p[0] = (uint8_t)((v >> 24) & 0xffu); // Highest byte.
    p[1] = (uint8_t)((v >> 16) & 0xffu); // Next byte.
    p[2] = (uint8_t)((v >> 8) & 0xffu);  // Next byte.
    p[3] = (uint8_t)(v & 0xffu);         // Lowest byte.
}

// Initialize a SHA-256 context to the algorithm's fixed IV (initial hash values).
// These constants are the first 32 bits of the fractional parts of the square
// roots of the first 8 primes (2, 3, 5, 7, 11, 13, 17, 19).
__device__ __forceinline__ void sha256_init(Sha256Ctx& ctx) {
    ctx.state[0] = 0x6a09e667u; // H0
    ctx.state[1] = 0xbb67ae85u; // H1
    ctx.state[2] = 0x3c6ef372u; // H2
    ctx.state[3] = 0xa54ff53au; // H3
    ctx.state[4] = 0x510e527fu; // H4
    ctx.state[5] = 0x9b05688cu; // H5
    ctx.state[6] = 0x1f83d9abu; // H6
    ctx.state[7] = 0x5be0cd19u; // H7
    ctx.block_len = 0;          // No bytes buffered yet.
    ctx.total_len = 0;          // No bytes processed yet.
}

// ---------------------------------------------------------------------------
// Optimization: rolling 16-word message schedule.
//
// Standard SHA-256 pre-expands all 64 message words into w[64] (256 bytes)
// before compression.  The rolling schedule keeps only w[16] (64 bytes) and
// updates w[i & 15] in-place during rounds 16-63, saving ~192 bytes of
// register pressure per sha256_compress call.  On NVIDIA GPUs this directly
// improves occupancy (more warps can be resident per SM).
//
// CUDA note: register pressure is even more critical on NVIDIA architectures
// because each SM has a fixed register file (e.g. 65536 32-bit registers).
// The compiler allocates registers per-thread; fewer registers per thread
// means more warps can be scheduled concurrently, which improves latency
// hiding through warp-level multithreading (32 threads per warp execute in
// lockstep, and the SM switches between warps to hide memory latency).
//
// Rounds 0-15:  use the caller-loaded w[i] directly (SHA256_ROUND_R).
// Rounds 16-63: recurrence w[i&15] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15])
//               + w[i-16], then round (SHA256_EXPAND_ROUND).
// ---------------------------------------------------------------------------

// SHA-256 round using rolling w[i & 15] indexing.
#define SHA256_ROUND_R(i, a, b, c, d, e, f, g, h) do { \
    unsigned int t1_ = (h) + bsig1((e)) + ch32((e), (f), (g)) + SHA256_K[(i)] + w[(i) & 15]; \
    unsigned int t2_ = bsig0((a)) + maj32((a), (b), (c)); \
    (d) += t1_; \
    (h) = t1_ + t2_; \
} while (0)

// Rolling expand + round: update w[i&15] in-place from the recurrence, then compress.
// w[(i-16)&15] == w[i&15] still holds the value from 16 rounds ago.
#define SHA256_EXPAND_ROUND(i, a, b, c, d, e, f, g, h) do { \
    w[(i) & 15] = ssig1(w[((i) - 2) & 15]) + w[((i) - 7) & 15] + ssig0(w[((i) - 15) & 15]) + w[(i) & 15]; \
    SHA256_ROUND_R(i, a, b, c, d, e, f, g, h); \
} while (0)

// 8 consecutive rounds for the first 16 rounds (no expansion needed).
#define SHA256_ROUND_8_INIT(base, a, b, c, d, e, f, g, h) \
    SHA256_ROUND_R((base) + 0, a, b, c, d, e, f, g, h); \
    SHA256_ROUND_R((base) + 1, h, a, b, c, d, e, f, g); \
    SHA256_ROUND_R((base) + 2, g, h, a, b, c, d, e, f); \
    SHA256_ROUND_R((base) + 3, f, g, h, a, b, c, d, e); \
    SHA256_ROUND_R((base) + 4, e, f, g, h, a, b, c, d); \
    SHA256_ROUND_R((base) + 5, d, e, f, g, h, a, b, c); \
    SHA256_ROUND_R((base) + 6, c, d, e, f, g, h, a, b); \
    SHA256_ROUND_R((base) + 7, b, c, d, e, f, g, h, a)

// 8 consecutive rounds with rolling expansion (rounds 16-63).
#define SHA256_ROUND_8_ROLLING(base, a, b, c, d, e, f, g, h) \
    SHA256_EXPAND_ROUND((base) + 0, a, b, c, d, e, f, g, h); \
    SHA256_EXPAND_ROUND((base) + 1, h, a, b, c, d, e, f, g); \
    SHA256_EXPAND_ROUND((base) + 2, g, h, a, b, c, d, e, f); \
    SHA256_EXPAND_ROUND((base) + 3, f, g, h, a, b, c, d, e); \
    SHA256_EXPAND_ROUND((base) + 4, e, f, g, h, a, b, c, d); \
    SHA256_EXPAND_ROUND((base) + 5, d, e, f, g, h, a, b, c); \
    SHA256_EXPAND_ROUND((base) + 6, c, d, e, f, g, h, a, b); \
    SHA256_EXPAND_ROUND((base) + 7, b, c, d, e, f, g, h, a)

// Compress one 512-bit block using a rolling 16-word message schedule.
// Takes bare state[8] and w[16] (caller loads the first 16 message words).
// NOTE: w[16] is modified in-place by the rolling expansion (rounds 16-63).
//
// Learning note: SHA-256 compression step by step
//
// 1. Copy current state into working variables a..h.
// 2. Rounds 0-15:  use the 16 message words directly with the round function.
//    Each round updates one working variable using: rotations of e (bsig1),
//    a choice function on e,f,g, the round constant K[i], and the message word w[i].
// 3. Rounds 16-63: first expand w[i&15] using a recurrence relation:
//      w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16]
//    then apply the same round function.
// 4. Add the working variables back to the state (this is the "Davies-Meyer" pattern).
//
// After processing all blocks, state[0..7] holds the final hash digest.
__device__ __forceinline__ void sha256_compress_rolling(unsigned int state[8], unsigned int w[16]) {
    unsigned int a = state[0];
    unsigned int b = state[1];
    unsigned int c = state[2];
    unsigned int d = state[3];
    unsigned int e = state[4];
    unsigned int f = state[5];
    unsigned int g = state[6];
    unsigned int h = state[7];

    // Rounds 0-15: use w[i] directly (no expansion needed).
    SHA256_ROUND_8_INIT(0,  a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_INIT(8,  a, b, c, d, e, f, g, h);
    // Rounds 16-63: expand w[i&15] in-place before each round.
    SHA256_ROUND_8_ROLLING(16, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(24, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(32, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(40, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(48, a, b, c, d, e, f, g, h);
    SHA256_ROUND_8_ROLLING(56, a, b, c, d, e, f, g, h);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// Compress one block from a Sha256Ctx's internal block buffer.
//
// CUDA note: unlike Metal, we do not need separate overloads for different
// pointer address spaces.  CUDA uses a unified "generic" address space
// for device pointers, so one function handles local arrays, global memory
// pointers, and __constant__ pointers uniformly.  The hardware determines
// the actual memory partition at runtime via a TLB-like mechanism.
__device__ __forceinline__ void sha256_compress(Sha256Ctx& ctx, const uint8_t* block) {
    unsigned int w[16];
    for (unsigned int i = 0; i < 16; ++i) {
        w[i] = load_be_u32(block + (i * 4u));
    }
    sha256_compress_rolling(ctx.state, w);
}

// Update SHA-256 state from local memory (per-thread registers/stack).
// We use this for small local arrays like ipad/opad and inner digest.
//
// CUDA note: in CUDA, local variables and global memory pointers use the
// same generic pointer type.  This single update function replaces the three
// separate Metal overloads (sha256_update_thread, sha256_update_device,
// sha256_update_constant) that were needed because Metal enforces distinct
// address-space-qualified pointer types at compile time.
__device__ __forceinline__ void sha256_update(Sha256Ctx& ctx, const uint8_t* data, unsigned int len) {
    if (len == 0u) {
        return;
    }
    ctx.total_len += (uint64_t)(len);             // Count total bytes for final bit-length encoding.

    unsigned int offset = 0u;
    if (ctx.block_len != 0u) {                    // Fill pending partial block first.
        const unsigned int need = 64u - ctx.block_len;
        const unsigned int take = (len < need) ? len : need;
        for (unsigned int i = 0; i < take; ++i) {
            ctx.block[ctx.block_len + i] = data[i];
        }
        ctx.block_len += take;
        offset += take;
        if (ctx.block_len == 64u) {
            sha256_compress(ctx, ctx.block);
            ctx.block_len = 0u;
        }
    }

    while ((len - offset) >= 64u) {               // Compress whole source blocks directly.
        sha256_compress(ctx, data + offset);
        offset += 64u;
    }

    for (; offset < len; ++offset) {              // Buffer only trailing bytes.
        ctx.block[ctx.block_len++] = data[offset];
    }
}

// Finalize SHA-256: add padding + length, compress final block(s), and write 32-byte digest.
//
// Learning note: SHA-256 padding rules
// 1. Append a single 0x80 byte (a 1-bit followed by zeros).
// 2. Pad with zero bytes until the block is 56 bytes long (leaving room for 8 bytes of length).
//    If appending 0x80 already pushed past byte 56, finish this block and start a new one.
// 3. Append the total message length in bits as a 64-bit big-endian integer in the last 8 bytes.
// 4. Compress the final block.  state[0..7] is the digest.
__device__ __forceinline__ void sha256_final(Sha256Ctx& ctx, uint8_t out[32]) {
    const uint64_t bit_len = ctx.total_len * 8ull;

    ctx.block[ctx.block_len++] = 0x80u;
    if (ctx.block_len > 56u) {
        while (ctx.block_len < 64u) ctx.block[ctx.block_len++] = 0u;
        sha256_compress(ctx, ctx.block);
        ctx.block_len = 0;
    }
    while (ctx.block_len < 56u) ctx.block[ctx.block_len++] = 0u;
    for (unsigned int i = 0; i < 8; ++i)
        ctx.block[56u + i] = (uint8_t)((bit_len >> (56u - 8u * i)) & 0xffull);
    sha256_compress(ctx, ctx.block);
    ctx.block_len = 0;

    for (unsigned int i = 0; i < 8; ++i)
        store_be_u32(out + (i * 4u), ctx.state[i]);
}

// Finalize SHA-256 without serializing output to bytes.
// After this call, ctx.state[0..7] contains the digest as big-endian unsigned int words.
// This avoids the store_be_u32 output loop when the digest will be consumed as words.
__device__ __forceinline__ void sha256_finalize(Sha256Ctx& ctx) {
    const uint64_t bit_len = ctx.total_len * 8ull;
    ctx.block[ctx.block_len++] = 0x80u;
    if (ctx.block_len > 56u) {
        while (ctx.block_len < 64u) ctx.block[ctx.block_len++] = 0u;
        sha256_compress(ctx, ctx.block);
        ctx.block_len = 0u;
    }
    while (ctx.block_len < 56u) ctx.block[ctx.block_len++] = 0u;
    for (unsigned int i = 0; i < 8; ++i)
        ctx.block[56u + i] = (uint8_t)((bit_len >> (56u - 8u * i)) & 0xffull);
    sha256_compress(ctx, ctx.block);
    ctx.block_len = 0u;
}

// Load an HMAC key from global memory into 16 big-endian unsigned int words (zero-padded).
// Eliminates the byte-level key_block[64] array in favor of word-level operations.
__device__ __forceinline__ void load_key_words(const uint8_t* key, unsigned int key_len, unsigned int kw[16]) {
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) kw[i] = 0u;

    const unsigned int full_words = key_len >> 2;
    for (unsigned int i = 0; i < full_words; ++i)
        kw[i] = load_be_u32(key + i * 4u);

    const unsigned int remaining = key_len & 3u;
    if (remaining > 0u) {
        unsigned int val = 0u;
        const uint8_t* tail = key + (full_words * 4u);
        for (unsigned int i = 0; i < remaining; ++i)
            val |= (unsigned int)(tail[i]) << (24u - 8u * i);
        kw[full_words] = val;
    }
}

// ---------------------------------------------------------------------------
// Optimization: flat specialized HMAC (no Sha256Ctx on the hot path).
//
// Learning note: why two HMAC implementations?
//
// The generic streaming Sha256Ctx carries 76 bytes of per-thread state
// (block[64] + block_len + total_len) that the HMAC hot path never needs:
// ipad/opad blocks are always exactly 64 bytes, message length is known, and
// finalization layout is deterministic.  This flat version uses bare
// unsigned int istate[8] / ostate[8] (32 bytes each) and constructs each
// 512-bit block in a local w[16] directly from constant memory with manual
// padding, eliminating ~152 bytes of register pressure across the
// inner+outer hashes.
//
// On GPUs, register pressure is the primary bottleneck for occupancy.  Fewer
// registers per thread = more threads can execute simultaneously = higher
// throughput.  This optimization alone can yield a 1.5-2x speedup.
//
// CUDA note: on NVIDIA GPUs, when a thread uses too many registers, the
// compiler "spills" excess values to local memory (which is actually slow
// global memory backed by L1/L2 cache).  The flat HMAC path minimizes this
// spilling by keeping live state small.
//
// Sha256Ctx + the streaming update/finalize API are kept for the long-key
// fallback path (key_len > 64 -> hash-then-HMAC per RFC 2104).
//
// HMAC-SHA256 conceptually:
//   inner = SHA256( (key XOR ipad) || message )
//   outer = SHA256( (key XOR opad) || inner_digest )
// where ipad = 0x36 repeated to block size, opad = 0x5c repeated to block size.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool hmac_sha256_from_key_words(
    const unsigned int kw[16],
    const uint8_t* message,
    unsigned int message_len,
    const unsigned int target[8]
) {
    // --- Inner hash (flat, no Sha256Ctx) ---
    unsigned int istate[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

    // Compress ipad block: kw XOR 0x36363636.
    {
        unsigned int w[16];
        #pragma unroll
        for (unsigned int i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x36363636u;
        sha256_compress_rolling(istate, w);
    }

    // Process full 64-byte message blocks directly from global memory.
    unsigned int offset = 0u;
    while ((message_len - offset) >= 64u) {
        unsigned int w[16];
        for (unsigned int i = 0; i < 16; ++i) {
            w[i] = load_be_u32(message + offset + i * 4u);
        }
        sha256_compress_rolling(istate, w);
        offset += 64u;
    }

    // Finalize inner hash: remaining bytes + 0x80 padding + bit length.
    const unsigned int rem = message_len - offset;
    const uint64_t inner_bit_len = (64ull + (uint64_t)(message_len)) * 8ull;
    {
        unsigned int w[16];
        #pragma unroll
        for (unsigned int i = 0; i < 16; ++i) w[i] = 0u;

        // Load remaining full words from message.
        const unsigned int full_words = rem >> 2;
        for (unsigned int i = 0; i < full_words; ++i) {
            w[i] = load_be_u32(message + offset + i * 4u);
        }

        // Load partial trailing bytes and append 0x80 padding byte.
        const unsigned int tail = rem & 3u;
        unsigned int pad_word = 0u;
        const uint8_t* tail_ptr = message + offset + (full_words * 4u);
        for (unsigned int i = 0; i < tail; ++i) {
            pad_word |= (unsigned int)(tail_ptr[i]) << (24u - 8u * i);
        }
        pad_word |= 0x80u << (24u - 8u * tail);
        w[full_words] = pad_word;

        if (rem > 55u) {
            // Two-block finalization: data+0x80 don't fit with the length field.
            sha256_compress_rolling(istate, w);
            #pragma unroll
            for (unsigned int i = 0; i < 16; ++i) w[i] = 0u;
        }
        w[14] = (unsigned int)((inner_bit_len >> 32u) & 0xffffffffull);
        w[15] = (unsigned int)(inner_bit_len & 0xffffffffull);
        sha256_compress_rolling(istate, w);
    }
    // istate[0..7] now holds the inner digest as words.

    // --- Outer hash (flat, no Sha256Ctx) ---
    unsigned int ostate[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

    // Compress opad block: kw XOR 0x5c5c5c5c.
    {
        unsigned int w[16];
        #pragma unroll
        for (unsigned int i = 0; i < 16; ++i) w[i] = kw[i] ^ 0x5c5c5c5cu;
        sha256_compress_rolling(ostate, w);
    }

    // Compress final outer block: inner digest + padding.
    // Outer message = 64 (opad) + 32 (inner digest) = 96 bytes total.
    {
        unsigned int w[16];
        for (unsigned int i = 0; i < 8; ++i) w[i] = istate[i];
        w[8] = 0x80000000u;
        w[9] = 0u; w[10] = 0u; w[11] = 0u; w[12] = 0u; w[13] = 0u;
        const uint64_t outer_bit_len = (64ull + 32ull) * 8ull;
        w[14] = (unsigned int)((outer_bit_len >> 32u) & 0xffffffffull);
        w[15] = (unsigned int)(outer_bit_len & 0xffffffffull);
        sha256_compress_rolling(ostate, w);
    }

    // Compare against precomputed target words.
    #pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        if (ostate[i] != target[i]) return false;
    }
    return true;
}

// HMAC-SHA256 that handles any key length and returns match result directly.
// For keys > 64 bytes, hashes the key first (RFC 2104), then uses word-level path.
__device__ __forceinline__ bool hmac_sha256_matches(
    const uint8_t* key,
    unsigned int key_len,
    const uint8_t* message,
    unsigned int message_len,
    const unsigned int target[8]
) {
    unsigned int kw[16];

    if (key_len > 64u) {
        // Long keys: hash first, then use the 32-byte hash as the key.
        #pragma unroll
        for (unsigned int i = 0; i < 16; ++i) kw[i] = 0u;
        Sha256Ctx key_ctx;
        sha256_init(key_ctx);
        sha256_update(key_ctx, key, key_len);
        sha256_finalize(key_ctx);
        #pragma unroll
        for (unsigned int i = 0; i < 8; ++i) kw[i] = key_ctx.state[i];
    } else {
        load_key_words(key, key_len, kw);
    }

    return hmac_sha256_from_key_words(kw, message, message_len, target);
}

// Short-key specialization: keys guaranteed <= 64 bytes, no hash-long-key branch.
__device__ __forceinline__ bool hmac_sha256_short_key_matches(
    const uint8_t* key,
    unsigned int key_len,
    const uint8_t* message,
    unsigned int message_len,
    const unsigned int target[8]
) {
    unsigned int kw[16];
    load_key_words(key, key_len, kw);
    return hmac_sha256_from_key_words(kw, message, message_len, target);
}

// ===========================================================================
// KERNEL ENTRY POINTS
//
// Learning note: CUDA kernel dispatch model
//
// The host (Rust code) launches a kernel with a grid of thread blocks:
//   kernel<<<grid_dim, block_dim>>>(...);
// or equivalently via the driver API:
//   cuLaunchKernel(function, grid_dim, 1, 1, block_dim, 1, 1, ...);
//
// CUDA creates `grid_dim * block_dim` threads total.  Each thread computes
// its global index as `blockIdx.x * blockDim.x + threadIdx.x`.  This is
// the CUDA equivalent of Metal's `[[thread_position_in_grid]]`.
//
// For this kernel:
//   grid_dim       = ceil(num_candidates / block_dim)
//   block_dim      = tunable (e.g. 256), chosen by autotune
//   gid            = which candidate this thread should test
//
// CUDA note on warps: within each block, threads execute in groups of 32
// called "warps".  All 32 threads in a warp execute the same instruction
// in lockstep (SIMT model).  This is directly analogous to Metal's SIMD
// groups (also 32 threads wide on Apple GPUs).  Divergent branches within
// a warp cause serialization -- both paths execute with some threads masked.
//
// Buffer bindings (set by the host launcher):
//   params         (target signature, message length, candidate count)
//   message_bytes  (JWT header.payload bytes, shared by all threads)
//   word_bytes     (concatenated candidate secret bytes)
//   word_offsets   (byte offset of each candidate in word_bytes)
//   word_lengths   (byte length of each candidate)
//   result_index   (atomic: lowest matching gid, or 0xFFFFFFFF)
//
// CUDA note: `extern "C"` prevents C++ name mangling, which is required
// for NVRTC (NVIDIA Runtime Compilation).  NVRTC compiles kernels from
// source at runtime, and the host looks up entry points by their plain
// string names.  Without `extern "C"`, the compiler would mangle the
// function name (e.g. "_Z16hs256_wordlistPK...") making it impossible
// to find by the simple name "hs256_wordlist".
//
// CUDA note: `__global__` marks these as kernel entry points that can be
// launched from host code.  This is the CUDA equivalent of Metal's
// `kernel` function qualifier.
//
// CUDA note: `__restrict__` is a pointer qualifier that tells the compiler
// no other pointer aliases this memory.  This enables more aggressive
// optimizations (register allocation, load reordering) because the
// compiler knows writes through one pointer cannot affect reads through
// another.
// ===========================================================================

// Main compute kernel: one GPU thread tests exactly one candidate secret.
extern "C" __global__ void hs256_wordlist(
    const Hs256BruteForceParams* __restrict__ params,
    const uint8_t* __restrict__ message_bytes,
    const uint8_t* __restrict__ word_bytes,
    const unsigned int* __restrict__ word_offsets,
    const unsigned short* __restrict__ word_lengths,
    unsigned int* __restrict__ result_index
) {
    // Compute global thread ID. This is the CUDA equivalent of Metal's
    // `uint gid [[thread_position_in_grid]]`.
    //
    // blockIdx.x  = which block this thread belongs to (0-indexed)
    // blockDim.x  = number of threads per block (e.g. 256)
    // threadIdx.x = thread's index within its block (0 to blockDim.x-1)
    //
    // Together: gid = blockIdx.x * blockDim.x + threadIdx.x gives a
    // unique global index across the entire grid, just like Metal's
    // thread_position_in_grid.
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: the grid may contain more threads than candidates
    // (grid size is rounded up to a multiple of block size).  Extra threads
    // must exit early to avoid out-of-bounds memory access.
    if (gid >= params->candidate_count) return;

    const unsigned int candidate_offset = word_offsets[gid];
    const unsigned int candidate_length = (unsigned int)(word_lengths[gid]);
    const uint8_t* candidate_secret = word_bytes + candidate_offset;

    if (hmac_sha256_matches(candidate_secret, candidate_length,
                            message_bytes, params->message_length,
                            params->target_signature)) {
        // CUDA note: `atomicMin` performs an atomic minimum operation on the
        // value at `result_index`.  If `gid` is less than the current value,
        // it replaces it.  This is the CUDA equivalent of Metal's
        // `atomic_fetch_min_explicit(result_index, gid, memory_order_relaxed)`.
        //
        // The `min` semantic ensures that if multiple threads find a match
        // (shouldn't happen with correct JWTs), the lowest-indexed match
        // wins deterministically.  CUDA's `atomicMin` provides relaxed
        // ordering by default (no memory fence), matching Metal's
        // `memory_order_relaxed`.
        atomicMin(result_index, gid);
    }
}

// Short-key specialization. Host dispatches this when every key in the batch is <= 64 bytes.
// Learning note: by guaranteeing key_len <= 64, this kernel eliminates the
// hash-the-long-key branch, saving one conditional and one full SHA-256
// computation per thread.  The host checks max_word_len at batch level and
// selects this kernel when safe.  This is a common GPU optimization: branch
// elimination via kernel specialization.
//
// CUDA note on warp divergence: eliminating the key_len > 64 branch is
// especially valuable on NVIDIA GPUs because divergent branches within a
// warp cause serialization.  If even one thread in a 32-thread warp takes
// the long-key path, all 32 threads must wait.  By guaranteeing all keys
// are short, we ensure uniform control flow across the entire warp.
extern "C" __global__ void hs256_wordlist_short_keys(
    const Hs256BruteForceParams* __restrict__ params,
    const uint8_t* __restrict__ message_bytes,
    const uint8_t* __restrict__ word_bytes,
    const unsigned int* __restrict__ word_offsets,
    const unsigned short* __restrict__ word_lengths,
    unsigned int* __restrict__ result_index
) {
    // Compute global thread ID (CUDA equivalent of Metal's thread_position_in_grid).
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: exit early if this thread index exceeds the candidate count.
    if (gid >= params->candidate_count) return;

    const unsigned int candidate_offset = word_offsets[gid];
    const unsigned int candidate_length = (unsigned int)(word_lengths[gid]);
    const uint8_t* candidate_secret = word_bytes + candidate_offset;

    if (hmac_sha256_short_key_matches(candidate_secret, candidate_length,
                                      message_bytes, params->message_length,
                                      params->target_signature)) {
        // Atomic minimum write: see comment in hs256_wordlist above.
        atomicMin(result_index, gid);
    }
}
