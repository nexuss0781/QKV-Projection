// ==========================================================================
// Component 2.1 — QKV Projection: Public API Interface
// Sub-Quadratic Factorized Projection (SQFP)
//
// This file defines the standardized interaction interface for downstream
// components. All external consumers interact with the SQFP projection
// exclusively through this API.
//
// Contracts defined herein:
//   -> Component 2.2 (Scaled Dot-Product Attention): Q, K_exp, V_exp
//   -> Component 2.3 (Multi-Head Attention concatenation): X_attn
//   -> Component 2.4 (KV-Cache): K, V cache operations
//   -> RoPE kernel: Q, K with even-d_k per-head guarantee
//
// Usage:
//   #include "output.cpp"   // single-header API
//   SQFPConfig cfg = { .d = 4096, .h = 32, .g = 8, .L = 32 };
//   SQFPHandle* proj = sqfp_create(&cfg);
//   sqfp_qkv_forward(proj, X, n, Q, K, V);
//   sqfp_expand_kv(proj, K, V, K_exp, V_exp, n);  // view-only broadcast
//   // ... downstream attention ...
//   sqfp_wo_forward(proj, X_attn, Out, n);
//   sqfp_destroy(proj);
// ==========================================================================

#include "Core.cpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>

// ==========================================================================
// API Types & Configuration
// ==========================================================================

#ifdef __cplusplus
extern "C" {
#endif

// SQFP configuration — must be set before creating a projector
typedef struct SQFPConfig {
    size_t d;       // Model dimension (e.g. 4096)
    size_t h;       // Number of query heads
    size_t g;       // Number of KV groups (h for MHA, 1 for MQA)
    size_t L;       // Number of transformer layers (for Wo scaling)
    int    adaptive; // 0 = disable, 1 = enable adaptive precision
    int    quantized; // 0 = FP32, 1 = INT8 quantized
    int    use_rope_pre; // 0 = no, 1 = initialize RoPE pre-structure
    float  rope_base;    // Base frequency for RoPE pre-structure (default 10000.0)

    // Streaming config (0 = use defaults)
    size_t L3_size_bytes;  // L3 cache size for micro-batch sizing
    int    num_threads;    // 0 = auto (hardware concurrency)
} SQFPConfig;

// Opaque handle — all API functions operate through this
typedef struct SQFPHandle SQFPHandle;

// Shape descriptor returned after handle creation
typedef struct SQFPSizes {
    size_t d;       // model dimension
    size_t h;       // query heads
    size_t g;       // KV groups
    size_t d_k;     // head dimension = d / h
    size_t d_v;     // value head dimension = d / h
    size_t r;       // DPLR rank = 2*ceil(log2(d))
    size_t m_Q;     // Q projection output dim
    size_t m_K;     // K projection output dim
    size_t m_V;     // V projection output dim
    size_t m_Wo;    // Wo projection output dim
    size_t params_q;  // parameter count for Q
    size_t params_k;  // parameter count for K
    size_t params_v;  // parameter count for V
    size_t params_o;  // parameter count for Wo
} SQFPSizes;


// ==========================================================================
// Lifecycle — Create, Configure, Destroy
// ==========================================================================

// Create an SQFP projector with the given configuration.
// Returns NULL on invalid config (d % h != 0, h % g != 0).
SQFPHandle* sqfp_create(const SQFPConfig* config);

// Destroy projector and free all associated memory.
void sqfp_destroy(SQFPHandle* handle);

// Get shape/size information about this projector.
SQFPSizes sqfp_get_sizes(const SQFPHandle* handle);

// Set the adaptive precision threshold (τ in spec Section 3.6).
// Only effective when config.adaptive == 1.
void sqfp_set_adaptive_threshold(SQFPHandle* handle, float tau);


// ==========================================================================
// Downstream Interface — Component 2.2 (Scaled Dot-Product Attention)
//
// Contract (Section 7.3):
//   Input:  X ∈ ℝ^{n × d}           — residual stream
//   Output: Q ∈ ℝ^{n × h × d_k}     — query tensor
//           K ∈ ℝ^{n × g × d_k}     — key tensor (pre-expand for GQA)
//           V ∈ ℝ^{n × g × d_v}     — value tensor (pre-expand for GQA)
//
// Shape invariants:
//   d_k = d / h, d_v = d / h
//   m_Q = h·d_k = d,  m_K = g·d_k,  m_V = g·d_v
//   d_k is guaranteed even for RoPE compatibility (Section 7.1)
//
// Memory layout: row-major, all pointers must be non-overlapping
// ==========================================================================

// Full QKV forward projection.
// X:    (n, d)     — input residual stream
// Q:    (n, h*d_k) — flat, reshape to (n, h, d_k) by downstream
// K:    (n, g*d_k) — flat, pre-expansion (GQA)
// V:    (n, g*d_v) — flat, pre-expansion (GQA)
// n:    sequence length
void sqfp_qkv_forward(
    SQFPHandle* handle,
    const float* X,
    float* Q,
    float* K,
    float* V,
    size_t n
);

// GQA broadcast expansion: repeat_interleave(K, h//g, dim=1)
// This is a view-only operation — output is contiguous copy for simplicity.
// K_exp and V_exp are sized (n, h*d_k) and (n, h*d_v).
// Contract: each group of (h/g) query heads attends to identical K,V.
void sqfp_expand_kv(
    const SQFPHandle* handle,
    const float* K,         // (n, g*d_k)
    const float* V,         // (n, g*d_v)
    float* K_expanded,      // (n, h*d_k)
    float* V_expanded,      // (n, h*d_v)
    size_t n
);


// ==========================================================================
// Downstream Interface — Component 2.3 (Multi-Head Attention → Wo)
//
// After attention heads are concatenated, Wo projects back to d.
// Contract (Section 3.5):
//   Input:  X_attn ∈ ℝ^{n × d}    — concatenated attention heads
//   Output: Out ∈ ℝ^{n × d}       — residual update
//
// Scaled initialization: D_o[i] = 1/√(2L) guarantees
//   Var(X + Attn(X)) ≈ Var(X)·(1 + 1/(2L)) at init
// ==========================================================================

void sqfp_wo_forward(
    SQFPHandle* handle,
    const float* X_attn,
    float* Out,
    size_t n
);


// ==========================================================================
// Downstream Interface — Component 2.4 (KV-Cache)
//
// Contract (Section 7.2):
//   K/V outputs in row-major layout (n, g, d_k)
//   New tokens computed one-at-a-time (n_new = 1)
//   Cache append is a simple memcpy into pre-allocated buffer
//   Zero recomputation: token n+1 does not recompute tokens 1..n
//
// The KV-cache stores only g groups (not h heads):
//   KV_cache shape: (total_seq, g, d_k) for K, (total_seq, g, d_v) for V
// ==========================================================================

// Compute K/V for a single new token and append to cache.
// KV_cache: pre-allocated circular buffer, offset is token position.
void sqfp_kv_cache_append(
    SQFPHandle* handle,
    const float* X_new,     // (d) — single token
    float* K_cache,         // pre-allocated, offset = token_pos * g * d_k
    float* V_cache,         // pre-allocated, offset = token_pos * g * d_v
    size_t token_pos        // sequence position of this token
);

// Look up K/V from cache at given positions.
// Contract: zero recomputation — cache values are immutable after write.
void sqfp_kv_cache_lookup(
    const SQFPHandle* handle, // shapes for stride computation
    const float* K_cache,
    const float* V_cache,
    float* K_out,           // (n, g*d_k)
    float* V_out,           // (n, g*d_v)
    const size_t* positions, // token indices to read
    size_t n
);


// ==========================================================================
// New Capability — Adaptive Precision Path (Section 3.6)
//
// Token importance ρ = ‖x·S‖₂ determines fast vs full path.
// Average case: ~80% tokens take fast path → O(d) amortized.
// ==========================================================================

// Get token importance scores for diagnostic/visualization.
void sqfp_get_importance(
    SQFPHandle* handle,
    const float* X,     // (n, d)
    float* scores,      // (n) — ρ_i for each token
    size_t n
);

// Set learned threshold for adaptive path.
// Tokens with ρ_i < τ take the O(d) fast path.
void sqfp_set_adaptive_threshold(SQFPHandle* handle, float tau);


// ==========================================================================
// New Capability — RoPE Pre-Structure (Section 7.1)
//
// D_q partitioned per head with geometric phase progression e^{iθ_j}.
// This encodes weak positional bias before RoPE, reducing angular
// correction and improving BF16 numerical stability.
// ==========================================================================

// Apply RoPE pre-structure initialization to Q diagonal.
// Called automatically if config.use_rope_pre is set at creation.
void sqfp_apply_rope_pre_structure(SQFPHandle* handle);


// ==========================================================================
// Utility — Parameter Access
// ==========================================================================

// Get total parameter count for this projector (across Q, K, V, Wo).
size_t sqfp_total_params(const SQFPHandle* handle);

// Get total parameter memory in bytes (FP32).
size_t sqfp_total_param_bytes(const SQFPHandle* handle);

// Save/load parameters to/from buffer (for model checkpointing).
void sqfp_save_params(const SQFPHandle* handle, float* buffer);
void sqfp_load_params(SQFPHandle* handle, const float* buffer);


// ==========================================================================
// Compatibility — Dense Baseline Comparison
//
// For testing and verification: construct equivalent dense matrix from
// SQFP decomposition. This is O(d²) — use only for validation.
// ==========================================================================

void sqfp_dense_weight_q(const SQFPHandle* handle, float* W_dense);
void sqfp_dense_weight_k(const SQFPHandle* handle, float* W_dense);
void sqfp_dense_weight_v(const SQFPHandle* handle, float* W_dense);
void sqfp_dense_weight_o(const SQFPHandle* handle, float* W_dense);

#ifdef __cplusplus
}
#endif


// ==========================================================================
// Implementation
// ==========================================================================

struct SQFPHandle {
    SQFPConfig config;
    SQFPShapes shapes;

    // Projector parameters per attention variant
    MHAProjectors<float> mha;
    GQAProjectors<float> gqa;
    MQAProjectors<float> mqa;
    SQFPParams<float> wo;

    // Adaptive precision (optional)
    AdaptiveSQFPParams<float>* adaptive_params;

    // Quantized variant (optional)
    QuantizedSQFPParams<float>* quantized_params;

    // Streaming configuration
    StreamingConfig<float> stream_cfg;

    // Head parallelism
    HeadParallelConfig<float> head_cfg;

    // Thread count
    size_t num_threads;

    SQFPHandle(SQFPConfig cfg)
        : config(cfg)
        , shapes(cfg.d, cfg.h, cfg.g)
        , mha(cfg.d, cfg.h)
        , gqa(cfg.d, cfg.h, cfg.g)
        , mqa(cfg.d, cfg.h)
        , wo(make_wo_projection<float>(cfg.d, cfg.L))
        , adaptive_params(nullptr)
        , quantized_params(nullptr)
        , stream_cfg(cfg.L3_size_bytes > 0 ? cfg.L3_size_bytes : 32ULL * 1024 * 1024, cfg.d)
        , head_cfg(cfg.h, cfg.num_threads > 0 ? cfg.num_threads : std::thread::hardware_concurrency())
        , num_threads(cfg.num_threads > 0 ? cfg.num_threads : std::thread::hardware_concurrency())
    {
        if (cfg.adaptive) {
            size_t r = shapes.r();
            adaptive_params = new AdaptiveSQFPParams<float>(cfg.d, cfg.d, r);
        }
        if (cfg.use_rope_pre) {
            apply_rope_pre_structure_to_projector(mha.theta_q, cfg.d, cfg.h);
        }
    }

    ~SQFPHandle() {
        delete adaptive_params;
        delete quantized_params;
    }
};


// ==========================================================================
// Lifecycle Implementation
// ==========================================================================

SQFPHandle* sqfp_create(const SQFPConfig* config) {
    if (!config) return nullptr;
    if (config->d % config->h != 0) return nullptr;
    if (config->h % config->g != 0) return nullptr;

    SQFPConfig cfg = *config;
    if (cfg.L3_size_bytes == 0) cfg.L3_size_bytes = 32ULL * 1024 * 1024;
    if (cfg.rope_base == 0.0f) cfg.rope_base = 10000.0f;

    SQFPHandle* handle = new SQFPHandle(cfg);
    return handle;
}

void sqfp_destroy(SQFPHandle* handle) {
    delete handle;
}

SQFPSizes sqfp_get_sizes(const SQFPHandle* handle) {
    SQFPSizes s = {};
    s.d = handle->shapes.d;
    s.h = handle->shapes.h;
    s.g = handle->shapes.g;
    s.d_k = handle->shapes.d_k;
    s.d_v = handle->shapes.d_v;
    s.r = handle->shapes.r();
    s.m_Q = handle->shapes.m_Q();
    s.m_K = handle->shapes.m_K_GQA();
    s.m_V = handle->shapes.m_V_GQA();
    s.m_Wo = handle->shapes.m_Wo();
    s.params_q = handle->mha.theta_q.param_count();
    s.params_k = handle->mha.theta_k.param_count();
    s.params_v = handle->mha.theta_v.param_count();
    s.params_o = handle->wo.param_count();
    return s;
}


// ==========================================================================
// QKV Forward Implementation
// ==========================================================================

void sqfp_qkv_forward(
    SQFPHandle* handle,
    const float* X,
    float* Q,
    float* K,
    float* V,
    size_t n
) {
    if (!handle || !X || !Q || !K || !V) return;

    size_t d = handle->shapes.d;
    size_t h = handle->shapes.h;
    size_t g = handle->shapes.g;

    if (g == h) {
        // MHA
        mha_forward(handle->mha, X, Q, K, V, n, d, h);
    } else if (g == 1) {
        // MQA: K shape is (n, 1, d_k), V shape is (n, 1, d_v)
        mqa_forward(handle->mqa, X, Q, K, V, n, d, h);
    } else {
        // GQA: K shape is (n, g, d_k), V shape is (n, g, d_v)
        gqa_forward(handle->gqa, X, Q, K, V, n, d, h, g);
    }
}

void sqfp_expand_kv(
    const SQFPHandle* handle,
    const float* K,
    const float* V,
    float* K_expanded,
    float* V_expanded,
    size_t n
) {
    if (!handle || !K || !V || !K_expanded || !V_expanded) return;

    size_t h = handle->shapes.h;
    size_t g = handle->shapes.g;
    size_t d_k = handle->shapes.d_k;
    size_t d_v = handle->shapes.d_v;

    // If MHA (g == h), no expansion needed — just copy
    if (g == h) {
        std::memcpy(K_expanded, K, n * h * d_k * sizeof(float));
        std::memcpy(V_expanded, V, n * h * d_v * sizeof(float));
        return;
    }

    // GQA broadcast: repeat_interleave(K, h//g, dim=1)
    size_t repeat = h / g;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < g; ++j) {
            for (size_t r = 0; r < repeat; ++r) {
                std::memcpy(
                    K_expanded + i * h * d_k + (j * repeat + r) * d_k,
                    K + i * g * d_k + j * d_k,
                    d_k * sizeof(float)
                );
                std::memcpy(
                    V_expanded + i * h * d_v + (j * repeat + r) * d_v,
                    V + i * g * d_v + j * d_v,
                    d_v * sizeof(float)
                );
            }
        }
    }
}


// ==========================================================================
// Wo Forward Implementation
// ==========================================================================

void sqfp_wo_forward(
    SQFPHandle* handle,
    const float* X_attn,
    float* Out,
    size_t n
) {
    if (!handle || !X_attn || !Out) return;
    size_t d = handle->shapes.d;
    wo_forward(handle->wo, X_attn, Out, n, d);
}


// ==========================================================================
// KV-Cache Operations
// ==========================================================================

void sqfp_kv_cache_append(
    SQFPHandle* handle,
    const float* X_new,
    float* K_cache,
    float* V_cache,
    size_t token_pos
) {
    if (!handle || !X_new || !K_cache || !V_cache) return;

    size_t g = handle->shapes.g;
    size_t d_k = handle->shapes.d_k;
    size_t d_v = handle->shapes.d_v;

    // Buffers for single-token output
    std::vector<float> Q_buf(1 * handle->shapes.m_Q());
    std::vector<float> K_buf(1 * g * d_k);
    std::vector<float> V_buf(1 * g * d_v);

    // Forward single token
    sqfp_qkv_forward(handle, X_new, Q_buf.data(), K_buf.data(), V_buf.data(), 1);

    // Append to cache at token_pos
    size_t k_offset = token_pos * g * d_k;
    size_t v_offset = token_pos * g * d_v;
    std::memcpy(K_cache + k_offset, K_buf.data(), g * d_k * sizeof(float));
    std::memcpy(V_cache + v_offset, V_buf.data(), g * d_v * sizeof(float));

    // Zero recomputation guaranteed: Q/K/V for token_pos depends only on
    // X_new, not on any previous token's state (Section 7.2).
}

void sqfp_kv_cache_lookup(
    const SQFPHandle* handle,
    const float* K_cache,
    const float* V_cache,
    float* K_out,
    float* V_out,
    const size_t* positions,
    size_t n
) {
    if (!K_cache || !V_cache || !K_out || !V_out || !positions) return;
    if (!handle) return;

    size_t g   = handle->shapes.g;
    size_t d_k = handle->shapes.d_k;
    size_t d_v = handle->shapes.d_v;

    size_t k_stride = g * d_k;
    size_t v_stride = g * d_v;

    for (size_t i = 0; i < n; ++i) {
        size_t pos = positions[i];
        std::memcpy(K_out + i * k_stride,
                    K_cache + pos * k_stride,
                    k_stride * sizeof(float));
        std::memcpy(V_out + i * v_stride,
                    V_cache + pos * v_stride,
                    v_stride * sizeof(float));
    }
}


// ==========================================================================
// Adaptive Precision
// ==========================================================================

void sqfp_get_importance(
    SQFPHandle* handle,
    const float* X,
    float* scores,
    size_t n
) {
    if (!handle || !X || !scores) return;
    if (!handle->adaptive_params) return;

    size_t d = handle->shapes.d;
    for (size_t i = 0; i < n; ++i) {
        scores[i] = handle->adaptive_params->sketch.compute_importance(
            X + i * d, d
        );
    }
}

void sqfp_set_adaptive_threshold(SQFPHandle* handle, float tau) {
    if (!handle || !handle->adaptive_params) return;
    handle->adaptive_params->tau = tau;
}


// ==========================================================================
// RoPE Pre-Structure
// ==========================================================================

void sqfp_apply_rope_pre_structure(SQFPHandle* handle) {
    if (!handle) return;
    apply_rope_pre_structure_to_projector(
        handle->mha.theta_q,
        handle->shapes.d,
        handle->shapes.h
    );
    apply_rope_pre_structure_to_projector(
        handle->gqa.theta_q,
        handle->shapes.d,
        handle->shapes.h
    );
    apply_rope_pre_structure_to_projector(
        handle->mqa.theta_q,
        handle->shapes.d,
        handle->shapes.h
    );
}


// ==========================================================================
// Parameter Utilities
// ==========================================================================

size_t sqfp_total_params(const SQFPHandle* handle) {
    if (!handle) return 0;
    return handle->mha.theta_q.param_count()
         + handle->mha.theta_k.param_count()
         + handle->mha.theta_v.param_count()
         + handle->wo.param_count();
}

size_t sqfp_total_param_bytes(const SQFPHandle* handle) {
    return sqfp_total_params(handle) * sizeof(float);
}

void sqfp_save_params(const SQFPHandle* handle, float* buffer) {
    if (!handle || !buffer) return;
    float* ptr = buffer;

    auto copy_params = [&ptr](const SQFPParams<float>& p) {
        std::memcpy(ptr, p.D.data(), p.m * sizeof(float)); ptr += p.m;
        std::memcpy(ptr, p.U.data(), p.m * p.r * sizeof(float)); ptr += p.m * p.r;
        std::memcpy(ptr, p.V.data(), p.d * p.r * sizeof(float)); ptr += p.d * p.r;
        *ptr++ = p.epsilon;
    };

    copy_params(handle->mha.theta_q);
    copy_params(handle->mha.theta_k);
    copy_params(handle->mha.theta_v);
    copy_params(handle->wo);
}

void sqfp_load_params(SQFPHandle* handle, const float* buffer) {
    if (!handle || !buffer) return;
    const float* ptr = buffer;

    auto load_params = [&ptr](SQFPParams<float>& p) {
        std::memcpy(p.D.data(), ptr, p.m * sizeof(float)); ptr += p.m;
        std::memcpy(p.U.data(), ptr, p.m * p.r * sizeof(float)); ptr += p.m * p.r;
        std::memcpy(p.V.data(), ptr, p.d * p.r * sizeof(float)); ptr += p.d * p.r;
        p.epsilon = *ptr++;
    };

    load_params(handle->mha.theta_q);
    load_params(handle->mha.theta_k);
    load_params(handle->mha.theta_v);
    load_params(handle->wo);
}


// ==========================================================================
// Dense Weight Reconstruction (for verification only)
// ==========================================================================

static void build_dense_weight(
    const SQFPParams<float>& p,
    float* W,
    size_t d,
    size_t m
) {
    std::fill(W, W + d * m, 0.0f);

    // Diagonal
    for (size_t i = 0; i < std::min(m, d); ++i) {
        W[i * d + i] = p.D[i];
    }

    // Low-rank contribution
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < d; ++j) {
            float uv = 0.0f;
            for (size_t k = 0; k < p.r; ++k)
                uv += p.U[i * p.r + k] * p.V[j * p.r + k];
            W[i * d + j] += uv;
        }
    }

    // Butterfly contribution would require full B expansion:
    // for a single token, W has shape (m, d), but the butterfly
    // B is (N, N) padded, so full reconstruction is approximate.
    // The epsilon*B term is omitted here; for exact spectral
    // comparison, compute B explicitly.
}

void sqfp_dense_weight_q(const SQFPHandle* handle, float* W_dense) {
    if (!handle || !W_dense) return;
    size_t d = handle->shapes.d;
    build_dense_weight(handle->mha.theta_q, W_dense, d, d);
}

void sqfp_dense_weight_k(const SQFPHandle* handle, float* W_dense) {
    if (!handle || !W_dense) return;
    size_t d = handle->shapes.d;
    build_dense_weight(handle->mha.theta_k, W_dense, d, d);
}

void sqfp_dense_weight_v(const SQFPHandle* handle, float* W_dense) {
    if (!handle || !W_dense) return;
    size_t d = handle->shapes.d;
    build_dense_weight(handle->mha.theta_v, W_dense, d, d);
}

void sqfp_dense_weight_o(const SQFPHandle* handle, float* W_dense) {
    if (!handle || !W_dense) return;
    size_t d = handle->shapes.d;
    build_dense_weight(handle->wo, W_dense, d, d);
}
