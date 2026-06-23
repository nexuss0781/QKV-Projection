// ==========================================================================
// Input.cpp — Phase-1 → Phase-2.1 Bridge
// Unified Input Layer: Tokenize → Embed (HFAQE) → Position-Encode (HDPE)
//   → QKV Projection (SQFP)
//
// Downstream contracts:
//   -> Component 2.2 (Scaled Dot-Product Attention): Q, K_exp, V_exp
//   -> Component 2.3 (MHA output projection): X_attn via sqfp_wo_forward
//   -> Component 2.4 (KV-Cache): K, V cache operations
//
// Pipeline:
//   text/token_ids
//     → InputEngine (Phase 1): tokenize → HFAQE embed → HDPE RoPE
//     → X ∈ ℝ^{B × n × d} (position-aware, fp32 row-major)
//     → sqfp_qkv_forward (Phase 2.1): SQFP projection → Q, K, V
//     → downstream attention
// ==========================================================================

#include "output.cpp"

// ---------------------------------------------------------------------------
// Phase-1 Input Pipeline (Tokenize → Embed → Position-Encode)
// ---------------------------------------------------------------------------
#include "../../Phase-1_Input-Processing/master-input/src/input_engine.cpp"

// ==========================================================================
// Bridge Types & Configuration
// ==========================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Unified configuration — combines InputEngine + SQFP configs
typedef struct BridgeConfig {
    // Embedding
    int V  = 16000;   // vocabulary size
    int d  = 512;     // model dimension
    int r  = 64;      // cold-tier rank (HFAQE)
    int K  = 1024;    // number of hot tokens (HFAQE)

    // Positional encoding (HDPE)
    int B_hdpe  = 64;
    int L_hdpe  = 4;
    int h       = 8;       // heads
    float rope_base = 10000.0f;

    // QKV (SQFP)
    int g          = 8;   // KV groups (h for MHA, 1 for MQA)
    int L_layers   = 32;  // number of transformer layers
    int adaptive   = 0;   // adaptive precision path
    int quantized  = 0;   // INT8 quantized path
    int use_rope_pre = 0; // RoPE pre-structure initialization

    // Streaming
    size_t L3_size_bytes = 0; // 0 = default (32MB)
    int    num_threads   = 0; // 0 = auto

    // Optional tokenizer vocab file for C++ tokenizer
    const char* vocab_path = nullptr;
} BridgeConfig;

// Opaque handle
typedef struct InputBridge InputBridge;

// Shape descriptor
typedef struct BridgeSizes {
    size_t d;        // model dimension
    size_t h;        // query heads
    size_t g;        // KV groups
    size_t d_k;      // head dimension
    size_t d_v;      // value head dimension
    size_t n_vocab;  // vocabulary size
    size_t m_Q;      // Q projection output dim
    size_t m_K;      // K projection output dim
    size_t m_V;      // V projection output dim
    size_t m_Wo;     // Wo projection output dim
} BridgeSizes;


// ==========================================================================
// Lifecycle
// ==========================================================================

// Create the unified input bridge.
// Returns NULL on invalid config (d % h != 0, h % g != 0).
InputBridge* bridge_create(const BridgeConfig* config);

// Destroy bridge and free all associated memory.
void bridge_destroy(InputBridge* bridge);

// Get shape/size information.
BridgeSizes bridge_get_sizes(const InputBridge* bridge);


// ==========================================================================
// Forward — Text → Token IDs → Embeddings → QKV
// ==========================================================================

// Full pipeline: text string → Q, K, V.
// Returns number of tokens, or 0 on failure.
//   text:       input string (UTF-8)
//   Q:          output buffer (max_tokens × h × d_k) — must be pre-allocated
//   K:          output buffer (max_tokens × g × d_k)
//   V:          output buffer (max_tokens × g × d_v)
//   max_tokens: capacity of output buffers
// Returns actual number of tokens processed.
int bridge_forward_text(
    InputBridge* bridge,
    const char* text,
    float* Q,
    float* K,
    float* V,
    int max_tokens
);

// Token IDs → Q, K, V (skip tokenization step).
//   token_ids: array of token IDs, length n
//   n:         number of tokens
//   Q, K, V:   output buffers sized (n × h × d_k), (n × g × d_k), (n × g × d_v)
void bridge_forward_tokens(
    InputBridge* bridge,
    const int* token_ids,
    int n,
    float* Q,
    float* K,
    float* V
);

// Token IDs → embeddings only (no QKV projection).
//   out: output buffer (n × d) — position-aware embeddings
void bridge_embed_tokens(
    InputBridge* bridge,
    const int* token_ids,
    int n,
    float* out
);


// ==========================================================================
// QKV Operations (delegated to SQFP)
// ==========================================================================

// GQA broadcast: repeat_interleave(K, h//g, dim=1)
void bridge_expand_kv(
    const InputBridge* bridge,
    const float* K,
    const float* V,
    float* K_expanded,
    float* V_expanded,
    size_t n
);

// Wo output projection (after attention heads are concatenated).
void bridge_wo_forward(
    InputBridge* bridge,
    const float* X_attn,
    float* Out,
    size_t n
);

// KV-cache: compute K/V for one token and append to cache.
void bridge_kv_cache_append(
    InputBridge* bridge,
    const float* X_new,
    float* K_cache,
    float* V_cache,
    size_t token_pos
);

// KV-cache: look up K/V from cache at given positions.
void bridge_kv_cache_lookup(
    const InputBridge* bridge,
    const float* K_cache,
    const float* V_cache,
    float* K_out,
    float* V_out,
    const size_t* positions,
    size_t n
);


// ==========================================================================
// Tokenizer Access
// ==========================================================================

// Tokenize text → token IDs (no embedding).
// Returns number of tokens written.
//   ids: output buffer, sized max_ids
int bridge_tokenize(
    InputBridge* bridge,
    const char* text,
    int* ids,
    int max_ids
);

// Get vocabulary size.
int bridge_vocab_size(const InputBridge* bridge);


// ==========================================================================
// Diagnostic
// ==========================================================================

// Get token importance scores (adaptive precision diagnostic).
void bridge_get_importance(
    InputBridge* bridge,
    const float* X,
    float* scores,
    size_t n
);

#ifdef __cplusplus
}
#endif


// ==========================================================================
// Implementation
// ==========================================================================

struct InputBridge {
    BridgeConfig config;
    InputEngine engine;
    SQFPHandle* sqfp;
    InputEngineConfig ie_cfg;

    InputBridge(const BridgeConfig& cfg)
        : config(cfg)
    {
        ie_cfg.V      = cfg.V;
        ie_cfg.d      = cfg.d;
        ie_cfg.r      = cfg.r;
        ie_cfg.K      = cfg.K;
        ie_cfg.B      = cfg.B_hdpe;
        ie_cfg.L      = cfg.L_hdpe;
        ie_cfg.h      = cfg.h;
        ie_cfg.base   = cfg.rope_base;

        if (cfg.vocab_path && cfg.vocab_path[0]) {
            ie_cfg.tokenizer_path = cfg.vocab_path;
        }

        engine.init(ie_cfg);

        SQFPConfig sqfp_cfg = {};
        sqfp_cfg.d           = cfg.d;
        sqfp_cfg.h           = cfg.h;
        sqfp_cfg.g           = cfg.g;
        sqfp_cfg.L           = cfg.L_layers;
        sqfp_cfg.adaptive    = cfg.adaptive;
        sqfp_cfg.quantized   = cfg.quantized;
        sqfp_cfg.use_rope_pre = cfg.use_rope_pre;
        sqfp_cfg.rope_base   = cfg.rope_base;
        sqfp_cfg.L3_size_bytes = cfg.L3_size_bytes;
        sqfp_cfg.num_threads = cfg.num_threads;
        sqfp = sqfp_create(&sqfp_cfg);
    }

    ~InputBridge() {
        if (sqfp) sqfp_destroy(sqfp);
    }
};


InputBridge* bridge_create(const BridgeConfig* config) {
    if (!config) return nullptr;
    if (config->d % config->h != 0) return nullptr;
    if (config->h % config->g != 0) return nullptr;

    return new InputBridge(*config);
}

void bridge_destroy(InputBridge* bridge) {
    delete bridge;
}

BridgeSizes bridge_get_sizes(const InputBridge* bridge) {
    BridgeSizes s = {};
    if (!bridge || !bridge->sqfp) return s;
    auto sq = sqfp_get_sizes(bridge->sqfp);
    s.d       = sq.d;
    s.h       = sq.h;
    s.g       = sq.g;
    s.d_k     = sq.d_k;
    s.d_v     = sq.d_v;
    s.n_vocab = bridge->ie_cfg.V;
    s.m_Q     = sq.m_Q;
    s.m_K     = sq.m_K;
    s.m_V     = sq.m_V;
    s.m_Wo    = sq.m_Wo;
    return s;
}


// ==========================================================================
// Forward Implementation
// ==========================================================================

int bridge_forward_text(
    InputBridge* bridge,
    const char* text,
    float* Q,
    float* K,
    float* V,
    int max_tokens
) {
    if (!bridge || !text || !Q || !K || !V) return 0;

    if (!bridge->sqfp) return 0;

    size_t d = bridge->config.d;

    // Step 1: Tokenize → position-aware embeddings
    // Use a temporary buffer sized for max_tokens
    std::vector<float> X(static_cast<size_t>(max_tokens) * d);
    int n = bridge->engine.forward_text(text, X.data(), max_tokens);
    if (n <= 0) return 0;

    // Step 2: QKV projection
    sqfp_qkv_forward(bridge->sqfp, X.data(), Q, K, V, static_cast<size_t>(n));

    return n;
}

void bridge_forward_tokens(
    InputBridge* bridge,
    const int* token_ids,
    int n,
    float* Q,
    float* K,
    float* V
) {
    if (!bridge || !token_ids || !Q || !K || !V) return;
    if (!bridge->sqfp || n <= 0) return;

    size_t d = bridge->config.d;

    // Embed + position-encode
    std::vector<float> X(static_cast<size_t>(n) * d);
    bridge->engine.forward(token_ids, n, X.data());

    // QKV projection
    sqfp_qkv_forward(bridge->sqfp, X.data(), Q, K, V, static_cast<size_t>(n));
}

void bridge_embed_tokens(
    InputBridge* bridge,
    const int* token_ids,
    int n,
    float* out
) {
    if (!bridge || !token_ids || !out) return;
    bridge->engine.forward(token_ids, n, out);
}


// ==========================================================================
// QKV Operations
// ==========================================================================

void bridge_expand_kv(
    const InputBridge* bridge,
    const float* K,
    const float* V,
    float* K_expanded,
    float* V_expanded,
    size_t n
) {
    if (!bridge || !bridge->sqfp) return;
    sqfp_expand_kv(bridge->sqfp, K, V, K_expanded, V_expanded, n);
}

void bridge_wo_forward(
    InputBridge* bridge,
    const float* X_attn,
    float* Out,
    size_t n
) {
    if (!bridge || !bridge->sqfp) return;
    sqfp_wo_forward(bridge->sqfp, X_attn, Out, n);
}

void bridge_kv_cache_append(
    InputBridge* bridge,
    const float* X_new,
    float* K_cache,
    float* V_cache,
    size_t token_pos
) {
    if (!bridge || !bridge->sqfp) return;
    sqfp_kv_cache_append(bridge->sqfp, X_new, K_cache, V_cache, token_pos);
}

void bridge_kv_cache_lookup(
    const InputBridge* bridge,
    const float* K_cache,
    const float* V_cache,
    float* K_out,
    float* V_out,
    const size_t* positions,
    size_t n
) {
    if (!bridge) return;
    sqfp_kv_cache_lookup(bridge->sqfp, K_cache, V_cache, K_out, V_out, positions, n);
}


// ==========================================================================
// Tokenizer Access
// ==========================================================================

int bridge_tokenize(
    InputBridge* bridge,
    const char* text,
    int* ids,
    int max_ids
) {
    if (!bridge || !text || !ids) return 0;

    auto token_vec = bridge->engine.tokenize(text);
    int n = static_cast<int>(token_vec.size());
    if (n > max_ids) n = max_ids;
    for (int i = 0; i < n; i++) ids[i] = token_vec[i];
    return n;
}

int bridge_vocab_size(const InputBridge* bridge) {
    if (!bridge) return 0;
    return bridge->ie_cfg.V;
}


// ==========================================================================
// Diagnostic
// ==========================================================================

void bridge_get_importance(
    InputBridge* bridge,
    const float* X,
    float* scores,
    size_t n
) {
    if (!bridge || !bridge->sqfp) return;
    sqfp_get_importance(bridge->sqfp, X, scores, n);
}
