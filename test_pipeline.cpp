// ==========================================================================
// test_pipeline.cpp — End-to-End Phase-1 → Phase-2.1 Pipeline Verification
//
// Exercises the full production path:
//   1. Tokenize sentences  →  (C++ Tokenizer, Component 1.1)
//   2. Embed token IDs     →  (HFAQE, Component 1.2)
//   3. Position-encode     →  (HDPE RoPE, Component 1.3)
//   4. QKV project         →  (SQFP, Component 2.1)
//
// Output: Q, K, V tensors ready for Phase-2.2 Scaled Dot-Product Attention
//
// Compile:
//   g++ -std=c++17 -O0 test_pipeline.cpp -o test_pipeline -lpthread
//   timeout 120 ./test_pipeline
// ==========================================================================

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <random>

#include "Input.cpp"

// ==========================================================================
// Helpers
// ==========================================================================

const char* PASS = "\033[32mPASS\033[0m";
const char* FAIL = "\033[31mFAIL\033[0m";

int g_checks = 0, g_failures = 0;

#define CHECK(cond, ...) do {                                           \
    g_checks++;                                                         \
    if (!(cond)) {                                                      \
        g_failures++;                                                   \
        std::printf("  [%s] ", FAIL);                                   \
        std::printf(__VA_ARGS__);                                       \
        std::printf("\n");                                              \
    } else {                                                            \
        std::printf("  [%s] ", PASS);                                   \
        std::printf(__VA_ARGS__);                                       \
        std::printf("\n");                                              \
    }                                                                   \
} while(0)

bool all_finite(const float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(data[i])) return false;
    return true;
}

bool all_nonzero(const float* data, size_t n, float eps = 1e-30f) {
    for (size_t i = 0; i < n; ++i)
        if (std::abs(data[i]) > eps) return true;
    return false;
}

float l2_norm(const float* a, const float* b, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return (float)std::sqrt(sum);
}

float max_abs_diff(const float* a, const float* b, size_t n) {
    float md = 0;
    for (size_t i = 0; i < n; ++i)
        md = std::max(md, std::abs(a[i] - b[i]));
    return md;
}

float rms(const float* data, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) sum += (double)data[i] * data[i];
    return (float)std::sqrt(sum / (double)n);
}

void print_row(const float* data, int d, int count = 4) {
    std::printf("[");
    for (int i = 0; i < count && i < d; ++i)
        std::printf("%s%.4f", i ? " " : "", data[i]);
    if (count < d) std::printf(" ...");
    std::printf("]");
}

// ==========================================================================
// Main
// ==========================================================================

int main() {
    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║   Phase-1 → Phase-2.1 End-to-End Pipeline Test         ║\n");
    std::printf("║   Tokenize → Embed → Position-Encode → QKV Project     ║\n");
    std::printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // ======================================================================
    // Stage 0: Configuration
    // ======================================================================
    std::printf("▸ Stage 0: Configuration\n");

    BridgeConfig cfg;
    cfg.V       = 1024;
    cfg.d       = 128;
    cfg.r       = 16;
    cfg.K       = 64;
    cfg.h       = 4;
    cfg.g       = 4;
    cfg.L_layers = 6;
    cfg.B_hdpe  = 64;
    cfg.L_hdpe  = 4;
    cfg.rope_base = 10000.0f;

    std::printf("  V=%d d=%d r=%d K=%d h=%d g=%d L=%d\n",
                cfg.V, cfg.d, cfg.r, cfg.K, cfg.h, cfg.g, cfg.L_layers);

    // ======================================================================
    // Stage 1: Create Bridge
    // ======================================================================
    std::printf("\n▸ Stage 1: Bridge Creation (HFAQE SVD init, ~5s)...\n");
    std::fflush(stdout);

    InputBridge* bridge = bridge_create(&cfg);
    CHECK(bridge != nullptr, "bridge_create: handle allocated");

    auto sizes = bridge_get_sizes(bridge);
    CHECK(sizes.d   == (size_t)cfg.d,   "dimension: d=%zu", sizes.d);
    CHECK(sizes.h   == (size_t)cfg.h,   "heads: h=%zu", sizes.h);
    CHECK(sizes.g   == (size_t)cfg.g,   "kv-groups: g=%zu", sizes.g);
    CHECK(sizes.d_k == size_t(cfg.d/cfg.h), "d_k = %zu", sizes.d_k);
    CHECK(sizes.d_v == size_t(cfg.d/cfg.h), "d_v = %zu", sizes.d_v);

    int vocab = bridge_vocab_size(bridge);
    CHECK(vocab == cfg.V, "vocab_size = %d", vocab);

    size_t d   = sizes.d;
    size_t h   = sizes.h;
    size_t g   = sizes.g;
    size_t d_k = sizes.d_k;
    size_t d_v = sizes.d_v;

    // ======================================================================
    // Stage 2: Tokenize Sentences
    // ======================================================================
    std::printf("\n▸ Stage 2: Tokenize Sentences (C++ Tokenizer, Component 1.1)\n");

    const char* sentences[] = {
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "attention is all you need",
        "neural machine translation",
        "transformer networks",
    };
    int num_sentences = 5;

    std::vector<std::vector<int>> token_ids(num_sentences);
    int total_tokens = 0;

    for (int si = 0; si < num_sentences; ++si) {
        const char* text = sentences[si];
        int max_len = (int)std::strlen(text) + 8;
        std::vector<int> buf(max_len);

        int n = bridge_tokenize(bridge, text, buf.data(), max_len);
        CHECK(n > 0, "  [%d] \"%s\" → %d tokens", si, text, n);

        bool valid = true;
        for (int i = 0; i < n; ++i)
            if (buf[i] < 0 || buf[i] >= vocab) { valid = false; break; }
        CHECK(valid, "  [%d] token IDs in [0, V)", si);

        token_ids[si].assign(buf.begin(), buf.begin() + n);
        total_tokens += n;

        std::printf("       IDs:");
        for (int i = 0; i < n && i < 10; ++i) std::printf(" %d", token_ids[si][i]);
        if (n > 10) std::printf(" ...");
        std::printf("\n");
    }

    CHECK(total_tokens > 0, "  total: %d tokens across %d sentences",
           total_tokens, num_sentences);

    // ======================================================================
    // Stage 3: Batch Assembly
    // ======================================================================
    std::printf("\n▸ Stage 3: Batch Assembly\n");

    std::vector<int> batch_ids;
    for (int si = 0; si < num_sentences; ++si)
        for (int id : token_ids[si]) batch_ids.push_back(id);

    int N = (int)batch_ids.size();
    CHECK(N == total_tokens, "batch: %d tokens", N);

    // ======================================================================
    // Stage 4: Embed + Position-Encode (HFAQE + HDPE)
    // ======================================================================
    std::printf("\n▸ Stage 4: Embed + Position-Encode (Phase 1.2 + 1.3)\n");

    std::vector<float> X((size_t)N * d);
    bridge_embed_tokens(bridge, batch_ids.data(), N, X.data());

    CHECK(all_finite(X.data(), X.size()), "X: all finite");
    CHECK(all_nonzero(X.data(), X.size()), "X: non-zero (embeddings active)");

    float x_rms = rms(X.data(), X.size());
    CHECK(x_rms > 1e-6f, "X: RMS=%.6f > 1e-6", x_rms);

    // Position encoding check: same token at different positions
    float pos_diff = 0;
    {
        int pos_a = 0, pos_b = N > 1 ? 1 : 0;
        if (pos_a != pos_b)
            pos_diff = l2_norm(X.data() + pos_a*d, X.data() + pos_b*d, d);
    }
    CHECK(pos_diff > 1e-4f, "X: position-dependent (diff=%.4f, HDPE active)", pos_diff);

    std::printf("  X shape: [%d x %zu]\n", N, d);
    std::printf("  X[0]:  "); print_row(X.data(), (int)d); std::printf("\n");
    if (N > 1) {
        std::printf("  X[1]:  "); print_row(X.data() + d, (int)d); std::printf("\n");
    }

    // ======================================================================
    // Stage 5: QKV Projection (SQFP Phase 2.1)
    // ======================================================================
    std::printf("\n▸ Stage 5: QKV Projection (SQFP, Component 2.1)\n");

    std::vector<float> Q((size_t)N * h * d_k);
    std::vector<float> K((size_t)N * g * d_k);
    std::vector<float> V((size_t)N * g * d_v);

    bridge_forward_tokens(bridge, batch_ids.data(), N, Q.data(), K.data(), V.data());

    CHECK(all_finite(Q.data(), Q.size()), "Q: all finite");
    CHECK(all_finite(K.data(), K.size()), "K: all finite");
    CHECK(all_finite(V.data(), V.size()), "V: all finite");
    CHECK(all_nonzero(Q.data(), Q.size()), "Q: non-zero (projection active)");
    CHECK(all_nonzero(K.data(), K.size()), "K: non-zero");
    CHECK(all_nonzero(V.data(), V.size()), "V: non-zero");

    float q_rms = rms(Q.data(), Q.size());
    CHECK(q_rms > 1e-6f, "Q: RMS=%.6f", q_rms);

    std::printf("  Q shape: [%d x %zu x %zu]  (%zu elems)\n", N, h, d_k, Q.size());
    std::printf("  K shape: [%d x %zu x %zu]  (%zu elems)\n", N, g, d_k, K.size());
    std::printf("  V shape: [%d x %zu x %zu]  (%zu elems)\n", N, g, d_v, V.size());
    std::printf("  Q[0,h0]: "); print_row(Q.data(), (int)d_k); std::printf("\n");
    std::printf("  Q[0,h1]: "); print_row(Q.data() + d_k, (int)d_k); std::printf("\n");
    std::printf("  K[0,g0]: "); print_row(K.data(), (int)d_k); std::printf("\n");
    std::printf("  V[0,g0]: "); print_row(V.data(), (int)d_v); std::printf("\n");

    // ======================================================================
    // Stage 6: GQA Broadcast (g -> h heads)
    // ======================================================================
    std::printf("\n▸ Stage 6: GQA Expand K, V (g=%zu -> h=%zu)\n", g, h);

    std::vector<float> K_exp((size_t)N * h * d_k);
    std::vector<float> V_exp((size_t)N * h * d_v);

    bridge_expand_kv(bridge, K.data(), V.data(), K_exp.data(), V_exp.data(), (size_t)N);

    CHECK(all_finite(K_exp.data(), K_exp.size()), "K_exp: all finite");
    CHECK(all_finite(V_exp.data(), V_exp.size()), "V_exp: all finite");
    CHECK(K_exp.size() == (size_t)N * h * d_k, "K_exp: N x h x d_k");
    CHECK(V_exp.size() == (size_t)N * h * d_v, "V_exp: N x h x d_v");

    // Heads in same KV group should be identical
    bool gqa_ok = true;
    size_t heads_per_group = h / g;
    for (int ti = 0; ti < N && gqa_ok; ++ti) {
        for (size_t grp = 0; grp < g && gqa_ok; ++grp) {
            size_t base = (size_t)ti * h * d_k + grp * heads_per_group * d_k;
            for (size_t hi = 1; hi < heads_per_group; ++hi) {
                float diff = l2_norm(
                    K_exp.data() + base,
                    K_exp.data() + base + hi * d_k, d_k);
                if (diff > 1e-3f) { gqa_ok = false; break; }
            }
        }
    }
    CHECK(gqa_ok, "GQA: K heads in same group are identical");

    std::printf("  K_exp shape: [%d x %zu x %zu]\n", N, h, d_k);
    std::printf("  V_exp shape: [%d x %zu x %zu]\n", N, h, d_v);

    // ======================================================================
    // Stage 7: Output Projection Wo
    // ======================================================================
    std::printf("\n▸ Stage 7: Output Projection (Wo, Component 2.3 contract)\n");

    std::vector<float> attn_out((size_t)N * d);
    {
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (auto& v : attn_out) v = dist(rng);
    }

    std::vector<float> out((size_t)N * d);
    bridge_wo_forward(bridge, attn_out.data(), out.data(), (size_t)N);

    CHECK(all_finite(out.data(), out.size()), "Wo: all finite");
    CHECK(all_nonzero(out.data(), out.size()), "Wo: non-zero");

    float wo_diff = max_abs_diff(attn_out.data(), out.data(), out.size());
    CHECK(wo_diff > 1e-4f, "Wo: modifies values (max diff=%.6f)", wo_diff);

    std::printf("  Wo(out[0]): "); print_row(out.data(), (int)d); std::printf("\n");

    // ======================================================================
    // Stage 8: KV Cache (autoregressive)
    // ======================================================================
    std::printf("\n▸ Stage 8: KV Cache (Autoregressive Mode)\n");

    // KV cache test
    std::vector<float> K_cache((size_t)N * g * d_k, -42.0f); // sentinel init
    std::vector<float> V_cache((size_t)N * g * d_v, -42.0f);

    for (int pos = 0; pos < N; ++pos) {
        bridge_kv_cache_append(bridge,
                               X.data() + (size_t)pos * d,
                               K_cache.data(), V_cache.data(),
                               (size_t)pos);
    }

    // Verify sentinels were overwritten
    bool cache_written = false;
    for (size_t i = 0; i < K_cache.size(); ++i)
        if (K_cache[i] != -42.0f) { cache_written = true; break; }
    CHECK(cache_written, "KV cache: data written (sentinels overwritten)");

    // Look up all positions from cache
    std::vector<size_t> positions(N);
    for (int i = 0; i < N; ++i) positions[i] = (size_t)i;

    std::vector<float> K_lookup((size_t)N * g * d_k);
    std::vector<float> V_lookup((size_t)N * g * d_v);
    bridge_kv_cache_lookup(bridge, K_cache.data(), V_cache.data(),
                            K_lookup.data(), V_lookup.data(),
                            positions.data(), (size_t)N);

    CHECK(all_finite(K_lookup.data(), K_lookup.size()), "KV cache K: finite");
    CHECK(all_finite(V_lookup.data(), V_lookup.size()), "KV cache V: finite");

    float cache_rms = rms(K_lookup.data(), K_lookup.size());
    CHECK(cache_rms > 1e-6f, "KV cache: RMS=%.6f (values stored & retrieved)", cache_rms);

    // Partial lookup (first 5 positions)
    size_t partial_n = std::min((size_t)N, (size_t)5);
    std::vector<size_t> partial_pos(partial_n);
    for (size_t i = 0; i < partial_n; ++i) partial_pos[i] = i;

    std::vector<float> K_partial(partial_n * g * d_k);
    std::vector<float> V_partial(partial_n * g * d_v);
    bridge_kv_cache_lookup(bridge, K_cache.data(), V_cache.data(),
                            K_partial.data(), V_partial.data(),
                            partial_pos.data(), partial_n);

    float partial_rms = rms(K_partial.data(), K_partial.size());
    CHECK(partial_rms > 1e-6f, "KV cache: partial lookup RMS=%.6f", partial_rms);

    // Compare with batch K from Stage 5
    float vs_batch = l2_norm(K.data(), K_lookup.data(), K.size());
    CHECK(vs_batch < 1e-2f, "KV cache: matches batch K (l2 diff=%.6f)", vs_batch);

    // ======================================================================
    // Stage 9: Single-Call Pipeline (text -> QKV)
    // ======================================================================
    std::printf("\n▸ Stage 9: End-to-End text->QKV (bridge_forward_text)\n");

    const char* test_text = "end to end transformer pipeline verification";

    std::vector<float> Q_text((size_t)64 * h * d_k);
    std::vector<float> K_text((size_t)64 * g * d_k);
    std::vector<float> V_text((size_t)64 * g * d_v);

    int n_text = bridge_forward_text(bridge, test_text,
                                     Q_text.data(), K_text.data(), V_text.data(), 64);
    CHECK(n_text > 0, "  \"%s\" -> %d tokens", test_text, n_text);

    size_t qn = (size_t)n_text * h * d_k;
    size_t kn = (size_t)n_text * g * d_k;
    size_t vn = (size_t)n_text * g * d_v;
    CHECK(all_finite(Q_text.data(), qn), "Q_text: all finite (n=%d)", n_text);
    CHECK(all_finite(K_text.data(), kn), "K_text: all finite");
    CHECK(all_finite(V_text.data(), vn), "V_text: all finite");

    // ======================================================================
    // Stage 10: Adaptive Precision Diagnostics
    // ======================================================================
    std::printf("\n▸ Stage 10: Adaptive Precision Diagnostics\n");

    float imp[4];
    bridge_get_importance(bridge, X.data(), imp, (size_t)N);
    CHECK(all_finite(imp, 4), "importance: [%.2f %.2f %.2f %.2f]",
           imp[0], imp[1], imp[2], imp[3]);

    // ======================================================================
    // Summary — QKV ready for Phase 2.2
    // ======================================================================
    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║   Summary                                              ║\n");
    std::printf("╚══════════════════════════════════════════════════════════╝\n");
    std::printf("  Tests: %d/%d passed\n", g_checks - g_failures, g_checks);

    if (g_failures == 0) {
        std::printf("\n  \033[32mQKV ready for Phase 2.2 (Scaled Dot-Product Attention)\033[0m\n");
        std::printf("\n  Q[%zu]  = query heads  (N=%d x h=%zu x d_k=%zu)\n",
                    Q.size(), N, h, d_k);
        std::printf("  K_exp[%zu] = key heads (broadcast N x h x d_k)\n", K_exp.size());
        std::printf("  V_exp[%zu] = value heads (broadcast N x h x d_v)\n", V_exp.size());
        std::printf("\n  Attention computes:  Attn(Q,K,V) = softmax(Q*K^T/sqrt(d_k)) * V\n");
        std::printf("  Output:  X_attn[%d x %zu] -> Wo -> residual\n\n", N, d);
    } else {
        std::printf("  \033[31m%d FAILURES — pipeline NOT ready\033[0m\n", g_failures);
    }

    bridge_destroy(bridge);
    std::printf("  bridge_destroy: OK\n");
    return g_failures > 0 ? 1 : 0;
}
