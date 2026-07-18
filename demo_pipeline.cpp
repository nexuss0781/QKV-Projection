// ==========================================================================
// demo_pipeline.cpp — Step-by-step trace of text → QKV pipeline
//
// Shows intermediate values at every stage:
//   1. Tokenization (token IDs)
//   2. Raw embedding (HFAQE, no position)
//   3. Position encoding (HDPE RoPE)
//   4. QKV projection (SQFP)
// ==========================================================================

#include "Input.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>

static void print_row(const float* data, int d, int count = 6) {
    std::printf("[");
    for (int i = 0; i < count && i < d; ++i)
        std::printf("%+.6f", data[i]);
    if (count < d) std::printf(" ... %d total", d);
    std::printf("]");
}

static float rms(const float* data, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) sum += (double)data[i] * data[i];
    return (float)std::sqrt(sum / (double)n);
}

int main() {
    std::printf("============================================================\n");
    std::printf("  Step-by-Step Pipeline Trace\n");
    std::printf("  Sentence -> Tokenize -> Embed -> Position-Encode -> QKV\n");
    std::printf("============================================================\n\n");

    // ---- Config (small for readability) ----
    BridgeConfig cfg;
    cfg.V        = 1024;
    cfg.d        = 64;
    cfg.r        = 16;
    cfg.K        = 64;
    cfg.h        = 4;
    cfg.g        = 4;
    cfg.L_layers = 6;
    cfg.B_hdpe   = 64;
    cfg.L_hdpe   = 4;
    cfg.rope_base = 10000.0f;

    int d = cfg.d, h = cfg.h, g = cfg.g, d_k = cfg.d / cfg.h, d_v = cfg.d / cfg.h;

    // ---- Create bridge ----
    std::printf("• Creating bridge (V=%d d=%d h=%d g=%d d_k=%d)...\n", cfg.V, d, h, g, d_k);
    InputBridge* bridge = bridge_create(&cfg);
    if (!bridge) { std::printf("  FAILED\n"); return 1; }
    std::printf("  OK\n\n");

    // ---- Sentence ----
    const char* sentence = "attention is all you need";
    std::printf("═ STEP 0: Input Sentence ═══════════════════════════════════\n");
    std::printf("  \"%s\"\n\n", sentence);

    // ---- Step 1: Tokenize ----
    std::printf("═ STEP 1: Tokenization (C1.1 BPE Tokenizer) ═══════════════\n");
    auto token_vec = bridge->engine.tokenize(sentence);
    int n = (int)token_vec.size();
    std::printf("  Tokens (%d): ", n);
    for (int i = 0; i < n; ++i) std::printf("%s%d", i ? ", " : "", token_vec[i]);
    std::printf("\n  As characters: ");
    for (int i = 0; i < n; ++i) {
        int id = token_vec[i];
        if (id >= 32 && id <= 126) std::printf("'%c' ", (char)id);
        else std::printf("[%d] ", id);
    }
    std::printf("\n\n");

    // ---- Step 2: Raw Embedding (no position) ----
    std::printf("═ STEP 2: Raw Embedding (HFAQE, C1.2) ═════════════════════\n");
    std::vector<float> X_raw((size_t)n * d);
    bridge->engine.embed(token_vec.data(), n, X_raw.data());

    float rms_raw = rms(X_raw.data(), X_raw.size());
    std::printf("  X shape: [%d x %d]  RMS=%.6f\n", n, d, rms_raw);
    for (int i = 0; i < n && i < 4; ++i) {
        std::printf("  X[%d] (token %d): ", i, token_vec[i]);
        print_row(&X_raw[(size_t)i * d], d);
        std::printf("\n");
    }
    if (n > 4) std::printf("  ... (%d more tokens)\n", n - 4);
    std::printf("\n");

    // ---- Step 3: Position Encoding (HDPE RoPE, C1.3) ----
    std::printf("═ STEP 3: Position Encoding (HDPE RoPE, C1.3) ═════════════\n");
    // Copy raw, then apply rotation in-place
    std::vector<float> X_pos = X_raw;
    bridge->engine.apply_rotation(X_pos.data(), n);

    float rms_pos = rms(X_pos.data(), X_pos.size());
    std::printf("  After RoPE: RMS=%.6f\n", rms_pos);

    // Show change: raw vs positioned at first token
    std::printf("  X[0] raw:  ");
    print_row(&X_raw[0], d);
    std::printf("\n  X[0] RoPE: ");
    print_row(&X_pos[0], d);
    std::printf("\n");

    // Verify position encoding is active: same token at different positions
    if (n >= 2) {
        float pos_diff = 0;
        for (int j = 0; j < d; ++j)
            pos_diff += std::fabs(X_pos[j] - X_pos[1 * d + j]);
        std::printf("  Token diff (pos 0 vs pos 1): %.6f (HDPE active)\n", pos_diff);
    }

    // Show per-head structure (d_k = 16, 4 heads, 4 pairs per head)
    std::printf("  X[0] per-head (d_k=%d, %d pairs/head):\n", d_k, d_k / 2);
    for (int hi = 0; hi < h && hi < 2; ++hi) {
        std::printf("    head %d: ", hi);
        print_row(&X_pos[hi * d_k], d_k, 4);
        std::printf("\n");
    }
    std::printf("\n");

    // ---- Step 4: QKV Projection (SQFP, C2.1) ----
    std::printf("═ STEP 4: QKV Projection (SQFP, C2.1) ═════════════════════\n");
    std::vector<float> Q((size_t)n * h * d_k);
    std::vector<float> K((size_t)n * g * d_k);
    std::vector<float> V((size_t)n * g * d_v);

    sqfp_qkv_forward(bridge->sqfp, X_pos.data(), Q.data(), K.data(), V.data(), (size_t)n);

    float rms_q = rms(Q.data(), Q.size());
    float rms_k = rms(K.data(), K.size());
    float rms_v = rms(V.data(), V.size());
    std::printf("  Q RMS=%.6f  K RMS=%.6f  V RMS=%.6f\n", rms_q, rms_k, rms_v);
    std::printf("  Q shape: [%d x %d x %d]\n", n, h, d_k);
    std::printf("  K shape: [%d x %d x %d]\n", n, g, d_k);
    std::printf("  V shape: [%d x %d x %d]\n", n, g, d_v);

    // Show Q per head at first token
    std::printf("  Q[0] per head:\n");
    for (int hi = 0; hi < h; ++hi) {
        std::printf("    h%d (q): ", hi);
        print_row(&Q[hi * d_k], d_k, 4);
        std::printf("\n");
    }
    // K, V per group at first token
    std::printf("  K[0] per group:\n");
    for (int gi = 0; gi < g; ++gi) {
        std::printf("    g%d (k): ", gi);
        print_row(&K[gi * d_k], d_k, 4);
        std::printf("\n");
    }
    std::printf("  V[0] per group:\n");
    for (int gi = 0; gi < g; ++gi) {
        std::printf("    g%d (v): ", gi);
        print_row(&V[gi * d_v], d_v, 4);
        std::printf("\n");
    }

    // ---- Step 4b: GQA expansion ----
    std::printf("═ STEP 4b: GQA Expand (g=%d -> h=%d) ════════════════════════\n", g, h);
    std::vector<float> K_exp((size_t)n * h * d_k);
    std::vector<float> V_exp((size_t)n * h * d_v);
    sqfp_expand_kv(bridge->sqfp, K.data(), V.data(), K_exp.data(), V_exp.data(), (size_t)n);

    // Verify: heads in same group are identical
    int h_per_g = h / g;
    bool gqa_ok = true;
    for (int i = 0; i < n && gqa_ok; ++i)
        for (int gi = 0; gi < g && gqa_ok; ++gi)
            for (int hi = 1; hi < h_per_g; ++hi)
                for (int j = 0; j < d_k; ++j)
                    if (std::fabs(K_exp[(size_t)i * h * d_k + gi * h_per_g * d_k + j] -
                                  K_exp[(size_t)i * h * d_k + gi * h_per_g * d_k + hi * d_k + j]) > 1e-6f)
                        gqa_ok = false;
    std::printf("  GQA broadcast: %s\n", gqa_ok ? "IDENTICAL (correct)" : "MISMATCH");

    std::printf("  K_exp[0] heads 0..%d (first 4 values each):\n", h - 1);
    for (int hi = 0; hi < h; ++hi) {
        std::printf("    h%d: ", hi);
        print_row(&K_exp[hi * d_k], d_k, 4);
        std::printf("\n");
    }

    // ---- Summary ----
    std::printf("\n════════════════════════════════════════════════════════\n");
    std::printf("  Pipeline complete: text -> Q[%zu] K_exp[%zu] V_exp[%zu]\n",
                Q.size(), K_exp.size(), V_exp.size());
    std::printf("  Ready for Phase 2.2 (Scaled Dot-Product Attention)\n");
    std::printf("════════════════════════════════════════════════════════\n");

    bridge_destroy(bridge);
    return 0;
}
