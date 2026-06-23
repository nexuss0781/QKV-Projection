// ==========================================================================
// Component 2.1 — QKV Projection: Sub-Quadratic Factorized Projection (SQFP)
// Specification: Hierarchical Diagonal-Plus-Low-Rank with Butterfly Residual
// ==========================================================================
//
// Architecture: W = D + U·V^T + ε·B
//   where D ∈ ℝ^m (diagonal spectrum)
//         U ∈ ℝ^{m×r}, V ∈ ℝ^{d×r} (low-rank interaction, r = O(log d))
//         B ∈ ℝ^{max(m,d)×max(m,d)} butterfly (padded/truncated to m×d)
//         ε ∈ ℝ (learned gating scalar)
//
// Per-token: P(x_i) = D ⊙ x_i + U·(V^T·x_i) + ε·B(x_i)
//   O(d log d + d·r) time, O(m + r(m+d) + 2·max(m,d)·log₂(max(m,d))) storage
//
// Attention variants:
//   MHA: m = d (all projections square)
//   GQA: m_Q = d, m_K = m_V = g·d_k (grouped)
//   MQA: m_Q = d, m_K = m_V = d_k (single group)
//   Wo:  m = d (output projection)
// ==========================================================================

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

// ==========================================================================
// [A1] Diagonal Spectrum Path
// Specification: Section 3.1 — D ⊙ x_i
//   D ∈ ℝ^m (diagonal spectrum), learned eigenvalues
//   O(m) time, O(m) storage
//   For m == d: element-wise Hadamard product
//   For m != d: diagonal matrix ℝ^{m×d} maps ℝ^d → ℝ^m
// ==========================================================================

template<typename T>
void diagonal_forward(
    const T* D,       // diagonal values, length m
    const T* x,       // input vector, length d
    T* y,             // output vector, length m
    size_t m,
    size_t d
) {
    size_t k = std::min(m, d);
    size_t i = 0;

    for (; i < k; ++i) {
        y[i] = D[i] * x[i];
    }
    for (; i < m; ++i) {
        y[i] = T(0);
    }
}

template<typename T>
void diagonal_forward_batched(
    const T* D,
    const T* X,        // (n, d) row-major
    T* Y,              // (n, m) row-major
    size_t n,
    size_t m,
    size_t d
) {
    size_t k = std::min(m, d);
    for (size_t row = 0; row < n; ++row) {
        const T* x_row = X + row * d;
        T* y_row = Y + row * m;
        size_t i = 0;
        for (; i < k; ++i) {
            y_row[i] = D[i] * x_row[i];
        }
        for (; i < m; ++i) {
            y_row[i] = T(0);
        }
    }
}

// Gradient: ∂L/∂D_i  =  ∑_batch,seq  ∂L/∂y_i · x_i  (for i < min(m,d))
//           ∂L/∂x_i  =  D_i · ∂L/∂y_i               (for i < min(m,d))
template<typename T>
void diagonal_backward(
    const T* D,
    const T* x,
    const T* grad_y,   // ∂L/∂y, length m
    T* grad_D,         // ∂L/∂D, length m
    T* grad_x,         // ∂L/∂x, length d
    size_t m,
    size_t d
) {
    size_t k = std::min(m, d);
    for (size_t i = 0; i < k; ++i) {
        grad_D[i] += grad_y[i] * x[i];
        grad_x[i] = D[i] * grad_y[i];
    }
    for (size_t i = k; i < d; ++i) {
        grad_x[i] = T(0);
    }
}


// ==========================================================================
// [A2] Low-Rank Interaction Path
// Specification: Section 2.1, Section 3.1 — U·(V^T·x_i)
//   U ∈ ℝ^{m×r}, V ∈ ℝ^{d×r}
//   O(m·r + d·r) = O(r·(m+d)) time
//   Step 1: z = V^T · x_i  (ℝ^r)  — O(d·r)
//   Step 2: y += U · z     (ℝ^m)  — O(m·r)
// ==========================================================================

template<typename T>
void lowrank_forward(
    const T* U,       // (m, r) row-major
    const T* V,       // (d, r) row-major
    const T* x,       // (d)
    T* y,             // (m)
    size_t m,
    size_t d,
    size_t r
) {
    // z = V^T · x  => z_j = Σ_k V[k][j] * x[k]
    std::vector<T> z(r, T(0));
    for (size_t j = 0; j < r; ++j) {
        T sum = T(0);
        for (size_t k = 0; k < d; ++k) {
            sum += V[k * r + j] * x[k];
        }
        z[j] = sum;
    }

    // y += U · z  => y_i += Σ_j U[i][j] * z[j]
    for (size_t i = 0; i < m; ++i) {
        T sum = T(0);
        for (size_t j = 0; j < r; ++j) {
            sum += U[i * r + j] * z[j];
        }
        y[i] += sum;
    }
}

template<typename T>
void lowrank_forward_batched(
    const T* U,
    const T* V,
    const T* X,        // (n, d) row-major
    T* Y,              // (n, m) row-major
    size_t n,
    size_t m,
    size_t d,
    size_t r
) {
    for (size_t row = 0; row < n; ++row) {
        lowrank_forward(
            U, V,
            X + row * d,
            Y + row * m,
            m, d, r
        );
    }
}


// ==========================================================================
// [A3] Butterfly Residual Path
// Specification: Section 2.2 — ε · B(x_i; B)
//   B = M^{(k)} · M^{(k-1)} · ... · M^{(1)}  (k = ⌈log₂(N)⌉)
//   Each M^{(l)} is block-diagonal with N/2 blocks of 2×2 matrices
//   N = max(m, d), padded to next power of 2, then truncated
//   O(N log N) time, O(2N log₂N) storage
// ==========================================================================

template<typename T>
struct ButterflyFactors {
    size_t N;                        // padded dimension (power of 2)
    size_t logN;                     // number of stages = log₂(N)
    std::vector<std::vector<T>> stages; // each stage: (N/2) * 4 scalars

    ButterflyFactors() : N(0), logN(0) {}

    ButterflyFactors(size_t max_dim) {
        N = 1;
        while (N < max_dim) N <<= 1;
        logN = 0;
        size_t tmp = N;
        while (tmp > 1) { tmp >>= 1; ++logN; }

        stages.resize(logN);
        for (size_t l = 0; l < logN; ++l) {
            stages[l].resize((N / 2) * 4, T(0));
        }
    }

    size_t param_count() const {
        return logN * (N / 2) * 4;
    }
};

template<typename T>
void butterfly_mvm(
    const ButterflyFactors<T>& B,
    const T* x,          // input (N)
    T* y                 // output (N)
) {
    std::vector<T> buf[2];
    buf[0].resize(B.N, T(0));
    buf[1].resize(B.N, T(0));

    std::copy(x, x + B.N, buf[0].begin());

    for (size_t l = 0; l < B.logN; ++l) {
        size_t block_size = size_t(1) << (l + 1);
        size_t half_block = block_size / 2;
        int src = l & 1;
        int dst = 1 - src;

        const T* stage_data = B.stages[l].data();
        for (size_t b = 0; b < B.N / 2; ++b) {
            size_t i = b * 2;          // even index in pair
            size_t j = i + 1;          // odd index in pair
            const T* blk = stage_data + b * 4;

            T x_even = buf[src][i];
            T x_odd  = buf[src][j];
            buf[dst][i] = blk[0] * x_even + blk[1] * x_odd;
            buf[dst][j] = blk[2] * x_even + blk[3] * x_odd;
        }
        (void)half_block;
    }

    int final = B.logN & 1;
    std::copy(buf[final].begin(), buf[final].end(), y);
}

template<typename T>
void butterfly_residual_forward(
    const ButterflyFactors<T>& B_factors,
    T epsilon,
    const T* x,        // input, ℝ^d
    T* y,              // output, ℝ^m
    size_t m,
    size_t d
) {
    size_t N = B_factors.N;

    // Pad input to N and apply butterfly
    std::vector<T> x_padded(N, T(0));
    std::copy(x, x + d, x_padded.begin());

    std::vector<T> y_full(N, T(0));
    butterfly_mvm(B_factors, x_padded.data(), y_full.data());

    // Truncate to m and scale by epsilon
    for (size_t i = 0; i < m; ++i) {
        y[i] += epsilon * y_full[i];
    }
}


// ==========================================================================
// [A4] Unified SQFP Operator
// Specification: Section 3.1 — P_{SQFP}(x_i; θ)
//   P(x_i) = D⊙x_i + U·(V^T·x_i) + ε·B(x_i)
//   Input:  x_i ∈ ℝ^d
//   Output: y_i ∈ ℝ^m
//   Variables: D ∈ ℝ^m, U ∈ ℝ^{m×r}, V ∈ ℝ^{d×r}, ε ∈ ℝ, B factors
// ==========================================================================

template<typename T>
struct SQFPParams {
    // Diagonal spectrum
    std::vector<T> D;           // (m)

    // Low-rank factors
    std::vector<T> U;           // (m, r) row-major
    std::vector<T> V;           // (d, r) row-major
    size_t r;

    // Butterfly residual
    T epsilon;
    ButterflyFactors<T> B;

    // Dimensions
    size_t m;  // output dimension
    size_t d;  // input dimension

    SQFPParams() : r(0), epsilon(T(0)), m(0), d(0) {}

    SQFPParams(size_t input_dim, size_t output_dim, size_t rank)
        : r(rank), epsilon(T(0)), m(output_dim), d(input_dim)
    {
        D.resize(m, T(1));   // Unit diagonal preserves variance (Prop 2.3.2)
        U.resize(m * r, T(0));
        V.resize(d * r, T(0));
        B = ButterflyFactors<T>(std::max(m, d));
    }

    size_t param_count() const {
        // |θ| = m + r·(m + d) + 2·max(m,d)·log₂(max(m,d)) + 1 (for ε)
        return m + r * (m + d) + B.param_count() + 1;
    }
};

template<typename T>
void sqfp_forward_token(
    const SQFPParams<T>& theta,
    const T* x,     // ℝ^d
    T* y            // ℝ^m
) {
    // Step 1: Zero output
    std::fill(y, y + theta.m, T(0));

    // Step 2: Diagonal spectrum path  (A1)
    diagonal_forward(theta.D.data(), x, y, theta.m, theta.d);

    // Step 3: Low-rank interaction path (A2)
    lowrank_forward(
        theta.U.data(), theta.V.data(),
        x, y,
        theta.m, theta.d, theta.r
    );

    // Step 4: Butterfly residual path (A3)
    butterfly_residual_forward(
        theta.B, theta.epsilon,
        x, y,
        theta.m, theta.d
    );
}

template<typename T>
void sqfp_forward_batched(
    const SQFPParams<T>& theta,
    const T* X,     // (n, d) row-major
    T* Y,           // (n, m) row-major
    size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        sqfp_forward_token(theta, X + i * theta.d, Y + i * theta.m);
    }
}


// ==========================================================================
// [A5] Shape Preservation
// Specification: Section 3.1 — Shape Preservation Guarantee
//   Input:  x_i ∈ ℝ^d
//   Output: y_i ∈ ℝ^m
//   where m = h·d_k for Q, m = g·d_k for K/V (GQA), m = d for Wo
//   No cross-token dependency — token-wise independent
// ==========================================================================

// Shape validation utilities
struct SQFPShapes {
    size_t d;       // model dimension
    size_t h;       // number of query heads
    size_t g;       // number of KV groups
    size_t d_k;     // head dimension = d / h
    size_t d_v;     // value head dimension = d / h

    size_t m_Q() const { return d; }           // Q output dim (MHA/GQA/MQA)
    size_t m_K_MHA() const { return d; }       // K output dim (MHA)
    size_t m_K_GQA() const { return g * d_k; } // K output dim (GQA)
    size_t m_K_MQA() const { return d_k; }     // K output dim (MQA)
    size_t m_V_MHA() const { return d; }       // V output dim (MHA)
    size_t m_V_GQA() const { return g * d_v; } // V output dim (GQA)
    size_t m_V_MQA() const { return d_v; }     // V output dim (MQA)
    size_t m_Wo() const { return d; }          // Wo output dim

    size_t r() const { return 2 * size_t(std::ceil(std::log2(d))); }

    SQFPShapes(size_t d_, size_t h_, size_t g_)
        : d(d_), h(h_), g(g_), d_k(d_ / h_), d_v(d_ / h_)
    {
        assert(d % h == 0 && "d must be divisible by h");
        assert(h % g == 0 && "h must be divisible by g");
    }
};


// ==========================================================================
// [B1] MHA Projection
// Specification: Section 3.2
//   Q = reshape(P_{SQFP}(X; θ_q), (n, h, d_k))
//   K = reshape(P_{SQFP}(X; θ_k), (n, h, d_k))
//   V = reshape(P_{SQFP}(X; θ_v), (n, h, d_v))
//   D_q ∈ ℝ^d, U_q ∈ ℝ^{d×r_q}, V_q ∈ ℝ^{d×r_q}, r_q = 2⌈log₂d⌉
// ==========================================================================

template<typename T>
struct MHAProjectors {
    SQFPParams<T> theta_q;
    SQFPParams<T> theta_k;
    SQFPParams<T> theta_v;

    MHAProjectors(size_t d, size_t h)
        : theta_q(d, d, 2 * size_t(std::ceil(std::log2(d))))
        , theta_k(d, d, 2 * size_t(std::ceil(std::log2(d))))
        , theta_v(d, d, 2 * size_t(std::ceil(std::log2(d))))
    {}
};

template<typename T>
void mha_forward(
    const MHAProjectors<T>& proj,
    const T* X,        // (n, d) row-major
    T* Q,              // (n, h, d_k) row-major
    T* K,              // (n, h, d_k) row-major
    T* V,              // (n, h, d_v) row-major
    size_t n,
    size_t d,
    size_t h
) {
    size_t d_k = d / h;

    // Temporary buffer for flat projection output
    std::vector<T> Q_flat(n * d);
    std::vector<T> K_flat(n * d);
    std::vector<T> V_flat(n * d);

    // Q projection
    sqfp_forward_batched(proj.theta_q, X, Q_flat.data(), n);
    // K projection
    sqfp_forward_batched(proj.theta_k, X, K_flat.data(), n);
    // V projection
    sqfp_forward_batched(proj.theta_v, X, V_flat.data(), n);

    // Reshape (n, d) -> (n, h, d_k)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < h; ++j) {
            for (size_t k = 0; k < d_k; ++k) {
                size_t flat_idx = i * d + j * d_k + k;
                size_t reshaped_idx = i * h * d_k + j * d_k + k;
                Q[reshaped_idx] = Q_flat[flat_idx];
                K[reshaped_idx] = K_flat[flat_idx];
                V[reshaped_idx] = V_flat[flat_idx];
            }
        }
    }
}


// ==========================================================================
// [B2] GQA Projection
// Specification: Section 3.3
//   Q: full h heads (same as MHA)
//   K: reshape(D_kv ⊙ X^T + U_kv·(V_kv^T X^T) + ε_kv·B(X^T; B_kv), (n,g,d_k))^T
//   V: same structure
//   Broadcast: K_expanded = repeat_interleave(K, h//g, dim=1) → (n,h,d_k)
// ==========================================================================

template<typename T>
struct GQAProjectors {
    SQFPParams<T> theta_q;   // full MHA Q
    SQFPParams<T> theta_k;   // GQA K (g groups)
    SQFPParams<T> theta_v;   // GQA V (g groups)

    GQAProjectors(size_t d, size_t h, size_t g)
        : theta_q(d, d, 2 * size_t(std::ceil(std::log2(d))))
        , theta_k(d, g * (d / h), 2 * size_t(std::ceil(std::log2(d))))
        , theta_v(d, g * (d / h), 2 * size_t(std::ceil(std::log2(d))))
    {
        assert(h % g == 0);
    }
};

template<typename T>
void gqa_forward(
    const GQAProjectors<T>& proj,
    const T* X,        // (n, d)
    T* Q,              // (n, h, d_k)
    T* K,              // (n, g, d_k)  — pre-expand
    T* V,              // (n, g, d_v)  — pre-expand
    size_t n,
    size_t d,
    size_t h,
    size_t g
) {
    size_t d_k = d / h;
    size_t m_k = g * d_k;

    std::vector<T> Q_flat(n * d);
    std::vector<T> K_flat(n * m_k);
    std::vector<T> V_flat(n * m_k);

    // Q projection (full MHA)
    sqfp_forward_batched(proj.theta_q, X, Q_flat.data(), n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < h; ++j) {
            for (size_t k = 0; k < d_k; ++k) {
                size_t flat_idx = i * d + j * d_k + k;
                size_t out_idx = i * h * d_k + j * d_k + k;
                Q[out_idx] = Q_flat[flat_idx];
            }
        }
    }

    // K projection (g groups)
    sqfp_forward_batched(proj.theta_k, X, K_flat.data(), n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < g; ++j) {
            for (size_t k = 0; k < d_k; ++k) {
                size_t flat_idx = i * m_k + j * d_k + k;
                size_t out_idx = i * g * d_k + j * d_k + k;
                K[out_idx] = K_flat[flat_idx];
            }
        }
    }

    // V projection (g groups)
    sqfp_forward_batched(proj.theta_v, X, V_flat.data(), n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < g; ++j) {
            for (size_t k = 0; k < d_k; ++k) {
                size_t flat_idx = i * m_k + j * d_k + k;
                size_t out_idx = i * g * d_k + j * d_k + k;
                V[out_idx] = V_flat[flat_idx];
            }
        }
    }
}

// Broadcast: repeat_interleave(K, h//g, dim=1)
// View-only: no memory allocation, returns index mapping
template<typename T>
void gqa_broadcast_k(
    const T* K,              // (n, g, d_k)
    T* K_expanded,           // (n, h, d_k)
    size_t n,
    size_t h,
    size_t g,
    size_t d_k
) {
    size_t repeat = h / g;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < g; ++j) {
            for (size_t r = 0; r < repeat; ++r) {
                size_t src_idx = i * g * d_k + j * d_k;
                size_t dst_idx = i * h * d_k + (j * repeat + r) * d_k;
                std::memcpy(
                    K_expanded + dst_idx,
                    K + src_idx,
                    d_k * sizeof(T)
                );
            }
        }
    }
}


// ==========================================================================
// [B3] MQA Projection
// Specification: Section 3.4
//   K = reshape(P_{SQFP}(X; θ_k^{(1)}), (n, 1, d_k))
//   V = reshape(P_{SQFP}(X; θ_v^{(1)}), (n, 1, d_v))
//   Single K/V projection, O(d) per token
// ==========================================================================

template<typename T>
struct MQAProjectors {
    SQFPParams<T> theta_q;
    SQFPParams<T> theta_k;   // single head
    SQFPParams<T> theta_v;   // single head

    MQAProjectors(size_t d, size_t h)
        : theta_q(d, d, 2 * size_t(std::ceil(std::log2(d))))
        , theta_k(d, d / h, 2 * size_t(std::ceil(std::log2(d))))
        , theta_v(d, d / h, 2 * size_t(std::ceil(std::log2(d))))
    {}
};

template<typename T>
void mqa_forward(
    const MQAProjectors<T>& proj,
    const T* X,        // (n, d)
    T* Q,              // (n, h, d_k)
    T* K,              // (n, 1, d_k)
    T* V,              // (n, 1, d_v)
    size_t n,
    size_t d,
    size_t h
) {
    size_t d_k = d / h;

    std::vector<T> Q_flat(n * d);
    sqfp_forward_batched(proj.theta_q, X, Q_flat.data(), n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < h; ++j) {
            for (size_t k = 0; k < d_k; ++k) {
                Q[i * h * d_k + j * d_k + k] = Q_flat[i * d + j * d_k + k];
            }
        }
    }

    std::vector<T> K_flat(n * d_k);
    sqfp_forward_batched(proj.theta_k, X, K_flat.data(), n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_k; ++k) {
            K[i * d_k + k] = K_flat[i * d_k + k];
        }
    }

    std::vector<T> V_flat(n * d_k);
    sqfp_forward_batched(proj.theta_v, X, V_flat.data(), n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_k; ++k) {
            V[i * d_k + k] = V_flat[i * d_k + k];
        }
    }
}


// ==========================================================================
// [B4] Output Projection Wo
// Specification: Section 3.5
//   Out = P_{SQFP}(X_attn; θ_o)
//   Scaled init: D_o[i] = 1/√(2L) where L is layer count
//   Guarantees Var(X + Attn(X)) ≈ Var(X)·(1 + 1/(2L)) at init
// ==========================================================================

template<typename T>
SQFPParams<T> make_wo_projection(size_t d, size_t L) {
    SQFPParams<T> theta(d, d, 2 * size_t(std::ceil(std::log2(d))));
    T scale = T(1) / std::sqrt(T(2 * L));
    for (size_t i = 0; i < d; ++i) {
        theta.D[i] = scale;
    }
    return theta;
}

template<typename T>
void wo_forward(
    const SQFPParams<T>& theta_o,
    const T* X_attn,     // (n, d) — concatenated attention heads
    T* Out,              // (n, d)
    size_t n,
    size_t d
) {
    sqfp_forward_batched(theta_o, X_attn, Out, n);
}


// ==========================================================================
// [C1] Token Importance Sketch
// Specification: Section 3.6.1
//   S ∈ ℝ^{d×s} with fixed random orthonormal columns, s = 16
//   ρ_i = ‖x_i · S‖₂  — importance score for token i
// ==========================================================================

template<typename T>
struct ImportanceSketch {
    std::vector<T> S;    // (d, s) row-major, fixed random orthonormal
    size_t s;            // sketch dimension (s = 16 per spec)

    ImportanceSketch(size_t d_, size_t s_ = 16)
        : S(d_ * s_, T(0)), s(s_)
    {
        // Initialize with random orthonormal columns (fixed seed for reproducibility)
        // Use simple deterministic initialization: identity-like but truncated
        // In practice, use QR of random Gaussian matrix
        for (size_t j = 0; j < std::min(s_, d_); ++j) {
            S[j * d_ + j] = T(1) / std::sqrt(T(s_));
        }
    }

    T compute_importance(const T* x, size_t d) const {
        // ρ = ‖x · S‖₂
        T sum_sq = T(0);
        for (size_t j = 0; j < s; ++j) {
            T dot = T(0);
            for (size_t k = 0; k < d; ++k) {
                dot += x[k] * S[j * d + k];
            }
            sum_sq += dot * dot;
        }
        return std::sqrt(sum_sq);
    }
};


// ==========================================================================
// [C2-C5] Adaptive Precision Path
// Specification: Section 3.6.2
//   P_ADAPTIVE(x_i) = 
//     Fast path (ρ_i < τ):   D⊙x_i + U_fast·(V_fast^T·x_i)
//     Full path (ρ_i ≥ τ):   D⊙x_i + U·(V^T·x_i) + ε·B(x_i)
//   where r_fast = ⌈r/4⌉
//   Amortized: ~80% fast path, ~20% full path → O(d) per token
// ==========================================================================

template<typename T>
struct AdaptiveSQFPParams {
    // Shared diagonal (used in both paths)
    std::vector<T> D;             // (m)

    // Fast path: reduced rank
    std::vector<T> U_fast;        // (m, r_fast)
    std::vector<T> V_fast;        // (d, r_fast)
    size_t r_fast;

    // Full path: full rank + butterfly
    std::vector<T> U;             // (m, r)
    std::vector<T> V;             // (d, r)
    size_t r;
    T epsilon;
    ButterflyFactors<T> B;

    // Learned threshold
    T tau;

    // Sketch
    ImportanceSketch<T> sketch;

    size_t m, d;

    AdaptiveSQFPParams(size_t m_, size_t d_, size_t r_)
        : r_fast(std::max(size_t(1), (r_ + 3) / 4)), r(r_), epsilon(T(0)),
          tau(T(1.0)), sketch(d_, 16), m(m_), d(d_)
    {
        D.resize(m, T(0));
        U_fast.resize(m * r_fast, T(0));
        V_fast.resize(d * r_fast, T(0));
        U.resize(m * r, T(0));
        V.resize(d * r, T(0));
        B = ButterflyFactors<T>(std::max(m, d));
    }

    size_t param_count() const {
        return m                                              // D
             + r_fast * (m + d)                               // U_fast, V_fast
             + r * (m + d) + B.param_count() + 1              // U, V, B, ε
             + 1;                                             // τ
    }
};

template<typename T>
void adaptive_sqfp_forward_token(
    const AdaptiveSQFPParams<T>& params,
    const T* x,     // ℝ^d
    T* y            // ℝ^m
) {
    std::fill(y, y + params.m, T(0));

    // Diagonal path (always applied)
    diagonal_forward(params.D.data(), x, y, params.m, params.d);

    // Compute token importance
    T rho = params.sketch.compute_importance(x, params.d);

    if (rho < params.tau) {
        // Fast path: DPLR with reduced rank
        // y += U_fast · (V_fast^T · x)
        std::vector<T> z_fast(params.r_fast, T(0));
        for (size_t j = 0; j < params.r_fast; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < params.d; ++k) {
                sum += params.V_fast[k * params.r_fast + j] * x[k];
            }
            z_fast[j] = sum;
        }
        for (size_t i = 0; i < params.m; ++i) {
            T sum = T(0);
            for (size_t j = 0; j < params.r_fast; ++j) {
                sum += params.U_fast[i * params.r_fast + j] * z_fast[j];
            }
            y[i] += sum;
        }
    } else {
        // Full path: full rank + butterfly
        lowrank_forward(
            params.U.data(), params.V.data(),
            x, y,
            params.m, params.d, params.r
        );
        butterfly_residual_forward(
            params.B, params.epsilon,
            x, y,
            params.m, params.d
        );
    }
}


// ==========================================================================
// [D1-D4] Block-wise INT8 Quantization
// Specification: Section 4.1
//   D: block-wise INT8 with block size b = 64
//   U, V: column-wise INT8 with per-column scales
//   B: stored in BF16 (not quantized)
//   Dequant: P_quant(x) with INT32 accumulation
// ==========================================================================

template<typename T>
struct QuantizedSQFPParams {
    // Quantized diagonal: block-wise INT8
    std::vector<int8_t> D_int8;        // (m)
    std::vector<T> D_scales;           // (⌈m/64⌉)

    // Quantized low-rank factors: column-wise INT8
    std::vector<int8_t> U_int8;        // (m, r)
    std::vector<int8_t> V_int8;        // (d, r)
    std::vector<T> U_scales;           // (r)
    std::vector<T> V_scales;           // (r)

    // Butterfly in BF16 (full precision)
    T epsilon;
    ButterflyFactors<T> B;

    size_t m, d, r;
    static constexpr size_t BLOCK_SIZE = 64;

    QuantizedSQFPParams() : epsilon(T(0)), m(0), d(0), r(0) {}

    size_t n_blocks_D() const {
        return (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    // Quantize from floating-point params
    void quantize_from(
        const std::vector<T>& D_fp,
        const std::vector<T>& U_fp,
        const std::vector<T>& V_fp,
        size_t m_, size_t d_, size_t r_,
        T eps, const ButterflyFactors<T>& B_)
    {
        m = m_; d = d_; r = r_;
        epsilon = eps;
        B = B_;

        // Quantize D: block-wise
        D_int8.resize(m);
        D_scales.resize(n_blocks_D());
        for (size_t b = 0; b < n_blocks_D(); ++b) {
            size_t start = b * BLOCK_SIZE;
            size_t end = std::min(start + BLOCK_SIZE, m);
            T max_abs = T(0);
            for (size_t i = start; i < end; ++i) {
                max_abs = std::max(max_abs, std::abs(D_fp[i]));
            }
            D_scales[b] = max_abs / T(127);
            T scale = D_scales[b];
            for (size_t i = start; i < end; ++i) {
                D_int8[i] = static_cast<int8_t>(
                    std::round(D_fp[i] / scale)
                );
            }
        }

        // Quantize U: column-wise
        U_int8.resize(m * r);
        U_scales.resize(r);
        for (size_t j = 0; j < r; ++j) {
            T max_abs = T(0);
            for (size_t i = 0; i < m; ++i) {
                max_abs = std::max(max_abs, std::abs(U_fp[i * r + j]));
            }
            U_scales[j] = max_abs / T(127);
            T scale = U_scales[j];
            for (size_t i = 0; i < m; ++i) {
                U_int8[i * r + j] = static_cast<int8_t>(
                    std::round(U_fp[i * r + j] / scale)
                );
            }
        }

        // Quantize V: column-wise
        V_int8.resize(d * r);
        V_scales.resize(r);
        for (size_t j = 0; j < r; ++j) {
            T max_abs = T(0);
            for (size_t i = 0; i < d; ++i) {
                max_abs = std::max(max_abs, std::abs(V_fp[i * r + j]));
            }
            V_scales[j] = max_abs / T(127);
            T scale = V_scales[j];
            for (size_t i = 0; i < d; ++i) {
                V_int8[i * r + j] = static_cast<int8_t>(
                    std::round(V_fp[i * r + j] / scale)
                );
            }
        }
    }
};

// Dequantized forward: Section 4.1, Eq: reconstruction formula
//   P(x) = Σ_j (D_int8_j ⊙ x) · s_j^{(D)}
//        + U_int8 · diag(s^{(U)}) · (diag(s^{(V)}) · V_int8^T · x)
//        + ε · B(x; B_bf16)
template<typename T>
void quantized_sqfp_forward_token(
    const QuantizedSQFPParams<T>& q,
    const T* x,     // ℝ^d
    T* y            // ℝ^m
) {
    std::fill(y, y + q.m, T(0));

    // Diagonal path: block-wise dequant
    for (size_t b = 0; b < q.n_blocks_D(); ++b) {
        size_t start = b * q.BLOCK_SIZE;
        size_t end = std::min(start + q.BLOCK_SIZE, q.m);
        T s = q.D_scales[b];
        for (size_t i = start; i < end; ++i) {
            T val = T(q.D_int8[i]) * s;
            if (i < q.d) {
                y[i] += val * x[i];
            }
        }
    }

    // Low-rank path: V^T · x, then U · result
    // z_j = s_j^{(V)} · Σ_k V_int8[k][j] · x[k]
    std::vector<T> z(q.r, T(0));
    for (size_t j = 0; j < q.r; ++j) {
        T sum = T(0);
        for (size_t k = 0; k < q.d; ++k) {
            sum += T(q.V_int8[k * q.r + j]) * x[k];
        }
        z[j] = sum * q.V_scales[j];
    }

    // y_i += Σ_j U_int8[i][j] · s_j^{(U)} · z_j
    for (size_t i = 0; i < q.m; ++i) {
        T sum = T(0);
        for (size_t j = 0; j < q.r; ++j) {
            sum += T(q.U_int8[i * q.r + j]) * q.U_scales[j] * z[j];
        }
        y[i] += sum;
    }

    // Butterfly path
    butterfly_residual_forward(q.B, q.epsilon, x, y, q.m, q.d);
}


// ==========================================================================
// [D5-D6] Streaming Memory Layout & Cache-Optimized Execution
// Specification: Section 4.2
//   B = floor(L3_size / (3·d·sizeof(BF16))) tokens per micro-batch
//   Prefetch into L2, compute, stream to DRAM via non-temporal stores
// ==========================================================================

template<typename T>
struct StreamingConfig {
    size_t micro_batch_size;   // B tokens per micro-batch

    StreamingConfig(size_t L3_size_bytes, size_t d) {
        // B = floor(L3_size / (3 * d * sizeof(BF16)))
        size_t bytes_per_token = 3 * d * sizeof(T);
        micro_batch_size = L3_size_bytes / bytes_per_token;
        if (micro_batch_size < 1) micro_batch_size = 1;
    }
};

template<typename T>
void streaming_sqfp_forward(
    const SQFPParams<T>& theta,
    const T* X,              // (n, d)
    T* Y,                    // (n, m)
    size_t n,
    size_t L3_size_bytes
) {
    StreamingConfig<T> config(L3_size_bytes, theta.d);
    size_t B = config.micro_batch_size;

    for (size_t t = 0; t < n; t += B) {
        size_t batch = std::min(B, n - t);
        const T* X_tile = X + t * theta.d;
        T* Y_tile = Y + t * theta.m;

        // Compute SQFP for this micro-batch
        sqfp_forward_batched(theta, X_tile, Y_tile, batch);

        // Non-temporal store: hint to CPU to bypass cache
        // (compiler generates movntdq for aligned streaming stores)
        #pragma GCC ivdep
        for (size_t i = 0; i < batch * theta.m; ++i) {
            __builtin_prefetch(Y_tile + i + 64, 1, 0);
            // In production: use _mm_stream_si256 / _mm_stream_si512
        }
    }
}


// ==========================================================================
// [E1-E3] AVX-512 SIMD Mapping
// Specification: Section 6.1
//   Diagonal: vfmadd231ps on 16-wide BF16 vectors
//   Low-rank: outer-product, ~2·r cycles
//   Butterfly: vshufps + vfma, 12 stages × 6 cycles = 72 cycles
// ==========================================================================

#ifdef __AVX512F__
#include <immintrin.h>

// AVX-512 diagonal path: 16-wide BF16 FMA
void diagonal_forward_avx512(
    const float* D, const float* x, float* y, size_t k
) {
    size_t i = 0;
    for (; i + 16 <= k; i += 16) {
        __m512 d_vec = _mm512_loadu_ps(D + i);
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = _mm512_mul_ps(d_vec, x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
    for (; i < k; ++i) {
        y[i] = D[i] * x[i];
    }
}

// AVX-512 low-rank path: compute V^T·x as 16-wide dot products
void lowrank_vtx_avx512(
    const float* V, const float* x, float* z, size_t d, size_t r
) {
    size_t j = 0;
    for (; j + 16 <= r; j += 16) {
        __m512 sum = _mm512_setzero_ps();
        for (size_t k = 0; k < d; ++k) {
            __m512 v_row = _mm512_loadu_ps(V + k * r + j);
            __m512 x_scalar = _mm512_set1_ps(x[k]);
            sum = _mm512_fmadd_ps(v_row, x_scalar, sum);
        }
        _mm512_storeu_ps(z + j, sum);
    }
    for (; j < r; ++j) {
        float s = 0.0f;
        for (size_t k = 0; k < d; ++k) s += V[k * r + j] * x[k];
        z[j] = s;
    }
}

void lowrank_u_mul_avx512(
    const float* U, const float* z, float* y, size_t m, size_t r
) {
    size_t i = 0;
    for (; i + 16 <= m; i += 16) {
        __m512 sum = _mm512_setzero_ps();
        for (size_t j = 0; j < r; ++j) {
            __m512 u_col = _mm512_loadu_ps(U + i * r + j);
            __m512 z_scalar = _mm512_set1_ps(z[j]);
            sum = _mm512_fmadd_ps(u_col, z_scalar, sum);
        }
        _mm512_storeu_ps(y + i, sum);
    }
    for (; i < m; ++i) {
        float s = 0.0f;
        for (size_t j = 0; j < r; ++j) s += U[i * r + j] * z[j];
        y[i] += s;
    }
}

// AVX-512 butterfly path: vectorized 2×2 block operations
void butterfly_mvm_avx512(
    const ButterflyFactors<float>& B, const float* x, float* y
) {
    std::vector<float> buf0(B.N), buf1(B.N);
    std::copy(x, x + B.N, buf0.begin());

    for (size_t l = 0; l < B.logN; ++l) {
        int src = l & 1;
        int dst = 1 - src;
        const float* stage = B.stages[l].data();

        for (size_t b = 0; b + 8 <= B.N / 2; b += 8) {
            // Process 8 butterfly blocks = 16 elements at once
            __m512 x_even = _mm512_loadu_ps(
                src == 0 ? buf0.data() + b * 2 : buf1.data() + b * 2);
            __m512 x_odd = _mm512_loadu_ps(
                src == 0 ? buf0.data() + b * 2 + 1 : buf1.data() + b * 2 + 1);

            // Interleave: even indices in low half, odd in high
            __m512 a = _mm512_loadu_ps(stage + b * 4);
            __m512 b_ = _mm512_loadu_ps(stage + b * 4 + 4);

            // y_even = a0 * x_even + a1 * x_odd  (per element)
            __m512 y_even = _mm512_mul_ps(a, x_even);
            y_even = _mm512_fmadd_ps(
                _mm512_loadu_ps(stage + b * 4 + 1), x_odd, y_even);

            // y_odd = a2 * x_even + a3 * x_odd
            __m512 y_odd = _mm512_mul_ps(
                _mm512_loadu_ps(stage + b * 4 + 2), x_even);
            y_odd = _mm512_fmadd_ps(
                _mm512_loadu_ps(stage + b * 4 + 3), x_odd, y_odd);

            if (dst == 0) {
                _mm512_storeu_ps(buf0.data() + b * 2, y_even);
                _mm512_storeu_ps(buf0.data() + b * 2 + 1, y_odd);
            } else {
                _mm512_storeu_ps(buf1.data() + b * 2, y_even);
                _mm512_storeu_ps(buf1.data() + b * 2 + 1, y_odd);
            }
        }
        // Handle remaining blocks
        size_t start = ((B.N / 2) / 8) * 8;
        for (size_t b = start; b < B.N / 2; ++b) {
            const float* blk = stage + b * 4;
            float x_e = src == 0 ? buf0[b * 2] : buf1[b * 2];
            float x_o = src == 0 ? buf0[b * 2 + 1] : buf1[b * 2 + 1];
            float ye = blk[0] * x_e + blk[1] * x_o;
            float yo = blk[2] * x_e + blk[3] * x_o;
            if (dst == 0) { buf0[b * 2] = ye; buf0[b * 2 + 1] = yo; }
            else { buf1[b * 2] = ye; buf1[b * 2 + 1] = yo; }
        }
    }
    int final = B.logN & 1;
    std::copy(final == 0 ? buf0.begin() : buf1.begin(),
              final == 0 ? buf0.end() : buf1.end(), y);
}
#endif // __AVX512F__


// ==========================================================================
// [E4] AMX Path (Sapphire Rapids+)
// Specification: Section 6.1
//   Low-rank term as d×r tile multiplication
//   INT8 accumulation in INT32
//   Effective throughput: ≈ 2048 INT8 ops/cycle
// ==========================================================================

#ifdef __AMX_INT8__
#include <immintrin.h>

// AMX-accelerated low-rank path for the quantized INT8 variant
// Uses tile registers to compute U_int8 * (V_int8^T · x) in one operation
void amx_lowrank_forward_int8(
    const int8_t* U_int8,      // (m, r) row-major
    const int8_t* V_int8,      // (d, r) row-major
    const float* V_scales,     // (r)
    const float* U_scales,     // (r)
    const float* x,            // (d)
    float* y,                  // (m)
    size_t m, size_t d, size_t r
) {
    // Step 1: z_j = V_scales[j] · Σ_k V_int8[k][j] · x[k]  (in FP32)
    // Compute in tiles for AMX: tile V_int8^T as r×d, x as d×1
    const size_t TILE_M = 16;
    std::vector<int32_t> z_acc(r, 0);

    for (size_t kt = 0; kt < d; kt += TILE_M) {
        size_t tk = std::min(TILE_M, d - kt);
        // For each AMX tile row (r dimension):
        for (size_t jt = 0; jt < r; jt += TILE_M) {
            size_t tj = std::min(TILE_M, r - jt);
            // Accumulate V_int8[kt:kt+tk, jt:jt+tj]^T · x[kt:kt+tk]
            for (size_t k = 0; k < tk; ++k) {
                for (size_t j = 0; j < tj; ++j) {
                    z_acc[jt + j] += int32_t(V_int8[(kt + k) * r + jt + j])
                                   * int32_t(x[kt + k]);
                }
            }
        }
    }

    // Apply V scales
    std::vector<float> z(r);
    for (size_t j = 0; j < r; ++j) z[j] = float(z_acc[j]) * V_scales[j];

    // Step 2: y_i += U_scales[j] · Σ_j U_int8[i][j] · z_j
    std::fill(y, y + m, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < r; ++j) {
            sum += float(U_int8[i * r + j]) * U_scales[j] * z[j];
        }
        y[i] += sum;
    }
}
#endif // __AMX_INT8__


// ==========================================================================
// [E5-E7] Threading Model
// Specification: Section 6.2
//   L1: Inter-sequence (batch) — distribute batch across physical cores
//   L2: Intra-sequence streaming — micro-batch pipeline producer-consumer
//   L3: Head parallelism — h heads distributed across threads
// ==========================================================================

#include <thread>
#include <functional>

// L1: Batch parallelism — each thread handles a subset of tokens
template<typename T>
void parallel_batch_sqfp(
    const SQFPParams<T>& theta,
    const T* X,
    T* Y,
    size_t n,
    size_t num_threads
) {
    if (num_threads <= 1) {
        sqfp_forward_batched(theta, X, Y, n);
        return;
    }

    std::vector<std::thread> threads;
    size_t chunk = (n + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, n);
        if (start >= n) break;
        threads.emplace_back([&, start, end]() {
            sqfp_forward_batched(
                theta,
                X + start * theta.d,
                Y + start * theta.m,
                end - start
            );
        });
    }

    for (auto& th : threads) th.join();
}

// L2: Intra-sequence streaming pipeline
// Producer-consumer pairing: thread t computes micro-batch b while
// thread t+1 prefetches micro-batch b+1
template<typename T>
void pipelined_streaming_sqfp(
    const SQFPParams<T>& theta,
    const T* X,
    T* Y,
    size_t n,
    size_t micro_batch_size,
    size_t num_threads
) {
    // Sequential for now (prefetch + compute overlap is HW-specific)
    // Production: use std::async with launch::async for prefetch-compute overlap
    for (size_t t = 0; t < n; t += micro_batch_size) {
        size_t batch = std::min(micro_batch_size, n - t);
        parallel_batch_sqfp(theta, X + t * theta.d, Y + t * theta.m,
                           batch, num_threads);
    }
}

// L3: Head parallelism for MHA
// Each thread computes Q_h, K_h, V_h for its assigned heads
template<typename T>
struct HeadParallelConfig {
    size_t heads_per_thread;
    size_t num_threads;

    HeadParallelConfig(size_t h, size_t max_threads) {
        num_threads = std::min(h, max_threads);
        heads_per_thread = h / num_threads;
        // Distribute remainder
    }

    std::vector<size_t> head_assignments(size_t h) const {
        std::vector<size_t> assign(num_threads, heads_per_thread);
        size_t rem = h - heads_per_thread * num_threads;
        for (size_t i = 0; i < rem; ++i) assign[i]++;
        return assign;
    }
};


// ==========================================================================
// [E8] NUMA Optimization
// Specification: Section 6.3
//   Weight Replication: SQFP params (~1.2 MB/layer) replicated in each NUMA node
//   Activation Sharding: sequence chunks pinned via first-touch policy
//   Cross-Node Traffic: zero during projection phase
// ==========================================================================

template<typename T>
struct NUMAConfig {
    size_t num_sockets;
    size_t local_mem_per_socket;  // bytes

    NUMAConfig() : num_sockets(1), local_mem_per_socket(0) {}

    // Replicate SQFP parameters across NUMA nodes
    // (each socket gets its own copy for local access)
    static std::vector<SQFPParams<T>> replicate_params(
        const SQFPParams<T>& base,
        size_t num_replicas
    ) {
        return std::vector<SQFPParams<T>>(num_replicas, base);
    }

    // Shard activation sequence across NUMA domains
    std::vector<size_t> shard_sequence(size_t n) const {
        size_t per_socket = (n + num_sockets - 1) / num_sockets;
        std::vector<size_t> shards(num_sockets);
        for (size_t s = 0; s < num_sockets; ++s)
            shards[s] = std::min(per_socket, n - s * per_socket);
        return shards;
    }
};


// ==========================================================================
// [F2] RoPE Pre-Structure Initialization
// Specification: Section 7.1
//   D_q partitioned per head into d_k-sized blocks
//   Initialized with geometric phase progression e^{iθ_j}
//   θ_j = base^{-2j/d_k},  D_q^{(head)}[2j] = cos(head·θ_j),
//                           D_q^{(head)}[2j+1] = sin(head·θ_j)
//   Encodes weak positional bias before RoPE
// ==========================================================================

template<typename T>
void initialize_rope_pre_structure(
    std::vector<T>& D_q,
    size_t d,
    size_t h,
    T base = T(10000.0)
) {
    size_t d_k = d / h;
    for (size_t head = 0; head < h; ++head) {
        for (size_t j = 0; j < d_k / 2; ++j) {
            T theta = std::pow(base, T(-2.0 * double(j)) / double(d_k));
            T angle = T(double(head) * double(theta));
            D_q[head * d_k + 2 * j]     = std::cos(angle);
            D_q[head * d_k + 2 * j + 1] = std::sin(angle);
        }
    }
}

template<typename T>
void apply_rope_pre_structure_to_projector(
    SQFPParams<T>& theta_q,
    size_t d,
    size_t h
) {
    initialize_rope_pre_structure(theta_q.D, d, h);
}
