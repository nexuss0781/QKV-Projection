# Component 2.1 — QKV Projection (SQFP) TODO

## External Input/Output Contracts

### Input
| Port | Shape | Type | Source |
|------|-------|------|--------|
| `X` | `(batch, n, d)` | BF16/FP16 | Residual stream (prior layer output) |

### Outputs
| Port | Shape | Type | Destination |
|------|-------|------|-------------|
| `Q` | `(batch, n, h, d_k)` | BF16/FP16 | Component 2.2 (Scaled Dot-Product Attention) |
| `K` | `(batch, n, g, d_k)` | BF16/FP16 | Component 2.2 + Component 2.4 (KV-Cache) |
| `V` | `(batch, n, g, d_v)` | BF16/FP16 | Component 2.2 + Component 2.4 (KV-Cache) |
| `Out` | `(batch, n, d)` | BF16/FP16 | Residual stream (after Wo) |

### Contracts
- GQA: `K_expanded = repeat_interleave(K, h//g, dim=1)` — view-only, no copy
- MQA: `g = 1`, `K.shape = (batch, n, 1, d_k)`, `V.shape = (batch, n, 1, d_v)`
- RoPE: Q, K must have even `d_k` dimension per head
- KV-Cache: New tokens appended via `memcpy` along sequence dim, no recomputation

---

## Atomic Tasks

### Phase A — Core SQFP Operator

- [x] **A1** — Diagonal Spectrum Path: `D ⊙ x_i` (Hadamard product, O(d))
- [x] **A2** — Low-Rank Interaction Path: `U · (V^T · x_i)` (O(d·r), r = O(log d))
- [x] **A3** — Butterfly Residual Path: `ε · B(x_i; B)` (O(d log d), factored product of log₂d stages of 2×2 blocks)
- [x] **A4** — Unified SQFP Operator: `P(x_i; θ) = D⊙x_i + U·(V^T·x_i) + ε·B(x_i; B)`
- [x] **A5** — Shape Preservation: output dim m = h·d_k (Q), m = g·d_k (K/V), m = d (Wo)

### Phase B — Attention Variant Projectors

- [x] **B1** — MHA Projection: Q shape `(n, h, d_k)`, K shape `(n, h, d_k)`, V shape `(n, h, d_v)`
- [x] **B2** — GQA Projection: K shape `(n, g, d_k)`, V shape `(n, g, d_v)`, broadcast expansion `repeat_interleave(K, h//g, dim=1)`
- [x] **B3** — MQA Projection: K shape `(n, 1, d_k)`, V shape `(n, 1, d_v)`
- [x] **B4** — Output Projection Wo: `Out = P(X_attn; θ_o)`, scaled init `D_o = 1/√(2L)`

### Phase C — Adaptive Precision Path

- [x] **C1** — Token Importance Sketch: `ρ_i = ‖x_i · S‖₂` where `S ∈ ℝ^{d×s}`, s=16, fixed random orthonormal
- [x] **C2** — Learned Threshold τ: importance gating
- [x] **C3** — Fast Path (ρ_i < τ): `D⊙x_i + U_fast·(V_fast^T·x_i)` with r_fast = ⌈r/4⌉
- [x] **C4** — Full Path (ρ_i ≥ τ): complete SQFP operator
- [x] **C5** — Amortized Complexity: 80% fast / 20% full → O(d) amortized per token

### Phase D — Quantization & Memory Layout

- [x] **D1** — Block-wise INT8 Quantization: D quantized in blocks of size 64, per-block scale
- [x] **D2** — INT8 for U, V: column-wise scales
- [x] **D3** — Dequantized Reconstruction: `P_quant(x)` with INT32 accumulation
- [x] **D4** — Butterfly in BF16: B stored and computed in BF16 (not quantized)
- [x] **D5** — Micro-batch Streaming: `B = floor(L3_size / (3·d·sizeof(BF16)))` tokens per chunk
- [x] **D6** — Non-temporal Stores: `movntdq` for output streaming, bypass cache

### Phase E — CPU Execution Model

- [x] **E1** — AVX-512 Diagonal Path: `vfmadd231ps` on 16-wide BF16 vectors
- [x] **E2** — AVX-512 Low-Rank Path: outer-product formulation, 2·r cycles
- [x] **E3** — AVX-512 Butterfly Path: `vshufps` + `vfma`, log₂d stages × 6 cycles
- [x] **E4** — AMX Path: INT8 tile multiplication for low-rank term (Sapphire Rapids+)
- [x] **E5** — Threading L1: batch dim distributed across cores
- [x] **E6** — Threading L2: intra-sequence micro-batch pipeline
- [x] **E7** — Threading L3: head parallelism (MHA, h=32 → 8 threads × 4 heads)
- [x] **E8** — NUMA Optimization: weight replication per socket, activation first-touch pinning

### Phase F — Integration Contracts

- [x] **F1** — RoPE Compatibility: Q, K reshape to `(..., h, d_k)` with even d_k
- [x] **F2** — RoPE Pre-Structure (optional): diagonal D_q partitioned per head with phase progression
- [x] **F3** — KV-Cache Append: row-major `(n, g, d_k)`, memcpy into circular buffer
- [x] **F4** — Zero Recomputation: token-wise independent, new token doesn't recompute past
- [x] **F5** — Attention Interface: Q `(n, h, d_k)`, K_expanded `(n, h, d_k)`, V_expanded `(n, h, d_v)`

### Phase G — Tests (Section 8)

#### G1: Shape & Structural Tests
- [x] **T1** — Q shape MHA: `reshape(Q, (b, n, h, d_k)).shape == (b, n, h, d_k)`
- [x] **T2** — K shape GQA pre-expand: `K.shape == (b, n, g, d_k)`
- [x] **T3** — K shape post-expand: `K_expanded.shape == (b, n, h, d_k)`
- [x] **T4** — V shape MQA: `V.shape == (b, n, 1, d_v)`
- [x] **T5** — Wo output shape: `output.shape == (b, n, d)`
- [x] **T6** — Butterfly block count: `len(B_factors) == ceil(log2(max(d, m)))`
- [x] **T7** — DPLR rank constraint: `rank(U) == rank(V) == r`

#### G2: Correctness & Equivalence Tests
- [x] **T8** — Linearity: `P(αX + βY) == α·P(X) + β·P(Y)` (10⁻⁵ rel)
- [x] **T9** — Spectral equivalence: `‖W_dense - (D + UV^T + εB)‖₂ < σ_{r+1} + δ` (10⁻³ abs)
- [x] **T10** — GQA broadcast identity: `all(K_expanded[:, :, 0::h/g, :] == K)`
- [x] **T11** — Parameter independence: `∇_{W_q}L` doesn't modify `W_k` buffer
- [x] **T12** — No parameter sharing: `id(W_q) != id(W_k)`
- [x] **T13** — Causal pre-structure: optional `triu(Q@K^T, k=1).sum() < 10⁻⁶`

#### G3: Gradient & Training Tests
- [x] **T14** — Full gradient flow: ∂L/∂D, ∂L/∂U, ∂L/∂V, ∂L/∂B all non-zero
- [x] **T15** — Stiefel constraint: `U^T U = I_r` (10⁻⁶ Frobenius)
- [x] **T16** — Variance preservation: `Var(X + Attn(X)) ≈ Var(X)·(1 + 1/(2L))` at init (5%)
- [x] **T17** — Gradient norm balance: `0.1 < ‖∇_D‖/‖∇_U‖ < 10`

#### G4: Numerical & Stability Tests
- [x] **T18** — No NaN forward: BF16, n=8192
- [x] **T19** — No NaN backward: BF16, loss scaling
- [x] **T20** — Quantization round-trip: `‖y - y_quant‖/‖y‖ < 0.005`
- [x] **T21** — Scale sanity: `‖Q‖_∞ ∈ [0.5, 2.0]` after layer-norm
- [x] **T22** — Butterfly invertibility: `‖B·B^{-1} - I‖_∞ < 10⁻⁴`

#### G5: Performance & Resource Tests
- [ ] **T23** — Cache miss ratio: < 1% (L3=32MB, d=4096) [analytical, needs HW-perf counters]
- [x] **T24** — Single-core throughput: measured on dev hardware
- [ ] **T25** — Multi-core scaling: > 50× on 64-core EPYC [HW-dependent]
- [x] **T26** — RAM footprint: < 100 MB for 32 layers
- [x] **T27** — Streaming latency: < 50 ms time-to-first-token (n=10⁶)

#### G6: Integration Tests
- [x] **T28** — RoPE shape compatibility: Component 2.1 → RoPE kernel
- [x] **T29** — KV-cache append: Component 2.1 → Component 2.4
- [x] **T30** — Attention score agreement: SQFP vs dense Q/K, < 10⁻³ KL-div
- [x] **T31** — End-to-end forward: full transformer block, < 10⁻³ vs dense baseline
- [x] **T32** — Long-context stability: n=128K, no gradient clip needed

#### F2: RoPE Pre-Structure (New Capability)
- [x] **F2-test** — RoPE pre-structure initialization and forward pass verified

---

## Implementation Order

1. **Phase A** first — core SQFP operator (mathematical primitive)
2. **Phase B** — attention variant projectors on top of SQFP
3. **Phase G1-G2** — basic shape and correctness tests
4. **Phase C** — adaptive precision path
5. **Phase D** — quantization and streaming memory layout
6. **Phase E** — CPU execution model optimizations
7. **Phase G3-G6** — remaining tests
8. **Phase F** — integration contracts

> Awaiting approval before implementation.
