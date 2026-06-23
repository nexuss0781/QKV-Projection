# Component 2.1 — QKV Projection

## Executive Summary

QKV Projection is the first operation inside each attention layer. It takes the residual stream `X ∈ ℝ^{n × d}` and linearly projects it into three separate representation spaces: Query (Q), Key (K), and Value (V). These three matrices are the sole inputs to the attention computation.

The projection is three independent linear transformations applied to every token position. In standard Multi-Head Attention (MHA), all three projections map from `d` to `d`. Modern LLMs widely use Grouped Query Attention (GQA) or Multi-Query Attention (MQA), which reduce the K and V projection output dimensions to cut KV-cache memory.

---

## Mathematical Breakdown & Specification

### Definitions

| Symbol | Meaning |
|---|---|
| `X` | Input residual stream, `X ∈ ℝ^{n × d}` |
| `d` | Model dimension |
| `h` | Number of query heads |
| `g` | Number of KV groups (GQA); `g = h` for MHA, `g = 1` for MQA |
| `d_k` | Key/Query head dimension, `d_k = d / h` |
| `d_v` | Value head dimension, `d_v = d / h` |
| `Wq` | Query weight matrix |
| `Wk` | Key weight matrix |
| `Wv` | Value weight matrix |

---

### Standard MHA Projection

```
Q = X · Wq      Wq ∈ ℝ^{d × (h · d_k)},    Q ∈ ℝ^{n × (h · d_k)}
K = X · Wk      Wk ∈ ℝ^{d × (h · d_k)},    K ∈ ℝ^{n × (h · d_k)}
V = X · Wv      Wv ∈ ℝ^{d × (h · d_v)},    V ∈ ℝ^{n × (h · d_v)}
```

Since `d_k = d_v = d/h`:
```
Wq, Wk, Wv ∈ ℝ^{d × d}     (all square)
Q, K, V ∈ ℝ^{n × d}
```

Split into heads for attention:
```
Q → [Q₁, Q₂, ..., Qₕ]    each Qᵢ ∈ ℝ^{n × d_k}
K → [K₁, K₂, ..., Kₕ]    each Kᵢ ∈ ℝ^{n × d_k}
V → [V₁, V₂, ..., Vₕ]    each Vᵢ ∈ ℝ^{n × d_v}
```

Reshape view (no copy):
```
Q.view(n, h, d_k).transpose(1, 0)  →  shape (h, n, d_k)
```

---

### Grouped Query Attention (GQA)

Introduced in GQA paper (Ainslie et al., 2023). Used in LLaMA-2, LLaMA-3, Mistral, Gemma.

```
Q = X · Wq      Wq ∈ ℝ^{d × (h · d_k)},    Q ∈ ℝ^{n × (h · d_k)}
K = X · Wk      Wk ∈ ℝ^{d × (g · d_k)},    K ∈ ℝ^{n × (g · d_k)}    g < h
V = X · Wv      Wv ∈ ℝ^{d × (g · d_v)},    V ∈ ℝ^{n × (g · d_v)}
```

Each KV group serves `h/g` query heads. K and V are expanded (broadcast) before attention:
```
K_expanded = repeat_interleave(K, h//g, dim=head_dim)   →  ℝ^{n × (h · d_k)}
V_expanded = repeat_interleave(V, h//g, dim=head_dim)   →  ℝ^{n × (h · d_v)}
```

No new parameters — just a view/broadcast. KV-cache stores only `g` groups, not `h` heads.

**KV-cache memory reduction:** factor `h / g`

| Model | h | g | Reduction |
|---|---|---|---|
| LLaMA-2 7B | 32 | 32 | 1× (MHA) |
| LLaMA-2 70B | 64 | 8 | 8× |
| LLaMA-3 8B | 32 | 8 | 4× |
| LLaMA-3 70B | 64 | 8 | 8× |
| Mistral 7B | 32 | 8 | 4× |
| Gemma 7B | 16 | 16 | 1× (MHA) |

---

### Multi-Query Attention (MQA)

Extreme case of GQA: `g = 1`. A single K and V shared by all `h` query heads.

```
K = X · Wk      Wk ∈ ℝ^{d × d_k}     (single head)
V = X · Wv      Wv ∈ ℝ^{d × d_v}
```

Maximum KV-cache savings (factor h), at some quality cost vs GQA.
Used in: PaLM, Falcon-7B, some StarCoder variants.

---

### Bias Terms

Most modern LLMs omit bias terms in QKV projections:
```
Q = X · Wq         (no bias)
```

Original Transformer used biases. Removing them saves parameters and marginally improves training stability at large scale with Pre-LN.

---

### Output Projection (Wo)

After multi-head attention concatenates its outputs, a final linear projection maps back to `d`:

```
Concat(head_1, ..., head_h) ∈ ℝ^{n × d}
Output = Concat · Wo        Wo ∈ ℝ^{d × d}
```

`Wo` is sometimes initialized to `1/√(2L)` (scaled initialization) to keep residual stream variance stable at initialization across L layers.

---

### Parameters Count per Attention Layer

| Component | MHA | GQA (g groups) | MQA |
|---|---|---|---|
| Wq | d² | d² | d² |
| Wk | d² | d · g · d_k = d² · g/h | d · d_k = d²/h |
| Wv | d² | d² · g/h | d²/h |
| Wo | d² | d² | d² |
| **Total** | **4d²** | **(2 + 2g/h) · d²** | **(2 + 2/h) · d²** |

For LLaMA-3 8B: d=4096, h=32, g=8 → (2 + 2·8/32) · 4096² = 2.5 · 16.7M ≈ 41.9M params/layer

---

## Complexity Analysis

| Operation | Time | Space |
|---|---|---|
| Q projection: X · Wq | O(n · d²) | O(n · d) |
| K projection (MHA): X · Wk | O(n · d²) | O(n · d) |
| K projection (GQA): X · Wk | O(n · d · g · d_k) = O(n · d² · g/h) | O(n · g · d_k) |
| V projection (GQA) | O(n · d² · g/h) | O(n · g · d_v) |
| Head split (reshape) | O(1) — view only | O(1) |
| Output projection Wo | O(n · d²) | O(n · d) |
| **Total (MHA)** | **O(n · d²)** | **O(n · d)** |

All operations are O(n) in sequence length — linear. d² is large but constant.

---

## Detailed Test Specification

### Shape Tests

- [ ] **Q shape MHA:** `Q.shape == (batch, n, h, d_k)` after reshape
- [ ] **K shape GQA:** `K.shape == (batch, n, g, d_k)` before expansion
- [ ] **K shape after expansion:** `K_expanded.shape == (batch, n, h, d_k)`
- [ ] **V shape MHA:** `V.shape == (batch, n, h, d_v)`
- [ ] **Output projection shape:** `(X_attn @ Wo).shape == (batch, n, d)`

### Correctness Tests

- [ ] **Linearity:** `QKV_proj(αX + βY) == α·QKV_proj(X) + β·QKV_proj(Y)` within float tolerance
- [ ] **Independence:** Q, K, V projections are independent — gradient through Wq does not affect Wk
- [ ] **GQA broadcast correctness:** Each of the `h/g` query heads in a group attends to identical K,V tensors (before attention weights are applied)
- [ ] **No parameter sharing between Q and K:** `Wq` and `Wk` are distinct tensors even if same shape

### Gradient Tests

- [ ] **Full gradient flow:** `∂L/∂Wq`, `∂L/∂Wk`, `∂L/∂Wv`, `∂L/∂Wo` are all nonzero after a forward+backward pass
- [ ] **Wo initialization:** At init, output projection does not blow up residual stream variance — `Var(X + Attn(X)) ≈ Var(X) · (1 + ε)`

### Numerical Tests

- [ ] **No NaN in projections:** Forward pass with standard init and bf16 produces no NaN
- [ ] **Scale sanity:** Output magnitude of Q,K projections is O(1) (not growing with d) — relevant for attention score scaling

### Integration Tests

- [ ] **RoPE compatibility:** Q and K output shapes are compatible with RoPE rotation — even `d_k` dimension
- [ ] **KV-cache compatibility:** K and V can be concatenated along the sequence dimension across decoding steps

---

## Checklist

- [ ] Executive summary written and reviewed
- [ ] Standard MHA projection documented
- [ ] GQA projection documented with expansion mechanism
- [ ] MQA as extreme case documented
- [ ] Output projection Wo documented
- [ ] Parameter count table complete
- [ ] Complexity analysis complete
- [ ] All shape tests written
- [ ] All correctness tests written
- [ ] All gradient tests written
- [ ] All numerical tests written
- [ ] All integration tests written
- [ ] Document reviewed against Transformer.md for consistency
