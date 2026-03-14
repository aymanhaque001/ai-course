# Module 2: The Transformer Architecture — A Deep Dive

> **Prerequisites:** Module 1 (neural network fundamentals, backpropagation, optimization)
> **Estimated Study Time:** 12–15 hours (this is the most critical module)

---

## 2.1 The Big Picture

The Transformer (Vaswani et al., 2017) replaced recurrence with **self-attention**, enabling:

- **Parallelized training** across sequence positions
- **Direct connections** between any two tokens (no information bottleneck)
- **Scalability** to billions of parameters

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER TAXONOMY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Encoder-Only          Encoder-Decoder         Decoder-Only     │
│  ┌───────────┐        ┌──────┬──────┐        ┌───────────┐     │
│  │ Encoder   │        │Encode│Decode│        │  Decoder   │     │
│  │           │        │  r   │  r   │        │           │     │
│  │ Bi-dir    │        │      │      │        │ Causal    │     │
│  │ attention │        │ Bi + │Causal│        │ attention │     │
│  └───────────┘        └──────┴──────┘        └───────────┘     │
│                                                                 │
│  BERT, RoBERTa         T5, BART               GPT, LLaMA,     │
│  (classification,      (translation,           Claude, Gemini   │
│   embeddings)          summarization)          (generation)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Modern LLMs (GPT-4, Claude, LLaMA, Gemini) are all decoder-only transformers.** This module focuses primarily on this architecture.

---

## 2.2 Self-Attention: The Core Mechanism

### Intuition

Self-attention lets each token "look at" every other token in the sequence and compute a weighted sum based on relevance.

Example: "The **cat** sat on the **mat** because **it** was soft."

- For the token "it", attention should focus heavily on "mat" (what "it" refers to).

### Queries, Keys, and Values

Each input token embedding is projected into three vectors:

```
                        ┌──── Q (Query): "What am I looking for?"
                        │
Input Embedding ────────┼──── K (Key):   "What do I contain?"
    xᵢ                 │
                        └──── V (Value): "What information do I provide?"

Q = xW_Q    K = xW_K    V = xW_V
```

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

Step by step:

```
Step 1: Compute similarity scores     Step 2: Scale          Step 3: Softmax        Step 4: Weighted sum
                                                              (normalize to          of Values
                                                               probabilities)

  QKᵀ =                               QKᵀ/√dₖ =             Attention weights =    Output =
  ┌              ┐                     ┌              ┐       ┌              ┐
  │ q₁·k₁ q₁·k₂ │                     │ s₁₁   s₁₂   │       │ 0.7    0.3   │       0.7·v₁ + 0.3·v₂
  │ q₂·k₁ q₂·k₂ │                     │ s₂₁   s₂₂   │       │ 0.1    0.9   │       0.1·v₁ + 0.9·v₂
  └              ┘                     └              ┘       └              ┘
```

**Why scale by √dₖ?** Without scaling, when dₖ is large, the dot products grow in magnitude, pushing the softmax into regions with extremely small gradients (near 0 or 1). Dividing by √dₖ keeps the variance of the dot products at ~1 regardless of dimension.

---

## 2.3 Multi-Head Attention

Instead of one attention function, we run **h** attention heads in parallel, each with its own learned projections:

```
                    ┌─── Head 1: Attention(QW₁Q, KW₁K, VW₁V) ──┐
                    │                                             │
Input ──────────────┼─── Head 2: Attention(QW₂Q, KW₂K, VW₂V) ──┼── Concat ── Wₒ ── Output
                    │                                             │
                    ├─── Head 3: Attention(QW₃Q, KW₃K, VW₃V) ──┤
                    │           ...                               │
                    └─── Head h: Attention(QWₕQ, KWₕK, VWₕV) ──┘
```

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wₒ
where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
```

**Dimensions:**

- Model dimension: d_model (e.g., 4096 for LLaMA-7B)
- Number of heads: h (e.g., 32)
- Per-head dimension: d_k = d_model / h (e.g., 128)

**Why multiple heads?** Each attention head has its own Q, K, V projection matrices, so each head computes attention in a _different learned subspace_. Empirically, different heads specialize in different linguistic relationships:

- Head 1 might learn syntactic dependencies (subject-verb agreement)
- Head 2 might learn coreference (which noun a pronoun refers to)
- Head 3 might learn positional proximity (attend to adjacent tokens)
- Head 4 might learn semantic similarity (attend to topically related words)

With a **single head**, all these patterns would compete for the same attention weights — the model would have to make a hard choice about which relationship to capture. With multiple heads, these patterns are captured **simultaneously and independently**, then combined via the output projection Wₒ. This is analogous to multiple convolutional filters learning different visual features (edges, textures, shapes) in CNNs.

---

## 2.4 Causal (Masked) Attention

For autoregressive language models (GPT, LLaMA, Claude), each token can only attend to itself and previous tokens — not future tokens. This is enforced with a causal mask:

```
Attention Mask (for sequence of length 5):

          Key positions
          k₁   k₂   k₃   k₄   k₅
    q₁  [  ✓    ✗    ✗    ✗    ✗  ]     ← token 1 sees only itself
    q₂  [  ✓    ✓    ✗    ✗    ✗  ]     ← token 2 sees tokens 1-2
Q   q₃  [  ✓    ✓    ✓    ✗    ✗  ]     ← token 3 sees tokens 1-3
    q₄  [  ✓    ✓    ✓    ✓    ✗  ]     ← token 4 sees tokens 1-4
    q₅  [  ✓    ✓    ✓    ✓    ✓  ]     ← token 5 sees all tokens

✗ positions are set to -∞ before softmax, making their attention weight 0.
```

**Implementation:** Before softmax, add a mask matrix where masked positions = -∞:

```python
scores = (Q @ K.T) / math.sqrt(d_k)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))
weights = F.softmax(scores, dim=-1)
output = weights @ V
```

---

## 2.5 The Full Decoder-Only Transformer Block

```
┌─────────────────────────────────────────────────┐
│              TRANSFORMER BLOCK (×N)              │
│                                                  │
│   Input (from previous block or embedding)       │
│     │                                            │
│     ▼                                            │
│   ┌─────────────────┐                            │
│   │   Layer Norm     │                            │
│   └────────┬────────┘                            │
│            │                                      │
│            ▼                                      │
│   ┌─────────────────┐      ┌─────┐               │
│   │  Masked Multi-  │      │     │               │
│   │  Head Self-     │      │ Add │◄── Residual   │
│   │  Attention      │──────│     │    Connection  │
│   └─────────────────┘      └──┬──┘               │
│                               │                   │
│                               ▼                   │
│   ┌─────────────────┐                            │
│   │   Layer Norm     │                            │
│   └────────┬────────┘                            │
│            │                                      │
│            ▼                                      │
│   ┌─────────────────┐      ┌─────┐               │
│   │  Feed-Forward    │      │     │               │
│   │  Network (FFN)   │──────│ Add │◄── Residual   │
│   │  (SwiGLU/GELU)  │      │     │    Connection  │
│   └─────────────────┘      └──┬──┘               │
│                               │                   │
│                               ▼                   │
│                           Output                  │
└─────────────────────────────────────────────────┘
```

### Feed-Forward Network (FFN)

The FFN in each block is a position-wise MLP applied independently to each token:

**Standard:**

```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

**SwiGLU (used in LLaMA, PaLM, modern LLMs):**

```
SwiGLU(x) = (Swish(xW₁) ⊙ xW₃) W₂
```

- ⊙ = element-wise multiplication
- Hidden dimension is typically ~2.67× the model dimension (for SwiGLU) vs 4× (for standard FFN)

**Why gating improves over standard FFN:** In a standard FFN, the transformation GELU(xW₁)W₂ applies the same non-linearity uniformly. SwiGLU introduces a **learned, input-dependent gate**: xW₁ produces a gate signal (passed through Swish to create values between 0 and 1), and xW₃ produces candidate values. The element-wise product means the gate _selectively suppresses or amplifies_ different features in the candidate. This is more expressive because the network can learn to route information: "for this input, activate these features and suppress those."

The cost is an extra weight matrix (3 matrices instead of 2), which is why the hidden dimension is reduced to ~2.67× to keep the parameter count equivalent. Despite this, SwiGLU consistently achieves measurably better loss than standard FFN at the same total parameter budget — Shazeer (2020) showed ~1-3% improvement across multiple model sizes.

---

## 2.6 Positional Encoding

Self-attention is permutation-invariant — without positional information, "dog bites man" = "man bites dog". We need to inject position information.

### Sinusoidal (Original Transformer)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

```
Position →   0    1    2    3    4    5    ...
Dim 0:    sin(0) sin(1) sin(2) sin(3) sin(4) sin(5)
Dim 1:    cos(0) cos(1) cos(2) cos(3) cos(4) cos(5)
Dim 2:    sin(0) sin(·) sin(·) sin(·) sin(·) sin(·)  (slower frequency)
Dim 3:    cos(0) cos(·) cos(·) cos(·) cos(·) cos(·)
  ...       ...    ...    ...    ...    ...    ...
```

Each dimension oscillates at a different frequency, creating a unique "fingerprint" for each position.

### Rotary Position Embeddings (RoPE) — Modern Standard

RoPE (Su et al., 2021) encodes position by **rotating** the query and key vectors in 2D subspaces:

```
For dimensions (2i, 2i+1), apply rotation by angle θᵢ·pos:

┌         ┐   ┌                  ┐   ┌      ┐
│ q'₂ᵢ    │ = │ cos(mθᵢ) -sin(mθᵢ) │ · │ q₂ᵢ   │
│ q'₂ᵢ₊₁  │   │ sin(mθᵢ)  cos(mθᵢ) │   │ q₂ᵢ₊₁ │
└         ┘   └                  ┘   └      ┘

where m = position index, θᵢ = 10000^(-2i/d)
```

**Key property:** The dot product qₘᵀkₙ depends only on the relative position (m-n), not absolute positions. This gives the model translational equivariance.

**Why RoPE won:** (1) Naturally captures relative positions; (2) Can be extended to longer sequences than seen during training; (3) No extra parameters; (4) Used by LLaMA, Mistral, and most modern open-source LLMs.

### ALiBi (Attention with Linear Biases)

Instead of modifying embeddings, ALiBi adds a linear bias directly to attention scores:

```
Attention score for (query at pos i, key at pos j):
score = qᵢᵀkⱼ - m·|i - j|
```

where m is a **head-specific slope** set as a geometric sequence: for h heads, the slopes are 2^(-8/h), 2^(-16/h), ..., 2^(-8). For 8 heads, m ∈ {1/2, 1/4, 1/8, ..., 1/256}. Different slopes mean different heads have different "attention windows" — heads with small m attend broadly, heads with large m focus locally.

**Why ALiBi enables length extrapolation:** Learned positional embeddings and sinusoidal encodings only work for positions seen during training. If you train with max length 2048, position 2049 has no embedding. RoPE can be extended but requires frequency adjustments. ALiBi's linear bias is a simple function — m·|i-j| — that naturally extends to any distance. At position 10000, the penalty is just 10000·m, which the softmax handles naturally. This makes ALiBi the simplest approach for length generalization, though RoPE with scaling tends to produce better quality for moderate extensions.

---

## 2.7 KV-Cache: Making Inference Fast

During autoregressive generation, we generate one token at a time. Naively, each new token requires recomputing attention over all previous tokens.

**Without KV-Cache (naive):**

```
Step 1: Compute attention for [The]                    → output token: "cat"
Step 2: Compute attention for [The, cat]               → output token: "sat"
Step 3: Compute attention for [The, cat, sat]          → output token: "on"
Step 4: Compute attention for [The, cat, sat, on]      → output token: "the"

Each step recomputes K and V for ALL previous tokens. O(T²) total.
```

**With KV-Cache:**

```
Step 1: Compute K₁,V₁ for [The], store in cache        → "cat"
Step 2: Compute K₂,V₂ for [cat], append to cache       → "sat"
         Cache: [K₁,K₂], [V₁,V₂]
Step 3: Compute K₃,V₃ for [sat], append to cache       → "on"
         Cache: [K₁,K₂,K₃], [V₁,V₂,V₃]

Only compute Q for the NEW token, reuse cached K,V. O(T) per step.
```

**Memory cost formula:** For each layer, we store K and V tensors of shape (batch, kv_heads, seq_len, d_head):

```
KV-cache = 2 × n_layers × n_kv_heads × seq_len × d_head × bytes_per_param
```

The factor of 2 is for K and V. Let's compute this for a 70B model (80 layers, 64 KV heads, d_head=128) with 32K context in FP16 (2 bytes):

```
= 2 × 80 × 64 × 32768 × 128 × 2 bytes
= ~86 GB per sequence!
```

For a batch of 8 concurrent users, that’s **~688 GB** of KV-cache alone — far exceeding GPU memory. This is why the KV-cache, not model weights, becomes the serving bottleneck for long-context models. The KV-cache determines:

- **Maximum context length:** Limited by available GPU memory after model weights
- **Maximum batch size:** Each concurrent request adds its own KV-cache
- **Throughput:** GPU memory spent on KV-cache can't be used for batching more requests

This is why KV-cache optimization (GQA, MQA, quantization, PagedAttention) is critical for serving.

---

## 2.8 Multi-Query & Grouped-Query Attention

Standard multi-head attention has separate K and V projections per head, making the KV-cache enormous. Modern architectures reduce this:

```
Multi-Head Attention (MHA):        Grouped-Query (GQA):          Multi-Query (MQA):
  H separate K,V per head           K,V shared within groups      1 K,V for all heads

  Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈        Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈      Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈
  │  │  │  │  │  │  │  │          │  │  │  │  │  │  │  │        │  │  │  │  │  │  │  │
  K₁ K₂ K₃ K₄ K₅ K₆ K₇ K₈        K₁ K₁ K₂ K₂ K₃ K₃ K₄ K₄      K₁ K₁ K₁ K₁ K₁ K₁ K₁ K₁
  V₁ V₂ V₃ V₄ V₅ V₆ V₇ V₈        V₁ V₁ V₂ V₂ V₃ V₃ V₄ V₄      V₁ V₁ V₁ V₁ V₁ V₁ V₁ V₁

  KV-cache: 8× per layer          KV-cache: 4× per layer        KV-cache: 1× per layer
  (original, full quality)         (LLaMA 2 70B, Mistral)        (PaLM, Falcon)
```

**GQA** is the modern default — it achieves nearly the same quality as MHA with significantly lower KV-cache memory, enabling longer contexts and higher throughput.

---

## 2.9 The Complete Forward Pass (End-to-End)

```
Input text: "The cat sat"

Step 1: TOKENIZATION
  "The cat sat" → [464, 3857, 3332]           (token IDs)

Step 2: TOKEN EMBEDDING + POSITIONAL ENCODING
  [464, 3857, 3332] → lookup in embedding table → 3 vectors of size d_model
  Apply RoPE or add positional embeddings

Step 3: PASS THROUGH N TRANSFORMER BLOCKS
  For each block (layer):
    a) Layer Norm
    b) Multi-Head Causal Self-Attention (with KV-cache during inference)
    c) Residual Add
    d) Layer Norm
    e) FFN (SwiGLU)
    f) Residual Add

Step 4: FINAL LAYER NORM

Step 5: OUTPUT PROJECTION (LM HEAD)
  final_hidden_state → multiply by embedding matrix (or separate output matrix)
  → logits vector of size |vocabulary| (e.g., 32000 for LLaMA)

Step 6: SAMPLING
  logits → softmax → probability distribution over vocabulary
  → sample next token (greedy, top-k, top-p, temperature)

  Selected token: 319 ("on")

Step 7: APPEND AND REPEAT
  New input: [464, 3857, 3332, 319] → "The cat sat on"
  (With KV-cache, only process token 319 through the model)
```

---

## 2.10 Decoding Strategies

```
logits = [2.1, 0.5, 1.8, -0.3, 3.2, ...]  (one per vocab token)
probs  = softmax(logits / temperature)

Temperature controls randomness:
  T < 1.0: sharper distribution (more deterministic)
  T = 1.0: original distribution
  T > 1.0: flatter distribution (more random)
```

| Strategy            | Description                                                         | When to Use                  |
| ------------------- | ------------------------------------------------------------------- | ---------------------------- |
| **Greedy**          | Always pick highest-probability token                               | Code generation, factual QA  |
| **Top-k**           | Sample from top k highest-probability tokens                        | Creative text (k=40-100)     |
| **Top-p (nucleus)** | Sample from smallest set of tokens whose cumulative probability ≥ p | General-purpose (p=0.9-0.95) |
| **Temperature**     | Scale logits by 1/T before softmax                                  | Combine with top-k/top-p     |
| **Beam search**     | Maintain b best partial sequences                                   | Translation, summarization   |

```
Vocabulary: [the, a, cat, dog, sat, ran, on, ...]
Probabilities after softmax:

Greedy (T=0):     the=0.45, cat=0.20, a=0.15, sat=0.10, ...  → always picks "the"
Top-k (k=3):      the=0.45, cat=0.20, a=0.15  → samples from these 3
Top-p (p=0.8):    the=0.45, cat=0.20, a=0.15  → cumsum: 0.45, 0.65, 0.80 ≥ 0.8 → sample from 3
High temp (T=2):  the=0.22, cat=0.18, a=0.16, sat=0.14, ... → flatter, more random
```

---

## 2.11 Model Dimensions — Concrete Numbers

| Model       | d_model | Heads | Layers | FFN dim | Vocab | Params |
| ----------- | ------- | ----- | ------ | ------- | ----- | ------ |
| GPT-2 Small | 768     | 12    | 12     | 3072    | 50257 | 117M   |
| GPT-2 XL    | 1600    | 25    | 48     | 6400    | 50257 | 1.5B   |
| LLaMA 7B    | 4096    | 32    | 32     | 11008   | 32000 | 6.7B   |
| LLaMA 70B   | 8192    | 64    | 80     | 28672   | 32000 | 65B    |
| GPT-3 175B  | 12288   | 96    | 96     | 49152   | 50257 | 175B   |

**Parameter count formula (decoder-only):**

```
Attention:  4 × d_model² per layer  (W_Q, W_K, W_V, W_O)
FFN:        ~2.67 × 4 × d_model² per layer (SwiGLU has 3 matrices)
Embeddings: vocab_size × d_model

Total ≈ N_layers × (4d² + 8/3 × 4d²) + vocab × d
      ≈ N_layers × ~15d²  (rough approximation)
```

---

## 2.12 Flash Attention

Standard attention requires O(N²) memory to store the full attention matrix. For N=32K tokens, this is ~4 GB per layer in FP32.

**Flash Attention** (Dao et al., 2022) computes exact attention without materializing the full N×N matrix:

```
Standard Attention:                    Flash Attention:
┌──────────────┐                      Process in tiles, never store
│              │                      full matrix in HBM (GPU RAM)
│   N × N      │  Stored in GPU
│   attention  │  memory (HBM)        ┌───┐
│   matrix     │                      │ █ │ ← compute tile in SRAM
│              │                      │   │   (fast on-chip memory)
└──────────────┘                      └───┘

Memory: O(N²)                         Memory: O(N)
IO-bound on GPU                       Compute-bound (faster!)
```

**Key idea — the memory hierarchy insight:**

Modern GPUs have two levels of memory:

- **SRAM** (on-chip, ~20MB): Very fast (~19 TB/s bandwidth), but tiny.
- **HBM** (main GPU RAM, ~80GB on A100): Large, but slow (~2 TB/s bandwidth).

Standard attention computes the full N×N matrix and writes it to HBM, then reads it back for the softmax and V multiplication. The bottleneck is not compute — it's _reading and writing_ this huge matrix to slow HBM memory.

Flash Attention restructures the computation into **tiles** that fit entirely in SRAM:

1. Divide Q into blocks of rows, K and V into blocks of columns
2. For each block pair, compute that tile of attention scores in SRAM
3. Use the **online softmax trick** (Milakov & Gimelshein, 2018): maintain running max and sum statistics so the softmax can be computed incrementally without needing the full row of scores
4. Accumulate the weighted V contributions directly in SRAM
5. Write only the final output to HBM — never the N×N attention matrix

Because the N×N matrix is never materialized in HBM, memory drops from O(N²) to O(N). And because the algorithm is now compute-bound (doing useful FLOPs) rather than memory-bound (waiting for HBM reads/writes), it runs 2-4× faster despite doing the same amount of arithmetic.

**Impact:** Flash Attention is now the default in PyTorch 2.0+ (via `torch.nn.functional.scaled_dot_product_attention`), all major LLM frameworks, and has enabled context windows of 100K+ tokens that would be impossible with standard attention.

---

## Interview Questions

### Conceptual

1. **Walk me through the full forward pass of a decoder-only transformer, from input text to output probability distribution.**
   <details>
   <summary>Answer</summary>
   1) Tokenize input text into token IDs. 2) Look up token embeddings from embedding table (d_model-dimensional vectors). 3) Apply positional encoding (e.g., RoPE rotations to Q,K in each layer). 4) Pass through N transformer blocks, each consisting of: LayerNorm → Multi-Head Causal Self-Attention → Residual Add → LayerNorm → FFN (SwiGLU) → Residual Add. 5) Apply final LayerNorm. 6) Project final hidden states through the output/unembedding matrix to get logits of size |vocabulary|. 7) Apply softmax to get a probability distribution. 8) Sample from this distribution using a decoding strategy (greedy, top-p, etc.) to select the next token.
   </details>

2. **Why do we scale the dot product by √dₖ in attention? What would happen without it?**
   <details>
   <summary>Answer</summary>
   When Q and K vectors have dimension dₖ, the dot product q·k is the sum of dₖ terms. If the components of q and k have zero mean and unit variance, the dot product has variance dₖ. For large dₖ (e.g., 128), the dot products become large in magnitude, pushing the softmax into saturated regions where gradients are nearly zero. Dividing by √dₖ normalizes the variance of the dot products back to ~1, keeping the softmax in a regime where gradients flow well and the attention distribution is not too peaked.
   </details>

3. **Explain the difference between MHA, GQA, and MQA. Why has GQA become the standard?**
   <details>
   <summary>Answer</summary>
   MHA: Each attention head has its own Q, K, V projections (h sets of K,V). GQA: Q heads are grouped, and each group shares one set of K,V projections (g groups where 1 < g < h). MQA: All heads share a single K,V projection (g=1). GQA became standard because: (1) KV-cache memory during inference scales linearly with the number of KV heads — GQA with g=8 uses 1/4 the KV-cache of MHA with h=32; (2) GQA matches MHA quality much more closely than MQA; (3) the memory savings enable serving longer contexts and higher batch sizes, which is crucial for production deployment.
   </details>

4. **What is Flash Attention and why does it matter?**
   <details>
   <summary>Answer</summary>
   Flash Attention is an IO-aware exact attention algorithm that avoids materializing the full N×N attention matrix in GPU HBM (main memory). Instead, it computes attention in blocks that fit in SRAM (fast on-chip memory), using an online softmax trick to accumulate results without needing the full matrix. Key benefits: (1) Reduces memory from O(N²) to O(N), enabling much longer sequences; (2) 2-4× faster by being compute-bound instead of memory-bound; (3) Computes exact attention (not an approximation). It's become a standard component in all modern LLM training and inference frameworks.
   </details>

5. **Compare RoPE, sinusoidal positional encodings, and ALiBi. When would you choose each?**
   <details>
   <summary>Answer</summary>
   Sinusoidal: Fixed, additive encodings from the original Transformer. Simple but don't extrapolate well beyond training length. Suitable for fixed-length tasks. RoPE: Applies rotations to Q and K vectors, encoding relative position through the angle of rotation. Naturally captures relative positions, can be extended beyond training length (with NTK-aware scaling or YaRN), and adds no parameters. The modern default (LLaMA, Mistral). ALiBi: Adds a linear bias to attention scores based on distance between positions. Very simple, good length extrapolation without any fine-tuning, but slightly less expressive. Used in BLOOM. Choose RoPE for general-purpose LLMs (best quality-flexibility tradeoff), ALiBi for applications requiring strong length extrapolation with minimal complexity.
   </details>

6. **What is the KV-cache and why is it the bottleneck for LLM serving?**
   <details>
   <summary>Answer</summary>
   During autoregressive generation, each new token needs to attend to all previous tokens. The KV-cache stores the key and value projections for all past tokens across all layers so they don't need to be recomputed. The KV-cache is the serving bottleneck because: (1) Its size scales as O(batch_size × num_layers × num_kv_heads × seq_len × d_head), which can be tens of GBs for long-context models; (2) It consumes GPU memory that could otherwise be used for larger batches; (3) It determines the maximum context length at inference time. Mitigations: GQA/MQA (fewer KV heads), KV-cache quantization, paged attention (vLLM), and sliding window attention.
   </details>

### Coding

7. **Implement scaled dot-product attention from scratch in PyTorch.**
   <details>
   <summary>Solution</summary>

   ```python
   import torch
   import torch.nn.functional as F
   import math

   def scaled_dot_product_attention(Q, K, V, mask=None):
       """
       Args:
           Q: (batch, heads, seq_len_q, d_k)
           K: (batch, heads, seq_len_k, d_k)
           V: (batch, heads, seq_len_k, d_v)
           mask: (seq_len_q, seq_len_k) boolean, True = masked positions
       Returns:
           output: (batch, heads, seq_len_q, d_v)
           weights: (batch, heads, seq_len_q, seq_len_k)
       """
       d_k = Q.size(-1)
       scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

       if mask is not None:
           scores = scores.masked_fill(mask, float('-inf'))

       weights = F.softmax(scores, dim=-1)
       output = torch.matmul(weights, V)
       return output, weights

   # Causal mask for decoder
   def causal_mask(seq_len):
       return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
   ```

   </details>

8. **Implement a complete multi-head attention module with causal masking.**
   <details>
   <summary>Solution</summary>

   ```python
   import torch
   import torch.nn as nn
   import math

   class MultiHeadAttention(nn.Module):
       def __init__(self, d_model, num_heads):
           super().__init__()
           assert d_model % num_heads == 0
           self.d_model = d_model
           self.num_heads = num_heads
           self.d_k = d_model // num_heads

           self.W_q = nn.Linear(d_model, d_model, bias=False)
           self.W_k = nn.Linear(d_model, d_model, bias=False)
           self.W_v = nn.Linear(d_model, d_model, bias=False)
           self.W_o = nn.Linear(d_model, d_model, bias=False)

       def forward(self, x, mask=None):
           batch, seq_len, _ = x.shape

           # Project to Q, K, V
           Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
           K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
           V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
           # Shape: (batch, heads, seq_len, d_k)

           # Scaled dot-product attention
           scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

           # Causal mask
           if mask is None:
               mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
           scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

           weights = torch.softmax(scores, dim=-1)
           attn_output = torch.matmul(weights, V)

           # Concatenate heads
           attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
           return self.W_o(attn_output)
   ```

   </details>

9. **Given a pre-trained transformer, implement KV-cache for efficient autoregressive generation.**
   <details>
   <summary>Solution</summary>

   ```python
   class KVCache:
       def __init__(self):
           self.key_cache = {}    # layer_idx -> (batch, heads, cached_len, d_k)
           self.value_cache = {}

       def update(self, layer_idx, new_keys, new_values):
           if layer_idx in self.key_cache:
               self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], new_keys], dim=2)
               self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], new_values], dim=2)
           else:
               self.key_cache[layer_idx] = new_keys
               self.value_cache[layer_idx] = new_values
           return self.key_cache[layer_idx], self.value_cache[layer_idx]

   def generate_with_kv_cache(model, input_ids, max_new_tokens, temperature=1.0):
       cache = KVCache()
       generated = input_ids  # (batch, seq_len)

       for step in range(max_new_tokens):
           if step == 0:
               # First step: process full prompt
               logits = model.forward(generated, kv_cache=cache)
           else:
               # Subsequent steps: only process last token
               logits = model.forward(generated[:, -1:], kv_cache=cache)

           # Sample next token
           next_token_logits = logits[:, -1, :] / temperature
           probs = torch.softmax(next_token_logits, dim=-1)
           next_token = torch.multinomial(probs, num_samples=1)

           generated = torch.cat([generated, next_token], dim=1)

           # Check for EOS
           if next_token.item() == model.eos_token_id:
               break

       return generated
   ```

   </details>

### System Design

10. **You need to serve a 70B parameter model with 32K context length. The KV-cache alone would require ~160GB. How do you make this feasible?**
    <details>
    <summary>Answer</summary>
    Multiple complementary strategies: (1) **GQA**: The model likely uses GQA (8 KV heads instead of 64), reducing KV-cache by 8×. (2) **KV-cache quantization**: Quantize cached K,V to INT8 or INT4, reducing by 2-4×. (3) **Paged Attention (vLLM)**: Manage KV-cache like virtual memory pages — allocate on demand, avoid fragmentation, share pages across requests with common prefixes. (4) **Tensor parallelism**: Shard the model and KV-cache across multiple GPUs. (5) **Sliding window attention** (if architecture supports it, e.g., Mistral): Only cache last W tokens per layer. (6) **Model quantization**: Run model weights in INT4 (GPTQ, AWQ) to free GPU memory for KV-cache. Combined, these can reduce requirements from 160GB to ~5-10GB, fitting on 2-4 A100 GPUs.
    </details>

---

## Key Papers

- Vaswani et al. (2017) — "Attention Is All You Need" (the original Transformer)
- Radford et al. (2018, 2019) — GPT, GPT-2 (decoder-only transformers for language modeling)
- Su et al. (2021) — "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE)
- Shazeer (2019) — "Fast Transformer Decoding: One Write-Head is All You Need" (MQA)
- Ainslie et al. (2023) — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
- Dao et al. (2022) — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- Shazeer (2020) — "GLU Variants Improve Transformer" (SwiGLU)
