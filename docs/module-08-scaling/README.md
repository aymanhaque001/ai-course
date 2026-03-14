# Module 8: Scaling Laws & Efficient Inference

> **Prerequisites:** Modules 1–7
> **Estimated Study Time:** 8–10 hours

---

## 8.1 Neural Scaling Laws

Scaling laws describe the predictable relationship between compute, data, model size, and loss. Understanding them is essential for making billion-dollar training decisions.

### The Power-Law Relationship

```
Loss ∝ N^(-αₙ)    (model size)
Loss ∝ D^(-αd)    (dataset size)
Loss ∝ C^(-αc)    (compute budget)

Where αₙ ≈ 0.076, αd ≈ 0.095, αc ≈ 0.057 (Kaplan et al. 2020)

Log-log plot:

log(Loss)
    │  ●
    │   ●
    │    ●
    │      ●
    │         ●
    │               ●
    └─────────────────────── log(N or D or C)

A straight line on a log-log plot = power law
```

### Kaplan et al. (2020): Original Scaling Laws

Key findings:

1. **Smooth power laws** hold over many orders of magnitude of model size, dataset size, and compute
2. **Compute-optimal training:** for a fixed compute budget C, the optimal is N ∝ C^0.73 and D ∝ C^0.27
3. **Model size > dataset size:** should preferentially scale model parameters for compute-optimal training

```
Kaplan Optimal Allocation (compute C FLOPs):
  N* = (C / 6)^0.73     (# parameters)
  D* = (C / 6)^0.27     (# tokens)

GPT-3 (175B) trained on only ~300B tokens — was it compute-optimal?
By Kaplan: optimal N for 3.14×10²³ FLOPs ≈ 175B, optimal D ≈ 300B
Appears compute-optimal by Kaplan, but...
```

### Chinchilla Scaling Laws (Hoffmann et al. 2022)

**Revolutionary finding:** Kaplan was wrong about the N:D ratio. Models should be trained with more data on smaller models.

```
Chinchilla Law:
  For compute-optimal training:
    N* ∝ C^0.50    (# parameters)
    D* ∝ C^0.50    (# tokens)

  Optimal tokens ≈ 20 × parameters

  N* = (C / 6)^(1/2)

┌──────────────────────────────────────────────────────────────────┐
│          KAPLAN vs CHINCHILLA OPTIMAL TRAINING                    │
│                                                                   │
│  GPT-3 (Kaplan-era):  175B params × 300B tokens                  │
│  Chinchilla-optimal:  70B params × 1.4T tokens = same compute    │
│                                                                   │
│  Chinchilla result: 70B model outperforms GPT-3 on almost        │
│  every benchmark despite using the same compute budget!           │
│  → The field was undertrained, not underparameterized.           │
└──────────────────────────────────────────────────────────────────┘
```

### The Chinchilla Optimal Frontier

```
  Model Size (Params)
         │
    500B │                          ← overparameterized (Kaplan-era models)
         │                     ·
    175B │                 ·
     70B │             ·
         │         ·
     13B │     ·
      7B │   · ← LLaMA 2 7B (60% closer to Chinchilla-optimal than GPT-3)
         │·
         └──────────────────────────────────────────────── Training Tokens
             300B  500B  1T   2T   5T   10T

Chinchilla frontier: diagonal line where N × 20 = D

Models above the line: overtrained parameters (too big, not enough data)
Models below the line: undertrained on tokens (too small parameters)

LLaMA 3 8B trained on 15T tokens = 1875 tokens/parameter
→ Massively below Chinchilla-optimal at training time, but:
  → Very efficient at inference (small model)
  → MUCH better at deployment cost than 70B
```

### Inference-Adjusted Scaling

Chinchilla optimal is best for a one-time training run. But for **deployed models**, inference cost matters:

```
Deployment scenario:
  - Once-trained model serves 1 billion requests
  - Inference cost = cost_per_token × tokens_per_request × num_requests
  - For a 70B model vs 7B model: inference is ~10× more expensive per request

If serving 1B requests:
  Train more on a smaller model → lower training compute, much lower inference compute

This is why LLaMA 3 8B was trained on 15T tokens instead of ~160B (Chinchilla-optimal):
  The inference savings across deployment lifetime overwhelm the extra training compute.

The Conceptual Leap Beyond Chinchilla:
  Chinchilla optimizes: min L(N, D)  subject to  C_train = 6·N·D
  It asks: "Given a training budget, what's the best model?"

  Inference-adjusted scaling optimizes TOTAL cost across the model's lifetime:
    C_total = C_train + N_queries × C_infer(M)

  Where:
    C_train  = one-time cost to train the model (6·N·D FLOPs)
    N_queries = total inference requests over the model's deployment lifetime
    C_infer(M) = per-query cost, proportional to model size M

  The key insight: in production, inference dominates.
    If N_queries = 10⁹ and C_infer(70B) = 10× C_infer(7B):
      70B: C_total = C_train + 10⁹ × 10x = dominated by inference
       7B: C_total = C_train' + 10⁹ × 1x = much cheaper total
      (even if C_train' > C_train due to longer training on more tokens)

  This shifts the optimal point toward SMALLER, BETTER-TRAINED models:
    Chinchilla-optimal 70B (20 tok/param) → Inference-optimal 8B (1875 tok/param)
    The extra training tokens are a one-time cost; the inference savings compound
    across every single query for the lifetime of the model.

  This explains the industry trend: LLaMA 3, Mistral, Gemma all train small
  models far beyond Chinchilla-optimal ratios for deployment efficiency.
```

---

## 8.2 Emergent Abilities

Scaling produces unexpected, non-linear capability jumps — "emergent abilities" appear at certain model sizes.

```
Performance
    │
100%│                                        ●─────────────── Model C
    │                                   ●───
    │                              ●───
    │                         ●───
 50%│                    ●───                                Model B
    │               ●───
    │          ●───
    │ ●●●●●●●●                                              Model A (below threshold)
  0%│
    └────────────────────────────────────────────────────── Model Size (log scale)
      7B        13B       70B       175B

Model A: near-random performance (task too hard for small model)
Model B: performance suddenly jumps at ~13B parameters
Model C: near-perfect at ~70B
```

**Examples of emergent abilities:**

- **Arithmetic (3-digit):** emerges around 13B parameters
- **Instruction following:** qualitatively better at ~50B+
- **Chain-of-thought reasoning:** improves dramatically past ~100B
- **In-context learning with many shots:** scales well above 100B

**Debate:** Schaeffer et al. (2023) argue many "emergent" abilities are artifacts of using non-smooth metrics (discontinuous accuracy vs smooth cross-entropy). When measured in bits-per-byte, capabilities scale smoothly. True emergent behaviors may be rarer than claimed.

---

## 8.3 Efficient Inference: The Problem

Training a model is a one-time cost. Inference is ongoing and must be optimized:

```
A100 GPU:
  FP16 matrix ops:  312 TFLOPS
  Memory bandwidth: 2 TB/s

LLaMA 3 8B inference:
  Parameters: 8B × 2 bytes (FP16) = 16 GB
  Per-token computation: ~16 GFLOP (each token needs to multiply through ~8B params)

  Arithmetic intensity = FLOPs / Bytes = 16G / 16G = 1 FLOP/byte
  A100 ridge point = 312T / 2T = ~156 FLOP/byte (the break-even)

  Since 1 << 156, LLM generation is MEMORY BANDWIDTH BOUND.
  The bottleneck is moving weights from HBM to compute cores, not computation itself.
```

### The KV Cache

Without the KV Cache, every new token would require recomputing all previous token keys and values:

```
Without KV Cache (O(n²) per sequence):
  Token 1: compute K₁, V₁
  Token 2: compute K₁, V₁, K₂, V₂  ← recompute K₁, V₁!
  Token 3: compute K₁, V₁, K₂, V₂, K₃, V₃  ← re-recompute!

With KV Cache (O(n) per sequence):
  Token 1: compute K₁, V₁  → store in cache
  Token 2: compute K₂, V₂ → store; load K₁, V₁ from cache
  Token 3: compute K₃, V₃ → store; load K₁, V₁, K₂, V₂ from cache

KV Cache memory usage:
  = 2 × num_layers × num_heads × head_dim × seq_len × bytes_per_element
  LLaMA 3 8B, seq_len=4096, FP16:
  = 2 × 32 × 32 × 128 × 4096 × 2 = 2 GB per request
  → KV cache is a major memory bottleneck for serving
```

---

## 8.4 Quantization

Reduce precision to decrease memory and increase speed:

```
FP32 → FP16 → BF16 → FP8 → INT8 → INT4

Weight memory:
  FP32: 4 bytes/param  → 7B model = 28 GB
  FP16: 2 bytes/param  → 7B model = 14 GB
  INT8: 1 byte/param   → 7B model = 7 GB
  INT4: 0.5 byte/param → 7B model = 3.5 GB  (fits on a Mac with 8GB VRAM!)

Quantization methods:
┌────────────────────────────────────────────────────────────────┐
│  Post-Training Quantization (PTQ):                             │
│    Quantize after training, no retraining required             │
│    GPTQ: layer-wise quantization minimizing reconstruction error│
│    AWQ: protect salient weights from quantization error        │
│    GGUF (llama.cpp): mixed precision per layer                 │
│                                                                │
│  Quantization-Aware Training (QAT):                           │
│    Simulate quantization during training                       │
│    Better quality than PTQ, requires retraining               │
│    Used in: BitNet b1.58 (1.58-bit weights!)                  │
└────────────────────────────────────────────────────────────────┘

AWQ vs GPTQ — Why They Differ:

  GPTQ (Frantar et al., 2022):
    Quantizes weights column-by-column within each layer, using second-order
    (Hessian) information: H = 2·Xᵀ·X from calibration data.
    Key idea: after quantizing column j, propagate the rounding error to
    remaining columns using the inverse Hessian (Cholesky decomposition).
    Minimizes ||WX - W_qX||² (layer output reconstruction error).
    Strength: mathematically optimal error compensation per layer.

  AWQ (Lin et al., 2023):
    Key observation: ~1% of weights are disproportionately important, but
    importance is determined by ACTIVATION magnitude, not weight magnitude.
    A small weight w multiplied by a large activation x contributes more
    to the output (w·x) than a large weight multiplied by a tiny activation.

    Method: identify salient weight channels (those multiplied by large
    activations in calibration data), then apply per-channel scaling to
    protect them before quantization:
      w' = w · s,  x' = x / s   (output unchanged: w'·x' = w·x)
    Choosing s > 1 for salient channels reduces their relative quantization
    error at the cost of slightly more error on non-salient channels.

    Result: AWQ typically preserves quality better than GPTQ at the same
    bit-width because it focuses protection where it actually matters
    (activation-weighted importance), not just where weight magnitudes are large.
```

### Quantization Error

```
Original weight:  w = 3.14159
INT8 quantized:   w_q = round(3.14159 / scale) × scale ≈ 3.14

Scale = (max_val - min_val) / 255

Error is small for most weights but catastrophic for outlier weights.
AWQ and GPTQ handle outliers specially to preserve accuracy.
```

---

## 8.5 Efficient Attention

Attention is O(n²) in sequence length — a bottleneck for long contexts.

### Flash Attention

Flash Attention (Dao et al., 2022) computes exact attention but restructures computation to minimize HBM read/writes:

```
Standard attention:
  1. Load Q, K from HBM → compute S = QKᵀ → write S to HBM (n² mem)
  2. Load S from HBM → compute A = softmax(S) → write A to HBM
  3. Load A, V from HBM → compute O = AV → write O to HBM

Flash Attention:
  - Process in tiles that fit in SRAM (L1 cache, 20MB on A100)
  - Never write full n×n attention matrix to HBM
  - 2-4× speedup, O(n) memory instead of O(n²)
  - Exact same output (not approximate)

Flash Attention 2 & 3: further optimizations, near-theoretical peak throughput
```

### Multi-Query Attention (MQA) and Grouped Query Attention (GQA)

```
Standard Multi-Head Attention (MHA):
  32 heads for Q, K, V each
  KV cache: 32 K heads + 32 V heads per layer

Multi-Query Attention (MQA):
  32 Q heads, 1 K head, 1 V head (shared)
  KV cache: 1 K head + 1 V head per layer → 32× reduction
  Slightly lower quality

Grouped Query Attention (GQA) — used in LLaMA 2/3, Mistral:
  32 Q heads, 8 K heads, 8 V heads (shared within groups of 4)
  KV cache: 8 K + 8 V per layer → 4× reduction
  Near MHA quality, most of MQA's speedup

  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐  Q heads (32)
  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
     │     │     │     │     │     │     │     │
  ┌──┴─┐ ┌─┴──┐ ┌┴──┐ ┌┴──┐ ┌┴──┐ ┌┴──┐ ┌┴──┐ ┌┴──┐  KV heads (8)
  └────┘ └────┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
     Group of 4 Q heads share one K, V head
```

---

## 8.6 Speculative Decoding

LLM generation is sequential (one token at a time), leaving GPU cores mostly idle during memory reads. Speculative decoding uses a small "draft" model to propose multiple tokens in parallel, then validates them in a single forward pass:

```
Standard Decoding:
  Big Model → token 1 → token 2 → token 3 → token 4 → token 5
  (5 sequential forward passes of big model)

Speculative Decoding:
  Step 1: Draft model generates tokens (fast):  [t1, t2, t3, t4, t5]
  Step 2: Big model validates all 5 in ONE forward pass
          Accept tᵢ if P_big(tᵢ) ≥ P_draft(tᵢ), else reject and resample
  Result: ~3-4 tokens accepted per big model forward pass

  Speedup: 2-4× for free (exact same distribution as standard decoding)
  Requirement: small and big model must share vocabulary

  Correctness Guarantee (Rejection Sampling):
  For each draft token t at position i:
    Let p_large = P_big(t), p_small = P_draft(t)

    Case 1: p_large(t) ≥ p_small(t)  →  Accept unconditionally.
            The big model is at least as likely to produce this token.

    Case 2: p_large(t) < p_small(t)  →  Accept with probability p_large(t) / p_small(t).
            Otherwise, reject and resample from corrected distribution:
              P_corrected(t) = max(0, p_large(t) - p_small(t))  (then renormalize)
            This is the distribution (p_large - p_small)⁺ / Z.

  Why this works: This is classical rejection sampling. The acceptance/rejection
  scheme provably recovers the large model's EXACT output distribution — every
  token produced has exactly the same probability as if you ran the large model
  alone. There is zero quality loss. The draft model only affects speed (higher
  acceptance rate = more tokens per big-model forward pass), never correctness.

Example: Claude uses "Haiku" draft + "Sonnet" target, Meta uses 68M draft + 70B target
```

---

## 8.7 Model Distillation

Distillation transfers knowledge from a large "teacher" model to a small "student" model:

```
Standard Training:
  Student learns from hard labels (one-hot): P(cat) = 1.0, P(dog) = 0.0

Knowledge Distillation (Hinton et al. 2015):
  Student learns from soft labels (teacher's probabilities):
    P(cat) = 0.7, P(dog) = 0.1, P(fox) = 0.08, ...

  Soft labels carry more information:
    - Teacher's "uncertainty" reveals which classes are similar
    - Richer gradient signal than hard labels

  Loss = α × CE(student, hard_labels) + (1-α) × KL(student_probs ‖ teacher_probs)
  Temperature T softens teacher's distribution: logits → logits/T before softmax
```

### Modern LLM Distillation

```
Offline Distillation:
  1. Generate large dataset with teacher model
  2. Train student on teacher-generated data (supervised)
  Examples: Alpaca (GPT-4 → LLaMA), WizardLM, Orca

Online Distillation (token-level KL):
  Student sees teacher's full softmax distribution per token
  Much richer signal; requires running teacher during student training
  Used in: Gemini Nano (distilled from Gemini Ultra)

Speculative Decoding as Distillation:
  Draft model trained to match draft distribution with target model's distribution
```

---

## 8.8 Mixture of Experts (MoE)

MoE scales model capacity without proportionally scaling computation per token:

```
Dense Model (e.g., 70B):
  Every token uses ALL 70B parameters per forward pass.
  Compute = model_size × tokens

MoE Model (e.g., Mixtral 8×7B):
  8 expert FFN blocks, each 7B parameters.
  Total parameters: 8 × 7B ≈ 47B.
  Each token is routed to TOP-2 experts only.
  Active parameters per token: 2 × 7B = 14B.

  ┌─────────────────────────────────────────────────────────┐
  │          MoE Layer                                      │
  │                                                         │
  │          Input Token                                    │
  │               │                                        │
  │               ▼                                        │
  │          [ROUTER] ──→ gates: [0.0, 0.7, 0.0, 0.0,     │
  │                               0.3, 0.0, 0.0, 0.0]     │
  │               │                                        │
  │         ┌─────┤ (top-2: experts 1 and 4)              │
  │         │     │                                        │
  │         ▼     ▼                                        │
  │    [Expert 1][Expert 4]  (others inactive)             │
  │         │     │                                        │
  │         └──+──┘ (weighted sum by router gates)         │
  │               │                                        │
  │           Output                                       │
  └─────────────────────────────────────────────────────────┘

  Advantages:
    - 70B dense model compute, but 47B capacity
    - Or: much larger capacity (120B+) at 14B active compute
  Disadvantages:
    - All 47B need to fit in GPU memory (or use tensor parallelism)
    - Load balancing: must avoid all tokens routing to same experts
    - Router training instability
```

### Load Balancing Loss

```
Without load balancing: all tokens route to expert 0 → expert collapse
Auxiliary loss: penalize uneven expert utilization

L_balance = α × num_experts × Σᵢ (fraction_i × mean_gate_i)

Where fraction_i = fraction of tokens routed to expert i
      mean_gate_i = mean softmax gate value for expert i
Target: uniform distribution across experts

The Rich-Get-Richer Problem (Expert Collapse):
  Early in training, router weights are near-random. By chance, one expert
  may perform slightly better on a batch. What happens next:
    1. Router learns to send more tokens to that expert (higher gate values)
    2. That expert receives more gradient updates and improves further
    3. Other experts receive fewer tokens → fewer gradients → stagnate
    4. Router sends even more tokens to the winning expert
    → Positive feedback loop: most experts become "dead" (unused)
    → "Expert collapse": N experts trained, but only 1-2 actually used
    → Model degenerates to a dense model with wasted parameters

  Formally, the auxiliary loss:
    L_aux = N × Σᵢ fᵢ · Pᵢ
  where:
    N = number of experts
    fᵢ = fraction of tokens routed to expert i  (discrete, non-differentiable)
    Pᵢ = average router probability for expert i (continuous, differentiable)

  If tokens are uniformly distributed: fᵢ = 1/N, Pᵢ = 1/N → L_aux = 1/N × N = 1
  If all tokens go to one expert: f₁ = 1, P₁ ≈ 1 → L_aux ≈ N (heavily penalized)

  The product fᵢ · Pᵢ is key: it's differentiable through Pᵢ while tracking
  actual routing through fᵢ, pushing the router toward uniform utilization.
  Typical α = 0.01 to avoid overwhelming the main language modeling loss.
```

---

## 8.9 Inference Serving

### Continuous Batching (PagedAttention)

```
Problem: Different requests have different lengths.
         Padding to max length wastes GPU compute.

Naive batching:
  Request A: [tok1, tok2, tok3, tok4, tok5, PAD, PAD, PAD]  (max_len=8)
  Request B: [tok1, tok2, PAD, PAD, PAD, PAD, PAD, PAD]
  → 50% of compute wasted on padding

Continuous batching (vLLM):
  Process tokens in a stream, dynamically insert/remove requests.
  No padding waste; throughput 3-5× better than naive batching.
  PagedAttention: manage KV cache like OS virtual memory (paged allocation).

PagedAttention in Detail:
  Traditional KV-cache problem:
    Pre-allocates contiguous memory for the maximum sequence length (e.g., 4096)
    for every request, even if a request only generates 50 tokens.
    → Massive internal fragmentation: most allocated memory sits unused.
    → Cannot serve as many concurrent requests as GPU memory allows.

  PagedAttention solution (Kwon et al., 2023):
    Stores KV-cache in fixed-size non-contiguous "pages" (blocks), just like
    how an OS manages virtual memory with page tables.

    ┌──────────┐     Page Table        Physical Pages (GPU HBM)
    │ Request A│ ──→ [0→P5, 1→P2, 2→P9]  ──→  scattered in memory
    │ (3 pages)│                               no contiguous requirement
    ├──────────┤
    │ Request B│ ──→ [0→P1, 1→P7]       ──→  only allocates what's needed
    │ (2 pages)│
    └──────────┘

    Benefits:
    - Eliminates internal fragmentation: allocate pages on demand as sequence grows
    - Eliminates external fragmentation: pages need not be contiguous
    - Memory sharing: parallel sequences (beam search, parallel sampling) can
      share physical pages for their common prefix via copy-on-write
    - Near-optimal memory utilization: waste < 4% vs ~60-80% with pre-allocation
    - Enables 2-4× more concurrent requests on the same GPU
```

### Tensor Parallelism for Serving

```
Single-node (8× A100 80GB):
  Shard the model across 8 GPUs using tensor parallelism.
  Each GPU holds 1/8 of each weight matrix.
  All-reduce communication after each matmul.

Multi-node (32× A100):
  Pipeline parallelism: each node holds a different set of layers.
  Micro-batching to hide pipeline bubble.

Prefill vs decode disaggregation:
  Prefill (processing input): compute-intensive (all input tokens at once)
  Decode (generating): memory-bandwidth-intensive (one token per step)
  → Route large prefills to high-compute nodes, decodes to high-bandwidth nodes
```

---

## 8.10 The Scaling Horizon: What's Next?

```
┌──────────────────────────────────────────────────────────────────┐
│               DIRECTIONS IN SCALING (2025–2026+)                  │
│                                                                   │
│  Test-Time Compute Scaling (o1/o3/R1 paradigm)                   │
│    More thinking at inference, not just training                  │
│    Chain-of-thought, tree search, self-reflection                 │
│    Trade inference compute for better answers on hard problems    │
│                                                                   │
│  Multimodal Scaling                                              │
│    Text + image + video + audio + code in one model              │
│    GPT-4o, Gemini 2.0, Claude 3.5 all multimodal                │
│                                                                   │
│  Long-Context Scaling                                            │
│    1M–10M token context windows                                  │
│    Challenges: attention is O(n²), lost-in-the-middle           │
│                                                                   │
│  Agent Scaling                                                   │
│    Models that reason over many turns using tools                │
│    Accuracy compounds: 95%^100 = 0.6% (error over 100 steps)    │
│                                                                   │
│  Data Scaling Wall                                               │
│    Approaching the limit of high-quality internet text           │
│    Synthetic data, web-scale video becoming the next frontier    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Interview Questions

### Conceptual

1. **Explain the Chinchilla scaling law. How did it change how models are trained?**
   <details>
   <summary>Answer</summary>
   Chinchilla (Hoffmann et al., 2022) found that for compute-optimal training, model size and training tokens should scale equally — roughly 20 tokens per parameter. Before Chinchilla, the field followed Kaplan et al.'s scaling law which suggested models should scale faster than data, leading to large undertrained models like GPT-3 (175B parameters, only 300B tokens). Chinchilla showed that a 70B model trained on 1.4T tokens outperforms GPT-3 on almost all benchmarks, using the same compute. The impact: (1) The field shifted toward smaller but more thoroughly trained models — LLaMA 1-3, Mistral, Falcon all follow this philosophy; (2) Model families like LLaMA 3 push this further, training 8B models on 15T tokens (far beyond Chinchilla-optimal) for deployment efficiency — because inference cost over millions of requests dominates the training cost; (3) It revealed that previous models were fundamentally undertrained, not undersized.
   </details>

2. **Why is LLM generation memory-bandwidth-bound rather than compute-bound? What does this mean for optimization?**
   <details>
   <summary>Answer</summary>
   For autoregressive token generation, each new token requires reading all model weights (billions of parameters) from GPU HBM (high bandwidth memory) into compute cores. With batch_size=1, each token only requires ~2× model_size_bytes of memory reads, but only 2× model_parameters FLOPs of computation. The arithmetic intensity (FLOPs/byte) is ~1, far below the GPU's ridge point (156 for A100), meaning the GPU is waiting for memory, not computing. Optimizations that follow: (1) Quantization: reducing weights from FP16 to INT4 directly reduces memory bandwidth and fits more data in SRAM; (2) Increasing batch size: with larger batches, more tokens are computed per weight read, improving arithmetic intensity; (3) Speculative decoding: generates multiple tokens per memory read cycle; (4) Grouped query attention: reduces KV cache size, freeing memory bandwidth; (5) Flash Attention: restructures memory access patterns to reduce HBM traffic; (6) Model parallelism: split the model to increase effective memory bandwidth.
   </details>

3. **What is speculative decoding? What are its requirements and limitations?**
   <details>
   <summary>Answer</summary>
   Speculative decoding uses a small fast "draft" model to speculatively generate k tokens, then uses the large target model to verify all k tokens in one parallel forward pass (since the target model can process all k draft tokens simultaneously during verification). Accepted tokens are kept; if a draft token is rejected, we resample from the target distribution at that position. The key is that verification is much cheaper than k separate forward passes of the target model. Requirements: (1) Draft and target models must share vocabulary; (2) The draft model must have similar token distribution to the target for high acceptance rates; (3) You need both models in memory simultaneously. Limitations: (1) Latency savings only matter when the target model is memory-bandwidth-bound (batch_size=1 or small); (2) Large draft models reduce the gap enough that the parallelism benefit disappears; (3) If the draft model is too dissimilar from the target, acceptance rate drops and you do extra work for nothing; (4) Requires additional memory for the draft model.
   </details>

4. **Explain Mixture of Experts. What are the tradeoffs vs dense models?**
   <details>
   <summary>Answer</summary>
   MoE replaces each dense FFN layer with N expert FFN networks (e.g., 8 experts), and a learned router that routes each token to the top-k experts (typically k=2). Only the selected experts are computed, so active parameters ≠ total parameters. Advantages: (1) Increase model capacity (total parameters) without proportionally increasing per-token compute cost; (2) Experts can specialize — some process code, others math, others language; (3) Competitive with dense models at same training FLOPs (Mixtral 8×7B matches LLaMA 2 70B with 70B's compute). Disadvantages: (1) All experts must fit in GPU memory even if only 2 activate per token — high memory requirement; (2) Router training requires careful load balancing losses to prevent expert collapse; (3) More complex distributed serving (different experts may be on different GPUs); (4) Training instability more common than dense models; (5) Less interpretable (which expert handles what?). Best for: scenarios where training cost is fixed and inference is at scale on large GPU clusters.
   </details>

5. **What is emergent ability and why is the concept contested?**
   <details>
   <summary>Answer</summary>
   Emergent abilities are capabilities that appear suddenly and unpredictably at certain model scales, showing near-zero performance below a threshold and strong performance above it (as a function of parameters, FLOPs, etc.). Examples: 3-digit arithmetic suddenly "works" at ~7B parameters; chain-of-thought improves dramatically at ~100B+. The concept is contested because: (1) Schaeffer et al. (2023) showed that emergent abilities disappear when you use continuous metrics (bits-per-byte) instead of discontinuous ones (accuracy), and when you aggregate across multiple models of the same size; (2) The "emergence" may be an artifact of the specific metric choice — accuracy is a threshold metric that looks discontinuous even when the underlying computation is smooth; (3) However, some capabilities (like in-context learning) do seem to require a minimum level of general intelligence that only appears at scale, and are harder to dismiss as metric artifacts. The practical implication: don't rely on emergence for capability planning — use continuous metrics and smooth interpolation instead.
   </details>

### Coding

6. **Implement GPTQ-style post-training quantization (simplified, layer-wise).**
   <details>
   <summary>Solution</summary>

   ```python
   import torch
   import torch.nn as nn

   def quantize_weight(W: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
       """
       Symmetric per-row quantization of a weight matrix.
       Returns quantized weights, scale, and zero point.
       """
       min_val = W.min(dim=1, keepdim=True).values
       max_val = W.max(dim=1, keepdim=True).values
       abs_max = torch.maximum(min_val.abs(), max_val.abs())

       n_levels = 2 ** (bits - 1) - 1  # e.g., 127 for INT8
       scale = abs_max / n_levels
       scale = scale.clamp(min=1e-8)

       # Quantize and clamp
       W_q = torch.round(W / scale).clamp(-n_levels, n_levels).to(torch.int8)
       W_deq = W_q.float() * scale  # Dequantized (for error computation)

       return W_q, scale, W_deq

   def gptq_simple(
       layer: nn.Linear,
       calibration_data: torch.Tensor,  # (n_samples, seq_len, in_features)
       bits: int = 4,
   ) -> nn.Linear:
       """
       Simplified GPTQ: use Hessian information to minimize quantization error.
       Full GPTQ iterates over columns with Cholesky-updated inverse Hessian.
       This is a simplified per-row version for illustration.
       """
       W = layer.weight.data.float()  # (out_features, in_features)

       # Compute Hessian approximation: H = 2 * X^T X
       X = calibration_data.reshape(-1, calibration_data.shape[-1])  # (n, in_features)
       H = (X.T @ X).float() * 2 / X.shape[0]
       H += 1e-6 * torch.eye(H.shape[0])  # Damping for numerical stability

       # Simple greedy: quantize, compute error, compensate remaining columns
       W_q = torch.zeros_like(W)
       W_remaining = W.clone()

       for col in range(W.shape[1]):
           w_col = W_remaining[:, col]
           scale = w_col.abs().max() / (2 ** (bits - 1) - 1)
           scale = max(scale.item(), 1e-8)
           q_col = torch.round(w_col / scale).clamp(-(2**(bits-1)-1), 2**(bits-1)-1)
           q_col_deq = q_col * scale
           W_q[:, col] = q_col_deq

           # Propagate error to remaining columns (simplified Hessian correction)
           error = (w_col - q_col_deq).unsqueeze(1)  # (out, 1)
           h_col = H[col, col+1:]  # (remaining_cols,)
           h_diag = H[col, col]
           if col + 1 < W.shape[1]:
               W_remaining[:, col+1:] -= error * (h_col / h_diag).unsqueeze(0)

       new_layer = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
       new_layer.weight.data = W_q.half()
       if layer.bias is not None:
           new_layer.bias.data = layer.bias.data
       return new_layer
   ```

   </details>

7. **Calculate the memory and FLOPs for a forward pass of an LLM (given architecture parameters).**
   <details>
   <summary>Solution</summary>

   ```python
   def llm_compute_analysis(
       n_params: int,        # total parameters
       n_layers: int,        # transformer layers
       d_model: int,         # hidden dimension
       n_heads: int,         # attention heads
       d_ff: int,            # FFN intermediate dimension (usually 4 * d_model)
       seq_len: int,         # sequence length
       batch_size: int = 1,
       bytes_per_param: int = 2,  # FP16
   ) -> dict:
       """Approximate FLOPs and memory for LLM inference."""

       # Memory
       param_memory_gb = n_params * bytes_per_param / 1e9
       head_dim = d_model // n_heads
       kv_cache_gb = (2 * n_layers * n_heads * head_dim * seq_len * batch_size
                      * bytes_per_param / 1e9)

       # FLOPs per layer per token
       # Self-attention: QKV projections + attention + output projection
       attn_flops = (4 * seq_len * d_model * d_model  # QKV + O projections
                     + 2 * seq_len ** 2 * d_model)    # QK^T + AV (attention)
       # FFN: two linear layers
       ffn_flops = 2 * seq_len * d_model * d_ff

       total_flops_per_layer = attn_flops + ffn_flops
       total_flops = total_flops_per_layer * n_layers * batch_size

       return {
           "param_memory_gb": round(param_memory_gb, 2),
           "kv_cache_gb_per_request": round(kv_cache_gb / batch_size, 2),
           "total_flops_per_forward": total_flops,
           "flops_per_param": total_flops / n_params,
           "arithmetic_intensity_flop_per_byte": (
               total_flops / (param_memory_gb * 1e9 + kv_cache_gb * 1e9)
           ),
       }

   # LLaMA 3 8B
   result = llm_compute_analysis(
       n_params=8_000_000_000,
       n_layers=32,
       d_model=4096,
       n_heads=32,
       d_ff=14336,
       seq_len=4096,
   )
   print(result)
   # {'param_memory_gb': 16.0, 'kv_cache_gb_per_request': 2.0, ...}
   ```

   </details>

### System Design

8. **You need to serve a 70B LLM for 10,000 requests per second with P99 latency < 1 second. Design the infrastructure.**
   <details>
   <summary>Answer</summary>

   **Hardware estimation:** 70B in FP16 = 140GB. Minimum 2× A100 80GB per model replica with tensor parallelism. At batch_size=32 and ~50 tokens/s per request (time-to-first-token < 200ms + 800ms generation), each replica handles ~50 req/s (assuming 100 token output). For 10K RPS: ~200 model replicas = 400 A100 80GB GPUs. Use H100s to cut this ~3×: ~70 nodes of 8× H100 = ~130-140 H100s. (These are rough estimates; real workload profiling is essential.)

   **Software stack:** vLLM or TensorRT-LLM for continuous batching and PagedAttention. NVIDIA Triton Inference Server for scheduling. Load balancer (sticky routing for KV cache reuse). Use Speculative Decoding with a 7B draft model for 2-3× decode speedup — reduces GPU count proportionally.

   **Quantization tradeoff:** INT8 quantization halves memory, doubles throughput, ~1% quality loss. INT4 (GPTQ/AWQ) cuts to 35GB, 4× throughput, ~2-3% quality loss. For 10K RPS, INT8 likely sufficient and safer on quality.

   **Cost optimization:** Use spot instances with checkpointing for batch workloads. Scale replicas down during off-peak hours. Monitor and right-size: many requests are short, don't allocate for worst-case uniformly.

   **Latency optimization:** Prefill-decode disaggregation: route prefill (input processing) to compute-dense nodes, decode to bandwidth-dense (HBM) nodes. KV cache sharing for system prompt (prefix caching, saves 20-40% of prefill for shared system prompts).
   </details>

---

## Key Papers

- Kaplan et al. (2020) — "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022) — "Training Compute-Optimal Language Models" (Chinchilla)
- Wei et al. (2022) — "Emergent Abilities of Large Language Models"
- Schaeffer et al. (2023) — "Are Emergent Abilities of Large Language Models a Mirage?"
- Dao et al. (2022) — "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Dao (2023) — "FlashAttention-2: Faster Attention with Better Parallelism"
- Ainslie et al. (2023) — "GQA: Training Generalized Multi-Query Transformer Models"
- Leviathan et al. (2023) — "Fast Inference from Transformers via Speculative Decoding"
- Frantar et al. (2022) — "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- Kwon et al. (2023) — "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- Shazeer et al. (2017) — "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- Jiang et al. (2024) — "Mixtral of Experts"
