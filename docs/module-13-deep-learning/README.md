# Module 13: Deep Learning — CNNs, RNNs & the Road to Transformers

> **Prerequisites:** Module 1 (Neural Network Foundations), Module 11 (Math)  
> **Estimated Time:** 10-12 hours  
> **Relevance:** Understanding CNNs and RNNs is essential for grasping why transformers were revolutionary, and these architectures remain critical for vision encoders in multimodal LLMs

---

## 13.1 Convolutional Neural Networks (CNNs)

### The Convolution Operation

```
Input Image (5×5)           Filter/Kernel (3×3)         Feature Map (3×3)

┌──┬──┬──┬──┬──┐           ┌──┬──┬──┐                 ┌──┬──┬──┐
│1 │0 │1 │0 │1 │           │1 │0 │1 │                 │4 │3 │4 │
├──┼──┼──┼──┼──┤           ├──┼──┼──┤                 ├──┼──┼──┤
│0 │1 │0 │1 │0 │    *      │0 │1 │0 │      =          │2 │4 │3 │
├──┼──┼──┼──┼──┤           ├──┼──┼──┤                 ├──┼──┼──┤
│1 │0 │1 │0 │1 │           │1 │0 │1 │                 │4 │3 │4 │
├──┼──┼──┼──┼──┤           └──┴──┴──┘                 └──┴──┴──┘
│0 │1 │0 │1 │0 │
├──┼──┼──┼──┼──┤    Output[i,j] = Σ Input[i+m, j+n] × Kernel[m, n]
│1 │0 │1 │0 │1 │               for m,n in kernel size
└──┴──┴──┴──┴──┘

Slide the filter across the input, compute element-wise multiply + sum
```

**Key insight:** Convolutions exploit spatial locality — nearby pixels are more related than distant ones. The same filter is **shared** across all positions (weight sharing), dramatically reducing parameters.

### Why Convolutions Work for Images

```
Key Properties:

1. LOCAL CONNECTIVITY           2. WEIGHT SHARING
   Each output only depends        Same filter applied everywhere
   on a small local region         → translation invariance

   ┌─────────────┐                Filter: [edge detector]
   │ ████ │      │                Applied at position 1: detects edge
   │ ████ │      │                Applied at position 2: detects edge
   │      │      │                Same weights → same feature anywhere
   └─────────────┘
   Only look at 3×3 patch        3. HIERARCHICAL FEATURES
                                    Layer 1: edges, textures
                                    Layer 2: parts (eyes, wheels)
                                    Layer 3: objects (faces, cars)

Parameter comparison:
  Fully connected (224×224×3 → 1000): ~150 million parameters
  Convolutional (3×3 filter, 64):     ~1,728 parameters per filter!
```

### CNN Building Blocks

```
┌──────────────────────────────────────────────────────────────────┐
│                    CNN Architecture Components                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CONVOLUTION LAYER                                                │
│    Parameters: kernel_size, stride, padding, num_filters           │
│    Input: H×W×C_in → Output: H'×W'×C_out                        │
│    H' = (H - K + 2P) / S + 1                                     │
│                                                                   │
│  POOLING LAYER (reduces spatial dimensions)                       │
│    Max Pooling:   Take max value in each window                   │
│    Avg Pooling:   Take average value in each window               │
│    ┌──┬──┬──┬──┐                                                 │
│    │ 1│ 3│ 2│ 1│     Max Pool 2×2     ┌──┬──┐                    │
│    │ 4│ 6│ 5│ 2│     ──────────→      │ 6│ 5│                    │
│    │ 7│ 2│ 3│ 1│     stride=2         │ 7│ 4│                    │
│    │ 1│ 3│ 4│ 2│                      └──┴──┘                    │
│    └──┴──┴──┴──┘                                                 │
│                                                                   │
│  BATCH NORMALIZATION                                              │
│    Normalize activations within a mini-batch                      │
│    x̂ = (x - μ_batch) / √(σ²_batch + ε)                          │
│    y = γx̂ + β   (learnable scale and shift)                      │
│    Benefits: faster training, higher learning rates, regularizes  │
│                                                                   │
│  GLOBAL AVERAGE POOLING                                           │
│    Average entire feature map → single number per channel         │
│    Replaces fully connected layers at the end                     │
│    14×14×512 → 1×1×512                                           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Landmark CNN Architectures

```
LeNet-5 (1998) — Yann LeCun
  The first practical CNN. Digit recognition.
  Conv → Pool → Conv → Pool → FC → FC → Output
  ~60K parameters

AlexNet (2012) — Krizhevsky et al.
  Won ImageNet, started the deep learning revolution.
  5 conv layers + 3 FC layers, ReLU, Dropout, GPU training
  ~60M parameters

VGG-16 (2014) — Simonyan & Zisserman
  Key insight: stack many 3×3 convolutions
  Two 3×3 convs = one 5×5 receptive field but fewer params
  ~138M parameters

┌──────────────────────────────────────────────────────────────────┐
│  ResNet (2015) — He et al. — The MOST Important CNN              │
│                                                                   │
│  Problem: Very deep networks train WORSE (degradation problem)    │
│  Solution: Residual connections (skip connections)                │
│                                                                   │
│       x ──────────────────────┐                                   │
│       │                       │ (skip/residual connection)        │
│       ▼                       │                                   │
│  ┌─────────┐                  │                                   │
│  │  Conv   │                  │                                   │
│  │  BN     │                  │                                   │
│  │  ReLU   │                  │                                   │
│  │  Conv   │                  │                                   │
│  │  BN     │                  │                                   │
│  └────┬────┘                  │                                   │
│       │                       │                                   │
│       ▼                       ▼                                   │
│      (+) ──── F(x) + x  ← learn the RESIDUAL                    │
│       │                                                           │
│       ▼                                                           │
│      ReLU                                                         │
│                                                                   │
│  If F(x) = 0 is optimal, just learn zero (easy!)                 │
│  Enables training networks with 100+ layers                      │
│  ResNet-50, ResNet-101, ResNet-152                                │
│                                                                   │
│  This SAME idea is used in every transformer block!               │
│  x + Attention(x) is a residual connection                        │
└──────────────────────────────────────────────────────────────────┘

EfficientNet (2019) — Tan & Le
  Compound scaling: scale width, depth, and resolution together
  Much better accuracy/efficiency tradeoff
  EfficientNet-B0 to B7

Vision Transformer (ViT) (2020) — Dosovitskiy et al.
  Split image into patches → flatten → transformer encoder
  "An Image is Worth 16x16 Words"
  Now dominates image classification at scale

  ┌────┬────┬────┬────┐
  │ P1 │ P2 │ P3 │ P4 │     16×16 patches
  ├────┼────┼────┼────┤     → flatten each → linear projection
  │ P5 │ P6 │ P7 │ P8 │     → add position embeddings
  ├────┼────┼────┼────┤     → feed into transformer encoder
  │ P9 │P10 │P11 │P12 │     → [CLS] token for classification
  ├────┼────┼────┼────┤
  │P13 │P14 │P15 │P16 │     Pre-trained on large image datasets
  └────┴────┴────┴────┘     Used as vision encoder in multimodal LLMs
```

### Transfer Learning with CNNs

```
Transfer Learning Pipeline:

  ┌────────────────────┐       ┌────────────────────┐
  │  Pre-trained CNN   │       │   Your Task        │
  │  (ImageNet, 1M→)   │       │   (100 images)     │
  │                    │       │                    │
  │  ┌──────────────┐ │       │  ┌──────────────┐ │
  │  │ Conv layers  │ │  →→→  │  │ Conv layers  │ │  FREEZE
  │  │ (features)   │ │ copy  │  │ (features)   │ │  these
  │  └──────────────┘ │       │  └──────────────┘ │
  │  ┌──────────────┐ │       │  ┌──────────────┐ │
  │  │ FC + 1000    │ │       │  │ FC + YOUR k  │ │  TRAIN
  │  │ classes      │ │       │  │ classes      │ │  this
  │  └──────────────┘ │       │  └──────────────┘ │
  └────────────────────┘       └────────────────────┘

Strategy:
  Small dataset: Freeze all conv layers, train only new FC head
  Medium dataset: Freeze early layers, fine-tune later layers
  Large dataset: Fine-tune entire network with small learning rate

  This is conceptually the same as LLM fine-tuning!
  Pre-trained features → adapt to your specific task
```

---

## 13.2 Recurrent Neural Networks (RNNs)

### The Recurrence Concept

```
Feedforward NN:              Recurrent NN:
  Fixed-size input             Sequential input (variable length)

  x → [NN] → y               x₁ → [RNN] → h₁ → [RNN] → h₂ → ... → hₜ
                                      ↑                       ↑
                                     h₀                      hₜ₋₁
                                  (initial)            (carries memory)

RNN Equation:
  hₜ = tanh(Wₕ hₜ₋₁ + Wₓ xₜ + b)

  hₜ:    hidden state at time t (the "memory")
  xₜ:    input at time t
  Wₕ:    hidden-to-hidden weights (SHARED across all time steps)
  Wₓ:    input-to-hidden weights (SHARED across all time steps)
```

### The Vanishing Gradient Problem

```
Backpropagation Through Time (BPTT):

  ∂L/∂W = Σₜ ∂Lₜ/∂W

  For long sequences, gradients must flow through many time steps:
  ∂h₁₀₀/∂h₁ = ∂h₁₀₀/∂h₉₉ × ∂h₉₉/∂h₉₈ × ... × ∂h₂/∂h₁
             = Wₕ × Wₕ × ... × Wₕ  (100 matrix multiplications!)

  If max eigenvalue of Wₕ < 1 → gradients VANISH (→ 0)
  If max eigenvalue of Wₕ > 1 → gradients EXPLODE (→ ∞)

  Gradients │  Exploding
            │  ╱
            │ ╱
            │╱    Vanishing
            │──────────── → 0
            └───────────────── Time steps

  Result: RNNs can't learn long-range dependencies
          "The cat, which sat on the mat, ..." → forgets "cat"
```

---

## 13.3 LSTM — Long Short-Term Memory

The LSTM solves the vanishing gradient problem with a **gating mechanism**:

```
┌──────────────────────────────────────────────────────────────────┐
│                      LSTM Cell Architecture                       │
│                                                                   │
│    cₜ₋₁ ─────────(×)──────────(+)──────────── cₜ                │
│                    ↑            ↑                                  │
│              ┌─────┘      ┌────┘                                  │
│              │             │                                       │
│         ┌────┴───┐   ┌────┴───────────┐                           │
│         │ Forget │   │    Input Gate   │                           │
│         │  Gate  │   │    × tanh(new)  │                           │
│         │   fₜ   │   │   iₜ × c̃ₜ     │                           │
│         └────────┘   └────────────────┘                           │
│              ↑             ↑                                       │
│         ┌────┴─────────────┴──────────┐                           │
│         │    [hₜ₋₁, xₜ] concatenated  │                           │
│         └─────────────────┬───────────┘                           │
│                           │                                        │
│                     ┌─────┴──────┐                                │
│                     │ Output Gate│                                │
│                     │    oₜ      │                                │
│                     └─────┬──────┘                                │
│                           │                                        │
│    hₜ₋₁ ──────────────── hₜ = oₜ × tanh(cₜ)──────── hₜ         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

Gate Equations:
  fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)     ← Forget gate: what to erase
  iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)     ← Input gate: what to write
  c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)  ← Candidate new memory
  cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ        ← Updated cell state
  oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)     ← Output gate: what to reveal
  hₜ = oₜ ⊙ tanh(cₜ)               ← Hidden state output

Why it works:
  - Cell state cₜ is a "highway" — gradients flow through addition
  - Forget gate can keep gradient = 1 (no vanishing!)
  - Each gate is a learned sigmoid → learns WHEN to remember/forget
  - Cell state is like the RESIDUAL CONNECTION in transformers!
```

### GRU — Gated Recurrent Unit

```
GRU = Simplified LSTM (2 gates instead of 3):

  zₜ = σ(Wz · [hₜ₋₁, xₜ])       ← Update gate (combine forget+input)
  rₜ = σ(Wr · [hₜ₋₁, xₜ])       ← Reset gate
  h̃ₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ]) ← Candidate hidden state
  hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ ← Interpolate old and new

Comparison:
  ┌────────┬──────────────┬──────────────────────────────┐
  │        │    LSTM      │         GRU                   │
  ├────────┼──────────────┼──────────────────────────────┤
  │ Gates  │ 3 (f, i, o) │ 2 (z, r)                     │
  │ States │ h, c (both) │ h only                        │
  │ Params │ More (4 × ) │ Fewer (3 ×)                   │
  │ Speed  │ Slower       │ Faster                        │
  │ Memory │ More         │ Less                          │
  │ Perf   │ ≈ equal      │ ≈ equal (slightly faster)    │
  └────────┴──────────────┴──────────────────────────────┘

  In practice: performance difference is marginal.
  Both have been largely replaced by transformers.
```

---

## 13.4 Sequence-to-Sequence (Seq2Seq)

### Encoder-Decoder Architecture

```
The Architecture That Preceded Transformers:

  Source: "Je suis étudiant"
  Target: "I am a student"

  ┌────────────────────────┐     ┌────────────────────────────┐
  │       ENCODER          │     │         DECODER             │
  │                        │     │                             │
  │  "Je" → [LSTM] → h₁   │     │   <BOS> → [LSTM] → "I"    │
  │  "suis"→ [LSTM] → h₂  │     │   "I"   → [LSTM] → "am"   │
  │  "étudiant"→[LSTM]→ h₃│──c──│   "am"  → [LSTM] → "a"    │
  │                        │     │   "a"   → [LSTM] → "student"│
  │  h₃ = "context vector" │     │   "student"→[LSTM]→ <EOS> │
  └────────────────────────┘     └────────────────────────────┘

  c = final hidden state of encoder = compress ENTIRE input

  Problem: Bottleneck!
  All information squeezed through one fixed-size vector.
  Long sentences lose early information.

  Solution → Attention mechanism
```

### Bahdanau Attention (2014)

The **precursor to transformer attention** — and critically important for understanding why transformers work:

```
Instead of one context vector, ATTEND to all encoder states:

  Encoder states: [h₁, h₂, h₃, h₄, h₅]     (one per input token)

  At each decoder step t:
    1. SCORE each encoder state:
       eᵢ = score(sₜ₋₁, hᵢ)        ← how relevant is hᵢ?

    2. NORMALIZE scores:
       αᵢ = softmax(eᵢ)             ← attention weights

    3. WEIGHTED SUM:
       cₜ = Σ αᵢ hᵢ                ← context vector for this step

    4. DECODE:
       sₜ = LSTM(sₜ₋₁, [yₜ₋₁; cₜ])

  Visualization (translating "I am a student"):

            h₁(Je)  h₂(suis)  h₃(étudiant)
  "I"        0.8      0.1       0.1        ← attends to "Je"
  "am"       0.1      0.8       0.1        ← attends to "suis"
  "a"        0.1      0.1       0.3        ← split attention
  "student"  0.0      0.1       0.9        ← attends to "étudiant"

  Compare to Transformer Self-Attention:
    Bahdanau:     score = v · tanh(W₁sₜ₋₁ + W₂hᵢ)   (additive)
    Transformer:  score = qₜ · kᵢ / √d                (dot-product)

    SAME concept, different scoring function!
```

### Luong Attention (2015)

```
Simplified attention variants:

  Dot-product:     score = sₜᵀ hᵢ
  General:         score = sₜᵀ W hᵢ
  Concat:          score = vᵀ tanh(W[sₜ; hᵢ])

  Luong's key contribution: location of attention computation
    Global attention: attend to ALL encoder states
    Local attention:  attend to a WINDOW of encoder states

  → Dot-product attention became the basis for transformer attention
```

---

## 13.5 From Seq2Seq to Transformers — The Evolution

```
Timeline of Key Developments:

2014  Seq2Seq (Sutskever)    │ LSTM encoder-decoder
      Bahdanau Attention     │ Additive attention over encoder states
                             │
2015  Luong Attention        │ Dot-product attention variants
                             │
2017  Transformer            │ REMOVE recurrence entirely!
      "Attention Is All      │ Self-attention + cross-attention
       You Need"             │ Parallel computation (unlike RNNs)
                             │
2018  BERT                   │ Encoder-only transformer
      GPT-1                  │ Decoder-only transformer
                             │
2019  GPT-2                  │ Larger decoder-only
      T5                     │ Encoder-decoder "text-to-text"
                             │
2020  GPT-3                  │ Scaling + in-context learning

Why Transformers Won:
┌──────────────┬──────────────────────┬────────────────────────┐
│              │   RNNs/LSTMs        │   Transformers          │
├──────────────┼──────────────────────┼────────────────────────┤
│ Parallelism  │ Sequential (slow!)   │ Fully parallel ✓       │
│ Long-range   │ Degrades with dist   │ O(1) path length ✓     │
│ Training     │ Hard to parallelize  │ GPU-optimized ✓        │
│ Scaling      │ Diminishing returns  │ Power-law scaling ✓    │
│ Attention    │ Implicit (LSTM mem)  │ Explicit (interpretable)│
└──────────────┴──────────────────────┴────────────────────────┘
```

---

## 13.6 Encoder-Only Models: BERT & Family

```
BERT Architecture (2018):

  Input:  [CLS] The cat sat on the [MASK] . [SEP]
           │     │    │   │   │   │     │   │
           ▼     ▼    ▼   ▼   ▼   ▼     ▼   ▼
        ┌──────────────────────────────────────────┐
        │         Transformer Encoder              │
        │         12 layers (BERT-base)            │
        │         24 layers (BERT-large)           │
        │         Bidirectional self-attention      │
        │         (sees BOTH left AND right context)│
        └──────┬─────┬───┬──┬──┬──┬─────┬──┬──────┘
               │     │   │  │  │  │     │  │
               ▼     ▼   ▼  ▼  ▼  ▼     ▼  ▼
  Output: h_CLS h₁   h₂  h₃ h₄ h₅  h_MASK h₇ h_SEP
          │                          │
          ▼                          ▼
   [Classification]           [Predict: "mat"]
   (use CLS embedding)        (MLM objective)

Training Objectives:
  1. Masked Language Modeling (MLM):
     Randomly mask 15% of tokens, predict the originals
     "The [MASK] sat on the mat" → predict "cat"

  2. Next Sentence Prediction (NSP):
     Given two sentences, predict if B follows A
     (later found to be unnecessary — RoBERTa removed it)

Key Differences from GPT:
  BERT (encoder):     Bidirectional (sees full context) → understanding
  GPT (decoder):      Unidirectional (left-to-right only) → generation

  BERT is BETTER for:                    GPT is BETTER for:
    Text classification                    Text generation
    Named Entity Recognition (NER)         Dialogue
    Question answering (extractive)        Creative writing
    Sentence embeddings                    Code generation
    Semantic similarity                    Reasoning tasks
```

### BERT Fine-tuning

```
Pre-trained BERT → Fine-tune for your task:

  Classification:           Token-level (NER):         Sentence Pair:

  [CLS] text [SEP]         [CLS] w₁ w₂ w₃ [SEP]     [CLS] s1 [SEP] s2 [SEP]
    │                          │  │  │  │                  │
    ▼                          ▼  ▼  ▼  ▼                  ▼
  ┌────────┐              ┌──┬──┬──┬──┐              ┌────────┐
  │ Linear │              │O₁│O₂│O₃│O₄│              │ Linear │
  │+ softmax│             │PER│ORG│ O│ O│              │ 2-class│
  └────────┘              └──┴──┴──┴──┘              └────────┘
  sentiment              named entities              entailment
  (pos/neg)              (per-token labels)           (yes/no)
```

### BERT Variants

```
┌──────────────┬───────────────────────────────────────────────────┐
│ RoBERTa      │ Robustly optimized BERT:                          │
│ (2019)       │   Removed NSP, more data, longer training,        │
│              │   dynamic masking. Strictly better than BERT.      │
├──────────────┼───────────────────────────────────────────────────┤
│ ALBERT       │ A Lite BERT:                                      │
│ (2019)       │   Parameter sharing + factorized embeddings.      │
│              │   Same performance, much fewer parameters.         │
├──────────────┼───────────────────────────────────────────────────┤
│ DeBERTa      │ Disentangled attention:                           │
│ (2020)       │   Separate content and position attention.        │
│              │   Best performing BERT variant.                   │
├──────────────┼───────────────────────────────────────────────────┤
│ DistilBERT   │ Knowledge distillation from BERT:                 │
│ (2019)       │   40% smaller, 60% faster, 97% performance.      │
│              │   6 layers instead of 12.                         │
├──────────────┼───────────────────────────────────────────────────┤
│ Sentence-    │ Fine-tuned for sentence embeddings:               │
│ BERT (2019)  │   Siamese network + contrastive loss.             │
│              │   Foundation of modern embedding models.           │
└──────────────┴───────────────────────────────────────────────────┘
```

---

## 13.7 Encoder-Decoder Models: T5 & BART

### T5: Text-to-Text Transfer Transformer

```
T5's Key Insight: EVERY NLP task = text-to-text

  Translation:     "translate English to French: Hello" → "Bonjour"
  Summarization:   "summarize: [article]" → "[summary]"
  Classification:  "classify: I love this" → "positive"
  QA:              "question: What is AI? context: ..." → "AI is..."

  Architecture:

  Input tokens                        Output tokens
       │                                    │
       ▼                                    ▼
  ┌──────────┐    cross-attention     ┌──────────┐
  │ Encoder  │ ──────────────────────▶│ Decoder  │
  │ (bidir)  │    decoder attends     │ (causal) │
  │ 12 layers│    to encoder output   │ 12 layers│
  └──────────┘                        └──────────┘

  Cross-Attention:
    Q comes from DECODER
    K, V come from ENCODER
    → decoder can look at any part of the input
    → this is what's missing from decoder-only (GPT) models

  T5 sizes: Small (60M), Base (220M), Large (770M),
            XL (3B), XXL (11B)
```

### BART

```
BART = BERT-style encoder + GPT-style decoder

  Pre-training: Corrupt text → reconstruct original
  Corruption methods:
    Token masking:     "The cat sat" → "The [MASK] sat"
    Token deletion:    "The cat sat" → "The sat"
    Sentence shuffling: Reorder sentences
    Document rotation:  Start from middle
    Text infilling:    "The cat sat" → "The [MASK] sat"
                        (single mask for multiple tokens)

  Best for: Summarization, Q&A, translation
  Used in: PEGASUS (summarization), mBART (multilingual)
```

---

## 13.8 Autoencoders & Variational Autoencoders

### Autoencoder

```
Standard Autoencoder:

  Input x ──▶ [Encoder] ──▶ z (latent) ──▶ [Decoder] ──▶ x̂ ≈ x
  (784D)      (compress)    (32D)          (decompress)   (784D)

  ┌────────────────────────────────────────────────────────────┐
  │  Input    Encoder        Latent     Decoder       Output   │
  │  ████     ████            ██        ████          ████     │
  │  ████     ████            ██        ████          ████     │
  │  ████     ████            ██        ████          ████     │
  │  ████     ███             ██        ███           ████     │
  │  ████     ██              ██        ██            ████     │
  │  ████                                             ████     │
  │           compress ──→ bottleneck ──→ reconstruct          │
  └────────────────────────────────────────────────────────────┘

  Loss = ‖x - x̂‖²  (reconstruction error)

  Applications:
    Dimensionality reduction (like PCA but nonlinear)
    Denoising (train to reconstruct from corrupted input)
    Anomaly detection (high reconstruction error = anomaly)
    Feature learning
```

### Variational Autoencoder (VAE)

```
VAE = Autoencoder + Probabilistic Latent Space

  Key difference from autoencoder:
    Autoencoder: z = encoder(x)              (deterministic point)
    VAE:         z ~ N(μ(x), σ²(x))         (sample from distribution)

  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │  x → [Encoder] → μ, σ² → z = μ + σ × ε → [Decoder] → x̂ │
  │                           ↑                                │
  │                      ε ~ N(0,1)                            │
  │                  (reparameterization trick)                │
  │                                                            │
  └────────────────────────────────────────────────────────────┘

  Loss = Reconstruction + KL Divergence
       = E[‖x - x̂‖²] + D_KL(q(z|x) ‖ p(z))
         ↑                    ↑
    reconstruct well    keep latent space organized

  Why the KL term?
    Forces q(z|x) close to N(0,1) prior
    → Smooth, continuous latent space
    → Can SAMPLE new z and decode → GENERATE new data!
    → Can interpolate between data points meaningfully

  Connection to LLMs:
    VAEs influenced latent-space approaches in NLP
    The "reparameterization trick" concept appears in diffusion models
    DALL-E 1 used a discrete VAE for image tokenization
```

---

## 13.9 Generative Adversarial Networks (GANs)

```
GAN Architecture:

  Random noise z ──▶ [Generator G] ──▶ fake image
                                           │
                                     ┌─────┴─────┐
                Real image ────────▶ │Discriminator│──▶ "Real" or "Fake"?
                                     │     D       │
                                     └────────────┘

  Training: A minimax game (adversarial)

    min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]

    Generator:     tries to fool D (make fake images look real)
    Discriminator: tries to distinguish real from fake

    They improve EACH OTHER through competition!

Evolution of GANs:
  ┌──────────┬────────┬──────────────────────────────────────────┐
  │ Year     │ Model  │ Contribution                             │
  ├──────────┼────────┼──────────────────────────────────────────┤
  │ 2014     │ GAN    │ Original adversarial framework           │
  │ 2015     │ DCGAN  │ Convolutional architecture, training tips│
  │ 2017     │ WGAN   │ Wasserstein distance (stable training)   │
  │ 2018     │ProGAN  │ Progressive growing (high-res)           │
  │ 2019     │StyleGAN│ Style-based generator (photorealistic)   │
  │ 2020     │StyleGAN2│ Removed artifacts, weight demodulation  │
  │ 2021     │StyleGAN3│ Translation/rotation equivariance       │
  └──────────┴────────┴──────────────────────────────────────────┘

GAN Problems:
  Mode collapse:    Generator produces limited variety
  Training instability: Oscillations, non-convergence
  Evaluation:       Hard to measure quality (FID, IS scores)

  → Largely replaced by Diffusion Models for image generation (2022+)
  → But GAN concepts (adversarial training) still important:
    - Adversarial examples in safety
    - Discriminator as a reward model
    - Style transfer
```

---

## 13.10 Practical Implementation

<details>
<summary><strong>Complete Code: CNN, LSTM, and Attention from Scratch</strong></summary>

```python
import numpy as np

# ============================================================
# CONVOLUTION LAYER FROM SCRATCH
# ============================================================

class Conv2D:
    """2D Convolution layer (forward pass)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        # Xavier initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * scale
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        """x shape: (batch, channels, height, width)"""
        batch, C_in, H, W = x.shape
        K = self.kernel_size

        # Pad input
        if self.padding > 0:
            x = np.pad(x, ((0,0), (0,0),
                          (self.padding, self.padding),
                          (self.padding, self.padding)))

        H_out = (H + 2*self.padding - K) // self.stride + 1
        W_out = (W + 2*self.padding - K) // self.stride + 1

        output = np.zeros((batch, self.weights.shape[0], H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                patch = x[:, :, h_start:h_start+K, w_start:w_start+K]
                # (batch, C_in, K, K) × (C_out, C_in, K, K) → (batch, C_out)
                for f in range(self.weights.shape[0]):
                    output[:, f, i, j] = np.sum(
                        patch * self.weights[f], axis=(1,2,3)
                    ) + self.bias[f]

        return output

# ============================================================
# LSTM FROM SCRATCH
# ============================================================

class LSTM:
    """LSTM cell with all four gates."""

    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        scale = np.sqrt(1.0 / hidden_size)

        # Combined weights for efficiency: [forget, input, candidate, output]
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x_sequence):
        """
        x_sequence: (seq_len, batch, input_size)
        Returns: all hidden states (seq_len, batch, hidden_size)
        """
        seq_len, batch, _ = x_sequence.shape
        H = self.hidden_size

        h = np.zeros((batch, H))
        c = np.zeros((batch, H))
        all_h = []

        for t in range(seq_len):
            x = x_sequence[t]
            combined = np.concatenate([h, x], axis=1)  # (batch, H+input)

            gates = combined @ self.W.T + self.b  # (batch, 4*H)

            f = self.sigmoid(gates[:, 0:H])         # forget gate
            i = self.sigmoid(gates[:, H:2*H])        # input gate
            c_tilde = np.tanh(gates[:, 2*H:3*H])     # candidate
            o = self.sigmoid(gates[:, 3*H:4*H])      # output gate

            c = f * c + i * c_tilde                   # cell state update
            h = o * np.tanh(c)                        # hidden state

            all_h.append(h)

        return np.array(all_h)  # (seq_len, batch, H)

# ============================================================
# BAHDANAU ATTENTION FROM SCRATCH
# ============================================================

class BahdanauAttention:
    """Additive attention — the precursor to transformer attention."""

    def __init__(self, hidden_size, attention_size):
        scale = np.sqrt(1.0 / attention_size)
        self.W1 = np.random.randn(attention_size, hidden_size) * scale  # encoder
        self.W2 = np.random.randn(attention_size, hidden_size) * scale  # decoder
        self.v = np.random.randn(attention_size) * scale

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (batch, hidden_size) — current decoder state
        encoder_outputs: (seq_len, batch, hidden_size) — all encoder states
        Returns: context vector (batch, hidden_size)
        """
        seq_len = encoder_outputs.shape[0]

        # Score each encoder state
        scores = np.zeros((seq_len, decoder_hidden.shape[0]))
        for i in range(seq_len):
            # score = v · tanh(W1·encoder + W2·decoder)
            combined = np.tanh(
                encoder_outputs[i] @ self.W1.T + decoder_hidden @ self.W2.T
            )
            scores[i] = combined @ self.v  # (batch,)

        # Attention weights (softmax over encoder positions)
        scores = scores.T  # (batch, seq_len)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Weighted sum of encoder outputs
        context = np.zeros_like(decoder_hidden)
        for i in range(seq_len):
            context += attention_weights[:, i:i+1] * encoder_outputs[i]

        return context, attention_weights

# Example usage
lstm = LSTM(input_size=128, hidden_size=256)
sequence = np.random.randn(20, 4, 128)  # 20 timesteps, batch=4, dim=128
hidden_states = lstm.forward(sequence)
print(f"LSTM output shape: {hidden_states.shape}")  # (20, 4, 256)
```

</details>

---

## 13.11 Interview Questions

### Conceptual Questions

**Q1: Explain the vanishing gradient problem in RNNs and how LSTMs solve it.**

In vanilla RNNs, gradients are multiplied by the weight matrix Wh at each time step during backpropagation. Over T steps, the gradient includes $W_h^T$. If the largest eigenvalue of Wh < 1, gradients decay exponentially to zero (vanishing). LSTMs solve this with the cell state highway: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. The forget gate $f_t$ can be close to 1, allowing gradients to flow through the additive path without multiplication by Wh. This is conceptually identical to residual connections in transformers ($x + F(x)$).

**Q2: Why did transformers replace RNNs? What was the key limitation of recurrence?**

Three fundamental limitations: (1) **Sequential computation** — RNN step t depends on step t-1, preventing parallelism. A 1000-token sequence needs 1000 sequential operations. Transformers compute all positions simultaneously. (2) **Long-range dependencies** — despite LSTMs, information still degrades over very long sequences (100+ tokens). Transformers have O(1) path length between any two positions via attention. (3) **Scaling** — RNN compute is inherently sequential, so more hardware doesn't help. Transformers benefit from GPU parallelism and scale with more compute/data following power laws.

**Q3: Compare encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) architectures. When would you use each?**

**Encoder-only (BERT):** Bidirectional attention → sees full context → best for understanding tasks. Use for: classification, NER, semantic similarity, embeddings. Can't generate text. **Decoder-only (GPT):** Causal attention → left-to-right → best for generation. Use for: text generation, dialogue, code, reasoning. Can do understanding tasks via prompting but less natural. **Encoder-decoder (T5):** Encoder processes input bidirectionally, decoder generates output with cross-attention → best for sequence-to-sequence tasks. Use for: translation, summarization, structured output from complex input. In 2024+, decoder-only models dominate due to scaling advantages and flexibility.

**Q4: What is the difference between BatchNorm and LayerNorm? Why do transformers use LayerNorm?**

**BatchNorm:** Normalizes across the batch dimension — compute mean and variance for each feature across all examples in the mini-batch. **LayerNorm:** Normalizes across the feature dimension — compute mean and variance for each example across all features. Transformers use LayerNorm because: (1) Sequence lengths vary, so batch statistics are inconsistent, (2) BatchNorm doesn't work well with small batch sizes (LLM training uses gradient accumulation), (3) At inference with batch=1, BatchNorm must use running statistics which can mismatch, (4) LayerNorm is consistent between training and inference.

**Q5: Explain how a Vision Transformer (ViT) works and why it's relevant to multimodal LLMs.**

ViT: (1) Split image into fixed-size patches (e.g., 16×16), (2) Flatten each patch into a 1D vector, (3) Linear projection to embedding dimension, (4) Prepend [CLS] token, add position embeddings, (5) Feed through standard transformer encoder, (6) Use [CLS] output for classification. Relevance to multimodal LLMs: ViT (or SigLIP/CLIP variants) serves as the **vision encoder** in models like LLaVA, GPT-4V, etc. Image patches become "visual tokens" that are projected into the LLM's embedding space and processed alongside text tokens. Understanding patch-based image tokenization is essential for multimodal architecture design.

### Coding Questions

**Q6: Implement a complete seq2seq model with Bahdanau attention for sequence translation.**

```python
class Seq2SeqWithAttention:
    """Sequence-to-sequence with Bahdanau attention."""

    def __init__(self, src_vocab, tgt_vocab, embed_dim, hidden_dim):
        self.encoder = LSTM(embed_dim, hidden_dim)
        self.decoder_lstm = LSTM(embed_dim + hidden_dim, hidden_dim)
        self.attention = BahdanauAttention(hidden_dim, hidden_dim)

        # Embedding tables
        scale_src = np.sqrt(1.0 / embed_dim)
        scale_tgt = np.sqrt(1.0 / embed_dim)
        self.src_embed = np.random.randn(src_vocab, embed_dim) * scale_src
        self.tgt_embed = np.random.randn(tgt_vocab, embed_dim) * scale_tgt

        # Output projection
        self.output_proj = np.random.randn(tgt_vocab, hidden_dim) * np.sqrt(1.0/hidden_dim)

    def encode(self, src_tokens):
        """src_tokens: (seq_len, batch) integer token IDs"""
        embeddings = self.src_embed[src_tokens]  # (seq_len, batch, embed_dim)
        return self.encoder.forward(embeddings)   # (seq_len, batch, hidden_dim)

    def decode_step(self, tgt_token, decoder_hidden, encoder_outputs):
        """One step of decoding with attention."""
        # Get context from attention
        context, attn_weights = self.attention.forward(decoder_hidden, encoder_outputs)

        # Embed target token
        tgt_embed = self.tgt_embed[tgt_token]  # (batch, embed_dim)

        # Concatenate embedding + context as LSTM input
        lstm_input = np.concatenate([tgt_embed, context], axis=1)
        lstm_input = lstm_input[np.newaxis, :, :]  # (1, batch, embed+hidden)

        new_hidden = self.decoder_lstm.forward(lstm_input)[0]  # (batch, hidden)

        # Project to vocabulary
        logits = new_hidden @ self.output_proj.T  # (batch, vocab)
        return logits, new_hidden, attn_weights
```

### System Design Questions

**Q7: Design an image classification service that can handle 1000 requests per second with 99.9% availability.**

```
┌──────────────────────────────────────────────────────────────────┐
│           Image Classification Service Architecture              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. API LAYER                                                     │
│     Load balancer → API servers (auto-scaling)                    │
│     Rate limiting, authentication, request validation             │
│     Image preprocessing: resize, normalize (on API server)        │
│                                                                   │
│  2. MODEL SERVING                                                 │
│     ┌─────────┐    ┌──────────────────┐                          │
│     │ Request │───▶│ Model Inference  │                          │
│     │  Queue  │    │ (GPU cluster)    │                          │
│     │ (Redis) │    │                  │                          │
│     └─────────┘    │ Option A: TorchServe with batching          │
│                    │ Option B: Triton Inference Server            │
│                    │ Option C: ONNX Runtime (CPU fallback)       │
│                    └──────────────────┘                          │
│                                                                   │
│  3. MODEL ARCHITECTURE                                            │
│     EfficientNet-B2 (best speed/accuracy for classification)     │
│     Quantized to INT8 for 2-3× speedup                           │
│     Batch size: 32 (dynamic batching)                             │
│     Target latency: P50 < 20ms, P99 < 100ms                      │
│                                                                   │
│  4. CACHING                                                       │
│     Content-hash images → cache predictions                       │
│     LRU cache with 1M entries → ~80% cache hit rate               │
│     Reduces GPU load significantly                                │
│                                                                   │
│  5. HIGH AVAILABILITY                                             │
│     Multi-AZ deployment (3 zones minimum)                         │
│     CPU fallback model if GPU unavailable (DistilEfficientNet)   │
│     Circuit breaker: degrade gracefully under load                │
│     Health checks every 10s, auto-restart failed pods             │
│                                                                   │
│  6. MONITORING                                                    │
│     Accuracy: sample predictions → human review weekly            │
│     Drift: monitor input distribution (image size, color dist)    │
│     Performance: latency percentiles, throughput, error rate       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 13.12 Key Papers

| Paper                                                                                     | Year | Why It Matters                                                 |
| ----------------------------------------------------------------------------------------- | ---- | -------------------------------------------------------------- |
| _ImageNet Classification with Deep CNNs_ (Krizhevsky et al.)                              | 2012 | AlexNet — started the deep learning revolution                 |
| _Deep Residual Learning_ (He et al.)                                                      | 2015 | ResNet — residual connections, enabled 100+ layer networks     |
| _Long Short-Term Memory_ (Hochreiter & Schmidhuber)                                       | 1997 | LSTM — solved vanishing gradients, dominated NLP until 2017    |
| _Sequence to Sequence Learning_ (Sutskever et al.)                                        | 2014 | Seq2seq — foundation of neural machine translation             |
| _Neural Machine Translation by Jointly Learning to Align and Translate_ (Bahdanau et al.) | 2014 | Bahdanau attention — direct precursor to transformer attention |
| _Attention Is All You Need_ (Vaswani et al.)                                              | 2017 | The Transformer — removed recurrence entirely                  |
| _BERT: Pre-training of Deep Bidirectional Transformers_ (Devlin et al.)                   | 2018 | BERT — bidirectional pre-training revolution                   |
| _Exploring the Limits of Transfer Learning with T5_ (Raffel et al.)                       | 2019 | T5 — text-to-text framework, comprehensive study               |
| _An Image is Worth 16x16 Words_ (Dosovitskiy et al.)                                      | 2020 | ViT — transformers for vision, bridge to multimodal            |
| _Generative Adversarial Nets_ (Goodfellow et al.)                                         | 2014 | GANs — adversarial training paradigm                           |
| _Auto-Encoding Variational Bayes_ (Kingma & Welling)                                      | 2013 | VAE — principled generative modeling with latent spaces        |
| _EfficientNet: Rethinking Model Scaling_ (Tan & Le)                                       | 2019 | Compound scaling — optimal architecture efficiency             |

---

[← Module 12: Classical ML](../module-12-classical-ml/README.md) | [Module 14: Generative AI →](../module-14-generative-ai/README.md)
