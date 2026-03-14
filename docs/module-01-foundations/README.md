# Module 1: Foundations of Neural Networks & Deep Learning

> **Prerequisites:** Linear algebra, calculus, basic probability & statistics, Python
> **Estimated Study Time:** 8–10 hours

---

## 1.1 The Neuron — From Biology to Math

A single artificial neuron computes a weighted sum of inputs, adds a bias, and passes the result through a non-linear activation function.

```
                    ┌─────────────┐
   x₁ ──w₁──►      │             │
   x₂ ──w₂──►      │  z = Σwᵢxᵢ + b  ──► a = σ(z) ──► output
   x₃ ──w₃──►      │             │
     ...            └─────────────┘
   xₙ ──wₙ──►           bias b
```

**Mathematically:**

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  =  wᵀx + b
a = σ(z)
```

Where:

- **x** = input vector
- **w** = weight vector (learnable)
- **b** = bias (learnable)
- **σ** = activation function

---

## 1.2 Activation Functions

| Function       | Formula                     | Range       | Pros                                 | Cons                                   |
| -------------- | --------------------------- | ----------- | ------------------------------------ | -------------------------------------- |
| **Sigmoid**    | σ(z) = 1/(1+e⁻ᶻ)            | (0, 1)      | Smooth, probabilistic interpretation | Vanishing gradients, not zero-centered |
| **Tanh**       | tanh(z) = (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) | (-1, 1)     | Zero-centered                        | Vanishing gradients                    |
| **ReLU**       | max(0, z)                   | [0, ∞)      | Fast, no vanishing gradient for z>0  | Dead neurons (z<0 → gradient=0)        |
| **Leaky ReLU** | max(αz, z), α≈0.01          | (-∞, ∞)     | No dead neurons                      | Extra hyperparameter                   |
| **GELU**       | z·Φ(z)                      | ≈(-0.17, ∞) | Smooth ReLU; used in BERT, GPT       | Slightly more expensive                |
| **SiLU/Swish** | z·σ(z)                      | ≈(-0.28, ∞) | Smooth, self-gated                   | Slightly more expensive                |

```
ReLU                    GELU                    Sigmoid
  │    /                  │    ·····               │     ·········
  │   /                   │  ··                    │   ··
  │  /                    │··                      │  ·
  │ /                    ·│                        │·
──┼──────            ··──┼──────             ─────┼──────
  │                      │                        │
  │                      │                        │
```

**Why GELU matters for LLMs:** GELU (Gaussian Error Linear Unit) is defined as:

```
GELU(x) = x · Φ(x)
```

where Φ(x) is the CDF of the standard normal distribution — i.e., the probability that a standard Gaussian random variable is ≤ x. Intuitively, each input is _scaled by its own percentile rank under a Gaussian_: highly positive values (high percentile) pass through almost unchanged, highly negative values (low percentile) are suppressed toward zero, and values near zero get partially dampened.

This creates a **soft stochastic gate**: during training, you can interpret GELU as the expectation of multiplying the input by a Bernoulli mask whose probability depends on the input's magnitude. This provides implicit regularization — an effect similar to dropout, but continuous and deterministic.

**Why it outperforms ReLU in transformers:** ReLU has a hard cutoff at zero with a non-smooth kink that can create optimization difficulties in deep networks. GELU's smoothness means gradients never abruptly vanish at the activation boundary. It's the default in GPT-2, GPT-3, BERT, and most modern transformers.

---

## 1.3 Feedforward Neural Networks (MLPs)

An MLP stacks multiple layers of neurons. Each layer performs a linear transformation followed by a non-linear activation.

```
  Input Layer        Hidden Layer 1      Hidden Layer 2       Output Layer
  (features)         (learned repr.)     (learned repr.)      (predictions)

    x₁ ─────┐     ┌──── h₁⁽¹⁾────┐    ┌──── h₁⁽²⁾────┐    ┌──── ŷ₁
             ├─────┤              ├────┤              ├────┤
    x₂ ─────┤     ├──── h₂⁽¹⁾────┤    ├──── h₂⁽²⁾────┤    ├──── ŷ₂
             ├─────┤              ├────┤              ├────┤
    x₃ ─────┤     ├──── h₃⁽¹⁾────┤    ├──── h₃⁽²⁾────┤    └──── ŷ₃
             ├─────┤              ├────┤              │
    x₄ ─────┘     └──── h₄⁽¹⁾────┘    └──── h₄⁽²⁾────┘

  Each arrow = weight (wᵢⱼ)
  Each node  = σ(wᵀx + b)
```

**Forward pass for layer l:**

```
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = σ(z⁽ˡ⁾)
```

**Universal Approximation Theorem:** A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of Rⁿ, given a non-polynomial activation function. This does NOT guarantee that SGD will find those weights, nor that the architecture is efficient.

---

## 1.4 Loss Functions

The loss function quantifies how wrong the model's predictions are.

### Regression

- **Mean Squared Error (MSE):** L = (1/n) Σ(yᵢ - ŷᵢ)²
- **Mean Absolute Error (MAE):** L = (1/n) Σ|yᵢ - ŷᵢ|

### Classification

- **Binary Cross-Entropy:** L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
- **Categorical Cross-Entropy:** L = -Σ yₖ·log(ŷₖ)

### Language Modeling

- **Cross-Entropy over vocabulary:** L = -Σₜ log P(wₜ | w₁, ..., wₜ₋₁)

This is the core loss for training LLMs — it measures how well the model predicts the next token given all preceding tokens.

---

## 1.5 Backpropagation & the Chain Rule

Backpropagation computes gradients of the loss with respect to every parameter by applying the chain rule of calculus backward through the computational graph.

```
Forward Pass:                    Backward Pass:
                                 (gradients flow backward)

x ──► [W₁,b₁] ──► z₁ ──► σ ──► a₁ ──► [W₂,b₂] ──► z₂ ──► L
                                                              │
x ◄── ∂L/∂W₁ ◄── ∂L/∂z₁ ◄── ∂L/∂a₁ ◄── ∂L/∂W₂ ◄── ∂L/∂z₂ ◄┘
```

**Chain rule example for a 2-layer network:**

```
∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁
```

**Key insight:** Each layer only needs the gradient from the layer above it (∂L/∂a⁽ˡ⁾) to compute its own parameter gradients. This enables efficient computation in O(n) time where n is the number of layers.

### Worked Numerical Example — Backprop Through a 2-Layer Network

Let's walk through a tiny example with actual numbers to make gradient flow concrete.

```
Setup: 1 input, 1 hidden neuron (ReLU), 1 output (identity), MSE loss
Weights: W₁ = 0.5, b₁ = 0.1, W₂ = -0.3, b₂ = 0.2
Input:  x = 2.0,  Target: y = 1.0

═══ FORWARD PASS ═══

Hidden layer:
  z₁ = W₁·x + b₁ = 0.5·2.0 + 0.1 = 1.1
  a₁ = ReLU(1.1) = 1.1

Output layer:
  z₂ = W₂·a₁ + b₂ = -0.3·1.1 + 0.2 = -0.13
  ŷ  = z₂ = -0.13  (identity activation)

Loss:
  L = ½(y - ŷ)² = ½(1.0 - (-0.13))² = ½(1.13)² = 0.638

═══ BACKWARD PASS ═══

Output gradient:
  ∂L/∂ŷ = -(y - ŷ) = -(1.0 - (-0.13)) = -1.13
  ∂L/∂z₂ = ∂L/∂ŷ · 1  = -1.13  (identity activation derivative = 1)

Gradients for W₂, b₂:
  ∂L/∂W₂ = ∂L/∂z₂ · a₁ = -1.13 · 1.1 = -1.243
  ∂L/∂b₂ = ∂L/∂z₂ · 1  = -1.13

Propagate to hidden layer:
  ∂L/∂a₁ = ∂L/∂z₂ · W₂ = -1.13 · (-0.3) = 0.339
  ∂L/∂z₁ = ∂L/∂a₁ · ReLU'(z₁) = 0.339 · 1 = 0.339  (z₁=1.1 > 0)

Gradients for W₁, b₁:
  ∂L/∂W₁ = ∂L/∂z₁ · x = 0.339 · 2.0 = 0.678
  ∂L/∂b₁ = ∂L/∂z₁ · 1 = 0.339

Update (lr = 0.1):
  W₁ ← 0.5  - 0.1·0.678  = 0.432
  b₁ ← 0.1  - 0.1·0.339  = 0.066
  W₂ ← -0.3 - 0.1·(-1.243) = -0.176
  b₂ ← 0.2  - 0.1·(-1.13)  = 0.313
```

Notice how each gradient depends on the chain of derivatives from the loss all the way back. If any factor in the chain is very small (e.g., sigmoid derivative ≤ 0.25), the gradient shrinks at every layer — this is the vanishing gradient problem.

### Vanishing & Exploding Gradients

When networks are deep, gradients can:

- **Vanish** (→ 0): happens with sigmoid/tanh because |σ'(z)| ≤ 0.25. After many layers, gradients become exponentially small.
- **Explode** (→ ∞): happens when weight magnitudes are large, causing gradients to grow exponentially.

**Solutions:**

- Residual connections (skip connections) — used in all modern transformers
- Layer normalization
- Careful weight initialization (Xavier, He, etc.)
- Gradient clipping

### Weight Initialization — Why It Matters

If weights are initialized too large or too small, signals and gradients either explode or vanish from the very first forward pass.

```
┌──────────────┬──────────────────────────────────────────────────────┐
│ Method       │ Formula & When to Use                                │
├──────────────┼──────────────────────────────────────────────────────┤
│ Xavier       │ W ~ N(0, 2/(n_in + n_out))                           │
│ (Glorot)     │ Designed for sigmoid/tanh activations                │
│              │ Keeps variance constant across layers                │
├──────────────┼──────────────────────────────────────────────────────┤
│ He (Kaiming) │ W ~ N(0, 2/n_in)                                     │
│              │ Designed for ReLU activations                        │
│              │ Accounts for ReLU zeroing out half the neurons       │
├──────────────┼──────────────────────────────────────────────────────┤
│ LeCun        │ W ~ N(0, 1/n_in)                                     │
│              │ For SELU activations (self-normalizing networks)     │
├──────────────┼──────────────────────────────────────────────────────┤
│ Orthogonal   │ W = orthogonal matrix (SVD-based)                    │
│              │ Preserves gradient norms perfectly                   │
│              │ Used in some RNN/LSTM initializations               │
└──────────────┴──────────────────────────────────────────────────────┘

Intuition:
  Too small init     Just right init     Too large init
  Layer 1: [0.5]     Layer 1: [0.5]      Layer 1: [0.5]
  Layer 5: [0.001]   Layer 5: [0.48]     Layer 5: [10000]
  → vanished!        → stable ✓          → exploded!
```

**The signal propagation argument (why variance scaling works):**

Consider a single layer: y = Wx where x has n input dimensions. If we assume inputs and weights are independent with zero mean, then:

```
Var(yⱼ) = n · Var(wᵢⱼ) · Var(xᵢ)
```

To keep the output variance equal to the input variance across layers (Var(y) = Var(x)), we need:

```
n · Var(w) = 1   →   Var(w) = 1/n
```

This is exactly the LeCun initialization. Xavier extends this to account for both forward and backward variance stability: Var(w) = 2/(n_in + n_out). He initialization uses Var(w) = 2/n_in because ReLU zeros out roughly half the neurons, effectively halving the number of active inputs.

**What happens without proper initialization:** In a 50-layer network with Var(w) = 1 (too large), the activation variance multiplies by ~n at each layer. With n=512, after just 10 layers the variance would be 512¹⁰ ≈ 10²⁷ — complete numerical overflow. Conversely, with Var(w) = 0.001 (too small), variance decays geometrically to zero and all gradients vanish.

For transformers specifically, a common practice is to scale the residual connection initialization by `1/√(2N)` where N is the number of layers, preventing the residual stream from growing too large in deep networks.

### Batch Normalization vs Layer Normalization

```
Batch Normalization (BN):
  Normalize across the BATCH dimension for each feature
  For a batch of inputs x ∈ R^{B×D}:
    μ = mean over batch (per feature)
    → Requires sufficiently large batches
    → Not suitable for variable-length sequences
    → Great for CNNs (fixed-size inputs)

Layer Normalization (LN):
  Normalize across the FEATURE dimension for each sample
  For a single input x ∈ R^D:
    μ = mean over features (per sample)
    → Independent of batch size
    → Works with any sequence length
    → Standard for transformers

  BN: normalize ↓ (across batch)
  ┌────────────────────┐
  │ sample1: [a b c d] │ ← normalize each column
  │ sample2: [e f g h] │    across all samples
  │ sample3: [i j k l] │
  └────────────────────┘

  LN: normalize → (across features)
  ┌────────────────────┐
  │ sample1: [a b c d] │ ← normalize each row
  │ sample2: [e f g h] │    independently
  │ sample3: [i j k l] │
  └────────────────────┘

RMSNorm (used in LLaMA, Gemma):
  Simplified LayerNorm — skip the mean subtraction
  RMSNorm(x) = x / RMS(x) · γ
  where RMS(x) = √(1/d Σ xᵢ²)
  ~10% faster than LayerNorm, similar quality
```

**Why RMSNorm works without mean centering:** LayerNorm subtracts the mean and divides by the standard deviation, but empirical studies show that the re-centering (mean subtraction) contributes very little to the normalization benefit — it's the _re-scaling_ (dividing by the magnitude) that matters most for training stability. Since LayerNorm already has learned shift parameter β, the mean subtraction is redundant: β can absorb any necessary centering.

RMSNorm removes the mean computation entirely, using just the root mean square: RMS(x) = √(1/d · Σ xᵢ²). This saves ~10-15% compute because computing the mean requires an extra reduction across the feature dimension. LLaMA and LLaMA 2 demonstrated that RMSNorm produces equivalent quality to LayerNorm at every scale tested, making it the de facto choice for modern LLM architectures.

---

## 1.6 Optimization Algorithms

### Stochastic Gradient Descent (SGD)

```
θ ← θ - η · ∇L(θ)
```

- η = learning rate
- Computed on mini-batches, not full dataset

### SGD with Momentum

```
v ← βv + ∇L(θ)        (accumulate velocity)
θ ← θ - η · v
```

- β ≈ 0.9 typical
- Accelerates convergence, dampens oscillations

### Adam (Adaptive Moment Estimation) — The Default for LLMs

```
m ← β₁m + (1-β₁)∇L          (1st moment — mean of gradients)
v ← β₂v + (1-β₂)(∇L)²       (2nd moment — variance of gradients)
m̂ = m / (1 - β₁ᵗ)            (bias correction)
v̂ = v / (1 - β₂ᵗ)
θ ← θ - η · m̂ / (√v̂ + ε)
```

- Default: β₁=0.9, β₂=0.999, ε=1e-8
- Adapts learning rate per-parameter
- Almost universally used for training transformers

### AdamW (Weight Decay Decoupled)

```
θ ← θ - η · (m̂ / (√v̂ + ε) + λθ)
```

- Decouples weight decay from the gradient update
- Standard for LLM training (GPT, LLaMA, etc.)

**Why L2 regularization ≠ weight decay in Adam (and why this matters):**

With vanilla SGD, adding an L2 penalty λ‖w‖² to the loss produces the same update as weight decay (multiplying weights by (1-λ) each step). But with Adam, this equivalence **breaks down**:

- **L2 regularization in Adam:** The gradient of the L2 term (2λw) gets processed by Adam's adaptive learning rate — divided by √v̂. Parameters with large historical gradients get _less_ regularization, and parameters with small historical gradients get _more_. This is inconsistent and unintended.

- **AdamW (decoupled weight decay):** Applies weight decay _directly_ to the weights, bypassing Adam's adaptive scaling: θ ← θ(1 - ηλ) - η·m̂/(√v̂+ε). Every parameter gets the same proportional decay regardless of its gradient history.

In practice, this distinction affects training stability and generalization in LLMs significantly. Loshchilov & Hutter (2017) showed that decoupled weight decay produces better generalization across all learning rates tested.

### Learning Rate Schedules

```
Learning Rate
     │
  η₀ │──┐
     │  │╲         Cosine decay
     │  │  ╲·····
     │  │       ····
     │  │           ····
     │  │               ·──
     └──┴───────────────────► Steps
     warmup  main training
```

- **Warmup:** Linearly increase LR from 0 to η₀ over first few thousand steps
- **Cosine decay:** Smoothly reduce LR following a cosine curve
- **Why warmup?** Early gradients are noisy (random weights) — large LR could destabilize training

---

## 1.7 Regularization

Techniques to prevent overfitting (memorizing training data instead of learning general patterns):

| Technique             | How It Works                                                 | Where Used                                            |
| --------------------- | ------------------------------------------------------------ | ----------------------------------------------------- |
| **L2 / Weight Decay** | Add λ‖w‖² to loss → penalizes large weights                  | AdamW (standard for LLMs)                             |
| **Dropout**           | Randomly zero out neurons during training with probability p | Used in some transformer layers                       |
| **Layer Norm**        | Normalize activations to zero mean, unit variance            | Every transformer sublayer                            |
| **Data Augmentation** | Create modified copies of training data                      | More common in vision                                 |
| **Early Stopping**    | Stop training when validation loss increases                 | Less common for LLMs (train for fixed compute budget) |

### Dropout Visualized

```
Training:                         Inference:
  ●──●──●──●──●                    ●──●──●──●──●
  │╲ │╲ │╲ │╲ │                    │╲ │╲ │╲ │╲ │
  ●──○──●──○──●   (○ = dropped)    ●──●──●──●──●  (all active, scaled)
  │╲ │╲ │╲ │╲ │                    │╲ │╲ │╲ │╲ │
  ●──●──○──●──●                    ●──●──●──●──●
```

---

## 1.8 Key Architectural Concepts for LLMs

### Residual (Skip) Connections

```
        ┌──────────────────────────┐
        │                          │
  x ────┤──► [Sublayer] ──► + ◄───┘ ──► Layer Norm ──► output
        │     (attention       ↑
        │      or FFN)         │
        │                      │
        └──────────────────────┘
            identity shortcut

output = LayerNorm(x + Sublayer(x))
```

**Why they matter:** Without residual connections, a 96-layer transformer would be nearly impossible to train. Skip connections allow gradients to flow directly through the network, solving the vanishing gradient problem.

### Layer Normalization

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β
```

- μ, σ = mean and std computed across the feature dimension (not the batch)
- γ, β = learnable scale and shift parameters
- Applied after every sublayer in modern transformers

### Pre-Norm vs Post-Norm

```
Post-Norm (original transformer):     Pre-Norm (GPT-2+, more stable):
x → Sublayer → Add → LayerNorm        x → LayerNorm → Sublayer → Add
```

Pre-Norm is more stable during training and is the standard in modern LLMs.

---

## 1.9 From MLPs to Sequence Models — Why We Needed Something Better

MLPs treat input as a fixed-size, unordered vector. Language is sequential and variable-length. The evolution:

```
Fixed input ──► RNNs ──► LSTMs/GRUs ──► Attention ──► Transformers
(MLPs)         (sequential,  (gates solve     (parallel,    (full
               vanishing     vanishing         captures      architecture)
               gradients)    gradients)        long-range
                                               dependencies)
```

### RNN Limitations That Motivated Transformers

1. **Sequential processing** — can't parallelize across time steps
2. **Long-range dependencies** — information decays over distance despite LSTMs
3. **Training speed** — O(T) sequential operations for sequence of length T

Transformers solve all three with self-attention: O(1) sequential operations, direct connections between any two positions, and full parallelization during training.

---

## Interview Questions

### Conceptual

1. **Explain the vanishing gradient problem. Why do residual connections help?**
   <details>
   <summary>Answer</summary>
   During backpropagation through deep networks, gradients are multiplied through many layers. If these multiplied values are < 1 (common with sigmoid/tanh activations), the gradient exponentially decays to near zero, preventing early layers from learning. Residual connections create a shortcut path: output = x + F(x). During backprop, ∂output/∂x = 1 + ∂F(x)/∂x, so the gradient is always at least 1, ensuring it flows through the full network.
   </details>

2. **Why is Adam preferred over SGD for training transformers?**
   <details>
   <summary>Answer</summary>
   Adam maintains per-parameter adaptive learning rates using first and second moment estimates of gradients. Transformers have heterogeneous parameter groups (attention weights, FFN weights, embeddings, layer norms) with very different gradient scales. Adam's adaptive rates handle this naturally. SGD with a single global learning rate would require extensive tuning. Additionally, Adam's momentum helps navigate the complex, high-dimensional loss landscapes of large models.
   </details>

3. **What is the difference between Layer Normalization and Batch Normalization? Why do transformers use LayerNorm?**
   <details>
   <summary>Answer</summary>
   BatchNorm normalizes across the batch dimension (computes mean/variance over all samples in a batch for each feature). LayerNorm normalizes across the feature dimension (computes mean/variance over all features for each sample independently). Transformers use LayerNorm because: (1) it doesn't depend on batch size, enabling inference with batch_size=1; (2) for variable-length sequences, BatchNorm statistics would be ill-defined across padded positions; (3) LayerNorm stabilizes the activations at each position independently.
   </details>

4. **Explain the Universal Approximation Theorem. Does it mean a single hidden layer is always sufficient?**
   <details>
   <summary>Answer</summary>
   The theorem states that an MLP with one hidden layer and sufficient neurons can approximate any continuous function to arbitrary precision. However, it does NOT mean a single layer is practical: (1) the required number of neurons may be exponentially large; (2) SGD may not find the right weights; (3) deeper networks can represent the same functions with exponentially fewer parameters due to compositionality. In practice, depth is far more parameter-efficient than width.
   </details>

5. **Why do modern LLMs use Pre-Norm instead of Post-Norm?**
   <details>
   <summary>Answer</summary>
   Pre-Norm (applying LayerNorm before the sublayer) produces more stable training dynamics. In Post-Norm, the residual connection adds unnormalized sublayer output to the residual stream, which can cause the magnitudes to grow unpredictably. Pre-Norm ensures the input to each sublayer is well-conditioned regardless of depth. Empirically, Pre-Norm eliminates the need for careful learning rate warmup and allows training much deeper models without divergence.
   </details>

### Coding

6. **Implement a simple feedforward neural network from scratch using only NumPy (no PyTorch/TF). Include forward pass, loss computation, and backpropagation.**

   <details>
   <summary>Solution</summary>

   ```python
   import numpy as np

   class SimpleNN:
       def __init__(self, input_dim, hidden_dim, output_dim):
           # Xavier initialization
           self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
           self.b1 = np.zeros((1, hidden_dim))
           self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
           self.b2 = np.zeros((1, output_dim))

       def relu(self, z):
           return np.maximum(0, z)

       def relu_derivative(self, z):
           return (z > 0).astype(float)

       def softmax(self, z):
           exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
           return exp_z / np.sum(exp_z, axis=1, keepdims=True)

       def cross_entropy_loss(self, y_pred, y_true):
           n = y_true.shape[0]
           log_probs = -np.log(y_pred[range(n), y_true] + 1e-9)
           return np.sum(log_probs) / n

       def forward(self, X):
           self.z1 = X @ self.W1 + self.b1
           self.a1 = self.relu(self.z1)
           self.z2 = self.a1 @ self.W2 + self.b2
           self.a2 = self.softmax(self.z2)
           return self.a2

       def backward(self, X, y_true, lr=0.01):
           n = X.shape[0]
           # Output layer gradient (softmax + cross-entropy simplification)
           dz2 = self.a2.copy()
           dz2[range(n), y_true] -= 1
           dz2 /= n

           dW2 = self.a1.T @ dz2
           db2 = np.sum(dz2, axis=0, keepdims=True)

           # Hidden layer gradient
           da1 = dz2 @ self.W2.T
           dz1 = da1 * self.relu_derivative(self.z1)
           dW1 = X.T @ dz1
           db1 = np.sum(dz1, axis=0, keepdims=True)

           # Update weights
           self.W1 -= lr * dW1
           self.b1 -= lr * db1
           self.W2 -= lr * dW2
           self.b2 -= lr * db2
   ```

   </details>

7. **Implement AdamW from scratch given a list of parameters and their gradients.**

   <details>
   <summary>Solution</summary>

   ```python
   class AdamW:
       def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
           self.params = params
           self.lr = lr
           self.beta1, self.beta2 = betas
           self.eps = eps
           self.wd = weight_decay
           self.t = 0
           self.m = [np.zeros_like(p) for p in params]  # 1st moment
           self.v = [np.zeros_like(p) for p in params]  # 2nd moment

       def step(self, grads):
           self.t += 1
           for i, (param, grad) in enumerate(zip(self.params, grads)):
               # Update moments
               self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
               self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

               # Bias correction
               m_hat = self.m[i] / (1 - self.beta1 ** self.t)
               v_hat = self.v[i] / (1 - self.beta2 ** self.t)

               # AdamW: decoupled weight decay applied to param directly
               param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * param)
   ```

   </details>

### System Design

8. **You're training a model and observe that training loss is decreasing but validation loss starts increasing after epoch 5. Diagnose and propose solutions.**
   <details>
   <summary>Answer</summary>
   This is classic overfitting. The model is memorizing training data rather than learning generalizable patterns. Solutions (in order of priority): (1) Add regularization — dropout, weight decay; (2) Get more training data or augment existing data; (3) Reduce model capacity (fewer layers/neurons); (4) Use early stopping (save checkpoint at epoch 5); (5) Apply learning rate reduction. For LLMs specifically, overfitting is less common during pre-training (massive datasets) but common during fine-tuning, where LoRA or low learning rates help.
   </details>

---

## Key Papers

- Rumelhart, Hinton, Williams (1986) — "Learning representations by back-propagating errors"
- Glorot & Bengio (2010) — "Understanding the difficulty of training deep feedforward neural networks" (Xavier init)
- He et al. (2015) — "Deep Residual Learning for Image Recognition" (ResNets)
- Kingma & Ba (2014) — "Adam: A Method for Stochastic Optimization"
- Loshchilov & Hutter (2017) — "Decoupled Weight Decay Regularization" (AdamW)
- Ba, Kiros, Hinton (2016) — "Layer Normalization"
