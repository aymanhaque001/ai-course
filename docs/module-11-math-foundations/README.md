# Module 11: Mathematical & Statistical Foundations for AI

> **Prerequisites:** High school algebra  
> **Estimated Time:** 10-12 hours  
> **Relevance:** Every AI concept — from gradient descent to attention mechanisms — is built on these fundamentals

---

## 11.1 Linear Algebra for AI

Linear algebra is the **language** of deep learning. Every forward pass through a neural network is a sequence of matrix multiplications and element-wise operations.

### Vectors and Vector Spaces

A vector is an ordered list of numbers representing a point or direction in space.

```
Scalar:       x = 3.14

Vector:       x = [1, 2, 3]ᵀ    (column vector, 3 dimensions)

Vector operations:
  Addition:       [1, 2] + [3, 4] = [4, 6]
  Scalar mult:    3 × [1, 2] = [3, 6]
  Dot product:    [1, 2] · [3, 4] = 1×3 + 2×4 = 11
```

**Why it matters for AI:**

- Word embeddings are vectors in ℝᵈ (e.g., d=768 for BERT)
- "King - Man + Woman ≈ Queen" is vector arithmetic in embedding space
- Cosine similarity between vectors measures semantic similarity

### Dot Product & Cosine Similarity

The dot product is the most important operation in transformers:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$$

```
Cosine Similarity:

           a · b            Σ aᵢbᵢ
cos(θ) = ───────── = ─────────────────────
          ‖a‖ ‖b‖    √(Σ aᵢ²) × √(Σ bᵢ²)

Range: [-1, 1]
  +1 = identical direction (same meaning)
   0 = orthogonal (unrelated)
  -1 = opposite direction (opposite meaning)
```

**In transformers:** The attention score between query q and key k is exactly their dot product: $\text{score} = \mathbf{q} \cdot \mathbf{k}$

### Matrices and Matrix Multiplication

A matrix is a 2D array of numbers. Matrix multiplication is the core computation in neural networks:

```
Matrix Multiplication (A × B = C):

A: (m × n)    B: (n × p)    →    C: (m × p)

┌         ┐   ┌         ┐       ┌         ┐
│ 1  2  3 │   │ 7   8   │       │ 58  64  │
│ 4  5  6 │ × │ 9  10   │   =   │ 139 154 │
└         ┘   │ 11 12   │       └         ┘
              └         ┘

C[i,j] = Σ_k A[i,k] × B[k,j]
```

**Key properties:**

- Not commutative: AB ≠ BA (in general)
- Associative: (AB)C = A(BC) — useful for optimization
- Dimensions must align: inner dimensions must match

**In neural networks:**

```
Linear layer:    y = Wx + b

Where:
  x: input vector     (n × 1)
  W: weight matrix     (m × n)     ← these are the learned parameters
  b: bias vector       (m × 1)
  y: output vector     (m × 1)
```

### Matrix Types Important for AI

```
Identity Matrix (I):              Diagonal Matrix:
┌       ┐                        ┌       ┐
│ 1 0 0 │                        │ a 0 0 │
│ 0 1 0 │   AI = A for any A    │ 0 b 0 │
│ 0 0 1 │                        │ 0 0 c │
└       ┘                        └       ┘

Transpose (Aᵀ):                  Symmetric (A = Aᵀ):
Swap rows and columns            Covariance matrices are symmetric
(m × n) → (n × m)               Used in PCA, kernel methods
```

### Eigenvalues and Eigenvectors

An eigenvector of a matrix A is a vector that, when multiplied by A, only changes in scale:

$$A\mathbf{v} = \lambda\mathbf{v}$$

where $\lambda$ is the eigenvalue and $\mathbf{v}$ is the eigenvector.

```
Geometric intuition:

  Most vectors change direction when multiplied by A:
      A × [1, 1]ᵀ → [3, 2]ᵀ  (rotated)

  Eigenvectors only stretch/compress:
      A × v → λv  (same direction, scaled by λ)
```

**AI applications:**

- **PCA:** Eigenvectors of the covariance matrix = principal components (directions of maximum variance)
- **Spectral clustering:** Eigenvectors of the graph Laplacian
- **Singular Value Decomposition (SVD):** Foundation of matrix factorization, used in recommendations and LoRA

### Singular Value Decomposition (SVD)

Any matrix A can be decomposed as:

$$A = U \Sigma V^T$$

```
SVD Decomposition:

A          =    U        ×    Σ        ×    Vᵀ
(m × n)       (m × m)       (m × n)       (n × n)
              orthogonal    diagonal       orthogonal

Low-rank approximation (keep top k singular values):
  Aₖ = Uₖ Σₖ Vₖᵀ     ← best rank-k approximation (Eckart-Young theorem)

Applications in AI:
  LoRA:  W₀ + ΔW ≈ W₀ + BA   where B(d×r), A(r×d), r << d
         This IS a low-rank matrix factorization!

  Dimensionality reduction:  Keep top-k components
  Recommender systems:  Matrix factorization of user-item matrix
```

### Norms

Norms measure the "size" of vectors — used everywhere in regularization and optimization:

| Norm                 | Formula                                | AI Usage                                |
| -------------------- | -------------------------------------- | --------------------------------------- | --- | -------------------------------------------- |
| L1 (Manhattan)       | $\|\mathbf{x}\|\_1 = \sum              | x_i                                     | $   | Lasso regularization, sparse solutions       |
| L2 (Euclidean)       | $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$ | Ridge regularization, gradient clipping |
| L∞ (Max)             | $\|\mathbf{x}\|\_\infty = \max         | x_i                                     | $   | Adversarial robustness (perturbation bounds) |
| Frobenius (matrices) | $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$ | Weight decay, matrix comparison         |

---

## 11.2 Calculus for Machine Learning

### Derivatives and Gradients

The derivative measures the rate of change — it tells us how to adjust parameters to reduce loss:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

For multivariate functions, the **gradient** is the vector of partial derivatives:

$$\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]$$

```
Gradient Descent Intuition:

Loss Surface (2D cross-section):

Loss │
     │  ╲
     │   ╲     ╱
     │    ╲   ╱
     │     ╲_╱  ← minimum (goal)
     │
     └──────────── parameter value

The gradient points UPHILL, so we go OPPOSITE:
  θ_new = θ_old - α × ∇L(θ)

  α = learning rate (step size)
```

### The Chain Rule — Heart of Backpropagation

The chain rule lets us compute gradients through composed functions:

$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

```
Neural Network as Function Composition:

  Input x → [Linear₁] → [ReLU] → [Linear₂] → [Softmax] → Loss
     x    →   z₁ = W₁x  → a₁     →  z₂ = W₂a₁ →  ŷ      →  L

Backpropagation applies chain rule backwards:

  ∂L/∂W₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂W₂

  ∂L/∂W₁ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂a₁ × ∂a₁/∂z₁ × ∂z₁/∂W₁
                                    ↑
                         gradient flows through all layers
```

### The Jacobian and Hessian

```
Jacobian Matrix:
  For f: ℝⁿ → ℝᵐ, the Jacobian is the matrix of all first-order partial derivatives

  J[i,j] = ∂fᵢ/∂xⱼ     (m × n matrix)

  Used in: Understanding layer-wise gradient flow, normalizing flows

Hessian Matrix:
  Matrix of second-order partial derivatives

  H[i,j] = ∂²f/∂xᵢ∂xⱼ   (n × n symmetric matrix)

  Used in:
    - Second-order optimization (Newton's method, L-BFGS)
    - Understanding loss landscape curvature
    - Pruning (OBS: Optimal Brain Surgeon uses Hessian)
    - Eigenvalues of Hessian → sharpness of minima
```

### Key Derivatives for AI

| Function                                               | Derivative                                            | Where It Appears                          |
| ------------------------------------------------------ | ----------------------------------------------------- | ----------------------------------------- |
| $\sigma(x) = \frac{1}{1+e^{-x}}$                       | $\sigma(x)(1-\sigma(x))$                              | Sigmoid activation, binary classification |
| $\tanh(x)$                                             | $1 - \tanh^2(x)$                                      | LSTM gates, older architectures           |
| $\text{ReLU}(x) = \max(0,x)$                           | $\begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ | Most common activation                    |
| $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | $s_i(\delta_{ij} - s_j)$                              | Output layer, attention weights           |
| $\text{CE}(y, \hat{y}) = -\sum y_i \log \hat{y}_i$     | $\hat{y}_i - y_i$ (with softmax)                      | Classification loss                       |

---

## 11.3 Probability & Statistics

### Probability Fundamentals

```
Probability Axioms:
  1. P(A) ≥ 0              (non-negative)
  2. P(Ω) = 1              (total probability = 1)
  3. P(A ∪ B) = P(A) + P(B) if A ∩ B = ∅  (additivity)

Conditional Probability:
  P(A|B) = P(A ∩ B) / P(B)

Bayes' Theorem:
  P(A|B) = P(B|A) × P(A) / P(B)
           ──────────────────────
           posterior = likelihood × prior / evidence

  AI application: Naive Bayes classifier, Bayesian optimization,
                  understanding LLM probability distributions
```

### Probability Distributions

```
┌──────────────────────────────────────────────────────────────────┐
│                     Key Distributions for AI                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Bernoulli(p):     Binary outcome (dropout mask, coin flip)       │
│    P(X=1) = p, P(X=0) = 1-p                                      │
│                                                                   │
│  Categorical(p₁..pₖ): One-of-K outcome (next token prediction)   │
│    P(X=k) = pₖ, Σpₖ = 1                                          │
│                                                                   │
│  Gaussian/Normal(μ, σ²):  The "default" continuous distribution   │
│    p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))                        │
│    Used in: Weight initialization, VAEs, noise modeling           │
│                                                                   │
│  Uniform(a, b):    Equal probability over [a, b]                  │
│    Used in: Random sampling, some initializations                 │
│                                                                   │
│  Multinomial:      Multiple draws from categorical                │
│    Used in: Bag-of-words models, topic models                     │
│                                                                   │
│  Poisson(λ):       Count of events in fixed interval              │
│    Used in: Event modeling, NLP word frequencies                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Expected Value, Variance, and Covariance

$$E[X] = \sum_x x \cdot P(X=x) \quad \text{(discrete)} \qquad E[X] = \int x \cdot p(x) \, dx \quad \text{(continuous)}$$

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

```
Covariance Matrix (Σ):
  Σ[i,j] = Cov(Xᵢ, Xⱼ) = E[(Xᵢ - μᵢ)(Xⱼ - μⱼ)]

  Diagonal: variances of each feature
  Off-diagonal: covariances between features

  Used in: PCA (eigen-decomposition of Σ), Gaussian processes,
           Mahalanobis distance, multivariate normal distribution
```

### Maximum Likelihood Estimation (MLE)

MLE finds parameters that make observed data most probable:

$$\hat{\theta}_{MLE} = \arg\max_\theta \prod_{i=1}^{n} P(x_i | \theta) = \arg\max_\theta \sum_{i=1}^{n} \log P(x_i | \theta)$$

```
Why log-likelihood?
  1. Products → sums (numerically stable)
  2. Same optimum (log is monotonic)
  3. Connection to cross-entropy loss!

MLE for language models:
  Training objective: maximize P(x₁, x₂, ..., xₙ | θ)
  = maximize Σᵢ log P(xᵢ | x₁, ..., xᵢ₋₁; θ)
  = minimize -Σᵢ log P(xᵢ | x₁, ..., xᵢ₋₁; θ)   ← this IS cross-entropy loss!

  So LLM training IS maximum likelihood estimation.
```

### Hypothesis Testing & A/B Testing

```
Hypothesis Testing Framework:
  H₀ (null): No difference (model A = model B)
  H₁ (alt):  There IS a difference

  p-value: Probability of observing data this extreme IF H₀ is true
    p < 0.05 → reject H₀ (statistically significant)
    p ≥ 0.05 → fail to reject H₀

  Type I Error (α):  False positive — see effect when there isn't one
  Type II Error (β): False negative — miss a real effect
  Power = 1 - β:     Probability of detecting a real effect

A/B Testing for ML Models:
  ┌──────────────────────────────────────────┐
  │            User Traffic                  │
  │                │                         │
  │        ┌───────┴───────┐                 │
  │        ▼               ▼                 │
  │   ┌─────────┐   ┌─────────┐             │
  │   │ Model A │   │ Model B │             │
  │   │(control)│   │ (test)  │             │
  │   └────┬────┘   └────┬────┘             │
  │        │              │                  │
  │        ▼              ▼                  │
  │   Metric: CTR,    Metric: CTR,           │
  │   Latency, etc.   Latency, etc.          │
  │                                          │
  │   Statistical test → significant?        │
  │   Sample size calculation: n = f(α,β,δ) │
  └──────────────────────────────────────────┘

  Key decisions:
    - Metric selection (primary + guardrail metrics)
    - Sample size: need enough data for statistical power
    - Duration: long enough for temporal effects
    - Randomization unit: user-level, session-level, request-level
```

---

## 11.4 Information Theory

Information theory quantifies "surprise" and "information content" — foundational to understanding loss functions and generative models.

### Entropy

Entropy measures the average information content (uncertainty) of a distribution:

$$H(X) = -\sum_{x} P(x) \log_2 P(x)$$

```
Entropy Examples:

Fair coin:    H = -0.5 log₂(0.5) - 0.5 log₂(0.5) = 1 bit
              (maximum uncertainty for binary)

Biased coin (90/10):  H = -0.9 log₂(0.9) - 0.1 log₂(0.1) ≈ 0.47 bits
                       (more predictable → less entropy)

Uniform over vocab (50,000 tokens):
  H = log₂(50,000) ≈ 15.6 bits  (maximum entropy)

Well-trained LLM (perplexity = 10):
  H = log₂(10) ≈ 3.32 bits per token  (much less uncertain)
```

### Cross-Entropy

Cross-entropy measures how well distribution q approximates true distribution p:

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

```
Connection to LLM training:

  True distribution p:  one-hot vector [0, 0, 1, 0, ..., 0]
                        (the actual next token)

  Model distribution q: softmax output [0.01, 0.02, 0.85, 0.03, ...]
                        (model's predicted probabilities)

  Cross-entropy loss = -log q(correct token)
                     = -log(0.85)
                     ≈ 0.163

  Lower cross-entropy → model assigns higher probability to correct tokens

  H(p, q) = H(p) + D_KL(p ‖ q)
  ↑ CE loss   ↑ true entropy   ↑ "wasted bits" (what we minimize)
             (constant)
```

### KL Divergence

KL divergence measures how one distribution differs from another:

$$D_{KL}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$

```
Properties:
  - D_KL(p ‖ q) ≥ 0  (Gibbs' inequality)
  - D_KL(p ‖ q) = 0  iff p = q
  - NOT symmetric: D_KL(p ‖ q) ≠ D_KL(q ‖ p)

AI Applications:
  ┌─────────────────────────────────────────────────────┐
  │  RLHF:   KL penalty between policy and reference   │
  │          L = R(y) - β × D_KL(π_θ ‖ π_ref)         │
  │          (prevent policy from diverging too far)    │
  │                                                     │
  │  VAE:    KL term in ELBO loss                       │
  │          L = Reconstruction + D_KL(q(z|x) ‖ p(z))  │
  │          (keep latent space close to prior)         │
  │                                                     │
  │  Knowledge Distillation:                            │
  │          L = KL(teacher_probs ‖ student_probs)      │
  │          (student mimics teacher's distribution)    │
  │                                                     │
  │  DPO:    Implicit KL constraint in preference       │
  │          optimization objective                     │
  └─────────────────────────────────────────────────────┘
```

### Mutual Information

$$I(X; Y) = H(X) - H(X|Y) = D_{KL}(P(X,Y) \| P(X)P(Y))$$

```
Mutual Information Intuition:

  How much does knowing Y reduce uncertainty about X?

  I(X; Y) = 0  → X and Y are independent
  I(X; Y) > 0  → knowing Y tells us something about X

  AI applications:
  - Feature selection: pick features with highest MI with target
  - InfoNCE loss (contrastive learning): maximizes MI between views
  - Representation learning: learn representations with high MI
```

### Perplexity — The "Intuitive" Metric

$$\text{PPL} = 2^{H(p, q)} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i})\right)$$

```
Perplexity Intuition:

  "How many tokens is the model effectively choosing between?"

  PPL = 1:      Perfect prediction (only 1 possible next token)
  PPL = 10:     Model is "confused" between ~10 equally likely tokens
  PPL = 50,000: Random guessing over entire vocabulary

  GPT-2 on WebText: PPL ≈ 18
  GPT-3 on WebText: PPL ≈ 10
  GPT-4:            PPL likely < 8 (not publicly reported)

  Lower perplexity → better language model
```

---

## 11.5 Optimization Theory

### Convex vs Non-Convex Optimization

```
Convex Function:                    Non-Convex Function (Neural Networks):

Loss │                              Loss │
     │ ╲                                 │    ╲     local
     │  ╲                                │     ╲   minimum
     │   ╲      ╱                        │   ╲  ╲_╱  ╱
     │    ╲    ╱                          │    ╲     ╱ ╲
     │     ╲__╱ ← unique global          │     ╲___╱   ╲___  ← global
     │          minimum                   │     saddle      minimum
     └──────────────                      │     point
                                          └────────────────────

Convex: Any local minimum IS the global minimum
  Examples: Linear regression, logistic regression, SVMs

Non-convex: Multiple local minima, saddle points
  Examples: Neural networks (all of deep learning!)

  Key insight: In high dimensions, saddle points are MORE common
  than local minima (Dauphin et al., 2014). This is actually
  favorable — SGD can escape saddle points but not bad local minima.
```

### Gradient Descent Variants

```
┌──────────────────────────────────────────────────────────────────┐
│              Gradient Descent Variants                             │
├─────────────┬────────────────────────────────────────────────────┤
│ Batch GD    │ Uses ALL data for each update                      │
│             │ Stable but slow, no escaping minima                │
│             │ θ ← θ - α∇L_total(θ)                              │
├─────────────┼────────────────────────────────────────────────────┤
│ SGD         │ Uses ONE sample per update                         │
│             │ Noisy but fast, can escape local minima            │
│             │ θ ← θ - α∇L_i(θ)                                  │
├─────────────┼────────────────────────────────────────────────────┤
│ Mini-batch  │ Uses B samples per update (B = 32...4096)          │
│ SGD         │ Best of both: stable enough, fast enough           │
│             │ θ ← θ - α∇L_batch(θ)                              │
├─────────────┼────────────────────────────────────────────────────┤
│ Momentum    │ Accumulates past gradients (like a ball rolling)   │
│             │ v ← βv + ∇L(θ);  θ ← θ - αv                      │
│             │ Accelerates in consistent gradient directions      │
├─────────────┼────────────────────────────────────────────────────┤
│ Adam        │ Adaptive learning rate per parameter               │
│             │ m ← β₁m + (1-β₁)g     (1st moment, mean)          │
│             │ v ← β₂v + (1-β₂)g²    (2nd moment, variance)      │
│             │ θ ← θ - α × m̂/√(v̂ + ε)                            │
│             │ Default for deep learning, especially LLMs         │
├─────────────┼────────────────────────────────────────────────────┤
│ AdamW       │ Adam + decoupled weight decay                      │
│             │ θ ← θ - α(m̂/√(v̂ + ε) + λθ)                       │
│             │ Standard for LLM training (GPT, LLaMA, etc.)      │
└─────────────┴────────────────────────────────────────────────────┘
```

### Learning Rate Schedules

```
Constant:           Warmup + Cosine Decay:         Warmup + Linear Decay:
                    (Most common for LLMs)
LR │ ─────────     LR │    ╱╲                     LR │    ╱╲
   │                  │   ╱  ╲                        │   ╱  ╲
   │                  │  ╱    ╲                        │  ╱    ╲
   │                  │ ╱      ╲___                    │ ╱      ╲
   └──────────        │╱              ╲                │╱        ╲____
     Steps            └───────────────── Steps         └──────────── Steps
                      ↑                                ↑
                    warmup                           warmup

Warmup prevents early instability when gradients are large.
Cosine decay: lr(t) = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))
```

---

## 11.6 Numerical Computing Essentials

### Floating Point Representation

```
IEEE 754 Floating Point Formats:

              Sign  Exponent  Mantissa   Total bits  Range
FP32          1     8         23         32          ±3.4 × 10³⁸
FP16          1     5         10         16          ±6.5 × 10⁴
BF16          1     8         7          16          ±3.4 × 10³⁸
FP8 (E4M3)   1     4         3          8           ±240
FP8 (E5M2)   1     5         2          8           ±57344

┌─────────────────────────────────────────────────────────────────┐
│  Why BF16 dominates LLM training:                               │
│                                                                  │
│  FP32: ████████████████████████████████  (32 bits) ← too slow   │
│         1 sign │ 8 exponent │ 23 mantissa                       │
│                                                                  │
│  BF16: ████████████████  (16 bits) ← same range as FP32!       │
│         1 sign │ 8 exponent │ 7 mantissa                        │
│                                                                  │
│  FP16: ████████████████  (16 bits) ← can overflow!             │
│         1 sign │ 5 exponent │ 10 mantissa                       │
│                                                                  │
│  BF16 = same exponent range as FP32 + 2× memory savings        │
│  Less precision but rarely matters for neural network training  │
└─────────────────────────────────────────────────────────────────┘
```

### Numerical Stability

```python
# BAD: Naive softmax (overflow for large values)
def softmax_naive(x):
    return np.exp(x) / np.sum(np.exp(x))  # exp(1000) = inf!

# GOOD: Numerically stable softmax
def softmax_stable(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)  # shift by max → largest exp is exp(0) = 1
    return exp_x / np.sum(exp_x)

# BAD: Naive log-sum-exp
def logsumexp_naive(x):
    return np.log(np.sum(np.exp(x)))  # overflow!

# GOOD: Log-sum-exp trick
def logsumexp_stable(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))
```

**Why this matters:** Transformers compute attention as softmax(QKᵀ/√d). Without the √d scaling and numerical tricks, attention scores overflow in FP16/BF16.

---

## 11.7 Dimensionality Reduction

### The Curse of Dimensionality

```
As dimensions grow, distances become meaningless:

2D:   Points spread out,          100D:  All points are roughly
      clear clusters                      equidistant from each other

  ●                                "In high dimensions,
    ●  ●                           everyone is a loner."
  ●  ●
      ●  <cluster>                 max_dist / min_dist → 1
                                   as d → ∞
   ●
 ●  ●  <cluster>                   This breaks: KNN, clustering,
   ●                               distance-based retrieval
```

### PCA (Principal Component Analysis)

```
PCA Algorithm:
  1. Center data: X̄ = X - mean(X)
  2. Compute covariance: Σ = X̄ᵀX̄ / (n-1)
  3. Eigendecompose: Σ = VΛVᵀ
  4. Project: Z = X̄V_k  (keep top-k eigenvectors)

Intuition:
  Original (3D)           After PCA (2D)

      z                   y │    ● ●
      │  ●  ●                │  ●    ●  ●
      │ ● ● ●              │ ●  ●  ●   ●
      │● ● ●               │   ●  ●
      └───── y              └──────────── x
     ╱
    x                     Data is "flat" along z-axis
                          → drop z, keep x and y (PC1, PC2)

AI applications:
  - Visualizing embeddings (768D → 2D)
  - Noise reduction
  - Feature compression
  - Understanding what dimensions models use
```

### t-SNE and UMAP

```
Comparison for Embedding Visualization:

  PCA                    t-SNE                   UMAP
  ─────────             ─────────               ─────────
  Linear                Nonlinear               Nonlinear
  Global structure ✓    Local structure ✓        Local + Global ✓
  Fast O(nd²)           Slow O(n²)              Fast O(n log n)
  Deterministic         Stochastic              Stochastic

  Use for:              Use for:                Use for:
  Quick overview,       Finding clusters,       Best general-purpose
  preprocessing         visualization           visualization
```

---

## 11.8 Practical Implementation

<details>
<summary><strong>Complete Code: Core Math Operations for AI</strong></summary>

```python
import numpy as np

# ============================================================
# LINEAR ALGEBRA ESSENTIALS
# ============================================================

def cosine_similarity(a, b):
    """Cosine similarity — the core of semantic search."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def svd_low_rank(matrix, rank):
    """Low-rank approximation via SVD — the idea behind LoRA."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]

def softmax(x, temperature=1.0):
    """Numerically stable softmax with temperature."""
    x = x / temperature
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ============================================================
# INFORMATION THEORY
# ============================================================

def entropy(probs):
    """Shannon entropy of a probability distribution."""
    probs = probs[probs > 0]  # avoid log(0)
    return -np.sum(probs * np.log2(probs))

def cross_entropy(true_probs, predicted_probs):
    """Cross-entropy — THE loss function for language models."""
    predicted_probs = np.clip(predicted_probs, 1e-10, 1.0)
    return -np.sum(true_probs * np.log(predicted_probs))

def kl_divergence(p, q):
    """KL divergence D_KL(p || q) — used in RLHF, VAE, distillation."""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))

def perplexity(log_probs):
    """Perplexity from log probabilities — lower is better."""
    return np.exp(-np.mean(log_probs))

# ============================================================
# PCA FROM SCRATCH
# ============================================================

def pca(X, n_components):
    """PCA for dimensionality reduction (e.g., visualizing embeddings)."""
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    # Covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx[:n_components]]
    # Project
    return X_centered @ eigenvectors

# ============================================================
# GRADIENT DESCENT IMPLEMENTATION
# ============================================================

def gradient_descent(gradient_fn, x0, lr=0.01, n_steps=1000):
    """Basic gradient descent — the foundation of all neural net training."""
    x = x0.copy()
    history = [x.copy()]
    for _ in range(n_steps):
        grad = gradient_fn(x)
        x = x - lr * grad
        history.append(x.copy())
    return x, history

def adam_optimizer(gradient_fn, x0, lr=0.001, beta1=0.9, beta2=0.999,
                   eps=1e-8, n_steps=1000):
    """Adam optimizer — the standard for LLM training."""
    x = x0.copy()
    m = np.zeros_like(x)  # first moment
    v = np.zeros_like(x)  # second moment
    for t in range(1, n_steps + 1):
        grad = gradient_fn(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)     # bias correction
        v_hat = v / (1 - beta2**t)     # bias correction
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
    return x

# Example: Minimize f(x) = x² + 4x + 4  (minimum at x = -2)
gradient = lambda x: np.array([2*x[0] + 4])
result, _ = gradient_descent(gradient, np.array([10.0]), lr=0.1, n_steps=50)
print(f"Minimum found at: {result[0]:.4f}")  # ≈ -2.0
```

</details>

---

## 11.9 Interview Questions

### Conceptual Questions

**Q1: Why do we use cross-entropy loss for classification and language modeling? What's its connection to MLE?**

Minimizing cross-entropy is equivalent to maximum likelihood estimation. For a language model, minimizing $-\sum_i \log P(x_i | x_{<i}; \theta)$ maximizes the probability of the training data under the model. Cross-entropy also equals $H(p) + D_{KL}(p \| q)$, so minimizing it minimizes the KL divergence between the true and predicted distributions (since $H(p)$ is constant).

**Q2: Explain the relationship between eigenvalues/eigenvectors and PCA. Why do the top eigenvectors capture the most variance?**

PCA projects data onto the eigenvectors of the covariance matrix. The eigenvalue $\lambda_i$ equals the variance of data projected onto eigenvector $v_i$. By the Rayleigh quotient, the direction maximizing projected variance is exactly the top eigenvector. Each subsequent eigenvector captures the next most variance orthogonal to previous ones.

**Q3: What is the KL divergence penalty in RLHF, and why is it necessary?**

The KL penalty $\beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$ prevents the fine-tuned policy from diverging too far from the base model. Without it, the model can "hack" the reward model by generating outputs that score high on the imperfect reward model but are actually low quality (reward hacking/overoptimization). The KL term acts as a regularizer anchoring outputs to the pre-trained distribution.

**Q4: Why does BF16 work better than FP16 for LLM training?**

BF16 has the same 8-bit exponent as FP32 (range ±3.4×10³⁸) but only 7 mantissa bits. FP16 has a 5-bit exponent (range ±65,504) which causes overflow during training when gradient values exceed this range. BF16 trades precision for range, which is acceptable because neural network training is inherently noisy — the exact precision of individual values matters less than avoiding overflow.

**Q5: Explain the curse of dimensionality and its implications for nearest-neighbor retrieval in RAG systems.**

In high dimensions, distances between all points converge — $\frac{d_{max} - d_{min}}{d_{min}} \to 0$ as dimensions grow. This means nearest-neighbor search becomes less meaningful because all points are roughly equidistant. RAG systems mitigate this through: (1) learned embeddings that concentrate semantic meaning in fewer effective dimensions, (2) approximate nearest neighbor algorithms (HNSW) that exploit local structure, and (3) dimensionality reduction techniques like Matryoshka embeddings.

### Coding Questions

**Q6: Implement numerically stable softmax and cross-entropy loss from scratch.**

```python
def stable_softmax(logits):
    """Softmax that won't overflow or underflow."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

def cross_entropy_loss(logits, target_indices):
    """
    Cross-entropy loss for classification/language modeling.
    logits: (batch_size, vocab_size) raw model outputs
    target_indices: (batch_size,) integer labels
    """
    probs = stable_softmax(logits)
    batch_size = logits.shape[0]
    # Select probability of correct class for each example
    correct_probs = probs[np.arange(batch_size), target_indices]
    # Clip for numerical stability
    correct_probs = np.clip(correct_probs, 1e-10, 1.0)
    return -np.mean(np.log(correct_probs))
```

### System Design Questions

**Q7: Design a statistical testing framework for comparing ML models in production.**

```
┌─────────────────────────────────────────────────────────┐
│           ML Model Comparison Framework                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. DEFINE METRICS                                       │
│     Primary: task metric (accuracy, BLEU, user engagement)│
│     Guardrail: latency, cost, safety score               │
│     Sample size: compute n for desired power (1-β ≥ 0.8)│
│                                                          │
│  2. TRAFFIC SPLITTING                                    │
│     ┌──────────┐  50%  ┌──────────┐                     │
│     │  Users   │──────▶│ Model A  │─── Metrics A        │
│     │          │  50%  ┌──────────┐                      │
│     │          │──────▶│ Model B  │─── Metrics B        │
│     └──────────┘       └──────────┘                      │
│     Hash(user_id) for consistent assignment              │
│                                                          │
│  3. STATISTICAL TESTS                                    │
│     Continuous metrics: Welch's t-test or bootstrap CI   │
│     Proportions: Chi-squared or Fisher's exact test      │
│     Multiple comparisons: Bonferroni correction          │
│     Sequential testing: avoid peeking problem            │
│                                                          │
│  4. DECISION                                             │
│     p < α AND practical significance (effect size > δ)   │
│     Check all guardrail metrics pass                     │
│     Gradual rollout: 5% → 25% → 50% → 100%             │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 11.10 Key Papers & Resources

| Paper/Resource                                                    | Year | Why It Matters                                                                              |
| ----------------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------------- |
| _Deep Learning_ (Goodfellow, Bengio, Courville)                   | 2016 | The comprehensive textbook — chapters on linear algebra, probability, numerical computation |
| _Mathematics for Machine Learning_ (Deisenroth et al.)            | 2020 | Free textbook — rigorous but accessible math foundations                                    |
| _Identifying and Attacking Saddle Points_ (Dauphin et al.)        | 2014 | Why saddle points, not local minima, are the real challenge                                 |
| _Adam: A Method for Stochastic Optimization_ (Kingma & Ba)        | 2014 | The optimizer behind virtually all modern LLMs                                              |
| _Decoupled Weight Decay Regularization_ (Loshchilov & Hutter)     | 2017 | AdamW — the fix that made Adam work properly                                                |
| _Information Theory, Inference, and Learning Algorithms_ (MacKay) | 2003 | Best intuition-building book on information theory                                          |
| _An Introduction to Statistical Learning_ (James et al.)          | 2013 | Accessible statistics for ML practitioners                                                  |

---

[← Module 10: Advanced Topics](../module-10-advanced/README.md) | [Module 12: Classical ML →](../module-12-classical-ml/README.md)
