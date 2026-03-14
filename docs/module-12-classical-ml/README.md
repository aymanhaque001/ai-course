# Module 12: Classical Machine Learning

> **Prerequisites:** Module 11 (Math Foundations)  
> **Estimated Time:** 10-12 hours  
> **Relevance:** Classical ML is the foundation. Even when building LLM systems, you need ML fundamentals for feature engineering, baselines, evaluation, and knowing when NOT to use an LLM

---

## 12.1 The Machine Learning Framework

```
┌──────────────────────────────────────────────────────────────────┐
│                 THE ML PROBLEM FRAMEWORK                          │
│                                                                   │
│  1. DEFINE the problem                                            │
│     Classification? Regression? Ranking? Clustering?              │
│                                                                   │
│  2. COLLECT & prepare data                                        │
│     Features, labels, train/val/test split                        │
│                                                                   │
│  3. CHOOSE a model                                                │
│     Simple → complex (Occam's razor)                              │
│                                                                   │
│  4. TRAIN (fit parameters)                                        │
│     Minimize loss on training data                                │
│                                                                   │
│  5. EVALUATE (tune hyperparameters)                               │
│     Measure on validation data, iterate                           │
│                                                                   │
│  6. TEST (final assessment)                                       │
│     Report on held-out test data — ONE time only                  │
│                                                                   │
│  7. DEPLOY & monitor                                              │
│     Serve predictions, detect drift                               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### The Bias-Variance Tradeoff

```
Error = Bias² + Variance + Irreducible Noise

High Bias (Underfitting):       High Variance (Overfitting):
  Model is too simple             Model is too complex
  Misses true pattern             Memorizes noise

  Train Error: HIGH               Train Error: LOW
  Val Error:   HIGH               Val Error:   HIGH

  Example: Linear model           Example: Deep tree on
  on nonlinear data               small dataset

                   Sweet Spot
Error │  ╲Bias²         ╱
      │   ╲╲           ╱╱ Variance
      │    ╲╲         ╱╱
      │     ╲╲       ╱╱
      │      ╲╲_____╱╱  ← optimal complexity
      │        Total Error
      └────────────────────── Model Complexity
        Linear  ←→  Deep Neural Net
```

### Train / Validation / Test Split

```
┌─────────────────── All Data ───────────────────┐
│                                                 │
│  ┌──────────────┐  ┌──────────┐  ┌──────────┐  │
│  │   Training   │  │Validation│  │   Test    │  │
│  │    60-80%    │  │  10-20%  │  │  10-20%  │  │
│  └──────────────┘  └──────────┘  └──────────┘  │
│                                                 │
│  Fit model          Tune hyper-   Final eval    │
│  parameters         parameters    (report this) │
│                     Model select                │
└─────────────────────────────────────────────────┘

K-Fold Cross-Validation (k=5):
  Fold 1: [VAL][Train][Train][Train][Train]
  Fold 2: [Train][VAL][Train][Train][Train]
  Fold 3: [Train][Train][VAL][Train][Train]
  Fold 4: [Train][Train][Train][VAL][Train]
  Fold 5: [Train][Train][Train][Train][VAL]

  Average metrics across all folds → more reliable estimate
  Use when: small datasets, need reliable performance estimate
```

---

## 12.2 Supervised Learning: Regression

### Linear Regression

The simplest model — a hyperplane that minimizes squared error:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$

```
Geometric Intuition (2D):

  y │        ● ╱
    │      ●  ╱ ●
    │    ●   ╱
    │  ●    ╱ ●
    │   ●  ╱      minimize Σ(yᵢ - ŷᵢ)²
    │     ╱ ●
    │    ╱
    └──────────── x

Closed-form solution:  w = (XᵀX)⁻¹Xᵀy   (Normal Equation)
                       O(n³) — impractical for large n

Gradient descent:      w ← w - α × ∂L/∂w = w - α × (-2/N) Xᵀ(y - Xw)
                       O(n²) per step — scales better
```

### Regularized Regression

```
┌──────────────────────────────────────────────────────────────────┐
│                    Regularization Comparison                      │
├──────────────┬──────────────────────────────────────────────────┤
│  Ridge (L2)  │  L = MSE + λ Σ wᵢ²                               │
│              │  Shrinks all weights toward 0                     │
│              │  Keeps all features (no sparsity)                 │
│              │  Best when many features all contribute           │
├──────────────┼──────────────────────────────────────────────────┤
│  Lasso (L1)  │  L = MSE + λ Σ |wᵢ|                              │
│              │  Drives some weights to EXACTLY 0                 │
│              │  Automatic feature selection                      │
│              │  Best when few features are relevant              │
├──────────────┼──────────────────────────────────────────────────┤
│  ElasticNet  │  L = MSE + λ₁ Σ |wᵢ| + λ₂ Σ wᵢ²                │
│              │  Combines L1 + L2                                 │
│              │  Handles correlated features better than Lasso    │
└──────────────┴──────────────────────────────────────────────────┘

λ too small → overfitting (no regularization effect)
λ too large → underfitting (all weights → 0)

Connection to deep learning: Weight decay in AdamW IS L2 regularization
Connection to LLMs: LoRA adds low-rank constraints = implicit regularization
```

### Polynomial & Non-Linear Regression

```
Linear:        y = w₁x + b        (straight line)
Polynomial:    y = w₁x + w₂x² + w₃x³ + b   (curves)
               Still "linear" in parameters! Just engineer features.

Feature Engineering:
  Original feature: x = [temperature]
  Polynomial features: x' = [temp, temp², temp³, temp×humidity, ...]

  This is why feature engineering matters — even a linear model
  can fit complex patterns with good features.
```

---

## 12.3 Supervised Learning: Classification

### Logistic Regression

Despite the name, this is a **classification** algorithm:

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

```
Decision Boundary:

  x₂ │  ● ● ●          ● = class 1
     │  ●  ●  ╲         ○ = class 0
     │  ●    ╲ ○ ○
     │    ●  ╲  ○ ○     The sigmoid maps wᵀx + b
     │      ╲ ○  ○ ○    from (-∞, +∞) to (0, 1)
     │     ╲  ○ ○
     └────╲───────── x₁
          decision
          boundary (linear)

Loss: Binary Cross-Entropy
  L = -[y log(ŷ) + (1-y) log(1-ŷ)]

  y=1, ŷ=0.9 → L = -log(0.9) = 0.105  (small loss, correct!)
  y=1, ŷ=0.1 → L = -log(0.1) = 2.302  (large loss, wrong!)
```

### Multiclass Classification

```
One-vs-Rest (OvR):           Softmax Regression:
  K binary classifiers         Single K-class model

  Class 1 vs Rest              P(y=k) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)
  Class 2 vs Rest
  Class 3 vs Rest              This IS the output layer of
  ...                          neural networks / LLMs!
  Pick highest score
```

### Support Vector Machines (SVMs)

```
SVM Intuition — Maximum Margin Classifier:

  x₂ │  ●           ○
     │   ●    ←margin→    ○
     │  ● ●    │    ○  ○
     │   ●     │     ○
     │    ●    │    ○ ○
     └─────────────────── x₁
            hyperplane

Find the hyperplane that:
  1. Correctly separates classes
  2. Maximizes the margin (distance to nearest points)

  Support vectors = the points closest to the boundary
  (these define the solution — other points don't matter!)

The Kernel Trick — Handle Non-Linear Data:

  Original space (not separable):     Feature space (separable!):

      ○  ○                               ○  ○
    ○ ● ● ○                              ╱────╲
    ○ ● ● ○         φ(x)→             ○ ╱● ●  ╲○
      ○  ○                            ○╱ ● ●   ╲○
                                       ╲  ○  ○  ╱
                                        ╲______╱

  Kernels:   K(x, x') = φ(x) · φ(x')
    Linear:       K = xᵀx'
    Polynomial:   K = (xᵀx' + c)ᵈ
    RBF/Gaussian: K = exp(-γ‖x - x'‖²)    ← most common

  Compute in high-D space WITHOUT actually transforming!
```

**When SVMs beat deep learning:** Small datasets, tabular data, when interpretability of margins matters.

### Classification Metrics

```
Confusion Matrix:
                    Predicted
                  Pos    Neg
Actual  Pos  │   TP   │  FN   │
        Neg  │   FP   │  TN   │

Accuracy  = (TP + TN) / Total      ← misleading if classes imbalanced!
Precision = TP / (TP + FP)         ← "of predictions, how many correct?"
Recall    = TP / (TP + FN)         ← "of actual positives, how many found?"
F1 Score  = 2 × P × R / (P + R)   ← harmonic mean of precision & recall

┌───────────────────────────────────────────────────────────────────┐
│  When to prioritize what:                                         │
│                                                                    │
│  HIGH PRECISION needed:  Spam filter (don't block real email)     │
│  HIGH RECALL needed:     Cancer screening (don't miss cases)      │
│  BOTH:                   Content moderation (balanced F1)          │
│                                                                    │
│  For LLM applications:                                             │
│    Retrieval (RAG):    Precision@k, Recall@k, MAP, NDCG           │
│    Generation:         BLEU, ROUGE (but mostly human eval)        │
│    Classification:     Macro/Micro F1 across categories           │
└───────────────────────────────────────────────────────────────────┘

ROC Curve & AUC:
  TPR │  ╱────── ← good model (AUC ≈ 0.95)
      │ ╱
      │╱  ╱ ← random (AUC = 0.5)
      │  ╱
      │ ╱
      └──────── FPR

  AUC = Area Under ROC Curve (threshold-independent metric)
  AUC = 1.0 (perfect) ... 0.5 (random) ... 0.0 (perfectly wrong)
```

---

## 12.4 Decision Trees & Ensemble Methods

### Decision Trees

```
Tree Structure:

         ┌─────────────────┐
         │ Income > $50K?  │ ← root node
         └────────┬────────┘
              ╱        ╲
           Yes          No
            ╱              ╲
  ┌──────────────┐   ┌──────────────┐
  │ Age > 30?    │   │ Credit score │
  │              │   │   > 700?     │
  └──────┬───────┘   └──────┬───────┘
      ╱      ╲           ╱      ╲
    Yes      No        Yes      No
     ╱          ╲       ╱          ╲
 [Approve]  [Deny]  [Approve]  [Deny]    ← leaf nodes

Splitting Criteria:
  Classification: Gini impurity or Information Gain (entropy)
    Gini(S) = 1 - Σ pᵢ²
    IG(S, feature) = H(S) - Σ (|Sᵥ|/|S|) × H(Sᵥ)

  Regression: Mean Squared Error reduction
    Pick split that minimizes MSE in child nodes

Pros:                           Cons:
✓ Interpretable (visualize!)    ✗ Overfits easily (high variance)
✓ No feature scaling needed     ✗ Limited expressiveness
✓ Handles mixed feature types   ✗ Unstable (small data change →
✓ Built-in feature importance       different tree)
```

### Random Forests

```
Random Forest = Ensemble of Decision Trees + Bagging + Feature Randomness

Training:
  ┌─── Dataset ───┐
  │                │
  │  Bootstrap     │  ← sample WITH replacement
  │  Sample 1      │     (each tree sees ~63% of data)
  │  ┌──────┐      │
  │  │Tree 1│      │  + random feature subset at each split
  │  └──────┘      │    (√n features for classification)
  │                │
  │  Bootstrap     │
  │  Sample 2      │
  │  ┌──────┐      │
  │  │Tree 2│      │
  │  └──────┘      │
  │                │
  │  ...           │
  │  ┌──────────┐  │
  │  │Tree 500  │  │
  │  └──────────┘  │
  └────────────────┘

Prediction:
  Classification: Majority vote across all trees
  Regression: Average prediction across all trees

Why it works:
  - Each tree overfits differently (different data, different features)
  - Averaging reduces variance (errors cancel out)
  - bias stays similar to individual trees

  "Many weak learners → one strong learner"
```

### Gradient Boosting (XGBoost, LightGBM, CatBoost)

```
Gradient Boosting = Sequentially fit trees to RESIDUALS

Step 1: Fit Tree₁ to data        → predictions F₁
Step 2: Fit Tree₂ to RESIDUALS of F₁  → F₂ = F₁ + αTree₂
Step 3: Fit Tree₃ to RESIDUALS of F₂  → F₃ = F₂ + αTree₃
...
Final:  F = Σ αᵢ Treeᵢ

┌──────────────────────────────────────────────────────────────────┐
│            Gradient Boosting Comparison                            │
├──────────────┬──────────────┬──────────────┬─────────────────────┤
│              │   XGBoost    │  LightGBM    │    CatBoost         │
├──────────────┼──────────────┼──────────────┼─────────────────────┤
│ Tree growth  │ Level-wise   │ Leaf-wise    │ Symmetric           │
│ Speed        │ Fast         │ Faster       │ Fast                │
│ Categoricals │ Manual encode│ Native       │ Native (best)       │
│ Overfitting  │ Regularized  │ Can overfit  │ Ordered boosting    │
│ GPU support  │ ✓            │ ✓            │ ✓                   │
│ Best for     │ General      │ Large data   │ Categorical-heavy   │
├──────────────┴──────────────┴──────────────┴─────────────────────┤
│                                                                    │
│  STILL THE CHAMPION FOR TABULAR DATA in 2026.                     │
│  Deep learning has not clearly beaten gradient boosting            │
│  on structured/tabular datasets. Always try XGBoost/LightGBM      │
│  as a baseline before reaching for neural networks.                │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Ensemble Methods Summary

```
┌────────────┬────────────────┬──────────────────────────────────┐
│  Method    │  Strategy      │  Effect                          │
├────────────┼────────────────┼──────────────────────────────────┤
│ Bagging    │ Parallel trees │ Reduces variance (overfitting)   │
│ (RF)       │ on bootstraps  │                                  │
├────────────┼────────────────┼──────────────────────────────────┤
│ Boosting   │ Sequential     │ Reduces bias (underfitting)      │
│ (XGB)      │ error correct  │                                  │
├────────────┼────────────────┼──────────────────────────────────┤
│ Stacking   │ Meta-learner   │ Combines diverse model types     │
│            │ on predictions │                                  │
└────────────┴────────────────┴──────────────────────────────────┘
```

---

## 12.5 Unsupervised Learning

### K-Means Clustering

```
Algorithm:
  1. Initialize k centroids randomly
  2. ASSIGN each point to nearest centroid
  3. UPDATE centroids = mean of assigned points
  4. Repeat 2-3 until convergence

Iteration 1:          Iteration 5 (converged):

    ●  +  ●              ●●● +
  ●  ●    ●  ○          ●●●●
  ●    ●  ○ ○ ○              ○○ +  ○
    ●  +  ○  ○              ○○○  ○

  + = centroid            + = centroid (stable)

Choosing k:
  Elbow method:  Plot cost vs k, find the "elbow"
  Silhouette score: Measure cluster cohesion vs separation

  Cost │  ╲
       │   ╲
       │    ╲___        ← elbow at k=3
       │        ╲___________
       └──────────────────── k
         1  2  3  4  5  6

Limitations:
  - Assumes spherical clusters
  - Must specify k
  - Sensitive to initialization (use k-means++)
  - Doesn't handle different cluster sizes well
```

### Hierarchical Clustering

```
Agglomerative (Bottom-Up):

  Start: each point is its own cluster
  Merge closest clusters until one remains

  Dendrogram (tree of merges):

  Height │         ┌─────────────────┐
         │    ┌────┤                 │
         │    │    │    ┌────────────┤
         │  ┌─┤    │    │            │
         │  │ │    │  ┌─┤            │
         └──┴─┴────┴──┴─┴────────────
            A B    C  D E

  Cut the dendrogram at desired height → get clusters

  Linkage methods:
    Single:   min distance between clusters  (can chain)
    Complete: max distance between clusters  (compact)
    Average:  mean distance between clusters (balanced)
    Ward:     minimize within-cluster variance (most common)
```

### Dimensionality Reduction: PCA

```
PCA in Practice:

  1. Standardize features (zero mean, unit variance)
  2. Compute covariance matrix
  3. Find eigenvectors (principal components)
  4. Project onto top-k components

Explained Variance:

  % Var │ ████████   PC1 explains 45% of variance
        │ █████      PC2 explains 25%
        │ ███        PC3 explains 15%
        │ ██         PC4 explains 10%
        │ █          PC5 explains 5%
        └───────
          PC1 PC2 PC3 PC4 PC5

  Keep enough PCs to explain ~95% of variance

  Applications in AI:
    - Visualize high-dimensional embeddings (768D → 2D)
    - Reduce feature dimensionality before ML
    - Noise reduction (discard low-variance components)
    - Compression (images, signals)
```

### Anomaly Detection

```
┌──────────────────────────────────────────────────────────────────┐
│                    Anomaly Detection Methods                      │
├──────────────┬──────────────────────────────────────────────────┤
│ Statistical  │ Z-score, IQR — flag points > 3σ from mean       │
│              │ Simple, works for univariate data                │
├──────────────┼──────────────────────────────────────────────────┤
│ Isolation    │ Random trees: anomalies are EASY to isolate      │
│ Forest       │ Short path in tree → anomaly                     │
│              │ Scales well, works with high dimensions          │
├──────────────┼──────────────────────────────────────────────────┤
│ One-Class    │ Learn boundary of normal data                    │
│ SVM         │ Kernel-based, works with small normal datasets    │
├──────────────┼──────────────────────────────────────────────────┤
│ Autoencoders│ High reconstruction error → anomaly              │
│ (deep)      │ Learn compressed representation of "normal"      │
│              │ Works well for images, time series                │
├──────────────┼──────────────────────────────────────────────────┤
│ LLM-based   │ Perplexity-based anomaly detection               │
│              │ Unusual text patterns → high perplexity           │
└──────────────┴──────────────────────────────────────────────────┘

Real-world applications:
  - Fraud detection (credit cards, transactions)
  - Content moderation (unusual posts/patterns)
  - Model monitoring (data drift detection)
  - Security (intrusion detection)
```

---

## 12.6 Feature Engineering

Feature engineering is often **more important than model choice** for classical ML:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Feature Engineering Toolkit                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  NUMERICAL:                                                       │
│    Scaling:     StandardScaler (z-score), MinMaxScaler (0-1)      │
│    Transform:   Log (skewed data), power transforms (Box-Cox)     │
│    Binning:     Age → [child, teen, adult, senior]                │
│    Interactions: feature₁ × feature₂                              │
│    Polynomial:  x, x², x³, x₁×x₂                                 │
│                                                                   │
│  CATEGORICAL:                                                     │
│    One-Hot:     color → [is_red, is_blue, is_green]               │
│    Label:       [low, medium, high] → [0, 1, 2]                  │
│    Target:      Encode with mean of target variable               │
│    Frequency:   Encode with occurrence count                      │
│    Embedding:   Learn dense representation (small NN)             │
│                                                                   │
│  TEXT (Pre-LLM):                                                  │
│    Bag of Words: Count word occurrences → sparse vector           │
│    TF-IDF:      Weight by term frequency × inverse doc frequency  │
│    n-grams:     "not good" captured as bigram                     │
│                                                                   │
│  TEMPORAL:                                                        │
│    Time features: hour, day, month, is_weekend, is_holiday        │
│    Lag features:  value_t-1, value_t-7 (weekly pattern)           │
│    Rolling:       7-day moving average, expanding statistics      │
│                                                                   │
│  MISSING DATA:                                                    │
│    Imputation:  mean, median, mode, KNN-based, model-based        │
│    Indicator:   Add is_missing binary feature                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Feature Selection

```
Methods:

Filter methods:    Statistical tests (chi², mutual info, correlation)
                   Fast, model-independent

Wrapper methods:   Forward selection, backward elimination, RFE
                   Slow but finds best feature subsets for given model

Embedded methods:  L1 regularization (Lasso), tree feature importance
                   Feature selection built into training

Modern approach:   XGBoost/LightGBM feature importance + SHAP values
                   → understand WHY features matter, not just IF
```

---

## 12.7 Hyperparameter Tuning

```
┌──────────────────────────────────────────────────────────────────┐
│                    Hyperparameter Search Strategies                │
├──────────────┬──────────────────────────────────────────────────┤
│ Grid Search  │  Try all combinations on a grid                   │
│              │  ● ● ● ● ●    Exhaustive, expensive              │
│              │  ● ● ● ● ●    O(nᵈ) for d hyperparameters       │
│              │  ● ● ● ● ●                                       │
├──────────────┼──────────────────────────────────────────────────┤
│ Random       │  Random points in hyperparameter space             │
│ Search       │    ●     ●     More efficient than grid!          │
│              │  ●    ●        Bergstra & Bengio (2012)           │
│              │      ●    ●    Better coverage of important dims  │
├──────────────┼──────────────────────────────────────────────────┤
│ Bayesian     │  Build probability model of objective             │
│ Optimization │  GP (Gaussian Process) models the loss surface    │
│              │  Acquisition function picks next point            │
│              │  Most sample-efficient                            │
│              │  Tools: Optuna, Ray Tune, W&B Sweeps              │
├──────────────┼──────────────────────────────────────────────────┤
│ Hyperband /  │  Aggressive early stopping of bad configs         │
│ ASHA        │  Allocate more budget to promising runs            │
│              │  Best for expensive models (deep learning)        │
└──────────────┴──────────────────────────────────────────────────┘

Common Hyperparameters to Tune:

  Trees (RF/XGB):        Learning rate, max depth, n_estimators,
                          min_samples_leaf, subsample

  Linear models:          Regularization strength (λ/C)

  SVMs:                   C (regularization), γ (RBF kernel width)

  Neural Networks:        Learning rate, batch size, layers,
                          dropout, weight decay

  LLM Fine-tuning:       Learning rate (1e-5 to 5e-5), epochs (1-5),
                          LoRA rank (4-64), LoRA alpha (16-128)
```

---

## 12.8 When to Use What

```
┌──────────────────────────────────────────────────────────────────┐
│              MODEL SELECTION DECISION GUIDE                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Data type?                                                       │
│    │                                                              │
│    ├── Tabular/Structured ──→ XGBoost/LightGBM first!            │
│    │     │                    (still best for tabular in 2026)    │
│    │     ├── Need interpretability? → Decision Tree / Linear      │
│    │     ├── Small data (<1K)?      → SVM, Random Forest         │
│    │     └── Very high dim?         → L1 regularization → RF     │
│    │                                                              │
│    ├── Text ──────────────→ LLM (fine-tuned or prompted)         │
│    │     ├── Quick baseline?        → TF-IDF + Logistic Reg      │
│    │     ├── Need embeddings?       → Sentence Transformers      │
│    │     └── Simple classification? → BERT fine-tune or LLM      │
│    │                                                              │
│    ├── Images ───────────→ CNN or Vision Transformer              │
│    │     ├── Transfer learning?     → ResNet/EfficientNet + head  │
│    │     └── Multimodal?            → CLIP / LLaVA                │
│    │                                                              │
│    ├── Time Series ──────→ XGBoost with lag features              │
│    │     ├── Classical?             → ARIMA, Prophet              │
│    │     └── Deep learning?         → Transformer, LSTM           │
│    │                                                              │
│    └── Graph ────────────→ GNN (Graph Neural Network)            │
│                                                                   │
│  RULE: Start simple, add complexity only when needed.             │
│  RULE: If LLM works with prompting, don't fine-tune.             │
│  RULE: If fine-tuned BERT works, don't use GPT-4.                │
│  RULE: If XGBoost works on tabular data, don't use deep learning.│
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 12.9 Practical Implementation

<details>
<summary><strong>Complete Code: Classical ML Pipeline</strong></summary>

```python
import numpy as np
from collections import Counter

# ============================================================
# LOGISTIC REGRESSION FROM SCRATCH
# ============================================================

class LogisticRegression:
    """Binary logistic regression with L2 regularization."""

    def __init__(self, lr=0.01, n_iters=1000, reg_lambda=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.reg_lambda = reg_lambda

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            z = X @ self.weights + self.bias
            predictions = self.sigmoid(z)

            # Gradients (cross-entropy loss + L2 regularization)
            dw = (1 / n_samples) * (X.T @ (predictions - y)) + \
                 self.reg_lambda * self.weights
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# ============================================================
# DECISION TREE FROM SCRATCH
# ============================================================

class DecisionTree:
    """Decision tree classifier using Gini impurity."""

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def gini(self, y):
        """Gini impurity: 1 - Σ pᵢ²"""
        counts = Counter(y)
        n = len(y)
        return 1 - sum((c / n) ** 2 for c in counts.values())

    def best_split(self, X, y):
        """Find best feature and threshold to split on."""
        best_gain, best_feat, best_thresh = -1, None, None
        parent_gini = self.gini(y)
        n = len(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                left_mask = X[:, feature] <= thresh
                right_mask = ~left_mask

                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                # Weighted Gini impurity of children
                left_gini = self.gini(y[left_mask])
                right_gini = self.gini(y[right_mask])
                weighted = (sum(left_mask) * left_gini +
                          sum(right_mask) * right_gini) / n

                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature
                    best_thresh = thresh

        return best_feat, best_thresh

    def build_tree(self, X, y, depth=0):
        # Stopping conditions
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(set(y)) == 1):
            return Counter(y).most_common(1)[0][0]

        feat, thresh = self.best_split(X, y)
        if feat is None:
            return Counter(y).most_common(1)[0][0]

        left_mask = X[:, feat] <= thresh
        return {
            'feature': feat,
            'threshold': thresh,
            'left': self.build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self.build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        return self.predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

# ============================================================
# K-MEANS FROM SCRATCH
# ============================================================

class KMeans:
    """K-Means clustering."""

    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        n_samples = X.shape[0]
        # Initialize centroids (k-means++)
        idx = [np.random.randint(n_samples)]
        for _ in range(1, self.k):
            dists = np.min([np.sum((X - X[i])**2, axis=1) for i in idx], axis=0)
            probs = dists / dists.sum()
            idx.append(np.random.choice(n_samples, p=probs))
        self.centroids = X[idx].copy()

        for _ in range(self.max_iters):
            # Assign clusters
            distances = np.array([np.sum((X - c)**2, axis=1)
                                 for c in self.centroids])
            self.labels = np.argmin(distances, axis=0)

            # Update centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0)
                                     for k in range(self.k)])

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return self.labels

# ============================================================
# TF-IDF FROM SCRATCH
# ============================================================

def compute_tfidf(documents):
    """TF-IDF: Term Frequency × Inverse Document Frequency"""
    # Build vocabulary
    vocab = sorted(set(word for doc in documents
                       for word in doc.lower().split()))
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    n_docs = len(documents)
    n_vocab = len(vocab)

    # Term Frequency (TF)
    tf = np.zeros((n_docs, n_vocab))
    for i, doc in enumerate(documents):
        words = doc.lower().split()
        for word in words:
            tf[i, word_to_idx[word]] += 1
        tf[i] /= len(words)  # normalize by doc length

    # Inverse Document Frequency (IDF)
    idf = np.zeros(n_vocab)
    for j, word in enumerate(vocab):
        doc_count = sum(1 for doc in documents if word in doc.lower())
        idf[j] = np.log(n_docs / (1 + doc_count))  # +1 smoothing

    # TF-IDF
    return tf * idf, vocab
```

</details>

<details>
<summary><strong>Code: Feature Engineering & Model Evaluation Pipeline</strong></summary>

```python
import numpy as np

# ============================================================
# CROSS-VALIDATION
# ============================================================

def k_fold_cross_validation(model_class, X, y, k=5, **model_params):
    """K-fold cross-validation from scratch."""
    n = len(y)
    indices = np.random.permutation(n)
    fold_size = n // k
    scores = []

    for i in range(k):
        # Split
        val_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size],
                                    indices[(i + 1) * fold_size:]])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_val)
        accuracy = np.mean(preds == y_val)
        scores.append(accuracy)

    return np.mean(scores), np.std(scores)

# ============================================================
# METRICS
# ============================================================

def classification_report(y_true, y_pred):
    """Compute precision, recall, F1 for each class."""
    classes = sorted(set(y_true) | set(y_pred))

    for cls in classes:
        tp = sum((t == cls and p == cls) for t, p in zip(y_true, y_pred))
        fp = sum((t != cls and p == cls) for t, p in zip(y_true, y_pred))
        fn = sum((t == cls and p != cls) for t, p in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) \
             if (precision + recall) > 0 else 0

        print(f"Class {cls}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

def roc_auc(y_true, y_scores, n_thresholds=100):
    """Compute ROC AUC from scratch."""
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list, fpr_list = [], []

    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))

        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    # AUC via trapezoidal rule
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

    return abs(auc)
```

</details>

---

## 12.10 Interview Questions

### Conceptual Questions

**Q1: Explain the bias-variance tradeoff. How does it relate to model selection and regularization?**

Bias is error from wrong assumptions (underfitting) — a linear model fitting nonlinear data. Variance is error from sensitivity to training data (overfitting) — a deep tree memorizing noise. Total error = Bias² + Variance + Noise. Regularization (L1, L2, dropout) reduces variance at the cost of slight bias increase. The optimal model complexity minimizes total error. In practice: start simple (high bias), increase complexity until validation error stops improving.

**Q2: Why does Random Forest reduce variance compared to a single decision tree? Why not bias?**

Random Forest averages predictions from trees trained on different bootstrap samples with random feature subsets. If each tree has error = bias + noise_i, averaging N trees: E[average] still has the same bias, but Var[average] = Var[single]/N (if errors are independent). The feature randomization de-correlates trees, making independence assumption more valid. Bias stays similar because each tree is grown deep enough to capture the signal.

**Q3: When would you choose XGBoost over a neural network for a production system?**

XGBoost wins on: (1) Tabular/structured data — still unmatched in 2026, (2) Small-to-medium datasets (<100K rows), (3) When interpretability matters (feature importance, SHAP), (4) Training speed and compute efficiency, (5) Handling missing values natively. Neural networks win on: unstructured data (text, images, audio), very large datasets, transfer learning scenarios, and multimodal inputs. In production: XGBoost is simpler to deploy, needs less infrastructure, and has more predictable latency.

**Q4: Explain TF-IDF and why it was used before embeddings for text search. What are its limitations?**

TF-IDF = Term Frequency × Inverse Document Frequency. TF measures how often a word appears in a document; IDF downweights words common across all documents. Result: rare, document-specific words get high scores. Limitations: (1) Bag-of-words — loses word order ("dog bites man" = "man bites dog"), (2) No semantic understanding ("car" ≠ "automobile"), (3) Sparse and high-dimensional (vocabulary size), (4) No contextual meaning. Modern approach: dense embeddings (BERT, sentence-transformers) capture semantics, but TF-IDF is still used in BM25 for initial retrieval in RAG systems.

**Q5: What is the kernel trick in SVMs and why is it computationally efficient?**

The kernel trick computes dot products in a high-dimensional feature space without explicitly transforming the data. For the RBF kernel, the implicit feature space is _infinite-dimensional_, but $K(x, x') = \exp(-\gamma\|x - x'\|^2)$ is a simple computation in the original space. This works because SVM only needs dot products between data points (via the dual formulation), not the actual transformed features. Complexity goes from impossible (infinite features) to O(n² × d) for n support vectors.

### Coding Questions

**Q6: Implement a complete gradient-boosted regression ensemble from scratch.**

```python
class GradientBoostedRegressor:
    """Simple gradient boosted regression using decision stumps."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_prediction = None

    def fit(self, X, y):
        self.base_prediction = np.mean(y)  # F₀ = mean
        current_prediction = np.full(len(y), self.base_prediction)

        for _ in range(self.n_estimators):
            residuals = y - current_prediction  # pseudo-residuals
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions = tree.predict(X).astype(float)
            current_prediction += self.lr * predictions
            self.trees.append(tree)

    def predict(self, X):
        prediction = np.full(len(X), self.base_prediction)
        for tree in self.trees:
            prediction += self.lr * tree.predict(X).astype(float)
        return prediction
```

### System Design Questions

**Q7: Design a real-time fraud detection system for a payment platform processing 10K transactions per second.**

```
┌──────────────────────────────────────────────────────────────────┐
│              Fraud Detection System Architecture                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. FEATURE PIPELINE (< 10ms budget)                              │
│     Real-time features:                                           │
│       Transaction: amount, merchant category, location            │
│       User: spending velocity (last 1h, 24h), avg transaction     │
│       Device: fingerprint, IP geolocation, new device flag        │
│     Pre-computed (batch):                                         │
│       User risk score, merchant risk score, historical patterns   │
│     Feature store: Redis (real-time) + Feast (batch)              │
│                                                                   │
│  2. MODEL ARCHITECTURE                                            │
│     ┌────────┐    ┌──────────────┐    ┌───────────┐              │
│     │ Rules  │───▶│  XGBoost     │───▶│ Neural Net│              │
│     │ Engine │    │  (fast, P90  │    │ (complex  │              │
│     │ (<1ms) │    │   < 5ms)     │    │  cases)   │              │
│     └────────┘    └──────────────┘    └───────────┘              │
│     Block obvious │ Score all txns   │ Re-score uncertain         │
│     fraud/allow   │ (0-1 risk)       │ cases async                │
│     clear legit   │                                               │
│                                                                   │
│  3. DECISION LAYER                                                │
│     Score < 0.3 → approve                                         │
│     Score 0.3-0.7 → step-up auth (2FA, OTP)                      │
│     Score > 0.7 → block + alert fraud team                        │
│     Score > 0.9 → block + freeze account                          │
│                                                                   │
│  4. FEEDBACK LOOP                                                 │
│     Chargebacks (label=fraud, delayed 30-90 days)                │
│     User disputes → confirm/deny label                            │
│     Retrain model weekly with new labels                          │
│     A/B test new models on shadow traffic                         │
│                                                                   │
│  5. MONITORING                                                    │
│     Metric: fraud rate, false positive rate, approval rate        │
│     Drift: feature distribution monitoring (PSI)                  │
│     Latency: P50 < 5ms, P99 < 20ms                               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 12.11 Key Papers & Resources

| Paper/Resource                                                          | Year | Why It Matters                                               |
| ----------------------------------------------------------------------- | ---- | ------------------------------------------------------------ |
| _XGBoost: A Scalable Tree Boosting System_ (Chen & Guestrin)            | 2016 | The most important tabular ML algorithm                      |
| _Random Forests_ (Breiman)                                              | 2001 | Foundation of ensemble learning                              |
| _A Few Useful Things to Know About ML_ (Domingos)                       | 2012 | Best 12-page overview of practical ML                        |
| _An Introduction to Statistical Learning_ (James et al.)                | 2013 | Free textbook, accessible to all levels                      |
| _LightGBM_ (Ke et al.)                                                  | 2017 | Faster gradient boosting for production                      |
| _Random Search for Hyperparameter Optimization_ (Bergstra & Bengio)     | 2012 | Why random beats grid search                                 |
| _Do We Need Hundreds of Classifiers_ (Fernández-Delgado)                | 2014 | Massive comparison — Random Forest wins for most problems    |
| _Tabular Data: Deep Learning is Not All You Need_ (Shwartz-Ziv & Armon) | 2022 | XGBoost still competitive with deep learning on tabular data |

---

[← Module 11: Math Foundations](../module-11-math-foundations/README.md) | [Module 13: Deep Learning →](../module-13-deep-learning/README.md)
