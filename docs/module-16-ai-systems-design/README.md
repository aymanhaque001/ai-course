# Module 16: AI Systems Design

> **Prerequisites:** All previous modules recommended (especially 12, 15)  
> **Estimated Time:** 12-15 hours  
> **Relevance:** The #1 interview format at senior ML/AI roles. Also the skill that separates ML engineers from ML researchers

---

## 16.1 ML System Design Framework

```
┌──────────────────────────────────────────────────────────────────┐
│              ML System Design Framework (4 Steps)                 │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ STEP 1: PROBLEM FORMULATION (10 min)                   │      │
│  │                                                         │      │
│  │  ▸ Clarify requirements & constraints                   │      │
│  │  ▸ Define business objective → ML objective             │      │
│  │  ▸ Identify inputs and outputs                          │      │
│  │  ▸ Choose ML task type (classification, ranking, gen)   │      │
│  │  ▸ Define success metrics (online & offline)            │      │
│  │  ▸ Identify constraints (latency, cost, scale)          │      │
│  └────────────────────────────────────────────────────────┘      │
│                         │                                         │
│                         ▼                                         │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ STEP 2: DATA & FEATURES (10 min)                       │      │
│  │                                                         │      │
│  │  ▸ What data is available? What needs to be collected?  │      │
│  │  ▸ Data schema and exploration                          │      │
│  │  ▸ Feature engineering                                  │      │
│  │  ▸ Label acquisition (how to get ground truth)          │      │
│  │  ▸ Data pipeline design                                 │      │
│  │  ▸ Handle class imbalance, missing data                 │      │
│  └────────────────────────────────────────────────────────┘      │
│                         │                                         │
│                         ▼                                         │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ STEP 3: MODEL ARCHITECTURE & TRAINING (10 min)         │      │
│  │                                                         │      │
│  │  ▸ Model selection (start simple → increase complexity) │      │
│  │  ▸ Architecture design                                  │      │
│  │  ▸ Training pipeline                                    │      │
│  │  ▸ Loss function design                                 │      │
│  │  ▸ Handling scale (distributed training if needed)      │      │
│  │  ▸ Offline evaluation strategy                          │      │
│  └────────────────────────────────────────────────────────┘      │
│                         │                                         │
│                         ▼                                         │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ STEP 4: DEPLOYMENT & SERVING (10 min)                   │      │
│  │                                                         │      │
│  │  ▸ Serving architecture (batch vs real-time)            │      │
│  │  ▸ Online evaluation & A/B testing                      │      │
│  │  ▸ Monitoring & alerting                                │      │
│  │  ▸ Failure modes & fallbacks                            │      │
│  │  ▸ Scaling & cost optimization                          │      │
│  │  ▸ Iteration & improvement plan                         │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  Pro tips:                                                        │
│  - ALWAYS start with the simplest approach (baseline)             │
│  - Talk about trade-offs, not just solutions                      │
│  - Mention what you'd do with more time/data                      │
│  - Ask clarifying questions before diving in                      │
│  - Draw diagrams!                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 16.2 Common ML Task Formulations

```
┌──────────────────────────────────────────────────────────────────┐
│  Business Problem         → ML Formulation                        │
├───────────────────────────┼──────────────────────────────────────┤
│  "Show relevant posts"    → Ranking (pointwise, pairwise,        │
│                              listwise) on engagement probability  │
│  "Detect spam"            → Binary classification                │
│  "Recommend products"     → Collaborative filtering + ranking    │
│  "Moderate content"       → Multi-label classification +         │
│                              severity regression                  │
│  "Estimate delivery time" → Regression                           │
│  "Answer questions"       → LLM generation + RAG                 │
│  "Similar image search"   → Embedding similarity (ANN)           │
│  "Auto-complete"          → Language model (next token)           │
│  "Detect fraud"           → Anomaly detection / binary           │
│                              classification (imbalanced)          │
│  "Summarize document"     → Seq2seq generation / LLM             │
│  "Segment users"          → Clustering                           │
│  "Predict churn"          → Survival analysis / classification   │
│  "Price optimization"     → Causal inference + optimization      │
│  "Translate languages"    → Seq2seq / LLM                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 16.3 Metrics: Offline vs Online

```
┌──────────────────────────────────────────────────────────────────┐
│           Offline Metrics → Online Metrics                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Offline (evaluated on holdout data):                            │
│    Classification: AUC-ROC, F1, precision/recall, log-loss       │
│    Ranking: NDCG@K, MAP@K, MRR                                   │
│    Regression: MAE, RMSE, R²                                     │
│    Generation: BLEU, ROUGE, perplexity, BERTScore                │
│    Retrieval: Recall@K, Precision@K, MRR                         │
│                                                                   │
│  Online (measured in production):                                │
│    Engagement: CTR, time spent, scroll depth                     │
│    Business: Revenue, conversion rate, retention                 │
│    Quality: User ratings, complaints, refund rate                 │
│    Efficiency: Latency, cost per prediction                      │
│    Safety: Flagged content rate, false positive rate              │
│                                                                   │
│  KEY INSIGHT: Offline metrics ≠ online metrics!                  │
│    Better AUC offline doesn't guarantee higher CTR online.       │
│    Always validate with A/B tests.                                │
│                                                                   │
│  North Star Metric:                                               │
│    The ONE metric that best represents user value                 │
│    Everything else is a proxy or guardrail                        │
│    e.g., Netflix: hours of quality viewing                        │
│          Uber: successful rides per week                          │
│          Google: queries satisfied (no re-query)                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 16.4 Design Case Study #1: Recommendation System

### Problem: "Design a content recommendation system for a social media platform"

```
STEP 1: REQUIREMENTS
  - Recommend posts/content in a user's feed
  - ~100M daily active users
  - Feed refresh latency < 200ms
  - Optimize for meaningful engagement (not just clicks)
  - Must handle cold-start (new users, new content)

STEP 2: HIGH-LEVEL ARCHITECTURE

  ┌──────────────────────────────────────────────────────────────┐
  │                Recommendation Pipeline                        │
  │                                                               │
  │   ~1M candidates     ~1000          ~200         ~50          │
  │  ┌──────────┐    ┌──────────┐   ┌──────────┐  ┌──────────┐  │
  │  │Candidate │───▶│  First   │──▶│  Second  │─▶│  Re-rank │  │
  │  │Generation│    │  Stage   │   │  Stage   │  │  + Policy│  │
  │  │  (cheap) │    │ Ranking  │   │ Ranking  │  │  (rules) │  │
  │  │          │    │  (fast)  │   │ (precise)│  │          │  │
  │  └──────────┘    └──────────┘   └──────────┘  └──────────┘  │
  │   Recall          Rough score    Final score   Diversity,    │
  │   focused         Logistic Reg   Deep Neural   freshness,    │
  │   ANN, CF,        + simple       Network       dedup,        │
  │   graph           features                     safety         │
  │                                                               │
  └──────────────────────────────────────────────────────────────┘

STEP 3: CANDIDATE GENERATION

  Multi-source retrieval (recall-focused, not precision):

  Source 1: Collaborative Filtering
    User-item matrix → ALS/matrix factorization
    "Users who liked what you liked also liked..."

  Source 2: Content-based
    Embed user profile & items in same space
    Approximate Nearest Neighbor (ANN) search
    FAISS or ScaNN for sub-millisecond retrieval

  Source 3: Social graph
    Content from users you follow
    Content liked by users similar to you

  Source 4: Trending / Explore
    Popular content in your region/interests
    Cold-start content for new users

  → Union of all sources = ~1000-5000 candidates per request

STEP 4: RANKING MODEL

  Features:
  ┌──────────────────────────────────────────────────────────────┐
  │  User features:    demographics, activity history, prefs    │
  │  Item features:    content type, creator stats, age,        │
  │                    engagement stats (CTR/likes/shares)       │
  │  Cross features:   user-creator affinity, topic match,      │
  │                    user×time_of_day interaction              │
  │  Context features: time of day, device, session length      │
  └──────────────────────────────────────────────────────────────┘

  Model: Multi-task deep neural network

  Input: User features + Item features + Cross features
    │
    ▼
  ┌──────────────────────────────────────────────┐
  │           Shared Bottom Layers                │
  │           [Dense → ReLU → Dense]              │
  └─────┬──────────┬──────────┬──────────┬───────┘
        │          │          │          │
     ┌──▼──┐  ┌───▼──┐  ┌───▼──┐  ┌───▼──┐
     │Click│  │Like  │  │Share │  │Time  │
     │Prob │  │Prob  │  │Prob  │  │Spent │
     └─────┘  └──────┘  └──────┘  └──────┘

  Final score = w₁·P(click) + w₂·P(like) + w₃·P(share)
                + w₄·E[time_spent] - w₅·P(report)

  Weights tuned to optimize for "meaningful engagement"

  HOW multi-task loss is balanced:
    Naive L = Σ w_i·L_i with fixed weights is brittle — tasks at
    different scales dominate training (e.g., time_spent in seconds
    dwarfs binary click loss). Solutions:
    (1) Uncertainty-weighted loss (Kendall et al.):
          L = Σ (1/(2σ_i²))·L_i + log(σ_i)
          Learns per-task uncertainty σ_i from data; high-variance tasks
          automatically get downweighted. No manual tuning needed.
    (2) GradNorm: normalize gradient magnitudes across tasks at each
          step so no single task dominates the shared layers.
    (3) Manual tuning with task-specific validation metrics:
          set weights to reflect business priorities (purchase >> click)
          then sweep values on a held-out validation set.
    Weights also encode business logic: a like is worth more than a click,
    a share more than a like, and a report subtracts heavily.

STEP 5: RE-RANKING & POLICIES

  After ML ranking, apply business rules:
  - Diversity: don't show 10 posts from same creator
  - Freshness: boost recent content
  - Safety: filter content flagged by moderation model
  - Deduplication: don't show near-duplicate content
  - Frequency capping: limit ad-like content

STEP 6: SERVING ARCHITECTURE

  ┌──────────────────────────────────────────────┐
  │  Request: GET /feed?user_id=12345            │
  │    │                                          │
  │    ├─→ Feature Store (user features, <5ms)    │
  │    ├─→ Candidate Gen (ANN lookup, <20ms)      │
  │    ├─→ Feature enrichment (item features)     │
  │    ├─→ Ranking model inference (<50ms)         │
  │    ├─→ Re-ranking rules (<5ms)                │
  │    └─→ Response (top 50 items, <200ms total)  │
  │                                               │
  │  Caching: user embeddings cached (TTL: 1hr)   │
  │  Precompute: ANN index rebuilt hourly          │
  │  Scale: horizontal pod autoscaling on QPS     │
  └──────────────────────────────────────────────┘

STEP 7: COLD START

  New user: Use demographic info, sign-up interests,
            trending/popular content, explore-exploit
  New item: Use content features (text embedding,
            image embedding), creator history,
            small initial exposure for data collection

  Multi-armed bandit algorithms for cold start:

  Thompson Sampling (Bayesian exploration):
    Maintain a Beta(α_i, β_i) distribution for each item's click
    probability, where α = prior successes + observed clicks,
    β = prior failures + observed non-clicks.
    At each request: sample θ_i ~ Beta(α_i, β_i) for each candidate,
    recommend the item with highest sample θ_i.
    Items with few observations have wide Beta distributions → high
    variance samples → natural exploration without a separate ε.
    As clicks accumulate, distributions sharpen → exploitation.

  UCB (Upper Confidence Bound):
    Select item i* = argmax_i [ μ̂_i + c · √(log t / n_i) ]
    where μ̂_i = empirical mean reward, t = total rounds, n_i = times
    item i was shown, and c controls exploration-exploitation trade-off.
    Items shown rarely (small n_i) get a large exploration bonus;
    well-explored items are selected on mean reward alone.
    Both Thompson Sampling and UCB converge to the optimal item while
    collecting enough data to resolve cold-start uncertainty.

STEP 8: EVALUATION

  Offline: NDCG@50, AUC on click prediction,
           multi-task loss convergence
  Online A/B: engagement rate, time spent, DAU,
              user satisfaction survey (guardrail)
```

---

## 16.5 Design Case Study #2: Search Ranking

### Problem: "Design a search ranking system for an e-commerce platform"

```
STEP 1: REQUIREMENTS
  - User types a query → return ranked product results
  - ~10M products, ~50M queries/day
  - Latency < 200ms (including retrieval + ranking)
  - Optimize for purchase conversion
  - Must handle: typos, synonyms, multi-language

STEP 2: ARCHITECTURE

  User query: "wireless noise cancelling headphones"

  ┌──────────────────────────────────────────────────────────────┐
  │  ┌────────────┐                                              │
  │  │   Query    │ Spell correction, tokenization,              │
  │  │ Processing │ synonym expansion, intent classification     │
  │  └──────┬─────┘                                              │
  │         │                                                     │
  │         ▼                                                     │
  │  ┌────────────┐                                              │
  │  │ Retrieval  │ Return ~1000 candidates from ~10M products   │
  │  │ (Recall)   │                                              │
  │  │            │ L1: Inverted index (Elasticsearch)            │
  │  │            │     BM25 text matching + filters              │
  │  │            │ L2: Vector search (semantic embeddings)       │
  │  │            │     E5/GTE embedding model, HNSW index       │
  │  │            │ → Hybrid: RRF fusion of BM25 + vector        │
  │  └──────┬─────┘                                              │
  │         │                                                     │
  │         ▼                                                     │
  │  ┌────────────┐                                              │
  │  │  Ranking   │ Score ~1000 → top 100                         │
  │  │  (LTR)     │                                              │
  │  │            │ Stage 1: Lightweight model (LambdaMART)       │
  │  │            │   ~50 features, <10ms for 1000 items          │
  │  │            │ Stage 2: Deep model (cross-encoder)           │
  │  │            │   Query-product attention, top 100            │
  │  └──────┬─────┘                                              │
  │         │                                                     │
  │         ▼                                                     │
  │  ┌────────────┐                                              │
  │  │ Re-ranking │ Business rules, personalization               │
  │  │ & Blending │ Sponsored results, diversity                  │
  │  └────────────┘                                              │
  └──────────────────────────────────────────────────────────────┘

STEP 3: FEATURE ENGINEERING

  Query features:
    - Query length, token count
    - Query intent (navigational, informational, transactional)
    - Historical query popularity

  Document features:
    - Product title/description TF-IDF
    - Sales velocity, rating, review count
    - Price percentile within category
    - Return rate, stock level
    - Image quality score

  Query-Document features (most important!):
    - BM25 score
    - Semantic similarity (query embedding · product embedding)
    - Title exact match / partial match
    - Category match with query intent
    - Historical CTR for this query-product pair

  User features (personalization):
    - Purchase history (categories, price range)
    - Click history on similar queries
    - Location (for shipping relevance)

STEP 4: RANKING MODEL

  Learning to Rank (LTR) approaches:

  ┌──────────────┬──────────────────────────────────────────────┐
  │ Pointwise    │ Predict relevance score per item              │
  │              │ Binary classification (relevant/not)          │
  │              │ Simple, but ignores relative ordering          │
  ├──────────────┼──────────────────────────────────────────────┤
  │ Pairwise     │ Predict which of two items is more relevant  │
  │ (RankNet,    │ Learns relative preferences                   │
  │  LambdaRank) │ Better ranking quality                        │
  ├──────────────┼──────────────────────────────────────────────┤
  │ Listwise     │ Optimize list-level metrics directly          │
  │ (LambdaMART, │ Best ranking quality                          │
  │  ListMLE)    │ Optimizes NDCG directly                       │
  └──────────────┴──────────────────────────────────────────────┘

  HOW LambdaMART optimizes NDCG (lambda gradients):
    Standard gradient boosting needs a differentiable loss, but NDCG
    is not differentiable. LambdaMART solves this with "lambda gradients":
    For each pair of documents (i, j) where i is ranked above j:
      λ_ij = -∂C/∂s_ij · |ΔNDCG_ij|
    where |ΔNDCG_ij| is how much NDCG would change by swapping i and j.
    These λ_ij values replace the true loss gradient in gradient boosting.
    Result: position-aware optimization — swapping rank 1 vs rank 2
    (large |ΔNDCG|) drives bigger gradient updates than swapping rank
    50 vs 51. This is why LambdaMART outperforms pointwise approaches:
    it directly encodes that top-rank mistakes matter far more.

  Production: LambdaMART (XGBoost) for stage 1
              Cross-encoder (BERT-based) for stage 2

STEP 5: TRAINING DATA

  Implicit feedback (abundant):
    Click = weak positive
    Add to cart = medium positive
    Purchase = strong positive
    Skip = weak negative (position bias caveat!)

  Position bias correction:
    Users more likely to click top results regardless of relevance
    Fix: inverse propensity weighting, randomization experiments

    HOW inverse propensity weighting (IPW) works:
      Corrected loss = Σ (click_i / P(click | pos_i)) · relevance_signal_i
      Each click is weighted by 1/P(click|position): clicks on lower-ranked
      items (where propensity to click is low) get higher weight because
      users who clicked despite low position signal strong relevance.
      Propensity estimation: run randomization experiments — randomly
      shuffle result rankings for a small traffic slice, observe natural
      click rates at each position. This gives an unbiased estimate of
      P(click|pos) independent of item relevance. Clicks at position 5
      (propensity ~0.1) get 10× the weight of clicks at position 1
      (propensity ~1.0).

  Explicit labels (expensive but high quality):
    Human raters label query-product pairs (1-5 scale)
    Used for evaluation, not primary training

STEP 6: EVALUATION

  Offline: NDCG@10, MRR, Recall@100
  Online: CTR, add-to-cart rate, conversion rate,
          revenue per search, zero-result rate
  Guardrails: latency P95 < 200ms, diversity of results
```

---

## 16.6 Design Case Study #3: Fraud Detection System

### Problem: "Design a real-time fraud detection system for a payment platform"

```
STEP 1: REQUIREMENTS
  - Score every transaction in real-time (< 100ms)
  - ~10M transactions/day, spikes 3× during sales
  - Optimize for precision at high recall (catch fraud, minimize false positives)
  - Fraud rate: ~0.1% of transactions (extreme class imbalance)
  - Must handle evolving fraud patterns (concept drift)
  - Regulatory requirement: explain why a transaction was flagged

STEP 2: ARCHITECTURE

  ┌──────────────────────────────────────────────────────────────┐
  │              Fraud Detection Pipeline                         │
  │                                                               │
  │  Transaction ─┬──────────────────────────────────────────┐   │
  │  Event        │                                          │   │
  │               ▼                                          │   │
  │  ┌────────────────────┐                                  │   │
  │  │  Rules Engine      │  Hard rules (velocity, blocklist)│   │
  │  │  (Immediate)       │  <5ms, catches ~30% of fraud     │   │
  │  │  BLOCK / PASS      │  e.g., >5 txns in 1 min         │   │
  │  └────────┬───────────┘                                  │   │
  │           │ pass                                         │   │
  │           ▼                                              │   │
  │  ┌────────────────────┐                                  │   │
  │  │  Feature           │  Real-time features from         │   │
  │  │  Computation       │  Feature Store (Redis)           │   │
  │  │  (<20ms)           │  + Streaming features (Flink)    │   │
  │  └────────┬───────────┘                                  │   │
  │           │                                              │   │
  │           ▼                                              │   │
  │  ┌────────────────────┐                                  │   │
  │  │  ML Model          │  Ensemble: XGBoost + Neural Net  │   │
  │  │  Scoring (<30ms)   │  → Fraud probability P(fraud)    │   │
  │  └────────┬───────────┘                                  │   │
  │           │                                              │   │
  │           ▼                                              │   │
  │  ┌────────────────────┐                                  │   │
  │  │  Decision Engine   │  P < 0.3: APPROVE               │   │
  │  │                    │  0.3 < P < 0.8: REVIEW (queue)   │   │
  │  │                    │  P > 0.8: BLOCK                   │   │
  │  └────────────────────┘                                  │   │
  │           │                                              │   │
  │           ▼                                              │   │
  │  ┌─────────────────────────────────────────────────────┐ │   │
  │  │  Human Review Queue (manual analysis → label)       │ │   │
  │  │  Agent: Review + decision → feedback loop for model │ │   │
  │  └─────────────────────────────────────────────────────┘ │   │
  └──────────────────────────────────────────────────────────────┘

STEP 3: FEATURES

  Transaction features:
    - Amount, currency, merchant category
    - Payment method, device info, IP geolocation
    - Time of day, day of week

  User behavior features (from Feature Store):
    - Avg transaction amount (7d, 30d)
    - Transaction count (1h, 1d, 7d)
    - Unique merchants (7d)
    - Max single transaction (30d)
    - Device change frequency
    - Typical geolocation

  Derived features:
    - Amount deviation: (amount - user_avg) / user_std
    - Velocity: txns in last 1h vs user's average
    - Geo anomaly: distance from user's typical location
    - Time anomaly: transaction at unusual time for user
    - Merchant risk score: historical fraud rate at merchant

  Graph features:
    - Shared device/IP across accounts
    - Transaction network patterns
    - Known fraud ring connections

STEP 4: HANDLING CLASS IMBALANCE (0.1% positive rate)

  ┌──────────────┬────────────────────────────────────────────┐
  │ SMOTE        │ Synthetic oversampling of minority class    │
  │ Undersampling│ Random subset of majority class             │
  │ Class weights│ weight_fraud = 1000, weight_legit = 1       │
  │ Focal loss   │ Down-weight easy negatives automatically    │
  │ Anomaly det. │ Train only on legitimate → flag anomalies  │
  │ Ensemble     │ Multiple models on balanced subsamples      │
  └──────────────┴────────────────────────────────────────────┘

  In practice: class weights + focal loss + careful threshold tuning

STEP 5: EXPLAINABILITY (regulatory requirement)

  Model: XGBoost (primary) — inherently interpretable

  Explanation per prediction:
    "Transaction blocked because:
     1. Amount ($5,000) is 8× user's average ($625)
     2. New device (never seen before)
     3. Geolocation: 3,000 miles from usual location
     4. 5th transaction in last 30 minutes"

  Tools: SHAP values per feature per prediction

  SHAP output:
    Feature                  SHAP value
    amount_deviation         +0.35  ← biggest contributor
    new_device               +0.22
    geo_distance             +0.18
    velocity_1h              +0.15
    merchant_risk_score      +0.05
    base_value               0.001  (population fraud rate)
    prediction               0.961  (probability of fraud)

  HOW Shapley values are computed:
    SHAP comes from cooperative game theory. Each feature's contribution:
      φ_i = Σ_{S ⊆ N\{i}} |S|!(|N|-|S|-1)!/|N|! · [f(S∪{i}) - f(S)]
    This averages the marginal contribution of feature i over all possible
    subsets S of the other features — a weighted average over all insertion
    orders. Naively exponential (2^|N| subsets), but:
    - TreeSHAP: exploits tree structure (decision paths) for exact
      polynomial-time computation on XGBoost/LightGBM/RF models.
    - KernelSHAP: for neural nets, approximates Shapley values using
      weighted linear regression on sampled feature coalitions.
    Key property (efficiency): φ_base + Σφ_i = f(x), so contributions
    sum exactly to the prediction — unlike simple feature importances.

STEP 6: HANDLING CONCEPT DRIFT

  Fraud patterns evolve constantly:
    - Retrain weekly on recent labeled data
    - Monitor fraud catch rate daily
    - Rules engine updated for new patterns immediately
    - Champion-challenger model testing
    - Human review feedback → fast labeling pipeline
```

---

## 16.7 Design Case Study #4: LLM-Powered Chatbot at Scale

### Problem: "Design a customer support chatbot using LLMs for a large SaaS company"

```
STEP 1: REQUIREMENTS
  - Answer customer questions 24/7
  - 100K conversations/day
  - Must use company knowledge base (10K+ docs)
  - Minimize hallucination (it's customer-facing!)
  - Seamless handoff to human agents
  - Cost < $0.10 per conversation average
  - Response latency < 3 seconds

STEP 2: ARCHITECTURE

  ┌──────────────────────────────────────────────────────────────┐
  │           Customer Support AI Architecture                    │
  │                                                               │
  │  User message                                                 │
  │    │                                                          │
  │    ▼                                                          │
  │  ┌──────────────┐                                            │
  │  │   Router     │  Intent classification:                     │
  │  │   Model      │  - FAQ (→ RAG)                              │
  │  │   (small LM) │  - Account issue (→ API + LLM)              │
  │  │              │  - Complaint (→ human handoff)               │
  │  │              │  - Technical (→ RAG + code analysis)         │
  │  │              │  - Out of scope (→ polite decline)           │
  │  └──────┬───────┘                                            │
  │         │                                                     │
  │         ▼                                                     │
  │  ┌──────────────┐   ┌──────────────────┐                     │
  │  │  RAG Pipeline│   │  Tool Executor   │                     │
  │  │              │   │                  │                     │
  │  │  Query       │   │  check_order()   │                     │
  │  │  rewriting   │   │  update_ticket() │                     │
  │  │      │       │   │  get_account()   │                     │
  │  │      ▼       │   │  refund()        │                     │
  │  │  Retrieval   │   │                  │                     │
  │  │  (hybrid:    │   │  With auth +     │                     │
  │  │  BM25+vector)│   │  permission      │                     │
  │  │      │       │   │  checks          │                     │
  │  │      ▼       │   │                  │                     │
  │  │  Re-ranking  │   │                  │                     │
  │  │  (cross-enc) │   │                  │                     │
  │  └──────┬───────┘   └───────┬──────────┘                     │
  │         │                   │                                  │
  │         ▼                   ▼                                  │
  │  ┌──────────────────────────────────────┐                    │
  │  │          LLM Generation              │                    │
  │  │  System prompt + context + tools     │                    │
  │  │  Grounded in retrieved documents     │                    │
  │  │  Citations required                  │                    │
  │  └──────────────┬───────────────────────┘                    │
  │                 │                                              │
  │                 ▼                                              │
  │  ┌──────────────────────────────────────┐                    │
  │  │        Safety & Quality Gate         │                    │
  │  │  - Hallucination check (NLI model)   │                    │
  │  │  - PII detection & masking           │                    │
  │  │  - Tone check (professional)         │                    │
  │  │  - Confidence threshold              │                    │
  │  │  If fails → escalate to human agent  │                    │
  │  └──────────────────────────────────────┘                    │
  │                                                               │
  │  HUMAN HANDOFF TRIGGERS:                                     │
  │  - User requests human                                        │
  │  - Bot confidence < threshold (3 consecutive low)              │
  │  - Sensitive topic (billing dispute, legal)                   │
  │  - Negative sentiment detected                                │
  │  - Conversation exceeds N turns without resolution            │
  │                                                               │
  └──────────────────────────────────────────────────────────────┘

STEP 3: COST OPTIMIZATION

  Target: < $0.10/conversation (avg 5 turns)

  Strategy                     Cost/conversation
  ┌──────────────────────────┬────────────────┐
  │ GPT-4 for everything     │ $0.50 - $2.00  │  ← too expensive
  │ Route: 70% small model   │                │
  │        30% GPT-4o        │ $0.05 - $0.15  │  ← target
  │ Fine-tuned small model   │ $0.01 - $0.05  │  ← ideal
  │ Cached responses (30%)   │ $0.00          │
  └──────────────────────────┴────────────────┘

  Implementation:
  1. Router classifies query complexity
  2. Simple FAQs → fine-tuned 8B model (self-hosted, <$0.01)
  3. Complex queries → GPT-4o-mini ($0.03)
  4. Edge cases → GPT-4o ($0.10)
  5. Semantic cache for repeated questions (30% hit rate)

STEP 4: EVALUATION

  Automated evals (run daily):
    - Answer accuracy vs golden QA pairs
    - Hallucination rate (NLI against source docs)
    - Citation accuracy (does cited doc support answer?)
    - Response time P50/P95

  Human evals (weekly):
    - Sample 100 conversations
    - Rate: helpfulness, accuracy, tone
    - Track: resolution rate, handoff rate

  Business metrics:
    - Customer satisfaction score (CSAT)
    - First-contact resolution rate
    - Conversation-to-ticket escalation rate
    - Cost per resolved conversation
    - Human agent time savings
```

---

## 16.8 Design Case Study #5: Content Moderation Pipeline

### Problem: "Design an AI content moderation system for a platform with user-generated content (text, images, videos)"

```
STEP 1: REQUIREMENTS
  - Moderate text, images, and video uploads
  - ~50M posts/day
  - Latency: text <500ms, images <2s, video <30s
  - Categories: hate speech, violence, NSFW, spam,
                misinformation, self-harm, CSAM
  - False positive rate < 1% (user experience)
  - Recall > 95% for worst categories (safety)
  - Explainable decisions (appeals process)

STEP 2: ARCHITECTURE

  ┌──────────────────────────────────────────────────────────────┐
  │              Content Moderation Pipeline                       │
  │                                                               │
  │  Content Upload                                               │
  │       │                                                       │
  │       ▼                                                       │
  │  ┌──────────────────────┐                                    │
  │  │  Hash/Fingerprint    │  Check known-bad content            │
  │  │  (perceptual hash)   │  PhotoDNA, CSAM databases           │
  │  │  (<10ms)             │  Exact + near-duplicate detection    │
  │  └─────────┬────────────┘                                    │
  │    match?  │  no match                                       │
  │    BLOCK   │                                                  │
  │            ▼                                                  │
  │  ┌──────────────────────┐                                    │
  │  │  Fast Classifier     │  Lightweight model for each        │
  │  │  (per modality)      │  modality. Runs on all content.     │
  │  │                      │                                    │
  │  │  Text: distilBERT    │  Score per violation category       │
  │  │  Image: EfficientNet │  If score > 0.9 → auto-action      │
  │  │  Video: sample frames│  If 0.5 < score < 0.9 → Stage 2    │
  │  │  (<100ms text,       │  If score < 0.5 → pass              │
  │  │   <500ms image)      │                                    │
  │  └─────────┬────────────┘                                    │
  │            │ borderline                                       │
  │            ▼                                                  │
  │  ┌──────────────────────┐                                    │
  │  │  Deep Analysis       │  More expensive models              │
  │  │                      │  Multi-modal analysis                │
  │  │  LLM-based:          │  Context understanding               │
  │  │  "Does this text     │  Sarcasm/satire detection            │
  │  │   violate policy X?" │  Image+text together                 │
  │  │                      │                                    │
  │  │  (<2s)               │                                    │
  │  └─────────┬────────────┘                                    │
  │            │ still borderline                                  │
  │            ▼                                                  │
  │  ┌──────────────────────┐                                    │
  │  │  Human Review Queue  │  Prioritized by severity            │
  │  │                      │  SLA: CSAM < 1hr, hate < 4hr        │
  │  │                      │  Decision → model training data      │
  │  └──────────────────────┘                                    │
  │                                                               │
  │  ACTIONS:                                                    │
  │    Remove, Warning, Age-gate, Reduce distribution,            │
  │    Inform user, No action                                     │
  │                                                               │
  │  APPEALS:                                                    │
  │    User appeals → different human reviewer                    │
  │    Overturn rate tracking per category                         │
  │    Continuous calibration of thresholds                        │
  │                                                               │
  └──────────────────────────────────────────────────────────────┘

STEP 3: KEY DESIGN DECISIONS

  Multi-stage pipeline (not one model!):
    - Stage 1 handles 90% of content cheaply
    - Stage 2 handles 9% with precision
    - Humans handle 1% (hardest cases)
    - Each stage's false negatives caught by next

  Category-specific thresholds:
    CSAM: extremely low threshold (catch everything, accept FP)
    Spam: higher threshold (FP harms user experience more)

  Temporal context:
    - Same account posting rapidly → suspicious
    - Content going viral → prioritize review
    - Coordinated campaigns → graph analysis

  Cross-modal:
    - Text says "beautiful sunset" + Image is NSFW → catch
    - Meme: innocent image + hateful text overlay
    - Need multi-modal understanding, not just per-modality
```

---

## 16.9 Design Case Study #6: Real-Time Translation System

### Problem: "Design a real-time translation system supporting 100+ language pairs"

```
STEP 1: REQUIREMENTS
  - Support 100+ languages
  - Latency < 500ms for text, real-time for speech
  - Handle: documents, chat messages, voice calls
  - Special: code-switching, domain-specific terms
  - Quality: human-level for top 10 language pairs

STEP 2: ARCHITECTURE

  ┌──────────────────────────────────────────────────────────────┐
  │              Translation System Architecture                   │
  │                                                               │
  │  INPUT: text / speech / document                              │
  │    │                                                          │
  │    ▼                                                          │
  │  ┌──────────────┐                                            │
  │  │  Language     │  Identify source language                   │
  │  │  Detection    │  Handle code-switching (mixed language)     │
  │  └──────┬───────┘                                            │
  │         │                                                     │
  │    ┌────┴───────────────────────────┐                         │
  │    │                                │                         │
  │    ▼                                ▼                         │
  │  ┌──────────────┐          ┌──────────────┐                  │
  │  │ HIGH-resource│          │ LOW-resource │                   │
  │  │ pair (en↔es, │          │ pair (xyz↔abc)│                  │
  │  │  en↔zh, etc.)│          │              │                   │
  │  │              │          │ Route through │                   │
  │  │ Direct model │          │ English pivot │                   │
  │  │ (fine-tuned) │          │ xyz→en→abc   │                   │
  │  └──────┬───────┘          └──────┬───────┘                  │
  │         │                         │                           │
  │         ▼                         ▼                           │
  │  ┌──────────────────────────────────────┐                    │
  │  │         Post-Processing              │                    │
  │  │  - Terminology override (glossary)    │                    │
  │  │  - Formality adjustment               │                    │
  │  │  - Named entity preservation          │                    │
  │  │  - Number/date format localization     │                    │
  │  └──────────────────────────────────────┘                    │
  │                                                               │
  │  MODEL STRATEGY:                                              │
  │  ┌──────────────────────────────────────────────────────┐    │
  │  │  Base: Single multilingual model (NLLB-200 / mT5)    │    │
  │  │        Handles all 100+ languages                     │    │
  │  │  Fine-tuned: Per language pair for top-10 pairs       │    │
  │  │        Higher quality, justified by volume            │    │
  │  │  LLM fallback: GPT-4 for complex/creative content    │    │
  │  │        Higher quality but higher cost & latency       │    │
  │  └──────────────────────────────────────────────────────┘    │
  │                                                               │
  │  SPEECH-TO-SPEECH:                                            │
  │  ┌──────────────────────────────────────────────────────┐    │
  │  │  Speech       Text         Translated     Translated │    │
  │  │  Input  ────▶ (ASR) ────▶ Text  ────────▶ Speech    │    │
  │  │         Whisper           MT model        TTS         │    │
  │  │                                                       │    │
  │  │  Streaming: chunk audio → translate → synthesize      │    │
  │  │  Latency budget: ASR(200ms) + MT(100ms) + TTS(200ms) │    │
  │  └──────────────────────────────────────────────────────┘    │
  │                                                               │
  │  EVALUATION:                                                  │
  │  ┌──────────────────────────────────────────────────────┐    │
  │  │  Automated: BLEU, chrF, COMET (neural metric)         │    │
  │  │  Human: MQM (Multidimensional Quality Metrics)        │    │
  │  │  Per language pair: quality dashboard                  │    │
  │  │  Regression testing: known-good translations          │    │
  │  └──────────────────────────────────────────────────────┘    │
  │                                                               │
  └──────────────────────────────────────────────────────────────┘
```

---

## 16.10 System Design Patterns Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────┐
│           Common ML System Design Patterns                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  MULTI-STAGE PIPELINE (Funnel Pattern)                           │
│    1M items → cheap filter → 1K → expensive model → 50           │
│    Use: search, recommendation, content moderation                │
│    Why: balance quality vs latency/cost                            │
│                                                                   │
│  ENSEMBLE / MODEL STACKING                                       │
│    Multiple models → combiner → final prediction                  │
│    Use: fraud detection, critical decisions                       │
│    Why: robust, catches different patterns                        │
│                                                                   │
│  ROUTING / MIXTURE OF EXPERTS                                    │
│    Router → specialized model based on input type                │
│    Use: multi-domain chatbot, language-specific models           │
│    Why: specialized models outperform generalist                  │
│                                                                   │
│  EMBEDDING + ANN (Retrieval Pattern)                             │
│    Encode items → build index → query-time ANN search            │
│    Use: similar items, semantic search, RAG                       │
│    Why: sub-millisecond retrieval over millions of items          │
│                                                                   │
│  FEATURE STORE PATTERN                                           │
│    Pre-compute features → serve from cache at inference          │
│    Use: any real-time ML with complex features                   │
│    Why: low latency, consistency between train/serve              │
│                                                                   │
│  HUMAN-IN-THE-LOOP                                               │
│    Model decides easy cases, escalates hard cases                │
│    Use: moderation, medical, legal                                │
│    Why: safety, builds labeled data for improvement              │
│                                                                   │
│  FEEDBACK LOOP (Online Learning)                                 │
│    User interactions → labels → retrain → improved model          │
│    Use: recommendation, ads, search                               │
│    Why: continuously improving from real-world data              │
│                                                                   │
│  CACHING + PRE-COMPUTATION                                       │
│    Pre-compute results for common inputs                         │
│    Use: popular queries, batch recommendations                   │
│    Why: dramatically reduce cost and latency                     │
│                                                                   │
│  FALLBACK / GRACEFUL DEGRADATION                                 │
│    If ML model fails → rules-based or cached fallback            │
│    Use: any production system                                    │
│    Why: 100% availability even during model issues               │
│                                                                   │
│  SHADOW DEPLOYMENT                                               │
│    Run new model alongside old, compare without serving          │
│    Use: pre-A/B validation, risk reduction                        │
│    Why: validates model before exposing to users                  │
│                                                                   │
│  DISTILLATION PATTERN                                            │
│    Large model generates training data for small model           │
│    Use: cost optimization, edge deployment                       │
│    Why: production-quality at 10-100× lower cost                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 16.11 Interview Tips for System Design

```
┌──────────────────────────────────────────────────────────────────┐
│          ML System Design Interview Playbook                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  BEFORE YOU START (2-3 min):                                     │
│    Ask clarifying questions:                                      │
│    - Scale: How many users/requests?                              │
│    - Latency: Real-time or batch?                                │
│    - Data: What data is available?                                │
│    - Metrics: How do we measure success?                          │
│    - Constraints: Budget, team size, timeline?                    │
│                                                                   │
│  DON'Ts:                                                         │
│    ✗ Jump into model architecture immediately                     │
│    ✗ Propose the most complex solution first                      │
│    ✗ Ignore data collection and labeling                          │
│    ✗ Skip evaluation and monitoring                               │
│    ✗ Forget about failure modes                                   │
│    ✗ Only discuss offline metrics                                 │
│                                                                   │
│  DOs:                                                            │
│    ✓ Start with problem formulation & metrics                     │
│    ✓ Propose a simple baseline FIRST                              │
│    ✓ Discuss data sources and labeling strategy                   │
│    ✓ Draw diagrams (architecture, data flow)                      │
│    ✓ Mention trade-offs at every decision point                   │
│    ✓ Cover deployment, monitoring, and iteration                  │
│    ✓ Discuss failure modes and fallbacks                          │
│    ✓ Reference specific scale numbers                              │
│                                                                   │
│  SIGNAL YOU'RE LOOKING FOR:                                      │
│    Senior: Thinks about the full system, not just the model      │
│    Staff+: Identifies the right problem to solve, considers       │
│            organizational & operational complexity, trade-offs    │
│            between build vs buy, cross-team dependencies          │
│                                                                   │
│  FRAMEWORK TO MEMORIZE:                                          │
│    1. Clarify → 2. Formulate → 3. Data → 4. Features →          │
│    5. Model → 6. Evaluation → 7. Deployment → 8. Monitoring     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 16.12 Interview Questions

### Conceptual Questions

**Q1: How would you handle the cold-start problem in a recommendation system?**

Cold start has two variants: (1) New user — no interaction history. Solutions: use demographic/contextual features, popular/trending items as initial recommendations, onboarding survey asking preferences, exploration-exploitation (epsilon-greedy or Thompson sampling to collect signal), transfer learning from similar products. (2) New item — no engagement data. Solutions: use content features (embeddings of title, image, description), creator/seller history as proxy, initial small-traffic exposure to collect clicks, content-based similarity to existing items. Long-term: build a multi-armed bandit system that balances exploiting known-good recommendations vs exploring new items/users to reduce cold-start gap.

**Q2: Why use a multi-stage ranking architecture instead of one model?**

Practical necessity at scale. With 10M items, a single complex model scoring all items would be too slow (e.g., BERT cross-encoder at 10ms/item = 28 hours). Multi-stage: candidate generation (cheap recall, ANN retrieval in <10ms) → first-stage ranker (lightweight model on ~1000 items, <20ms) → second-stage ranker (expensive model on ~100 items, <50ms) → re-ranking rules. Total: <100ms. Each stage filters by ~10×. First stage optimizes recall (don't miss good items), later stages optimize precision (rank the best items highest). This decomposition also allows different teams to own different stages.

**Q3: How do you ensure fairness in an ML system?**

Multi-layered approach: (1) Data: audit training data for representation bias, measure label bias across demographic groups. (2) Model: apply fairness constraints during training (equalized odds, demographic parity), measure per-group metrics (F1 by gender, race, age). (3) Post-hoc: calibrate predictions per group, threshold tuning per group to equalize FPR/FNR. (4) Monitoring: track per-group metrics in production, alert on disparity increases. (5) Process: diverse review teams, fairness review as part of model launch process, red-teaming with adversarial inputs. Trade-off: strict fairness constraints may reduce overall accuracy. Discuss which fairness definition is appropriate for the specific use case (equal opportunity vs equalized odds vs demographic parity).

**Q4: When would you choose a rule-based system over ML?**

Rules are better when: (1) Logic is well-defined and stable (e.g., "block transactions over $10K without 2FA"), (2) Explainability is critical and non-negotiable (regulated industries), (3) Not enough labeled data to train a model, (4) Very few edge cases (simple decision boundary), (5) Need immediate deployment (no training time). ML is better when: patterns are complex and non-obvious, data is abundant, the problem evolves over time, there are too many rules to maintain manually. Best: combine both. Rules handle the obvious cases (fast, interpretable), ML handles the nuanced cases. Rules as guardrails on ML output. Start with rules, add ML as data accumulates.

**Q5: How would you design the evaluation strategy for a search ranking system?**

Layered approach: Offline — NDCG@10, MRR on held-out judged query-document pairs; train/validation/test split by time (not random) to avoid leakage; measure per-query-segment (head vs torso vs tail queries). Online — A/B test measuring: CTR, successful session rate (found what they wanted), zero-result rate, reformulation rate (lower = better), purchase-after-search rate. Interleaving — more efficient than A/B for ranking: mix results from two models in one SERP, measure which model's results get more clicks. Human evaluation — periodic expert annotation on a sample; MQM-style rating on relevance, freshness, diversity. Guardrails — latency P95, coverage (% queries with results), revenue impact. Key insight: offline metrics can mislead — always validate with online experiments before launch.

### System Design Questions

**Q6: Design an AI-powered code review assistant.**

```
┌──────────────────────────────────────────────────────────────────┐
│           AI Code Review Assistant Architecture                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PR Opened / Updated (webhook)                                    │
│       │                                                           │
│       ▼                                                           │
│  ┌──────────────────┐                                            │
│  │  Diff Parser      │  Extract changed files, diff hunks        │
│  │                   │  Language detection, file filtering       │
│  └────────┬──────────┘                                           │
│           │                                                       │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │  Context Builder  │  Get full file context (not just diff)     │
│  │                   │  Related files (imports, callers)          │
│  │                   │  PR description, linked issues             │
│  │                   │  Project style guide, past review comments │
│  └────────┬──────────┘                                           │
│           │                                                       │
│    ┌──────┴──────────────────────┐                                │
│    │                             │                                │
│    ▼                             ▼                                │
│  ┌──────────────┐   ┌───────────────────┐                        │
│  │ Static       │   │  LLM-based        │                        │
│  │ Analysis     │   │  Review           │                        │
│  │              │   │                   │                        │
│  │ Linting      │   │ Per-hunk analysis │                        │
│  │ Type errors  │   │ Bug detection     │                        │
│  │ Security     │   │ Improvement ideas │                        │
│  │ (Semgrep)    │   │ Style suggestions │                        │
│  └──────┬───────┘   └────────┬──────────┘                        │
│         │                    │                                    │
│         ▼                    ▼                                    │
│  ┌──────────────────────────────────────┐                        │
│  │  Comment Aggregator                  │                        │
│  │  - Deduplicate similar comments      │                        │
│  │  - Prioritize by severity            │                        │
│  │  - Filter noise (< confidence 0.7)   │                        │
│  │  - Group by file/concern             │                        │
│  │  - Rate limit (max 15 comments/PR)   │                        │
│  └──────────────────────────────────────┘                        │
│           │                                                       │
│           ▼                                                       │
│  Post inline comments on PR via GitHub/GitLab API                │
│                                                                   │
│  FEEDBACK LOOP:                                                   │
│    Developer accepts/dismisses comment → training signal          │
│    Dismissed comments analyzed monthly → improve prompts          │
│    Accept rate target: > 60% (otherwise too noisy)                │
│                                                                   │
│  COST CONTROL:                                                   │
│    - Only analyze changed hunks (not full files)                  │
│    - Cache common patterns                                       │
│    - Skip files: generated, lock, binary, config                  │
│    - Small model for triage, large model for detailed review     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 16.13 Key Resources

| Resource                                                           | Type   | Why It Matters                              |
| ------------------------------------------------------------------ | ------ | ------------------------------------------- |
| _Designing Machine Learning Systems_ (Chip Huyen)                  | Book   | Best end-to-end ML systems design book      |
| _Machine Learning System Design Interview_ (Ali Aminian & Alex Xu) | Book   | Structured approach to ML design interviews |
| _System Design Interview – ML_ (Alex Xu Vol 2)                     | Book   | Large-scale system design with ML focus     |
| _Stanford CS 329S: ML Systems Design_                              | Course | Academic depth on production ML systems     |
| _Made with ML_ (Goku Mohandas)                                     | Course | Free end-to-end MLOps course                |
| _ML Design Docs_ (Google)                                          | Guide  | How Google designs ML systems internally    |
| _Papers We Love: ML Systems_                                       | Papers | Collection of influential ML systems papers |
| _Eugene Yan's Blog_                                                | Blog   | Excellent practical ML systems writing      |
| _Chip Huyen's Blog_                                                | Blog   | ML systems, LLMOps, industry trends         |

---

[← Module 15: MLOps](../module-15-mlops/README.md) | [Back to Course Overview →](../index.md)
