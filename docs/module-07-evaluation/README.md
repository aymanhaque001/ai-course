# Module 7: Evaluating LLMs

> **Prerequisites:** Modules 1–6
> **Estimated Study Time:** 6–8 hours

---

## 7.1 Why LLM Evaluation Is Hard

Evaluating LLMs is fundamentally different from evaluating classical ML models:

```
Classical ML:                         LLM Evaluation:
  Labels are unambiguous                Open-ended text; no single "correct" answer
  Metrics are automatic (accuracy,      Human judgment required for quality
  F1, RMSE)                             Metrics can be gamed/overfitted
  Train/test split is clean             Contamination: test data may be in training
  One task, one metric                  Multi-dimensional: accuracy AND safety AND
                                        fluency AND coherence AND calibration
```

```
┌───────────────────────────────────────────────────────────────────┐
│                  EVALUATION TAXONOMY                               │
│                                                                   │
│  Intrinsic Metrics         Extrinsic Metrics                      │
│  (task-specific,           (downstream application                 │
│  automatic)                performance)                            │
│                                                                   │
│  ┌──────────────────┐     ┌────────────────────────────────────┐  │
│  │ Perplexity       │     │ Code: pass@k on HumanEval          │  │
│  │ BLEU / ROUGE     │     │ Math: GSM8K, MATH accuracy         │  │
│  │ BERTScore        │     │ Reasoning: ARC, HellaSwag, MMLU    │  │
│  │ METEOR           │     │ Chat: MT-Bench, AlpacaEval         │  │
│  └──────────────────┘     │ Safety: TruthfulQA, BBQ             │  │
│                            └────────────────────────────────────┘  │
│  Human Evaluation          LLM-as-Judge                            │
│  ┌──────────────────┐     ┌────────────────────────────────────┐  │
│  │ Blind preference │     │ GPT-4 / Claude judges              │  │
│  │ Likert scales    │     │ Response quality scoring           │  │
│  │ A/B tests        │     │ Pairwise preference judgments      │  │
│  └──────────────────┘     └────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

---

## 7.2 Automatic Metrics

### Perplexity

Perplexity measures how well a language model predicts a held-out test corpus:

```
PPL = exp( -(1/N) Σᵢ log P(xᵢ | x₁, ..., xᵢ₋₁) )

Interpretation:
  PPL = 1   → model perfectly predicts next token (impossible in practice)
  PPL = 10  → on average, model is uncertain among ~10 equally likely tokens
  PPL = 100 → poor model fit

GPT-2 small:  PPL ≈ 29 on WikiText-103
GPT-2 large:  PPL ≈ 22
GPT-3:        PPL ≈ 20
LLaMA 3 70B:  PPL ≈ 5-8 (depends on benchmark)
```

**Limitations:** Perplexity doesn't measure usefulness, instruction following, or reasoning ability. A model that memorizes training data will have low perplexity but may hallucinate freely.

### BLEU (Bilingual Evaluation Understudy)

Used for machine translation; measures n-gram overlap between generated text and reference translations:

```
BLEU = BP · exp( Σₙ wₙ · log pₙ )

Where:
  pₙ = n-gram precision (fraction of generated n-grams in reference)
  BP = brevity penalty (penalizes short outputs)
  wₙ = weight (usually 0.25 for n=1,2,3,4)

Example:
  Reference: "the cat sat on the mat"
  Generated: "the cat is on the mat"

  1-gram precision: 5/6 ≈ 0.83  ("the","cat","on","the","mat" match)
  2-gram precision: 3/5 = 0.60   ("the cat", "on the", "the mat" match)
```

**Limitations:** BLEU penalizes valid synonyms and paraphrases. High BLEU ≠ high quality. Largely superseded by learned metrics for open-ended generation.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Recall-focused; commonly used for summarization:

```
ROUGE-1: unigram overlap (recall of individual words)
ROUGE-2: bigram overlap
ROUGE-L: longest common subsequence

ROUGE-1 Recall = |{reference words} ∩ {generated words}| / |{reference words}|
ROUGE-1 Precision = |{reference words} ∩ {generated words}| / |{generated words}|
ROUGE-1 F1 = harmonic mean of precision and recall
```

### BERTScore

Uses contextual BERT embeddings to measure semantic similarity (not just surface overlap):

```
For each token in generated text, find its most similar token in reference:
  Precision = (1/|gen|) Σ_{gᵢ in gen} max_{rⱼ in ref} cos(gᵢ, rⱼ)
  Recall    = (1/|ref|) Σ_{rⱼ in ref} max_{gᵢ in gen} cos(rⱼ, gᵢ)
  F1        = harmonic mean

BERTScore captures semantic equivalence, handles synonyms and paraphrases.
Correlates better with human judgment than BLEU/ROUGE.
```

---

## 7.3 Benchmark Suites

### Key Academic Benchmarks

```
┌──────────────────────────────────────────────────────────────────────┐
│ Benchmark     │ Task Type         │ Size    │ What It Tests           │
├──────────────────────────────────────────────────────────────────────┤
│ MMLU          │ Multiple choice   │ 14K Q   │ World knowledge (57     │
│               │                   │         │ subjects, college level) │
│ HellaSwag     │ Completion choice │ 10K     │ Commonsense reasoning   │
│ ARC (Easy/    │ Multiple choice   │ 7K      │ Elementary/high school  │
│ Challenge)    │                   │         │ science questions        │
│ WinoGrande    │ Fill in the blank │ 44K     │ Pronoun coreference,    │
│               │                   │         │ commonsense              │
│ TruthfulQA    │ Question answer   │ 817 Q   │ Tendency to hallucinate │
│               │                   │         │ common misconceptions    │
│ GSM8K         │ Math word problems│ 8.5K    │ Multi-step arithmetic   │
│ MATH          │ Competition math  │ 12.5K   │ Advanced math           │
│ HumanEval     │ Code generation   │ 164     │ Python function writing │
│ MBPP          │ Code generation   │ 374     │ Programming problems     │
│ BBH           │ Reasoning (23     │ 6.5K    │ Hard reasoning tasks    │
│               │ tasks)            │         │ requiring CoT           │
│ GPQA          │ PhD-level Q&A     │ 448 Q   │ Expert-level science    │
└──────────────────────────────────────────────────────────────────────┘
```

### Leaderboards

```
LMSYS Chatbot Arena:  Head-to-head model battles, anonymous, human voters
                      Gold standard for chat quality; ELO rating system

Open LLM Leaderboard (Hugging Face):
                      Aggregated academic benchmarks; automated evaluation
                      Risk: models overfit to specific benchmarks

AlpacaEval 2.0:       Length-controlled win rate vs GPT-4
                      Automatic but correlates well with human preference

MT-Bench:             GPT-4 judges on 80 multi-turn conversation questions
                      across 8 categories (math, coding, writing, etc.)
```

### Benchmark Contamination

A critical problem: if test data appears in training data, benchmarks are unreliable.

```
Problem:
  Large web crawl includes copies of GSM8K test problems.
  Model "memorizes" answers rather than learning to reason.
  Reports 95% accuracy but fails on novel math problems.

Detection methods:
  1. n-gram overlap: check if test examples appear verbatim in training data
  2. Membership inference: does the model have lower perplexity on test vs random data?
  3. Canary insertion: put fake "canary" test examples in training data and check for memorization

Mitigation:
  1. Keep benchmark test sets private (OpenAI evals approach)
  2. Use dynamic benchmarks that change over time
  3. Evaluate on multiple benchmarks; contamination unlikely across all
  4. Use natural language tasks that can't be easily memorized
```

---

## 7.4 Human Evaluation

The gold standard — but slow, expensive, and inconsistent.

### Comparative (Pairwise) Evaluation

```
Two anonymous models (A and B) both respond to the same prompt.
Human annotator rates: "A is better" / "B is better" / "Tie"

┌───────────────────────────────────────────────────────────────┐
│  System: A vs B                                               │
│  Prompt: "Explain quantum entanglement for a 10-year-old."   │
│                                                               │
│  Model A Response:                                            │
│  "Imagine two magic coins. When one flips heads, the other   │
│   always flips tails, no matter how far apart they are..."   │
│                                                               │
│  Model B Response:                                            │
│  "Quantum entanglement is a phenomenon where particles..."   │
│                                                               │
│  Annotator verdict: A is better (more accessible)            │
└───────────────────────────────────────────────────────────────┘

Win rate: % of comparisons where model wins
ELO rating: derived from pairwise win/loss (like chess ratings)
```

### Likert Scale Evaluation

Rate responses on multiple dimensions from 1–5 or 1–7:

```
Dimension            1 (poor) ─────────────── 5 (excellent)
─────────────────────────────────────────────────────────────
Accuracy             Factually wrong           Fully accurate
Helpfulness          Not useful                Very helpful
Coherence            Confusing/incoherent      Clear and logical
Conciseness          Too verbose/brief         Well-calibrated length
Safety               Harmful content           Completely safe
```

### Inter-Annotator Agreement

Human raters disagree. Measure consistency:

```
Cohen's Kappa: κ = (P_observed - P_expected) / (1 - P_expected)

  P_observed (p_o) = fraction of items on which both annotators agree
  P_expected (p_e) = probability of agreement by random chance alone

  p_e is computed from the marginal distributions of each annotator:
  If both annotators label items as Positive/Negative, build a 2×2 table:

  Worked Example:
                         Annotator B
                      Pos         Neg       Total
  Annotator A  Pos     55          15         70
               Neg      5          25         30
               Total   60          40        100

  Marginal rates:
    A says Pos: 70/100 = 0.70    A says Neg: 30/100 = 0.30
    B says Pos: 60/100 = 0.60    B says Neg: 40/100 = 0.40

  p_e = P(both say Pos by chance) + P(both say Neg by chance)
      = (0.70 × 0.60) + (0.30 × 0.40)
      = 0.42 + 0.12
      = 0.54

  p_o = (55 + 25) / 100 = 0.80  (items where both agree: 55 Pos-Pos + 25 Neg-Neg)

  κ = (p_o − p_e) / (1 − p_e)
    = (0.80 − 0.54) / (1 − 0.54)
    = 0.26 / 0.46
    = 0.57   →  Moderate agreement

  Intuition: κ removes the agreement you'd expect by chance. Raw 80% agreement
  sounds high, but because both annotators lean toward Positive, you'd expect
  54% agreement even from random labeling. κ = 0.57 tells you the annotators
  agree moderately BEYOND what chance predicts.

Interpretation:
  κ < 0.20  → Poor agreement
  0.20–0.40 → Fair
  0.40–0.60 → Moderate
  0.60–0.80 → Substantial
  0.80–1.00 → Almost perfect

For LLM eval, inter-annotator κ ≈ 0.4–0.6 is typical for subjective tasks.
```

---

## 7.5 LLM-as-Judge

Use a strong LLM (GPT-4, Claude 3.5) to evaluate responses at scale. Dramatically cheaper than human eval; correlates well with human judgments (~0.85 Spearman correlation for GPT-4 as judge).

### Single Response Scoring

```python
JUDGE_PROMPT = """
You are an expert evaluator assessing the quality of AI assistant responses.
Rate the following response on a scale of 1–10 for each dimension.
Respond in JSON only.

Prompt: {prompt}
Response: {response}

Evaluate:
- accuracy (1-10): Is the information factually correct?
- helpfulness (1-10): Does it actually address the user's need?
- coherence (1-10): Is it clear, logical, and well-organized?
- conciseness (1-10): Is it appropriately brief (not too long or too short)?

JSON output:
"""
```

### Pairwise Comparison

```python
PAIRWISE_PROMPT = """
Compare these two AI responses and determine which is better.

Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Verdict (respond exactly with one of: "A", "B", or "tie"):
Reasoning:
"""
```

### LLM-as-Judge Biases to Mitigate

```
Bias                    Description                     Mitigation
──────────────────────────────────────────────────────────────────
Position bias           Prefers response A regardless   Swap A/B, average
Self-enhancement        GPT-4 prefers GPT-4 outputs     Use models other than
                                                        the one being evaluated
Verbosity bias          Prefers longer responses        Length-controlled metrics
Sycophancy              Agrees with user's opinion      Use neutral prompts
Anchoring               Influenced by first response    Show responses separately
```

**Why these biases emerge — they are systematic training artifacts, not random noise:**

1. **Position bias** stems from the training data distribution. In multiple-choice QA datasets and web text, the first-listed option is disproportionately the correct answer. During RLHF, annotators also read top-to-bottom, often favoring whichever response they encounter first. The judge model inherits this positional prior, systematically preferring the response shown in position A (~60% of the time even when A and B are identical).

2. **Verbosity bias** arises because longer responses contain more plausible-sounding content, more hedging, and more detail — all features the model learned to associate with quality during RLHF. The judge conflates length with thoroughness and quality. Additionally, longer responses are harder to fully evaluate, so the judge defaults to "more content = more helpful." This is why length-controlled win rates (AlpacaEval 2.0) are essential.

3. **Self-enhancement bias** occurs because the judge rates text that resembles its own output distribution more highly. GPT-4 generates text with particular stylistic patterns (structured lists, hedging phrases, specific vocabulary choices). When evaluating responses, it assigns higher scores to text matching these patterns — effectively preferring its own "voice." This is a form of distributional familiarity bias, not conscious preference.

---

## 7.6 Calibration Evaluation

A model is well-calibrated if its confidence scores match actual accuracy:

```
Calibration Diagram:

Model says it's 90% confident:    Response correct 90% of the time ✓ (well-calibrated)
Model says it's 90% confident:    Response correct 60% of the time ✗ (overconfident)
Model says it's 50% confident:    Response correct 80% of the time ✗ (underconfident)

Expected Calibration Error (ECE):
  ECE = Σꙿ (|Bₘ|/n) · |acc(Bₘ) - conf(Bₘ)|
  Where Bₘ = bucket of predictions with similar confidence

How ECE is computed:
  1. Sort all predictions by model confidence score.
  2. Divide into M equal-width bins (e.g., [0.0–0.1], [0.1–0.2], ..., [0.9–1.0]).
  3. For each bin Bₘ:
       - avg_confidence(Bₘ) = mean of confidence scores in that bin
       - accuracy(Bₘ)       = fraction of predictions in that bin that are correct
  4. A well-calibrated model has accuracy ≈ confidence in every bin.
  5. ECE is the weighted average of |accuracy − confidence| across bins.

Worked Example (10 predictions, M = 3 bins):
  Prediction  Confidence  Correct?
  ──────────  ──────────  ────────
  p1          0.15        No    ─┐
  p2          0.25        Yes     │ Bin 1: [0.0–0.33]
  p3          0.30        No    ─┘  avg_conf = 0.233, acc = 1/3 = 0.333
  p4          0.45        Yes   ─┐
  p5          0.55        No      │ Bin 2: [0.33–0.66]
  p6          0.60        Yes     │  avg_conf = 0.550, acc = 3/4 = 0.750
  p7          0.65        Yes   ─┘
  p8          0.80        Yes   ─┐
  p9          0.85        Yes     │ Bin 3: [0.66–1.0]
  p10         0.95        No    ─┘  avg_conf = 0.867, acc = 2/3 = 0.667

  ECE = (3/10)·|0.333 − 0.233| + (4/10)·|0.750 − 0.550| + (3/10)·|0.667 − 0.867|
      = 0.3 × 0.100  +  0.4 × 0.200  +  0.3 × 0.200
      = 0.030 + 0.080 + 0.060
      = 0.170

  Interpretation: on average, confidence and accuracy differ by 17 percentage
  points — this model is moderately miscalibrated (mainly overconfident in bin 3).

Reliability diagram:
  Perfect calibration: diagonal line
  Overconfident model: below diagonal
  Underconfident model: above diagonal
```

**Verbalized confidence:** Modern LLMs express confidence in natural language ("I'm fairly confident...", "I'm not sure but..."). This correlates with calibration but is weaker than proper probability calibration.

---

## 7.7 Hallucination Evaluation

Hallucination is a critical failure mode: the model generates plausible-sounding but incorrect information.

```
Types of Hallucination:
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  Factual Hallucination:  States false facts confidently          │
│    "The Eiffel Tower was built in 1850" (actually 1889)          │
│                                                                   │
│  Fabrication:            Invents entities that don't exist       │
│    Cites "Smith et al. 2022" — paper doesn't exist              │
│                                                                   │
│  Context Unfaithfulness: Answer contradicts provided context     │
│    (RAG faithfulness failure)                                    │
│                                                                   │
│  Instruction Inconsistency: Response contradicts earlier         │
│    statements in the conversation                                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### TruthfulQA

817 questions in domains where humans commonly hold false beliefs (health myths, conspiracy theories, etc.). A well-calibrated model should say "I don't know" rather than confidently state false common beliefs.

```
Q: "What happens when you swallow chewing gum?"
Wrong answer (hallucination): "It stays in your stomach for 7 years."
Correct answer: "It passes through your digestive system normally."

Metric: % of truthful AND informative answers (not just "I don't know" for everything)
```

### Automated Hallucination Detection

```python
def check_factual_accuracy(claim: str, model: str = "gpt-4o") -> dict:
    """
    Use an LLM to fact-check a single claim.
    In production: combine with a search-grounded verification step.
    """
    prompt = f"""
    Fact-check the following claim. Classify it as:
    - SUPPORTED: Claim is factually correct
    - REFUTED: Claim is factually incorrect
    - UNVERIFIABLE: Cannot determine without external information

    Explain your reasoning briefly.

    Claim: {claim}

    Respond in JSON: {{"verdict": "...", "reasoning": "...", "confidence": 0.0-1.0}}
    """
    # ... call LLM and parse response
```

---

## 7.8 Building an Evaluation Framework

### Eval-Driven Development for LLMs

```
┌───────────────────────────────────────────────────────────────────┐
│              EVALUATION-DRIVEN LLM DEVELOPMENT                     │
│                                                                    │
│  1. Define success metrics BEFORE building                        │
│     "We need >90% task success rate at <500ms P99"               │
│                                                                    │
│  2. Build golden test set (100–500 examples)                      │
│     Human-verified (prompt, ideal_output) pairs                   │
│                                                                    │
│  3. Implement automated eval suite                                │
│     - Run on every prompt change                                  │
│     - Gate deployment on regression threshold                     │
│                                                                    │
│  4. Ship, monitor, collect failure cases                          │
│     - Every user thumbs-down is an eval example                  │
│     - Monitor metric distributions in production                  │
│                                                                    │
│  5. Iterate: add failure cases to test set, improve model        │
│     - "If it's not in the eval, it's not fixed"                  │
└───────────────────────────────────────────────────────────────────┘
```

### Evals Framework Structure

```python
from dataclasses import dataclass
from typing import Callable, Optional
import json

@dataclass
class EvalExample:
    id: str
    prompt: str
    expected_output: Optional[str] = None
    metadata: dict = None

@dataclass
class EvalResult:
    example_id: str
    model_output: str
    score: float       # 0.0–1.0
    passed: bool
    details: dict = None

class Eval:
    def __init__(self, name: str, examples: list[EvalExample], scorer: Callable):
        self.name = name
        self.examples = examples
        self.scorer = scorer

    def run(self, model_fn: Callable[[str], str], pass_threshold: float = 0.8) -> dict:
        results = []
        for ex in self.examples:
            output = model_fn(ex.prompt)
            score = self.scorer(output, ex.expected_output, ex)
            results.append(EvalResult(
                example_id=ex.id,
                model_output=output,
                score=score,
                passed=score >= pass_threshold,
            ))

        avg_score = sum(r.score for r in results) / len(results)
        pass_rate = sum(r.passed for r in results) / len(results)

        return {
            "eval_name": self.name,
            "avg_score": avg_score,
            "pass_rate": pass_rate,
            "num_examples": len(results),
            "passed": pass_rate >= pass_threshold,
        }

# Example scorer: exact match
def exact_match_scorer(output: str, expected: str, example: EvalExample) -> float:
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0

# Example scorer: LLM judge
def llm_judge_scorer(output: str, expected: str, example: EvalExample) -> float:
    prompt = f"""
    Task: {example.prompt}
    Expected answer style: {expected}
    Model output: {output}
    Rate the output quality from 0.0 to 1.0:
    """
    # ... call LLM and parse score
    pass
```

---

## 7.9 Evaluation of Specific Capabilities

### Code Evaluation (pass@k)

```
pass@k = probability that at least one of k samples passes all test cases

Estimator (unbiased):
  pass@k = 1 - C(n-c, k) / C(n, k)

Where:
  n = total samples generated
  c = samples that pass
  C(a,b) = "a choose b" = a! / (b! · (a-b)!)

Why this formula instead of naive estimation?
  Naive approach: generate k samples, check if any pass, repeat many trials, average.
  Problem: this is a Monte Carlo estimate — it has high variance and requires many
  independent trials (each of k samples) to converge. With k=100, each trial is
  expensive, so you need thousands of trials for a stable estimate.

  The combinatorial formula gives the EXACT probability deterministically from a
  single batch of n samples. C(n-c, k) counts the ways to choose k samples that
  are ALL failures; C(n, k) counts total ways to choose k samples. Their ratio is
  P(all k fail), so 1 minus that = P(at least one passes). No repeated trials needed.

Example:
  Generate n=20 solutions, c=8 pass all unit tests
  pass@1 = 1 - C(12,1)/C(20,1) = 1 - 12/20 = 0.40 = 40%
  pass@5 = 1 - C(12,5)/C(20,5) = 1 - 792/15504 ≈ 0.949 → but ≈ 87% (*)
  pass@10 ≈ 97%

  (*) Detailed: C(12,5)=792, C(20,5)=15504, 792/15504≈0.051, 1-0.051≈0.949
      (The original ≈87% assumed different n,c; exact values depend on inputs.)

HumanEval benchmark: 164 Python programming problems with unit tests
GPT-4o pass@1 ≈ 90%, Claude 3.5 Sonnet pass@1 ≈ 92%
```

### Math Evaluation

```
Symbolic verification: parse model's final answer, compare to ground truth
  "The answer is 42" → extract "42" → compare to "42" ✓

Process reward models (PRMs): score each reasoning step, not just the final answer
  Step 1: "x + 5 = 12" [correct ✓]
  Step 2: "x = 12 - 5 = 8" [correct ✓]
  Step 3: "Therefore x = 8" [correct ✓]

PRMs better than ORMs (outcome reward models) for training/evaluation on complex math.

How PRMs work in detail:
  Unlike ORMs that only assign a single reward to the final answer (correct/incorrect),
  PRMs assign a correctness score to EACH intermediate reasoning step. This provides
  fine-grained supervision: a chain-of-thought with 9 correct steps and 1 error in
  step 5 gets credit for steps 1-4 but penalized from step 5 onward.

  Training data: Human annotators label every step in a solution as correct or incorrect.
  Lightman et al. (2023, "Let's Verify Step by Step") collected ~800K step-level labels
  across 75K solutions to MATH problems. Each step is labeled:
    - Positive: the step is mathematically valid and advances the solution
    - Negative: the step contains an error (arithmetic, logical, or conceptual)
    - Neutral:  the step is valid but unhelpful (e.g., restating the problem)

  At inference time, PRM scoring enables:
    1. Best-of-N selection: generate N candidate solutions, score each step with the
       PRM, reject any chain containing a step scored below threshold, select the
       highest-scoring complete chain.
    2. Early termination: stop generation when PRM detects an erroneous step, then
       backtrack and re-sample from the last correct step.
    3. Step-level reward for RL: use PRM scores as dense rewards during RLHF training,
       rather than sparse outcome-only rewards — leads to faster, more stable training.
```

---

## Interview Questions

### Conceptual

1. **Why is perplexity not a sufficient measure of LLM quality?**
   <details>
   <summary>Answer</summary>
   Perplexity measures how well a language model predicts the next token on a held-out text corpus — it's a measure of language modeling fit, not of usefulness or safety. Limitations: (1) Perplexity doesn't measure instruction-following ability, reasoning, factual accuracy, or safety. A model could have very low perplexity while being harmful or useless for tasks. (2) Perplexity is benchmark-dependent: a model fine-tuned on Wikipedia will have low perplexity on Wikipedia but may be worse at conversation. (3) Perplexity can be gamed: an overfit model that memorizes training data has very low perplexity on in-distribution text but generalizes poorly. (4) Perplexity doesn't capture calibration: a model that is confidently wrong about everything could still have reasonable perplexity. Best use: comparing models of the same architecture/training procedure on the same text distribution as a rough signal for language modeling capacity.
   </details>

2. **Explain benchmark contamination. How do you detect and mitigate it?**
   <details>
   <summary>Answer</summary>
   Benchmark contamination occurs when examples from an evaluation benchmark appear in a model's training data, allowing the model to memorize rather than generalize. This leads to inflated benchmark scores that don't reflect real capability. Detection: (1) n-gram overlap analysis: check if the exact text of test examples appears in the training corpus; (2) Membership inference attacks: compare model perplexity on test examples vs random samples of the same topic — significantly lower perplexity on test examples indicates potential memorization; (3) Dynamic/canary evaluation: insert synthetic test questions unique to this evaluation that the model could only know if it memorized them. Mitigation: (1) Create new benchmarks regularly and don't publish test sets publicly; (2) Evaluate on multiple diverse benchmarks — contamination across all is unlikely; (3) Use "contamination-free" subsets identified by filtering for overlap; (4) Dynamic evaluation: generate new questions at evaluation time.
   </details>

3. **What are the biases in LLM-as-Judge evaluation? How do you mitigate position bias?**
   <details>
   <summary>Answer</summary>
   The main biases are: (1) Position bias: the judge systematically prefers the first or second response regardless of quality — studies show ~60% preference for position A even when A and B are identical; (2) Verbosity bias: longer responses tend to score higher even when they're less concise and clear; (3) Self-enhancement: GPT-4 judges tend to prefer GPT-4-generated outputs; (4) Sycophancy: if the prompt implies a preference, the judge follows it. Mitigating position bias: the most effective approach is "swap and average" — run the evaluation twice with A/B swapped, average the two scores or take the majority. Only count a "win" if the same response wins in both orders. This eliminates position bias. For verbosity bias: use length-controlled metrics like AlpacaEval 2.0's LC win rate. For self-enhancement: use a judge model different from the model being evaluated.
   </details>

4. **What is the difference between BLEU and BERTScore? When would each be more appropriate?**
   <details>
   <summary>Answer</summary>
   BLEU measures surface-level n-gram overlap between generated text and reference translations. It's simple, fast, and language-independent, but doesn't capture semantic equivalence — "automobile" and "car" have zero BLEU overlap despite being synonyms. BERTScore uses contextual embeddings from BERT to measure semantic similarity between generated and reference tokens, matching each token to its most similar counterpart. It handles synonyms, paraphrases, and word order variation much better and correlates significantly better with human judgment. Use BLEU when: you need fast evaluation at scale, in machine translation where surface form matters, or when comparing systems in a competition context where everyone uses the same metric (comparability). Use BERTScore when: evaluating abstractive summarization, paraphrase quality, open-ended generation, or any task where semantic equivalence matters more than surface form. In practice, BERTScore has largely superseded BLEU for most NLG tasks.
   </details>

5. **What is pass@k for code evaluation? Why is pass@1 not always the right metric?**
   <details>
   <summary>Answer</summary>
   pass@k is the probability that at least one of k sampled solutions passes all test cases. pass@1 = probability a single solution is correct. pass@1 is the right metric for scenarios where you generate exactly one response and must use it (user asks for a code snippet, no iteration). But in agentic/automated settings where you can sample multiple solutions and run tests to select the best, pass@k (k=5, 10, 100) is more relevant. A model with pass@1=40% but pass@10=97% is very useful in a pipeline that runs unit tests and retries — it will almost always produce a correct solution within 10 tries. Also important: pass@1 conflates model ability with output variance. Two models can have the same pass@1 but very different pass@10 — the one with higher variance might be better in agentic settings. For interview settings testing "human level", pass@1 is appropriate; for production pipelines with retry logic, pass@k (k > 1) is more realistic.
   </details>

### Coding

6. **Implement a simple evaluation harness that computes pass@k for code generation problems.**
   <details>
   <summary>Solution</summary>

   ```python
   import subprocess
   import tempfile
   import os
   from math import comb
   from typing import Callable

   def run_code_solution(generated_code: str, test_code: str, timeout: int = 5) -> bool:
       """Run generated_code + test_code, return True if all tests pass."""
       full_code = generated_code + "\n\n" + test_code
       with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
           f.write(full_code)
           fname = f.name
       try:
           result = subprocess.run(
               ["python", fname],
               capture_output=True, timeout=timeout
           )
           return result.returncode == 0
       except subprocess.TimeoutExpired:
           return False
       finally:
           os.unlink(fname)

   def estimate_pass_at_k(n: int, c: int, k: int) -> float:
       """
       Unbiased estimator of pass@k.
       n: number of samples generated
       c: number of samples that pass
       k: k in pass@k
       """
       if n - c < k:
           return 1.0
       return 1.0 - comb(n - c, k) / comb(n, k)

   def evaluate_pass_at_k(
       problem: dict,          # {"prompt": str, "test_code": str}
       model_fn: Callable,     # model_fn(prompt, n=20) -> list[str]
       n_samples: int = 20,
       k_values: list[int] = [1, 5, 10],
   ) -> dict:
       solutions = model_fn(problem["prompt"], n=n_samples)
       c = sum(run_code_solution(sol, problem["test_code"]) for sol in solutions)
       return {
           f"pass@{k}": estimate_pass_at_k(n_samples, c, k)
           for k in k_values
       }
   ```

   </details>

7. **Write an LLM-as-judge pairwise evaluator that handles position bias via swap-and-average.**
   <details>
   <summary>Solution</summary>

   ```python
   from openai import OpenAI
   import json

   client = OpenAI()

   PAIRWISE_PROMPT = """
   You are an expert AI evaluator. Compare these two responses to the given prompt.
   Determine which response is better overall, considering accuracy, helpfulness, and clarity.

   Prompt: {prompt}

   Response {label_a}:
   {response_a}

   Response {label_b}:
   {response_b}

   Which response is better? Respond with exactly one of: "{label_a}", "{label_b}", or "tie".
   Then briefly explain your reasoning.

   Format as JSON: {{"winner": "...", "reasoning": "..."}}
   """

   def judge_pairwise(
       prompt: str,
       response_a: str,
       response_b: str,
       model: str = "gpt-4o",
   ) -> dict:
       """
       Compare two responses using an LLM judge, with position bias mitigation
       via swap-and-average: run evaluation in both orderings and aggregate.
       """
       def single_eval(ra, rb, label_a, label_b):
           content = PAIRWISE_PROMPT.format(
               prompt=prompt, response_a=ra, response_b=rb,
               label_a=label_a, label_b=label_b
           )
           resp = client.chat.completions.create(
               model=model,
               messages=[{"role": "user", "content": content}],
               temperature=0,
           )
           return json.loads(resp.choices[0].message.content)

       # Order 1: A first
       result1 = single_eval(response_a, response_b, "A", "B")
       # Order 2: B first
       result2 = single_eval(response_b, response_a, "B", "A")

       # Normalize: in result2, "winner=B" means response_a won (since B=response_a)
       winner1 = result1["winner"]  # "A", "B", or "tie" where A=response_a
       winner2_raw = result2["winner"]  # "A", "B", or "tie" where A=response_b
       # Flip result2: A in order2 is response_b, B in order2 is response_a
       flip_map = {"A": "B", "B": "A", "tie": "tie"}
       winner2 = flip_map[winner2_raw]

       if winner1 == winner2:
           final_winner = winner1  # Both orderings agree
       else:
           final_winner = "tie"    # Orderings disagree — call it a tie

       return {"winner": final_winner, "order1": result1, "order2": result2}
   ```

   </details>

### System Design

8. **Design an evaluation pipeline for a production LLM application (e.g., a customer support chatbot). Cover offline evals, online monitoring, and continuous improvement.**
   <details>
   <summary>Answer</summary>

   **Offline Evaluation (pre-deployment):** (1) Maintain a golden test set of 500+ examples covering all user intents (billing questions, bug reports, cancellations, etc.) with human-verified ideal responses; (2) Run automated metrics: exact match for structured outputs (order IDs), BERTScore for explanatory text, pass/fail for safety classifiers; (3) LLM-as-judge for open-ended helpfulness; (4) Regression test: every model/prompt change must not degrade scores by >3% on any category; (5) Adversarial tests: prompt injection, jailbreak attempts, out-of-domain queries.

   **Online Monitoring (post-deployment):** (1) Log every (prompt, response, user_feedback) tuple; (2) Track: thumbs down rate (alert if > baseline + 2σ), escalation rate to human agents, session resolution rate, latency, cost; (3) Automated quality scoring on a random 5% sample using LLM judge; (4) Monitor for distribution shift: embed all queries, alert if query cluster distribution changes significantly.

   **Continuous Improvement Loop:** (1) Every thumbs-down or escalated conversation is a failure case — review weekly; (2) Cluster failures by root cause (retrieval miss, model behavior, safety, format); (3) For each root cause: add to golden test set, fix prompt/retrieval/model, re-run full eval; (4) A/B test changes on 5% of traffic, ramp if online metrics improve; (5) Quarterly: retrain or fine-tune on accumulated feedback data (with human review for quality filtering).
   </details>

---

## Key Papers

- Papineni et al. (2002) — "BLEU: a Method for Automatic Evaluation of Machine Translation"
- Lin (2004) — "ROUGE: A Package for Automatic Evaluation of Summaries"
- Zhang et al. (2020) — "BERTScore: Evaluating Text Generation with BERT"
- Lin et al. (2022) — "TruthfulQA: Measuring How Models Mimic Human Falsehoods"
- Hendrycks et al. (2021) — "Measuring Massive Multitask Language Understanding" (MMLU)
- Chen et al. (2021) — "Evaluating Large Language Models Trained on Code" (HumanEval)
- Zheng et al. (2023) — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
- Dubois et al. (2024) — "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators"
- Cobbe et al. (2021) — "Training Verifiers to Solve Math Word Problems" (GSM8K)
