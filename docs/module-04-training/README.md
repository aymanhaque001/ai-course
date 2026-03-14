# Module 4: Training LLMs — Pre-training, Fine-tuning, RLHF & Alignment

> **Prerequisites:** Modules 1-3
> **Estimated Study Time:** 10–12 hours

---

## 4.1 The Three Stages of LLM Training

```
┌──────────────────────────────────────────────────────────────────────┐
│                     LLM TRAINING PIPELINE                            │
│                                                                      │
│  Stage 1              Stage 2                Stage 3                 │
│  PRE-TRAINING         SUPERVISED              ALIGNMENT              │
│                       FINE-TUNING (SFT)       (RLHF / DPO)          │
│                                                                      │
│  ┌──────────┐        ┌──────────┐           ┌──────────┐            │
│  │ Raw text │        │ (Prompt, │           │ (Prompt,  │            │
│  │ corpus   │        │  Answer) │           │  Good,    │            │
│  │          │        │  pairs   │           │  Bad)     │            │
│  └────┬─────┘        └────┬─────┘           └────┬─────┘            │
│       │                   │                      │                   │
│       ▼                   ▼                      ▼                   │
│  Next-token          Follow                  Preference              │
│  prediction          instructions            alignment               │
│                                                                      │
│  Cost: $$$$$         Cost: $$                Cost: $$                │
│  Data: ~10T tokens   Data: ~100K examples    Data: ~50K comparisons  │
│  Time: weeks-months  Time: hours-days        Time: hours-days        │
│                                                                      │
│  Output: Base model  Output: Chat model      Output: Aligned model   │
│  (LLaMA)             (LLaMA-Chat)            (Claude, GPT-4)        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4.2 Stage 1: Pre-training

### Objective: Next-Token Prediction (Causal Language Modeling)

```
Input:    "The  cat  sat  on   the"
Target:   "cat  sat  on   the  mat"

For each position, predict the NEXT token given all PREVIOUS tokens:

P(cat | The) × P(sat | The cat) × P(on | The cat sat) × ...

Loss = -Σₜ log P(xₜ | x₁, ..., xₜ₋₁)   (cross-entropy)
```

### Data Scale

```
Model          Training Tokens     Dataset Size
──────────────────────────────────────────────
GPT-2          ~40B               ~40 GB
GPT-3          300B               ~570 GB
LLaMA 1        1.0–1.4T           ~4.5 TB
LLaMA 2        2.0T               ~8 TB
LLaMA 3        15T+               ~50+ TB
```

### Data Sources and Quality

```
┌────────────────────────────────────────────────────┐
│              PRE-TRAINING DATA MIX                  │
│                                                     │
│  Common Crawl (web)        ████████████  ~60-70%    │
│  Books/Literature          ███            ~10%      │
│  Wikipedia                 ██             ~5%       │
│  Code (GitHub)             ███            ~10%      │
│  Scientific papers         █              ~3%       │
│  Social media/forums       █              ~2-5%     │
│                                                     │
│  Key: █ = ~3% of total data                        │
└────────────────────────────────────────────────────┘
```

**Data quality pipeline:**
```
Raw web crawl → Language detection → Deduplication → Quality filtering → Toxicity filtering → Final dataset
                     │                    │                │                   │
                Drop non-target     Remove exact &    Perplexity filter,   Remove PII,
                languages           near-duplicates    classifier-based     harmful content
                                    (MinHash)          quality scoring
```

### Distributed Training

Training a 70B+ model on a single GPU is impossible. Modern pre-training uses multiple forms of parallelism:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PARALLELISM STRATEGIES                        │
│                                                                  │
│  Data Parallelism (DP):                                         │
│    Same model on each GPU, different data, average gradients     │
│    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                             │
│    │GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│  ← full model copy each     │
│    │Bat 0│ │Bat 1│ │Bat 2│ │Bat 3│  ← different data batches   │
│    └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                              │
│       └───────┴───────┴───────┘                                  │
│              AllReduce gradients                                 │
│                                                                  │
│  Tensor Parallelism (TP):                                       │
│    Split individual layers across GPUs                           │
│    ┌──────────────────────┐                                      │
│    │     Attention Layer   │                                     │
│    │  ┌────┬────┬────┬────┤                                      │
│    │  │H0-1│H2-3│H4-5│H6-7│  ← heads split across GPUs        │
│    │  │GPU0│GPU1│GPU2│GPU3│                                      │
│    │  └────┴────┴────┴────┘                                      │
│    └──────────────────────┘                                      │
│                                                                  │
│  Pipeline Parallelism (PP):                                     │
│    Split layers across GPUs, process micro-batches in pipeline   │
│    GPU 0: Layers 0-7                                            │
│    GPU 1: Layers 8-15                                           │
│    GPU 2: Layers 16-23                                          │
│    GPU 3: Layers 24-31                                          │
│                                                                  │
│  FSDP (Fully Sharded Data Parallelism):                         │
│    Shard model parameters, gradients, and optimizer states       │
│    Each GPU holds 1/N of everything, gathers as needed           │
│    = ZeRO Stage 3 (DeepSpeed) = FSDP (PyTorch)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Mixed Precision Training

```
FP32 (32-bit):  1 sign + 8 exponent + 23 mantissa  → baseline
BF16 (16-bit):  1 sign + 8 exponent + 7 mantissa   → same range as FP32, less precision
FP16 (16-bit):  1 sign + 5 exponent + 10 mantissa  → more precision, less range

Modern LLM training uses BF16 because:
- Same dynamic range as FP32 (won't overflow)
- 2× memory savings
- 2× compute speedup on modern GPUs (A100, H100)
- Slightly less precise, but good enough for training

Master weights kept in FP32 for accumulation stability.
```

---

## 4.3 Stage 2: Supervised Fine-Tuning (SFT)

SFT transforms a base model (text completion) into a chat model (instruction following).

### Data Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."},
    {"role": "assistant", "content": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously..."}
  ]
}
```

### Training Details

```
Loss is computed ONLY on the assistant tokens (not system/user tokens):

Tokens:  <sys> You are helpful </sys> <user> Explain QC </user> <asst> Quantum computing uses ...
Mask:     ✗    ✗   ✗    ✗       ✗     ✗     ✗       ✗   ✗      ✓     ✓        ✓        ✓   ...

This ensures the model learns to GENERATE good responses, not to predict user messages.
```

**Typical hyperparameters for SFT:**
- Learning rate: 1e-5 to 2e-5 (10-100× lower than pre-training)
- Epochs: 2-5 (small dataset, don't overtrain)
- Batch size: 32-128
- Warmup: 3-10% of steps

---

## 4.4 Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates all parameters — expensive for large models. PEFT methods update only a small fraction.

### LoRA (Low-Rank Adaptation)

The key insight: weight updates during fine-tuning have low intrinsic rank.

```
Full fine-tuning:                    LoRA:
  W' = W + ΔW                         W' = W + BA
  ΔW ∈ ℝ^(d×d)                        B ∈ ℝ^(d×r), A ∈ ℝ^(r×d)
  d²  parameters                       2dr parameters (r << d)

  For d=4096, r=16:
  Full: 16,777,216 params              LoRA: 131,072 params (0.8%!)

  ┌─────────────────────┐
  │                     │
  │   Original W        │◄── Frozen (not updated)
  │   (d × d)           │
  │                     │
  └────────┬────────────┘
           │
           + ◄─── B × A  (trainable, low-rank)
           │      │    │
           │    (d×r)(r×d)
           ▼
      Output = Wx + BAx
```

**LoRA hyperparameters:**
- **r (rank):** 4-64. Higher = more capacity, more parameters
- **α (scaling):** Scales the LoRA output. Effective scaling = α/r
- **Target modules:** Usually Q, K, V, O projections. Can also include FFN

### QLoRA (Quantized LoRA)

```
Base model weights: quantized to 4-bit (NF4)   ← saves ~4× memory
LoRA adapters: kept in BF16                      ← full precision for training
Computation: dequantize on-the-fly to BF16       ← no quality loss during forward pass

Result: Fine-tune a 70B model on a single 48GB GPU!
```

### Comparison of PEFT Methods

```
Method          Trainable Params    Memory     Quality
────────────────────────────────────────────────────────
Full Fine-tune  100%               Very High   Best
LoRA            0.1-1%             Medium      Near full FT
QLoRA           0.1-1%             Low         Near LoRA
Prefix Tuning   <0.1%              Low         Good
Prompt Tuning   <0.01%             Very Low    Decent
Adapters        1-5%               Medium      Good
```

---

## 4.5 Stage 3a: RLHF (Reinforcement Learning from Human Feedback)

### The RLHF Pipeline

```
Step 1: Collect comparison data
  Prompt: "Write a poem about spring"

  Response A: "Spring brings flowers,     Response B: "Roses are red,
               birds sing their songs,                  violets are blue,
               warmth fills the air..."                 spring is nice too"

  Human annotator: A > B  ✓

Step 2: Train a Reward Model (RM)
  ┌──────────────────────────────────────────┐
  │  Reward Model (initialized from SFT)      │
  │                                           │
  │  Input: (prompt, response) pair           │
  │  Output: scalar reward score              │
  │                                           │
  │  Loss: -log(σ(r(preferred) - r(rejected)))│
  │  (Bradley-Terry model)                    │
  └──────────────────────────────────────────┘

Step 3: Optimize policy with PPO
  ┌───────────────────────────────────────────────────────────┐
  │                                                           │
  │  For each prompt:                                         │
  │    1. Generate response with current policy π_θ           │
  │    2. Score with reward model: R(prompt, response)        │
  │    3. Compute KL penalty: β·KL(π_θ ‖ π_ref)             │
  │    4. Total reward = R - β·KL                             │
  │    5. Update π_θ with PPO to maximize total reward        │
  │                                                           │
  │  The KL penalty prevents the model from diverging too     │
  │  far from the SFT model (avoiding reward hacking)         │
  └───────────────────────────────────────────────────────────┘
```

### Why KL Penalty?

```
Without KL penalty:                      With KL penalty:
  Model finds "shortcuts" to high        Model stays close to the
  reward that don't reflect quality      SFT policy while improving

  Example reward hack:                   Balances:
  - Repeat the prompt back               ┌──────────────────────┐
  - Use overly verbose language           │ Reward ←───────→ KL  │
  - Include "I'm confident"              │ (quality)   (safety) │
    in every response                     └──────────────────────┘
```

---

## 4.5b Stage 3b: DPO (Direct Preference Optimization)

DPO eliminates the need for a separate reward model and PPO — it directly optimizes the policy from preference data.

```
RLHF Pipeline:                          DPO Pipeline:
  SFT → Train RM → PPO → Aligned        SFT → DPO → Aligned
  (3 models, complex)                    (1 model, simpler)
```

### DPO Loss Function

```
L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

Where:
  y_w = preferred (winning) response
  y_l = dispreferred (losing) response
  π_θ = current policy (model being trained)
  π_ref = reference policy (frozen SFT model)
  β = temperature parameter
```

**Intuition:** DPO increases the probability of preferred responses relative to the reference model, while decreasing the probability of dispreferred responses. The implicit reward is the log-ratio of policy to reference probabilities.

```
┌────────────────────────────────────────────┐
│           RLHF vs DPO Tradeoffs            │
├────────────────────────────────────────────┤
│                                            │
│  RLHF (PPO):                              │
│    ✓ More flexible (reward model reusable) │
│    ✓ Can optimize for arbitrary rewards    │
│    ✗ Complex (3 models in memory)          │
│    ✗ Unstable training (PPO is tricky)     │
│    ✗ Reward model can be exploited         │
│                                            │
│  DPO:                                      │
│    ✓ Simple (standard cross-entropy-like)  │
│    ✓ Stable training                       │
│    ✓ Only 2 models (policy + frozen ref)   │
│    ✗ Less flexible                         │
│    ✗ Requires good-quality preference data │
│    ✗ Can overfit to preference format      │
│                                            │
│  Current trend: DPO variants (IPO, KTO,   │
│  ORPO) are increasingly popular            │
└────────────────────────────────────────────┘
```

---

## 4.6 Constitutional AI (CAI)

Used by Anthropic for Claude's alignment:

```
┌─────────────────────────────────────────────────┐
│          CONSTITUTIONAL AI PIPELINE              │
│                                                  │
│  Step 1: RED TEAMING (generate harmful prompts)  │
│    "How do I pick a lock?"                       │
│                                                  │
│  Step 2: INITIAL RESPONSE (from base model)      │
│    "Here's how to pick a lock: ..."              │
│                                                  │
│  Step 3: CRITIQUE (model critiques itself)       │
│    "This response could help someone break       │
│     into homes, which is harmful."               │
│                                                  │
│  Step 4: REVISION (model revises response)       │
│    "I can't provide lockpicking instructions     │
│     as they could be used for illegal entry.     │
│     If you're locked out, contact a locksmith."  │
│                                                  │
│  Step 5: Train on (prompt, revision) pairs       │
│    using RLHF with AI-generated preferences      │
│                                                  │
│  Key insight: Uses a "constitution" (set of      │
│  principles) to guide the critique & revision    │
└─────────────────────────────────────────────────┘
```

---

## 4.7 Training Infrastructure — Practical Details

### Compute Requirements

```
Model Size    GPUs (A100 80GB)    Training Time      Cost (estimate)
─────────────────────────────────────────────────────────────────────
7B            8-16 GPUs           ~1-2 weeks         $50K-$100K
13B           32-64 GPUs          ~2-3 weeks         $200K-$400K
70B           256-512 GPUs        ~1-2 months        $2M-$5M
175B+         1000+ GPUs          ~2-3 months        $10M-$50M
```

### Training Instabilities and Solutions

```
Problem                    Symptom                  Solution
─────────────────────────────────────────────────────────────────
Loss spikes               Sudden jump in loss       Reduce LR, skip batch,
                                                    gradient clipping
Divergence                Loss goes to NaN/inf      Lower LR, check data quality,
                                                    verify numerics (BF16 vs FP16)
Slow convergence          Loss plateaus early       Increase LR, check data mixing,
                                                    verify gradient flow
Gradient explosion        Very large gradients      Gradient clipping (max_norm=1.0)
```

---

## 4.8 Continued Pre-training & Domain Adaptation

```
General Base Model → Continued Pre-training on Domain Data → Domain-Adapted Model

Examples:
  LLaMA → + Medical papers   → Med-LLaMA
  LLaMA → + Legal documents  → Legal-LLaMA
  LLaMA → + Code corpus      → Code LLaMA

Typical recipe:
  - 10B-100B domain-specific tokens
  - Lower LR than initial pre-training (0.5-1× of final pre-training LR)
  - Include some general data to prevent catastrophic forgetting (~10-20%)
```

---

## Interview Questions

### Conceptual

1. **Walk through the three stages of LLM training. Why is each stage necessary?**
   <details>
   <summary>Answer</summary>
   Stage 1 (Pre-training): Train on massive text corpus with next-token prediction. This gives the model broad language understanding and world knowledge. Necessary because this is where the model learns grammar, facts, reasoning patterns, and code from trillions of tokens. Stage 2 (SFT): Fine-tune on high-quality (instruction, response) pairs. Necessary because pre-trained models just complete text — they don't follow instructions or have a conversational format. SFT teaches the model the chat format and how to be helpful. Stage 3 (RLHF/DPO): Align the model with human preferences. Necessary because SFT alone can produce plausible but harmful, verbose, or unhelpful responses. RLHF/DPO teaches the model to prefer responses that humans actually prefer — being concise, honest, harmless, and genuinely helpful. Each stage builds on the previous one: knowledge → format → quality.
   </details>

2. **Explain LoRA. Why does it work despite training <1% of parameters?**
   <details>
   <summary>Answer</summary>
   LoRA decomposes the weight update ΔW into a low-rank product BA where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×d) with r << d. It works because: (1) Research (Aghajanyan et al., 2020) showed that pre-trained models have a low "intrinsic dimensionality" — you only need to update along a small number of directions to adapt behavior. The weight changes during fine-tuning are inherently low-rank. (2) The pre-trained weights already encode vast knowledge; fine-tuning only needs to "steer" this knowledge, not rewrite it. (3) By targeting attention projections (Q, K, V, O), LoRA modifies the attention patterns which have the highest impact on model behavior per parameter. The rank r controls the expressiveness-efficiency tradeoff: r=8-32 is sufficient for most tasks.
   </details>

3. **Compare RLHF and DPO. When would you use each?**
   <details>
   <summary>Answer</summary>
   RLHF trains a separate reward model on preference data, then optimizes the policy using PPO to maximize reward while staying close to the reference model (KL penalty). DPO directly optimizes the policy from preference data without a reward model, using an analytical mapping between reward functions and optimal policies. Use RLHF when: (1) you want to reuse the reward model for other tasks (e.g., best-of-N sampling, filtering); (2) you need to optimize complex, multi-component rewards; (3) you have the engineering capacity to stabilize PPO training. Use DPO when: (1) you want simpler training (standard supervised learning pipeline); (2) you have limited compute (only 2 models vs 3-4); (3) you have high-quality preference data. In practice, DPO and its variants (IPO, KTO) have become more popular due to simplicity and competitive results.
   </details>

4. **What is catastrophic forgetting? How do you mitigate it during fine-tuning?**
   <details>
   <summary>Answer</summary>
   Catastrophic forgetting occurs when fine-tuning on a new task or domain causes the model to lose capabilities it had after pre-training. For example, a model fine-tuned on medical QA might lose its ability to write code. Mitigations: (1) Low learning rate (1e-5 to 2e-5) — small updates preserve pre-trained knowledge; (2) LoRA/PEFT — only modifying a small number of parameters limits what can be "forgotten"; (3) Data mixing — include general-purpose data alongside domain-specific data during fine-tuning; (4) Regularization — weight decay, KL penalty against the base model; (5) Short training — few epochs (2-5) to avoid over-specialization; (6) Replay — periodically mix in examples from the pre-training distribution.
   </details>

5. **What is reward hacking in RLHF? Give concrete examples and solutions.**
   <details>
   <summary>Answer</summary>
   Reward hacking occurs when the model exploits patterns in the reward model to achieve high scores without genuinely improving response quality. Examples: (1) The model learns that longer responses get higher rewards, so it becomes excessively verbose; (2) The model learns to use confident-sounding phrases ("I'm absolutely certain...") even when uncertain; (3) The model generates responses that "look good" to the reward model but are factually wrong. Solutions: (1) KL penalty — keeps the model close to the SFT policy; (2) Reward model ensembles — harder to exploit multiple models simultaneously; (3) Iterative RLHF — retrain reward model on new policy outputs; (4) Constitutional AI — use principles-based evaluation instead of a single scalar reward; (5) Better reward model training data that includes edge cases.
   </details>

### Coding

6. **Implement the DPO loss function in PyTorch.**
   <details>
   <summary>Solution</summary>

   ```python
   import torch
   import torch.nn.functional as F

   def dpo_loss(policy_logps_chosen, policy_logps_rejected,
                ref_logps_chosen, ref_logps_rejected,
                beta=0.1):
       """
       Compute DPO loss.

       Args:
           policy_logps_chosen: log P_θ(y_w | x) for preferred responses
           policy_logps_rejected: log P_θ(y_l | x) for rejected responses
           ref_logps_chosen: log P_ref(y_w | x)
           ref_logps_rejected: log P_ref(y_l | x)
           beta: temperature parameter

       Returns:
           loss: scalar DPO loss
       """
       # Log-ratio differences
       policy_ratios = policy_logps_chosen - policy_logps_rejected
       ref_ratios = ref_logps_chosen - ref_logps_rejected

       logits = beta * (policy_ratios - ref_ratios)
       loss = -F.logsigmoid(logits).mean()

       # Useful metrics
       with torch.no_grad():
           chosen_rewards = beta * (policy_logps_chosen - ref_logps_chosen)
           rejected_rewards = beta * (policy_logps_rejected - ref_logps_rejected)
           reward_margin = (chosen_rewards - rejected_rewards).mean()
           accuracy = (logits > 0).float().mean()

       return loss, {
           'reward_margin': reward_margin.item(),
           'accuracy': accuracy.item()
       }
   ```
   </details>

7. **Write code to set up LoRA fine-tuning for a LLaMA model using the PEFT library.**
   <details>
   <summary>Solution</summary>

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
   from peft import LoraConfig, get_peft_model, TaskType
   from trl import SFTTrainer

   # Load base model
   model_name = "meta-llama/Llama-2-7b-hf"
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=torch.bfloat16,
       device_map="auto",
   )
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   tokenizer.pad_token = tokenizer.eos_token

   # Configure LoRA
   lora_config = LoraConfig(
       task_type=TaskType.CAUSAL_LM,
       r=16,                          # Rank
       lora_alpha=32,                 # Scaling: effective lr multiplier = alpha/r = 2
       lora_dropout=0.05,
       target_modules=[               # Which layers to adapt
           "q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj",
       ],
       bias="none",
   )

   # Apply LoRA
   model = get_peft_model(model, lora_config)
   model.print_trainable_parameters()
   # Output: trainable params: 13,107,200 || all params: 6,751,637,504 || trainable%: 0.19%

   # Training
   training_args = TrainingArguments(
       output_dir="./lora-llama",
       num_train_epochs=3,
       per_device_train_batch_size=4,
       gradient_accumulation_steps=8,
       learning_rate=2e-4,            # Higher LR is OK for LoRA
       warmup_ratio=0.03,
       lr_scheduler_type="cosine",
       bf16=True,
       logging_steps=10,
       save_strategy="epoch",
   )

   trainer = SFTTrainer(
       model=model,
       args=training_args,
       train_dataset=dataset,
       tokenizer=tokenizer,
       max_seq_length=2048,
   )
   trainer.train()

   # Save adapter only (small file, ~50MB vs ~14GB for full model)
   model.save_pretrained("./lora-adapter")

   # Later: load and merge
   # base_model = AutoModelForCausalLM.from_pretrained(model_name)
   # model = PeftModel.from_pretrained(base_model, "./lora-adapter")
   # merged = model.merge_and_unload()  # Merge LoRA into base weights
   ```
   </details>

### System Design

8. **Design a fine-tuning pipeline for a company that wants to adapt an open-source LLM to their domain (e.g., legal, medical). Consider data, compute, evaluation, and deployment.**
   <details>
   <summary>Answer</summary>

   **Data Pipeline:** (1) Collect domain-specific documents (contracts, case law, medical records — with proper compliance/anonymization); (2) Generate instruction-response pairs using a stronger model (GPT-4) or expert annotators; (3) Create preference pairs for DPO by having domain experts rank outputs; (4) Quality filter: remove low-quality, contradictory, or sensitive examples; (5) Mix in ~20% general-purpose instruction data to prevent catastrophic forgetting.

   **Training:** (1) Start with a strong base model (LLaMA 3 70B or Mistral); (2) Continued pre-training on domain corpus (10-50B tokens) if domain vocabulary is very different; (3) SFT with LoRA (r=32-64) on instruction pairs; (4) DPO with domain expert preferences; (5) Use QLoRA if compute-constrained (single A100 node).

   **Evaluation:** (1) Domain-specific benchmarks (legal bar exam questions, medical licensing exam); (2) Human evaluation by domain experts (blind comparison vs base model); (3) Automated metrics: accuracy on domain QA, hallucination rate, citation accuracy; (4) Safety evaluation: ensure fine-tuning didn't degrade safety guardrails.

   **Deployment:** (1) Merge LoRA weights into base model; (2) Quantize to INT4/INT8 for serving; (3) Deploy with vLLM or TGI for efficient inference; (4) Monitor: track domain accuracy, user satisfaction, edge cases; (5) Set up feedback loop for continuous improvement.
   </details>

---

## Key Papers
- Radford et al. (2018) — "Improving Language Understanding by Generative Pre-Training" (GPT)
- Ouyang et al. (2022) — "Training language models to follow instructions with human feedback" (InstructGPT/RLHF)
- Hu et al. (2021) — "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers et al. (2023) — "QLoRA: Efficient Finetuning of Quantized Language Models"
- Rafailov et al. (2023) — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (DPO)
- Bai et al. (2022) — "Constitutional AI: Harmlessness from AI Feedback"
- Touvron et al. (2023) — "LLaMA: Open and Efficient Foundation Language Models"
