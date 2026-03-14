# Module 10: Advanced Topics — Agents, Multimodal LLMs & Frontier Research

> **Prerequisites:** Modules 1–9 (this is the capstone module; all prior modules are assumed)
> **Estimated Study Time:** 10–14 hours

---

## 10.1 LLM Agents

An LLM agent combines a language model with the ability to take actions — calling tools, searching the web, executing code, and interacting with external systems — to accomplish goals over multiple steps.

```
┌──────────────────────────────────────────────────────────────────────┐
│                      AGENT ARCHITECTURE                               │
│                                                                      │
│                    ┌─────────────────────┐                           │
│                    │       LLM Brain      │                           │
│                    │  (planning, reasoning│                           │
│                    │   tool selection)    │                           │
│                    └──────────┬──────────┘                           │
│                               │ action                               │
│                    ┌──────────▼──────────┐                           │
│                    │    Tool Executor     │                           │
│                    │  ┌──────────────┐   │                           │
│                    │  │ Web Search   │   │                           │
│                    │  │ Code Exec    │   │                           │
│                    │  │ DB Query     │   │                           │
│                    │  │ API Calls    │   │                           │
│                    │  │ File R/W     │   │                           │
│                    │  └──────────────┘   │                           │
│                    └──────────┬──────────┘                           │
│                               │ observation                          │
│                    ┌──────────▼──────────┐                           │
│                    │   Memory & Context   │                           │
│                    │  (working memory,    │                           │
│                    │   vector DB, logs)   │                           │
│                    └─────────────────────┘                           │
└──────────────────────────────────────────────────────────────────────┘
```

### The Agent Loop

```python
def agent_loop(goal: str, max_steps: int = 20) -> str:
    history = []
    for step in range(max_steps):
        # Think
        response = llm.generate(
            system="You are an agent. Use tools to accomplish the goal.",
            messages=history + [{"role": "user", "content": goal}]
        )
        # Parse action
        if is_final_answer(response):
            return extract_answer(response)

        tool_call = parse_tool_call(response)
        # Act
        observation = execute_tool(tool_call)
        # Observe
        history.extend([
            {"role": "assistant", "content": response},
            {"role": "tool", "content": observation},
        ])
    return "Max steps reached without resolution."
```

### Function Calling / Tool Use

Modern LLMs (GPT-4, Claude 3.5, Llama 3) support structured tool definitions:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
# Model returns structured tool_calls instead of text when it wants to use a tool
```

---

## 10.2 Agent Planning Strategies

### ReAct (Reasoning + Acting)

```
Thought: I need to find the population of Tokyo and compare it to NYC.
Action: web_search("Tokyo population 2026")
Observation: Tokyo population is approximately 13.96 million.
Thought: Now I need NYC's population.
Action: web_search("New York City population 2026")
Observation: NYC population is approximately 8.34 million.
Thought: I can now compare the two.
Final Answer: Tokyo (13.96M) is about 1.67× larger than NYC (8.34M).
```

### Plan-and-Execute

```
PLAN phase (one LLM call):
  Break goal into ordered steps:
    1. Search for Tokyo population
    2. Search for NYC population
    3. Compute the ratio
    4. Format the final answer

EXECUTE phase (one call per step):
  Execute each step in order with appropriate tools.

Advantages: More efficient for complex tasks; easier to monitor progress.
Disadvantages: Plan may need revision; rigid sequencing can fail on unexpected observations.
```

### Tree of Thoughts (ToT)

```
                        ┌── Branch A ──── terminal
                        │    (score: 7)
Goal ──► Root ──────────┼── Branch B ──── Branch B1 ── terminal*
          (expand)      │    (score: 9)   (score: 8)
                        └── Branch C ──── pruned
                             (score: 3)

BFS: expand most promising nodes first
DFS: explore one path deeply
A*: heuristic-guided search

Use case: complex problems with clear intermediate quality metrics
(math proofs, code with unit tests, multi-step reasoning)
```

---

## 10.3 Multi-Agent Systems

Instead of one agent doing everything, complex tasks can be distributed across specialized agents:

```
┌──────────────────────────────────────────────────────────────────┐
│               MULTI-AGENT ARCHITECTURE                            │
│                                                                   │
│   Human Request                                                  │
│        │                                                         │
│        ▼                                                         │
│  ┌────────────┐   task      ┌─────────────────────────────────┐ │
│  │ Orchestrator│ ──────────► │ Specialized Agents              │ │
│  │   Agent    │             │  ┌──────────┐ ┌──────────┐      │ │
│  │           │ ◄────────── │  │ Research │ │  Coder   │      │ │
│  │ (plans,   │  results    │  │  Agent   │ │  Agent   │      │ │
│  │  delegates│             │  └──────────┘ └──────────┘      │ │
│  │  reviews) │             │  ┌──────────┐ ┌──────────┐      │ │
│  └────────────┘             │  │ Writer   │ │ Reviewer │      │ │
│                              │  │  Agent   │ │  Agent   │      │ │
│                              │  └──────────┘ └──────────┘      │ │
│                              └─────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Agent Reliability: The Compounding Error Problem

```
Per-step success rate: 95%

Multi-step accuracy:
  10 steps:   0.95^10  = 59.9% success
  20 steps:   0.95^20  = 35.8% success
  50 steps:   0.95^50  = 7.7%  success
  100 steps:  0.95^100 = 0.6%  success

This is why reliability at each step is critical for long-horizon agents.
Strategies:
  1. Reduce max_steps (decompose into smaller tasks)
  2. Add verification steps ("check your work")
  3. Use stronger models (99% per-step → 90.4% at 10 steps)
  4. Add checkpointing and rollback
  5. Human-in-the-loop for high-stakes actions
```

### Frameworks for Building Agents

| Framework                     | Approach                        | Best For                     |
| ----------------------------- | ------------------------------- | ---------------------------- |
| **LangChain / LangGraph**     | Graph-based agent orchestration | Complex workflows with state |
| **AutoGen (Microsoft)**       | Multi-agent conversation        | Research, code gen teams     |
| **CrewAI**                    | Role-based agent crews          | Structured multi-agent tasks |
| **OpenAI Swarm**              | Lightweight multi-agent         | Simple agent handoffs        |
| **Smolagents (Hugging Face)** | Code-as-action agents           | Data science workflows       |
| **Claude's Computer Use**     | Native GUI interaction          | Browser automation           |

---

## 10.4 Long-Context LLMs

Context windows have grown dramatically:

```
Year     Model              Context Window
────────────────────────────────────────────
2020     GPT-3              4,096 tokens
2022     GPT-3.5 turbo      16,384 tokens
2023     GPT-4              32,768 / 128K tokens
2024     Gemini 1.5 Pro     1,000,000 tokens
         Claude 3           200,000 tokens
2025     Gemini 2.0 Pro     1,024,000 tokens
         Gemini 2.5 Pro     2,048,000 tokens
2026     Claude 4           500,000+ tokens
```

### Challenges at Long Context

```
1. Attention cost: O(n²) for standard attention — 1M tokens = 10¹²  operations
   Solutions: Flash Attention, sliding window attention (Mistral),
   sparse attention patterns (BigBird, Longformer), linear attention

2. Position encoding generalization:
   Models trained on 4K context may degrade at 100K even with extended training.
   Solutions: RoPE scaling (NTK, dynamic NTK), context extension fine-tuning (LongLoRA)

3. Lost-in-the-middle (Module 7):
   Retrieval accuracy degrades for content in the middle.

4. Distraction: with more context, models may be confused by irrelevant information.
   Solution: RAG (select only relevant context) vs long-context (use everything)
```

### RoPE (Rotary Position Encoding)

The dominant position encoding in LLaMA, Mistral, GPT-NeoX:

```
Standard positional encoding: add positional vector to token embedding
  x_t = embed(token_t) + pos_embed(t)

RoPE: rotate queries and keys by angle proportional to position
  q_rotated = q ⊗ r(t)    (element-wise complex multiplication)
  k_rotated = k ⊗ r(t')

  Attention: q_rotated · k_rotated = f(q, k, t - t')
  → attention depends only on RELATIVE position (t - t'), not absolute!

  r(t) = [cos(t·θ₁), sin(t·θ₁), cos(t·θ₂), sin(t·θ₂), ...]
         θᵢ = 10000^(-2(i-1)/d)  (same as sinusoidal PE)

Context length extension:
  Scale θᵢ by factor > 1 to handle longer sequences.
  YaRN, NTK-by-parts scaling: extend 4K → 128K with minimal quality loss
```

---

## 10.5 Multimodal LLMs

Modern LLMs process text, images, audio, and video in a unified architecture.

```
┌──────────────────────────────────────────────────────────────────┐
│              MULTIMODAL LLM ARCHITECTURE                          │
│                                                                   │
│   Input Modalities:                                              │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│   │  Text   │  │  Image  │  │  Audio  │  │  Video  │          │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘          │
│        │            │            │              │               │
│        ▼            ▼            ▼              ▼               │
│   Text tokens  Image encoder  Audio encoder  Frame encoder     │
│   (tokenizer)  (ViT/SigLIP)   (Whisper)     (video ViT)       │
│        │            │            │              │               │
│        └────────────┴────────────┴──────────────┘               │
│                              │                                   │
│                     Combined token sequence                      │
│                              │                                   │
│                     ┌────────▼────────┐                         │
│                     │  Language Model  │                         │
│                     │  (transformer)   │                         │
│                     └────────┬────────┘                         │
│                              │                                   │
│                         Text / Image output                      │
└──────────────────────────────────────────────────────────────────┘
```

### Vision Encoders

```
ViT (Vision Transformer):
  1. Split image into 16×16 pixel patches
  2. Embed each patch as a token (e.g., 196 patches for 224×224 image)
  3. Process patch tokens with transformer encoder
  4. Output: 196 visual embeddings ∈ ℝ^d_vision

Bridge to LLM:
  Visual embeddings (d_vision) → linear projection → LLM embedding space (d_model)
  196 visual tokens are prepended to the text tokens

  Example:
  [IMG_START, v₁, v₂, ..., v₁₉₆, IMG_END, "What is in this image?", ...]
```

### LLaVA Architecture

LLaVA (Large Language and Vision Assistant) is a simple but effective open-source multimodal architecture:

```python
class LLaVA:
    def __init__(self, vision_encoder, projection_layer, language_model):
        self.vision_encoder = vision_encoder    # CLIP ViT-L/14
        self.projection = projection_layer       # Linear: d_vision → d_llm
        self.llm = language_model               # Vicuna/LLaMA

    def forward(self, image, text_tokens):
        # 1. Encode image
        visual_embeds = self.vision_encoder(image)    # (N_patches, d_vision)
        # 2. Project to LLM space
        visual_tokens = self.projection(visual_embeds) # (N_patches, d_llm)
        # 3. Embed text tokens
        text_embeds = self.llm.embed_tokens(text_tokens)
        # 4. Concatenate and run LLM
        full_sequence = torch.cat([visual_tokens, text_embeds], dim=1)
        return self.llm.transformer(full_sequence)
```

### Training Multimodal Models

```
Stage 1: Pretraining (align vision and language)
  - Freeze LLM, train only the projection layer
  - Data: (image, caption) pairs (CC3M, LAION)
  - Goal: make visual tokens "speak the same language" as text tokens

Stage 2: Instruction fine-tuning
  - Unfreeze LLM (or use LoRA)
  - Data: high-quality visual instruction data (LLaVA-Instruct, ShareGPT4V)
  - Goal: follow instructions about images, answer visual questions

Stage 3 (optional): Alignment
  - RLHF on multimodal preference data
  - Reduce hallucination about image content
```

---

## 10.6 Test-Time Compute Scaling (o1/o3/R1 Paradigm)

A new scaling dimension: rather than just scaling training compute, scale **inference-time compute** to improve reasoning quality.

```
┌──────────────────────────────────────────────────────────────────┐
│           TEST-TIME COMPUTE SCALING                               │
│                                                                   │
│  Traditional: bigger model → better answers                      │
│    Train for months → fixed capability in model weights          │
│                                                                   │
│  New paradigm: longer thinking → better answers                  │
│    Standard query:                                               │
│      User: "What is 24 × 36?"                                   │
│      Model: "864" (immediate, 1 forward pass)                   │
│                                                                   │
│    Chain-of-thought (moderate thinking):                         │
│      Model: "24 × 36 = 24 × 30 + 24 × 6 = 720 + 144 = 864"    │
│                                                                   │
│    Extended reasoning (many thinking tokens):                    │
│      <think>                                                     │
│        Let me verify: 20×36=720, 4×36=144, 720+144=864.         │
│        Cross-check: 24×40=960, minus 24×4=96, 960-96=864. ✓    │
│      </think>                                                    │
│      Answer: 864                                                 │
│                                                                   │
│  The longer the model can "think", the better it performs        │
│  on hard problems — especially math, coding, logic puzzles.      │
└──────────────────────────────────────────────────────────────────┘
```

### How It's Trained (Process Reward Models)

```
Standard RLHF: reward based on final answer quality (ORM)
  Problem: wasteful search, doesn't distinguish good from bad reasoning

Process Reward Model (PRM):
  Score each reasoning STEP, not just the final answer.

  Step 1: "Let x = unknown variable"         [reward: 0.9]
  Step 2: "x × 5 = 25, so x = 5"            [reward: 0.95]
  Step 3: "Therefore the answer is 5 units"   [reward: 0.92]

  Train on step-level human labels (Math-Shepherd dataset)
  Use PRM for search: MCTS, beam search, best-of-N over reasoning traces

Best-of-N with PRM:
  Generate N reasoning traces → score each with PRM → return highest-scored answer
  N=256 small model can match N=1 large model accuracy on competition math
```

### MCTS for Reasoning

```
                            Root (problem)
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
      [Step A: try eq.]  [Step B: try factor.] [Step C: try graph]
      PRM score: 0.7     PRM score: 0.9        PRM score: 0.4
             │                  │
     [Step A1]  [Step A2]   [Step B1]  [Step B2]
                              0.95        0.60
                               │
                         [Step B1a: terminal]
                         PRM: 0.98 → correct answer!

Backpropagate scores, prioritize high-scoring branches.
AlphaZero-style MCTS applied to reasoning.
Used in: OpenAI o1/o3, DeepSeek R1
```

---

## 10.7 LLM Pre-training on Synthetic Data

As the internet's high-quality text approaches exhaustion, synthetic data becomes increasingly important:

```
┌──────────────────────────────────────────────────────────────────┐
│              SYNTHETIC DATA FOR LLM PRE-TRAINING                  │
│                                                                   │
│  Phi-1 (Microsoft, 2023):                                        │
│    Train on "textbook quality" synthetic data generated by GPT-4 │
│    1.3B params × 7B tokens → matches CodeLLaMA 34B on HumanEval │
│    Key insight: quality > quantity for small models              │
│                                                                   │
│  Phi-2/3/4:                                                      │
│    Progressively scale synthetic data quality + diversity        │
│    Phi-4 14B surpasses models 10× its size on reasoning tasks    │
│                                                                   │
│  Synthetic data generation pipeline:                             │
│    1. Define capability target (e.g., multi-step math reasoning) │
│    2. Prompt strong teacher model to generate diverse exercises  │
│    3. Generate worked solutions with explicit reasoning          │
│    4. Filter for quality (execute code, verify math, etc.)       │
│    5. Add diversity constraints (topic, difficulty, format)      │
│                                                                   │
│  Limitations:                                                    │
│    - Model collapse: training on AI output → less diversity      │
│    - Ceiling at teacher model's capability                       │
│    - Domain gaps: hard to generate authentic domain-specific data│
└──────────────────────────────────────────────────────────────────┘
```

---

## 10.8 Mechanistic Interpretability

Understanding what computations happen inside transformers:

### Superposition Hypothesis (Anthropic)

```
Problem: Models represent far more features than they have neurons.
  A model with d_model=4096 dimensions can represent millions of features.

Theory: Features are stored in SUPERPOSITION — many features share neurons,
  with different combinations active for different inputs.

Evidence:
  Single neurons activate for multiple unrelated concepts.
  Direction in activation space (not individual neurons) correspond to features.

┌─────────────────────────────────────────────────────────────────────┐
│  Neuron 1234 in GPT-2's layer 8 activates maximally for:           │
│    - The word "banana"                                              │
│    - The word "yellow" in color contexts                            │
│    - Mentions of the sun when discussing warmth                     │
│    (polysemantic neuron — represents multiple features)             │
└─────────────────────────────────────────────────────────────────────┘
```

### Circuits

Using activation patching and ablation to identify which circuits implement which algorithms:

```
Induction Heads (found in ~2-7 layers in GPT-2-small):
  Implement in-context pattern matching:
    "A B ... A → B" (predict B after seeing the pattern A→B earlier)

  Mechanism:
    Layer N attention head (k-composition):
      1. Pattern head: attends to previous token
      2. Induction head: uses Q=current token, K=next-token embeddings
         to find positions where same token appeared earlier
    Result: model can repeat arbitrary patterns after a few examples

Great Interpretability Tools:
  TransformerLens (Neel Nanda): hook into any residual stream
  Neuronpedia: visualize neuron activations
  Sparse Autoencoders (SAEs): decompose superposed features into
    monosemantic components
```

### Sparse Autoencoders (SAEs)

The current state-of-the-art tool for interpretability:

```
Goal: find a dictionary of "true" features underlying the superposed MLP activations

x ∈ ℝ^d   (activation vector)
        │
  ┌─────▼──────┐
  │   Encoder  │    h = ReLU(W_enc x + b_enc)   h ∈ ℝ^D, D >> d
  └─────┬──────┘    (sparse: most values = 0)
        │
  ┌─────▼──────┐
  │   Decoder  │    x̂ = W_dec h + b_dec
  └─────┬──────┘
        │
  Loss = ‖x - x̂‖² + λ‖h‖₁    (reconstruction + L1 sparsity)

The learned features in h are much more monosemantic than neurons in x.
Anthropic's "Scaling Monosemanticity" paper found millions of interpretable
features in Claude Sonnet by analyzing SAE features.
```

---

## 10.9 Frontier Models & Research Directions (2025–2026)

### Key Architectural Innovations

```
State Space Models (Mamba, Mamba-2):
  O(n) instead of O(n²) complexity for sequence processing
  Selective state spaces: S4, S6 architectures
  Status: competitive with transformers on some tasks but hasn't
         displaced transformers at scale

Diffusion Language Models (MDLM, PLAID):
  Generate text by progressively denoising token sequences
  Non-autoregressive generation (all tokens at once)
  Status: promising but behind autoregressive transformers in quality

Hybrid Architectures (Jamba, Zamba):
  Interleave Mamba SSM layers with transformer attention layers
  Combine O(n) per-token compute with powerful attention for hard tasks

RWKV (Linear Attention):
  Reformulate attention as linear time RNN that can run as transformer in training
  O(n) inference without KV cache
```

### The "World Models" Frontier

```
Current LLMs:
  Process patterns in text → generate plausible next tokens
  Limited physical world understanding (no 3D reasoning, no physics)

Direction:
  Train on video (space, time, causality) → build world model
  GPT-4o, Gemini 2.0: beginning to incorporate visual world understanding
  Sora, Veo: video generation models encode world physics implicitly

  Applications: robotics (embodied AI), simulation, planning
```

### Reasoning and Formal Verification

```
LLMs + Formal Methods:
  AlphaProof (Google DeepMind): solve IMO proofs with formal verification
  Lean Copilot: LLM assists Lean theorem prover

  Key insight: formal verifiers provide ground-truth reward signal.
  No reward hacking possible if the proof checker verifies correctly.

  This is one of the few areas where AI can be verifiably correct.
```

---

## 10.10 Building Production LLM Applications

### System Design Patterns

```
┌──────────────────────────────────────────────────────────────────┐
│                  PRODUCTION LLM APPLICATION                       │
│                                                                   │
│  Reliability Layer:                                              │
│    Retry with exponential backoff (transient API failures)       │
│    Fallback models (GPT-4o primary → Claude backup → local)     │
│    timeout handling (stream with max_tokens hard limit)          │
│                                                                   │
│  Cost Control:                                                   │
│    Route by complexity: GPT-4o-mini for simple tasks            │
│    Cache aggressively (semantic similarity cache)                │
│    Compression: summarize conversation history                   │
│    Use streaming to start delivering value before full response  │
│                                                                   │
│  Observability:                                                  │
│    Trace every LLM call: input, output, latency, cost, model    │
│    Log prompt versions → enable prompt rollback                  │
│    Alert on cost spikes, latency regressions, error rate changes │
│                                                                   │
│  Prompt Management:                                              │
│    Version prompts in code (git), not in databases              │
│    Evaluation suite for every prompt change                      │
│    A/B test prompt variants                                      │
└──────────────────────────────────────────────────────────────────┘
```

### Structured Output — The Right Way

```python
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Literal

client = OpenAI()

class ExtractedEntity(BaseModel):
    name: str
    entity_type: Literal["person", "organization", "location", "product"]
    confidence: float = Field(ge=0.0, le=1.0, description="0–1 confidence score")
    context: str = Field(description="Brief quote from text supporting this entity")

class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"]
    summary: str = Field(max_length=200)

def extract_entities(text: str) -> ExtractionResult:
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract entities and analyze sentiment."},
            {"role": "user", "content": text},
        ],
        response_format=ExtractionResult,  # OpenAI structured outputs
        temperature=0,
    )
    return response.choices[0].message.parsed

# Pydantic validation is automatic — malformed responses raise ParseError
result = extract_entities("Apple CEO Tim Cook announced record quarterly earnings.")
print(result.entities[0].name)  # "Tim Cook"
print(result.overall_sentiment) # "positive"
```

### Streaming with Backpressure

```python
import asyncio
from openai import AsyncOpenAI

async def stream_with_timeout(
    prompt: str, max_total_tokens: int = 1000, timeout_per_chunk: float = 5.0
) -> None:
    client = AsyncOpenAI()
    try:
        async with asyncio.timeout(timeout_per_chunk):
            stream = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=max_total_tokens,
            )

        tokens_received = 0
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                tokens_received += 1
                yield delta

            if tokens_received >= max_total_tokens:
                break

    except asyncio.TimeoutError:
        yield "\n[Response timed out]"
```

---

## 10.11 The LLM Ecosystem Map

```
┌──────────────────────────────────────────────────────────────────────┐
│                       LLM ECOSYSTEM (2026)                            │
│                                                                      │
│  Frontier Models (Proprietary):                                      │
│    OpenAI: GPT-4o, o1, o3, o4                                       │
│    Anthropic: Claude 3.5/4 Sonnet, Haiku, Opus                      │
│    Google: Gemini 2.0/2.5 Flash & Pro                               │
│    xAI: Grok 3                                                       │
│                                                                      │
│  Open-Weight Models:                                                 │
│    Meta: LLaMA 3.x (8B, 70B, 405B)                                  │
│    Mistral AI: Mistral 7B, Mixtral 8x7B/8x22B, Mistral Large        │
│    Alibaba: Qwen 2.5 (0.5B-72B)                                     │
│    DeepSeek: DeepSeek V3, R1 (competitive with GPT-4o)              │
│    Microsoft: Phi-4 (small, distilled)                               │
│                                                                      │
│  Infrastructure:                                                     │
│    Training: NVIDIA H100/H200, AMD MI300X, Google TPUv5             │
│    Serving: vLLM, TGI, TensorRT-LLM, SGLang                        │
│    Frameworks: PyTorch + DeepSpeed/FSDP, JAX + XLA                  │
│                                                                      │
│  Application Layer:                                                  │
│    Orchestration: LangChain, LlamaIndex, LangGraph                  │
│    APIs: OpenAI, Anthropic, Cohere, Mistral, Together AI            │
│    Deployment: Replicate, Modal, Fireworks, Groq                    │
│                                                                      │
│  Evaluation:                                                         │
│    LMSYS Chatbot Arena, Open LLM Leaderboard, Scale HELM             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Interview Questions

### Conceptual

1. **What is the key innovation in OpenAI's o1/o3 models? How does test-time compute scaling work?**
   <details>
   <summary>Answer</summary>
   The core insight is that spending more compute at inference time (generating longer internal reasoning traces) reliably improves performance on hard reasoning tasks. Traditional scaling adds parameters to the model (training-time compute); o1-style scaling adds reasoning tokens (test-time compute). The model is trained with reinforcement learning to produce long internal chain-of-thought reasoning before answering, using process reward models (PRMs) to score each reasoning step. The training objective is to learn when to spend more tokens on verification, when to backtrack (tree search in thought space), and how to structure multi-step reasoning. This is especially effective for: mathematical proofs, competitive programming, multi-hop logical deduction. The practical implications: (1) more compute at inference → better answers, without changing weights; (2) a small but well-trained reasoning model can match or beat a much larger standard model; (3) compute cost scales with problem difficulty, not just response length.
   </details>

2. **What is the superposition hypothesis in LLM interpretability? Why does it matter for safety?**
   <details>
   <summary>Answer</summary>
   The superposition hypothesis (Elhage et al., 2022) proposes that neural network layers store more learned features than they have neurons, by representing features as directions in the activation space rather than as individual neurons. Multiple features "superpose" — the network can recover them because they're nearly orthogonal and activate sparsely. Evidence: individual neurons respond to multiple unrelated concepts (polysemanticity); features can be identified as directions in activation space that don't align with any single neuron. Why it matters for safety: (1) Safety-relevant concepts (harmful intent, deception, self-interest) may be encoded as features in superposition — if we can identify these features via tools like Sparse Autoencoders, we could detect or intervene on them; (2) It explains why probing for simple concepts (like "is this harmful?") is difficult — no single neuron or circuit represents it cleanly; (3) Mechanistic interpretability via SAEs may eventually allow us to verify whether a model has "dangerous" features before deployment, addressing deceptive alignment concerns; (4) It motivates research into activation steering and feature suppression as alignment techniques.
   </details>

3. **How do multi-agent systems fail? What engineering patterns increase reliability?**
   <details>
   <summary>Answer</summary>
   Failure modes: (1) Error compounding — with N steps at 95% reliability, 10-step tasks succeed only 60% of the time; (2) Context drift — later steps lose context from earlier steps (context window limits, long conversation distortions); (3) Tool failures — external APIs fail, code executes with side effects, search returns irrelevant results; (4) Planner-executor mismatch — the plan doesn't account for failure modes, and the executor follows a broken plan off a cliff; (5) Resource exhaustion — agents spawn sub-agents recursively, consuming compute indefinitely; (6) Trust boundaries — multi-agent systems with different trust levels can be exploited via prompt injection between agents. Reliability patterns: (1) Checkpointing: save state after each major step, enable rollback; (2) Verification steps: make the model explicitly verify intermediate results before proceeding; (3) Shorter task decomposition: design subtasks that complete in <5 steps each; (4) Structured outputs at every step: use Pydantic/JSON Schema validation to catch tool call errors immediately; (5) Human-in-the-loop gates: require human approval for irreversible actions (send email, write to DB); (6) Budget limits: max_steps, max_tool_calls, max_cost per task.
   </details>

4. **What are the tradeoffs between a long-context approach and a RAG approach for a document Q&A system?**
   <details>
   <summary>Answer</summary>
   Long-context: (1) Simpler architecture — no indexing pipeline, no retrieval infrastructure; (2) Can reason across the entire document for questions requiring synthesis; (3) No retrieval failures — if the answer is in the document, it's always in context; (4) Benefits from co-location of distant context (no chunking boundary artifacts). Costs: quadratic attention cost, high token expense, latency proportional to document length, lost-in-the-middle degradation, practical limit even at 1M tokens. RAG: (1) Scales to arbitrarily large corpora (billions of documents); (2) Fast retrieval brings only relevant chunks (lower latency, lower cost); (3) Naturally provides citations; (4) Updatable without re-running expensive LLM calls. Costs: retrieval pipeline complexity, chunking artifacts (answers split across boundaries), retrieval failures, harder to synthesize across many documents. Decision framework: few long docs with complex cross-section questions → long context. Many docs OR need to scale to large corpora → RAG. Best system: hybrid — RAG for initial selection, then stuff relevant sections into long context for synthesis.
   </details>

5. **What distinguishes a language model from an agent? What additional capabilities and risks emerge?**
   <details>
   <summary>Answer</summary>
   A language model takes text in and produces text out — it's a passive responder with no persistent state or external impact. An agent adds: (1) Tool use — it can query APIs, execute code, search the web, modify files; (2) Multi-step execution — it acts over multiple turns toward a goal; (3) Memory — it maintains state across actions; (4) Autonomy — it makes decisions about what to do next without human approval at each step. Emergent capabilities: complex task completion, research, software development, data analysis in a way that simple LLM calls can't achieve. Emergent risks: (1) Real-world side effects — agents can send emails, make purchases, delete files; mistakes are harder to undo than rephrasings; (2) Compounding errors — mistakes cascade across steps; (3) Resource acquisition — an agent optimizing for a goal may acquire compute, money, or permissions beyond what's needed; (4) Unintended goal generalization — agents trained to "help the user" may help in ways that seem plausible but are harmful; (5) Prompt injection via tool outputs — attacker-controlled data the agent processes can hijack its action sequence. Mitigations: principle of least privilege, human-in-the-loop for irreversible actions, sandboxed execution environments, output validation at every step.
   </details>

### Coding

6. **Build a minimal ReAct agent with tool use from scratch.**
   <details>
   <summary>Solution</summary>

   ```python
   import json
   import re
   from openai import OpenAI

   client = OpenAI()

   # Tool definitions
   def web_search(query: str) -> str:
       """Mock web search (replace with real search API)"""
       return f"Search results for '{query}': [Sample result about {query}]"

   def calculate(expression: str) -> str:
       """Safely evaluate a math expression"""
       try:
           result = eval(expression, {"__builtins__": {}}, {})
           return str(result)
       except Exception as e:
           return f"Error: {e}"

   TOOLS = {"web_search": web_search, "calculate": calculate}

   SYSTEM = """You are a helpful agent. To answer questions, you can use tools.

   Use this format exactly:
   Thought: [your reasoning]
   Action: {"tool": "tool_name", "input": "tool input"}

   After receiving an observation, continue thinking:
   Thought: [reasoning about observation]
   Action: {"tool": "tool_name", "input": ...}

   When ready to give the final answer:
   Thought: [final reasoning]
   Final Answer: [your answer]

   Available tools: web_search(query), calculate(expression)"""

   def react_agent(question: str, max_steps: int = 10) -> str:
       messages = [
           {"role": "system", "content": SYSTEM},
           {"role": "user", "content": question},
       ]

       for step in range(max_steps):
           response = client.chat.completions.create(
               model="gpt-4o",
               messages=messages,
               temperature=0,
               stop=["Observation:"],
           )
           assistant_text = response.choices[0].message.content
           messages.append({"role": "assistant", "content": assistant_text})

           # Check for final answer
           if "Final Answer:" in assistant_text:
               return assistant_text.split("Final Answer:")[-1].strip()

           # Parse action
           action_match = re.search(r'Action: (\{.*?\})', assistant_text, re.DOTALL)
           if not action_match:
               return f"Agent error: could not parse action from: {assistant_text}"

           action = json.loads(action_match.group(1))
           tool_name = action["tool"]
           tool_input = action["input"]

           # Execute tool
           if tool_name in TOOLS:
               observation = TOOLS[tool_name](tool_input)
           else:
               observation = f"Error: unknown tool '{tool_name}'"

           messages.append({"role": "user", "content": f"Observation: {observation}"})

       return "Max steps reached without a final answer."

   # Usage
   answer = react_agent("What is 15% of 240, and what is Python's latest version?")
   print(answer)
   ```

   </details>

7. **Implement a streaming multimodal request with image input using the OpenAI API.**
   <details>
   <summary>Solution</summary>

   ```python
   import base64
   import asyncio
   from pathlib import Path
   from openai import AsyncOpenAI

   async def analyze_image_streaming(
       image_path: str,
       prompt: str = "Describe this image in detail.",
       model: str = "gpt-4o",
   ) -> None:
       """Stream a multimodal analysis of an image."""
       client = AsyncOpenAI()

       # Load and encode image
       image_data = Path(image_path).read_bytes()
       b64_image = base64.b64encode(image_data).decode("utf-8")
       suffix = Path(image_path).suffix.lower().replace(".", "")
       media_type = f"image/{suffix if suffix != 'jpg' else 'jpeg'}"

       messages = [
           {
               "role": "user",
               "content": [
                   {
                       "type": "image_url",
                       "image_url": {
                           "url": f"data:{media_type};base64,{b64_image}",
                           "detail": "high",  # "low" for faster, cheaper analysis
                       },
                   },
                   {
                       "type": "text",
                       "text": prompt,
                   },
               ],
           }
       ]

       print(f"Analyzing image: {image_path}")
       print("-" * 40)

       async with client.chat.completions.stream(
           model=model,
           messages=messages,
           max_tokens=500,
       ) as stream:
           async for text in stream.text_stream:
               print(text, end="", flush=True)

       print()  # newline at end

   asyncio.run(analyze_image_streaming("photo.jpg", "What objects are in this scene?"))
   ```

   </details>

### System Design

8. **Design an AI coding assistant (like Cursor or GitHub Copilot) from scratch. Cover architecture, context retrieval, multi-file context, and safety.**
   <details>
   <summary>Answer</summary>

   **Core architecture:** IDE plugin + cloud backend. Editor sends a context window to the backend, which assembles a prompt and calls an LLM, streaming back completions. The key engineering challenge is assembling the right context.

   **Context retrieval (RAG for code):** Index the entire codebase: (1) Parse AST of each file, emit chunks by function/class/block boundary rather than fixed token size; (2) Generate two types of embeddings: coarse (file-level, what does this file do?) and fine (function-level, what does this function do?); (3) At query time: exact path match (files explicitly mentioned/imported) + semantic retrieval (similar code patterns). Retrieve: current file, imported symbols' definitions, similar code from codebase, relevant tests, README. Use BM25 for symbol names (exact match important for code) + dense for conceptual similarity.

   **Multi-file context assembly:** Fill context window prioritizing: (1) Cursor position ± 2000 tokens (local context, highest priority); (2) Currently open tabs (user is working in these); (3) Retrieved semantically relevant sections; (4) Common utilities/types imported throughout codebase. Use token budget allocation: 30% local context, 20% open tabs, 30% retrieved context, 20% for response.

   **Prompt design:** System prompt establishes: language/framework context, coding style from repo (detect from existing code), relevant comments about codebase conventions. Use fill-in-the-middle (FIM) format for completion (prefix + suffix → middle) — enables completion cursor-in.

   **Safety:** Code can have significant security implications. Flag suggestions containing: SQL string concatenation (injection risk), hardcoded secrets, use of `eval()`, unsafe deserialization. Don't suggest code that matches known CVE patterns. Educate (show why something is unsafe) rather than just refuse.

   **Latency optimization:** Run small model (Starcoder 1B, CodeGemma 2B) locally for sub-50ms completions. Route complex requests (multi-file, explain, refactor) to cloud LLM. Precompute file embeddings on save, not at query time. Prefetch likely next-needed context in background.
   </details>

---

## Key Papers

- Yao et al. (2022) — "ReAct: Synergizing Reasoning and Acting in Language Models"
- Wei et al. (2023) — "Chain-of-Thought Hub: A Continuous Effort to Measure Large Language Models' Reasoning Performance"
- Shinn et al. (2023) — "Reflexion: Language Agents with Verbal Reinforcement Learning"
- Lightman et al. (2023) — "Let's Verify Step by Step" (Process Reward Models)
- Gu et al. (2024) — "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Liu et al. (2024) — "LLaVA: Visual Instruction Tuning"
- Elhage et al. (2022) — "Toy Models of Superposition" (Anthropic)
- Cunningham et al. (2023) — "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
- OpenAI (2024) — "OpenAI o1 System Card"
- DeepSeek-AI (2025) — "DeepSeek-R1: Incentivizing Reasoning Capability via RL"
- Microsoft Research (2023) — "Phi-1: Textbooks Are All You Need"
- Su et al. (2022) — "RoFormer: Enhanced Transformer with Rotary Position Embedding"

---

## Course Complete — What's Next?

```
┌──────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED NEXT STEPS                             │
│                                                                      │
│  Hands-on Practice:                                                  │
│    □ Fine-tune a model on a custom dataset (QLoRA + LLaMA 3 8B)     │
│    □ Build a complete RAG system on your own documents               │
│    □ Implement a multi-step agent that uses web search + code exec   │
│    □ Reproduce a paper: implement an eval benchmark from scratch     │
│                                                                      │
│  Papers to Read Next:                                                │
│    □ Attention Is All You Need (original transformer)               │
│    □ GPT-3 (Brown et al. 2020)                                       │
│    □ InstructGPT (Ouyang et al. 2022)                                │
│    □ LLaMA 3 (Meta AI 2024)                                          │
│    □ Constitutional AI (Bai et al. 2022)                             │
│    □ FlashAttention-2 (Dao 2023)                                     │
│                                                                      │
│  Communities & Resources:                                            │
│    □ Andrej Karpathy — "Neural Networks: Zero to Hero" (YouTube)    │
│    □ Hugging Face course: huggingface.co/learn                       │
│    □ Anthropic Alignment Forum posts                                 │
│    □ Lilian Weng's blog (lilianweng.github.io)                       │
│    □ Sebastian Raschka's LLM book / newsletter                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

[← Module 9: Safety & Alignment](../module-09-safety-alignment/README.md) | [Module 11: Math Foundations →](../module-11-math-foundations/README.md)
