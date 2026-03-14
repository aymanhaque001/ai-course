# 🧠 Complete AI & Machine Learning Course

> From mathematical foundations to production AI systems — a comprehensive, self-paced curriculum for mastering artificial intelligence, with interview preparation at every level.

---

## Course Overview

This course covers the **full AI/ML stack** — from linear algebra and classical machine learning through deep learning, transformers, LLMs, generative AI, and production systems. Every module includes:

- **Conceptual explanations** with ASCII diagrams and comparison tables
- **Code implementations** from scratch (Python/NumPy)
- **Interview questions** — conceptual, coding, and system design
- **Key papers** with context on why they matter

```
┌──────────────────────────────────────────────────────────────────┐
│                     COURSE STRUCTURE                              │
│                                                                   │
│  FOUNDATIONS        CORE LLM STACK       BEYOND LLMs    PROD     │
│  ┌────────────┐    ┌──────────────┐    ┌───────────┐  ┌──────┐  │
│  │ Math (11)  │    │ Transformers │    │Gen AI (14)│  │MLOps │  │
│  │ ClassicML  │    │   (2)       │    │           │  │ (15) │  │
│  │   (12)     │───▶│ Tokens (3)  │───▶│           │──│      │  │
│  │ Deep       │    │ Training (4)│    │           │  │Sys   │  │
│  │ Learn (13) │    │ Prompts (5) │    │           │  │Design│  │
│  │ Foundations│    │ RAG (6)     │    │           │  │ (16) │  │
│  │   (1)      │    │ Eval (7)    │    │           │  │      │  │
│  │            │    │ Scale (8)   │    │           │  │      │  │
│  │            │    │ Safety (9)  │    │           │  │      │  │
│  │            │    │ Advanced(10)│    │           │  │      │  │
│  └────────────┘    └──────────────┘    └───────────┘  └──────┘  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Recommended Learning Paths

### Path 1: Complete Beginner → AI Engineer (All 16 Modules)

```
Module 11 (Math) → Module 12 (Classical ML) → Module 13 (Deep Learning)
    → Module 1 (Foundations) → Module 2 (Transformers)
    → Module 3 (Tokenization) → Module 4 (Training)
    → Module 5 (Prompts) → Module 6 (RAG) → Module 7 (Eval)
    → Module 8 (Scaling) → Module 9 (Safety) → Module 10 (Advanced)
    → Module 14 (Generative AI) → Module 15 (MLOps) → Module 16 (Systems Design)
```

### Path 2: Software Engineer → LLM Engineer (Core LLM Track)

```
Module 1 → Module 2 → Module 3 → Module 5 → Module 6
    → Module 7 → Module 8 → Module 10 → Module 15 → Module 16
```

### Path 3: ML Engineer → LLM Specialist

```
Module 2 → Module 3 → Module 4 → Module 5 → Module 6
    → Module 8 → Module 9 → Module 10 → Module 14
```

### Path 4: Interview Prep (Systems Design Focus)

```
Module 12 (Classical ML) → Module 15 (MLOps) → Module 16 (Systems Design)
    → Interview sections from Modules 1-10
```

---

## All Modules

### Foundations & Prerequisites

| #   | Module                                                                           | Topics                                                                              | Est. Time |
| --- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------- |
| 11  | [Mathematical Foundations](module-11-math-foundations/README.md)                 | Linear algebra, calculus, probability, information theory, optimization             | 8-10 hrs  |
| 12  | [Classical Machine Learning](module-12-classical-ml/README.md)                   | Regression, classification, SVMs, trees, ensembles, clustering, feature engineering | 10-12 hrs |
| 13  | [Deep Learning (CNNs, RNNs, Pre-Transformer)](module-13-deep-learning/README.md) | CNNs, RNNs, LSTMs, seq2seq, attention, BERT, autoencoders, GANs                     | 10-12 hrs |

### Core LLM Stack

| #   | Module                                                                   | Topics                                                                     | Est. Time |
| --- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------- | --------- |
| 1   | [Neural Network Foundations](module-01-foundations/README.md)            | Neurons, activations, backprop, optimization, regularization, weight init  | 8-10 hrs  |
| 2   | [Transformer Architecture](module-02-transformers/README.md)             | Self-attention, multi-head, positional encoding, KV-cache, Flash Attention | 8-10 hrs  |
| 3   | [Tokenization & Embeddings](module-03-tokenization-embeddings/README.md) | BPE, tokenizers, Word2Vec, contextual embeddings, vector spaces            | 6-8 hrs   |
| 4   | [Training LLMs](module-04-training/README.md)                            | Pre-training, SFT, LoRA/QLoRA, RLHF, DPO, alignment                        | 8-10 hrs  |
| 5   | [Prompt Engineering](module-05-prompt-engineering/README.md)             | Few-shot, CoT, ReAct, prompt injection, DSPy, sampling                     | 6-8 hrs   |
| 6   | [Retrieval-Augmented Generation](module-06-rag/README.md)                | RAG architecture, chunking, vector DBs, hybrid search, Graph RAG           | 8-10 hrs  |
| 7   | [Evaluation & Benchmarks](module-07-evaluation/README.md)                | Metrics, benchmarks, human eval, LLM-as-judge, hallucination               | 6-8 hrs   |
| 8   | [Scaling & Inference](module-08-scaling/README.md)                       | Scaling laws, quantization, Flash Attention, speculative decoding, MoE     | 8-10 hrs  |
| 9   | [Safety & Alignment](module-09-safety-alignment/README.md)               | HHH framework, harm taxonomy, bias, red teaming, governance                | 6-8 hrs   |
| 10  | [Advanced Topics](module-10-advanced/README.md)                          | Agents, multi-agent systems, long-context, multimodal, mechanistic interp  | 8-10 hrs  |

### Beyond LLMs

| #   | Module                                                         | Topics                                                                  | Est. Time |
| --- | -------------------------------------------------------------- | ----------------------------------------------------------------------- | --------- |
| 14  | [Generative AI Beyond LLMs](module-14-generative-ai/README.md) | Diffusion models, Stable Diffusion, DALL-E, video gen, audio/speech, 3D | 8-10 hrs  |

### Production & Systems

| #   | Module                                                     | Topics                                                                         | Est. Time |
| --- | ---------------------------------------------------------- | ------------------------------------------------------------------------------ | --------- |
| 15  | [MLOps & Production ML](module-15-mlops/README.md)         | Pipelines, experiment tracking, data management, serving, CI/CD, monitoring    | 10-12 hrs |
| 16  | [AI Systems Design](module-16-ai-systems-design/README.md) | Design framework, recommendation, search, fraud detection, chatbot, moderation | 12-15 hrs |

**Total estimated study time: ~140-170 hours**

---

## Prerequisites Map

```
┌───────────────────────────────────────────────────────────────────┐
│                    PREREQUISITE GRAPH                              │
│                                                                    │
│  Module 11 (Math)                                                  │
│    └──▶ Module 12 (Classical ML)                                   │
│           └──▶ Module 13 (Deep Learning)                           │
│                  └──▶ Module 1 (Foundations)                        │
│                         └──▶ Module 2 (Transformers)               │
│                                ├──▶ Module 3 (Tokenization)        │
│                                │      └──▶ Module 4 (Training)     │
│                                │             ├──▶ Module 5 (Prompt)│
│                                │             │     └──▶ Module 6   │
│                                │             │           (RAG)     │
│                                │             └──▶ Module 9 (Safety)│
│                                └──▶ Module 8 (Scaling)             │
│                                                                    │
│  Module 7 (Evaluation) ◀── Modules 1-6                            │
│  Module 10 (Advanced) ◀── Modules 1-9                              │
│  Module 14 (Gen AI) ◀── Module 13                                  │
│  Module 15 (MLOps) ◀── Modules 4, 12                               │
│  Module 16 (Systems Design) ◀── All previous (recommended)         │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Interview Preparation Index

Every module includes interview questions. Here's a quick reference by category:

| Category                          | Modules with Interview Questions                                                             |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| **Conceptual**                    | All modules (5+ questions each)                                                              |
| **Coding (from scratch)**         | 1, 2, 3, 4, 6, 11, 12, 13, 14, 15                                                            |
| **System Design**                 | 1, 3, 6, 12, 15, 16 (full case studies)                                                      |
| **ML System Design Case Studies** | Module 16: Recommendation, Search, Fraud Detection, Chatbot, Content Moderation, Translation |

---

## How to Use This Course

1. **Pick a learning path** above based on your background
2. **Read each module sequentially** — concepts build on each other
3. **Implement the code examples** — don't just read them
4. **Answer the interview questions** before revealing solutions
5. **Read the key papers** (at least the abstracts and intro sections)
6. **Build something** after every 3-4 modules to solidify understanding

---

_Total modules: 16 | Total interview questions: 100+ | Key papers: 80+_
