# Module 3: Tokenization & Embeddings

> **Prerequisites:** Module 1-2
> **Estimated Study Time:** 6–8 hours

---

## 3.1 Why Tokenization Matters

LLMs don't see text — they see sequences of integers. Tokenization is the bridge between human-readable text and model-digestible numbers. It directly impacts:

- **Model capacity:** vocabulary size determines the output projection size
- **Sequence length efficiency:** better tokenization = more information per token = effectively longer context
- **Multilingual performance:** tokenizers trained on English may fragment non-English text into many tokens
- **Cost:** API pricing is per-token

```
"Hello, world!" → Tokenizer → [15496, 11, 995, 0] → Embedding → [v₁, v₂, v₃, v₄]
                                 4 tokens                         4 vectors ∈ ℝ^d_model
```

---

## 3.2 Tokenization Approaches — Evolution

```
Character-level          Word-level              Subword (Modern)
┌─────────────┐         ┌─────────────┐         ┌─────────────────┐
│ H,e,l,l,o   │         │ Hello World │         │ Hello, _world   │
│              │         │             │         │                 │
│ + Small vocab│         │ + Meaningful│         │ + Balanced vocab│
│ + No OOV     │         │   units     │         │ + No OOV        │
│ - Very long  │         │ - Huge vocab│         │ + Efficient     │
│   sequences  │         │ - OOV words │         │ + Handles rare  │
│ - Hard to    │         │ - Can't     │         │   words         │
│   learn      │         │   handle    │         │                 │
│   meaning    │         │   new words │         │                 │
└─────────────┘         └─────────────┘         └─────────────────┘

       ←─── Modern LLMs use subword tokenization ───→
```

---

## 3.3 Byte-Pair Encoding (BPE) — The Dominant Algorithm

BPE iteratively merges the most frequent pair of tokens to build a vocabulary.

### Training Algorithm

```
Corpus: "low low low low low lowest lowest newer newer newer wider wider wider"

Step 0: Start with character-level tokens
  Vocabulary: {l, o, w, e, s, t, n, r, i, d, _}
  Tokens: l o w _ l o w _ l o w _ l o w _ l o w e s t _ l o w e s t _
          n e w e r _ n e w e r _ n e w e r _
          w i d e r _ w i d e r _ w i d e r

Step 1: Most frequent pair: (l, o) → merge into "lo"
  Vocabulary: {..., lo}

Step 2: Most frequent pair: (lo, w) → merge into "low"
  Vocabulary: {..., low}

Step 3: Most frequent pair: (e, r) → merge into "er"
  Vocabulary: {..., er}

Step 4: Most frequent pair: (n, e) → merge into "ne"
  ...

Continue until vocabulary reaches desired size (e.g., 32,000 or 100,000)
```

### Key Properties

- Common words become single tokens: "the" → [1]
- Rare words are split into subwords: "unforgettable" → ["un", "forget", "table"] or similar
- Any string can be encoded (no out-of-vocabulary problem)
- Vocabulary size is a hyperparameter (typically 32K–100K)

### Tokenizer Comparison

| Tokenizer       | Used By            | Vocab Size | Algorithm       |
| --------------- | ------------------ | ---------- | --------------- |
| GPT-2 tokenizer | GPT-2              | 50,257     | BPE             |
| cl100k_base     | GPT-3.5/4          | 100,256    | BPE             |
| SentencePiece   | LLaMA, Mistral, T5 | 32,000     | BPE/Unigram     |
| tiktoken        | OpenAI models      | varies     | BPE (optimized) |

### SentencePiece vs Standard BPE

```
Standard BPE (GPT-2):              SentencePiece (LLaMA):
  Pre-tokenize on whitespace         Treats text as raw byte stream
  then apply BPE to each word        No pre-tokenization step
                                     Uses ▁ (U+2581) for word boundaries

  "Hello world" →                    "Hello world" →
  ["Hello", " world"]                ["▁Hello", "▁world"]
  (space attached to next word)      (▁ marks word start)

  Advantage: intuitive               Advantage: language-agnostic,
  Disadvantage: space handling        handles any script, reversible
  is irregular
```

---

## 3.4 Tokenization in Practice — Examples and Gotchas

```
Text: "ChatGPT is amazing!"

GPT-2 tokenizer (50K vocab):     cl100k (100K vocab):
  ["Chat", "G", "PT",             ["Chat", "GPT",
   " is", " amazing", "!"]         " is", " amazing", "!"]
  = 6 tokens                       = 5 tokens

  Larger vocab → fewer tokens → more efficient
```

### Common Gotchas

**1. Whitespace sensitivity:**

```
"Hello"     → [15496]           (1 token)
" Hello"    → [18435]           (1 token, but different!)
"  Hello"   → [220, 18435]     (2 tokens)
```

**2. Numbers are often poorly tokenized:**

```
"123456789" → ["123", "456", "789"]   (3 tokens)
"100000"    → ["100", "000"]          (2 tokens)

This means arithmetic is hard for LLMs — they don't "see" numbers as numbers.
```

**3. Code tokenization can be surprising:**

```python
# Python code: "    def hello():"
→ ["   ", " def", " hello", "():", ]  # 4 tokens

# 4 spaces of indentation = 1 token (learned from training data)
```

**4. Multilingual inefficiency:**

```
English: "Hello"       → 1 token
Chinese: "你好"         → 2-3 tokens (each character may be separate)
Arabic:  "مرحبا"       → 4-5 tokens (split at character/byte level)

→ Non-English text uses more tokens for equivalent information
→ Higher cost and shorter effective context for non-English users
```

---

## 3.5 Token Embeddings

Each token ID maps to a dense vector via an embedding table (a learnable lookup table):

```
Vocabulary size: V (e.g., 32,000)
Embedding dimension: d (e.g., 4096)

Embedding Matrix E ∈ ℝ^(V × d):

Token ID    →  Embedding vector
───────────────────────────────────
   0 ("▁")  →  [0.12, -0.34, 0.56, ..., 0.78]    ← d dimensions
   1 ("the") →  [0.45, 0.23, -0.67, ..., 0.11]
   2 ("▁a")  →  [-0.33, 0.89, 0.12, ..., -0.45]
   ...
   31999     →  [0.67, -0.12, 0.34, ..., 0.56]

Lookup is just matrix indexing: embedding = E[token_id]
```

**Weight tying:** Many models tie the input embedding matrix and the output projection matrix (the "LM head"). This means the same matrix is used to:

1. Convert token IDs → embeddings (input): multiply by E to map from token space to semantic space
2. Convert final hidden states → logits over vocabulary (output): multiply by Eᵀ to map from semantic space back to token space

**Why this works (beyond just saving parameters):** The input embedding maps tokens _into_ a semantic space, and the output projection maps hidden states _back out_ to tokens. These are conceptually **inverse operations on the same space**. A token's embedding should be close (high dot product) to hidden states that should predict that token. Using the same matrix enforces this symmetry explicitly.

Press & Wolf (2017) showed that weight tying consistently **improves** perplexity across model sizes — it's not just a memory optimization, it's a useful inductive bias. The parameter savings are also significant: for a 32K vocabulary with d_model=4096, the embedding matrix is 131M parameters, accounting for ~2% of a 7B model but ~10% of a 1B model.

---

## 3.6 Word Embeddings — The Foundational Concept

Before LLMs, word embeddings were trained as standalone models. Understanding them is foundational.

### Word2Vec (Mikolov et al., 2013)

Two architectures:

```
CBOW (Continuous Bag of Words):       Skip-gram:
  Context → predict center word        Center word → predict context

  "the cat [?] on the"                 "[sat]" → predict "the","cat","on","the"
  Context words → hidden → target      Target word → hidden → context words

  Faster, good for frequent words      Slower, better for rare words
```

### Training Word2Vec — Negative Sampling

Training Word2Vec on the full vocabulary is impractical because the output softmax requires computing a dot product with _every_ word in the vocabulary (typically >100K words) for each training example.

**Negative sampling** solves this by transforming the problem from multi-class classification to binary classification:

1. For a (center word, context word) pair from the real data, label it as **positive** (1)
2. Randomly sample k "noise" words from the vocabulary (typically k=5-15), and label them as **negative** (0)
3. Train a binary classifier: $\sigma(v_w^T v_c) \approx 1$ for real context words, $\sigma(v_w^T v_n) \approx 0$ for random noise words

Instead of computing a softmax over 100K words, you only compute k+1 sigmoid operations per training example. The noise words are sampled proportional to their frequency raised to the 3/4 power: $P(w) \propto f(w)^{3/4}$ — this slightly upweights rarer words to ensure they get used as negatives.

This is what makes Word2Vec tractable on large corpora: the original Word2Vec paper trained on 1.6 billion words in hours, not weeks.

### The Embedding Space

Word embeddings capture semantic relationships as geometric relationships:

```
           ┌─────────────────────────────────────────┐
           │                                         │
           │    king •───────────────► queen •        │
           │         │                    │          │
           │         │  same vector       │          │
           │         │  (gender)          │          │
           │         ▼                    ▼          │
           │    man  •───────────────► woman •        │
           │                                         │
           │    king - man + woman ≈ queen            │
           │                                         │
           │    paris •──────────────► france •       │
           │         │                    │          │
           │         │  same vector       │          │
           │         │  (capital-of)      │          │
           │         ▼                    ▼          │
           │    tokyo •──────────────► japan •        │
           │                                         │
           └─────────────────────────────────────────┘
```

**Why vector arithmetic captures analogies:**

The Skip-gram objective implicitly factorizes a **Pointwise Mutual Information (PMI) matrix** — each dimension of the embedding captures a particular co-occurrence pattern between words. The relationship "king" → "queen" and "man" → "woman" involves the same systematic difference in co-occurrence contexts (e.g., appearing with "she/her" vs "he/him", "wife" vs "husband"). Because these context differences are consistent, they encode as a consistent direction in the embedding space.

More concretely: if "king" and "queen" appear in the same contexts except for gendered words, and "man" and "woman" have the same pattern, then:

- king - man ≈ the "gendered royalty" direction
- woman + "gendered royalty" ≈ queen

This breaks down for irregular relationships or when the training data doesn't have consistent contextual patterns. It also works best for frequent words with rich co-occurrence statistics.

### Limitations of Static Embeddings

```
Static (Word2Vec/GloVe):              Contextual (BERT/GPT):
  "bank" → ONE vector                 "river bank" → vector₁
  regardless of context                "bank account" → vector₂
                                       (different vectors for same word!)

  Can't disambiguate polysemy          Each occurrence gets a unique
                                       representation based on context
```

---

## 3.7 Contextual Embeddings — How LLMs Create Them

In a transformer, each layer refines the representation of each token based on its context through the self-attention mechanism:

```
Input:  "The bank by the river was steep"

Layer 0 (token embedding):
  "bank" → generic embedding (no context)

Layer 1-4 (early layers):
  "bank" starts incorporating local context ("by", "the", "river")

Layer 5-16 (middle layers):
  "bank" embedding now strongly encodes "river bank" meaning
  Distinguished from financial "bank"

Layer 17-32 (later layers):
  "bank" embedding encodes full sentence semantics
  Ready for next-token prediction

This is the "residual stream" view: each layer ADDS information to the
embedding via residual connections, progressively enriching it.
```

**The mechanism in detail:** Self-attention creates contextual embeddings by computing a _weighted average_ of all token representations, with weights determined by relevance (the attention scores). Consider the word "bank":

1. "bank" generates a **query** vector: "what context am I looking for?"
2. Every other token ("The", "by", "river", "steep") generates **key** vectors: "here's what I offer as context"
3. The dot product query·key determines how relevant each token is to "bank"
4. "river" gets a high attention weight (very relevant), while "The" gets a lower weight
5. The output is a weighted sum of value vectors, dominated by the most relevant tokens

After layer 1, "bank" now contains information from its immediate neighbors. After layer 2, it contains information from _their_ neighbors (two hops away). By the final layer, each token's representation has been influenced by the entire sequence, with more influence from semantically relevant tokens. The same word "bank" in "bank account" would have "account" dominating the attention weights, pulling the representation toward the financial meaning instead.

---

## 3.8 Sentence & Document Embeddings

For retrieval and similarity tasks, we need fixed-size representations of variable-length text.

### Methods

```
1. [CLS] token pooling (BERT):
   Input: [CLS] The cat sat [SEP]
   Use the final hidden state of [CLS] as sentence embedding

2. Mean pooling (most common for modern embedding models):
   Average all token hidden states: emb = (1/T) Σᵢ hᵢ

3. Last token pooling (decoder models):
   Use the hidden state of the last token

4. Dedicated embedding models:
   Models trained specifically for embeddings with contrastive learning
   (e.g., text-embedding-3-large, BGE, E5, GTE)
```

### Similarity Metrics

```
Cosine Similarity:                    Dot Product:
  sim(a,b) = a·b / (‖a‖·‖b‖)         sim(a,b) = a·b

  Range: [-1, 1]                       Range: (-∞, ∞)
  Ignores magnitude                    Affected by magnitude
  Most common for embeddings           Used with normalized embeddings

Euclidean Distance:
  d(a,b) = ‖a - b‖₂

  Range: [0, ∞)
  Smaller = more similar
  Related to cosine sim for normalized vectors
```

### Embedding Model Comparison (for RAG/retrieval)

| Model                           | Dimensions | Max Tokens | Open Source |
| ------------------------------- | ---------- | ---------- | ----------- |
| text-embedding-3-large (OpenAI) | 3072       | 8191       | No          |
| BGE-large-en-v1.5               | 1024       | 512        | Yes         |
| E5-large-v2                     | 1024       | 512        | Yes         |
| GTE-large                       | 1024       | 512        | Yes         |
| Cohere embed-v3                 | 1024       | 512        | No          |

---

## 3.9 The Embedding Space — Geometric Intuition

```
High-dimensional space (d=4096) projected to 2D for visualization:

                    "programming"
                         •
                        / \
                       /   \
              "code" •     • "software"
                     |     |
                     |     |
            "python" •     • "javascript"


                                          "cooking"
                                              •
                                             / \
                                            /   \
                                   "recipe"•     •"kitchen"
                                           |     |
                                           |     |
                                    "bake" •     •"chef"

    ←── Semantically related words cluster together ──→
    ←── Unrelated concepts are distant ──→
```

### Dimensionality Reduction for Visualization

- **PCA:** Linear projection, preserves global structure
- **t-SNE:** Non-linear, preserves local neighborhoods, good for clusters
- **UMAP:** Non-linear, preserves both local and global structure, faster than t-SNE

---

## 3.10 Special Tokens

```
Token          Purpose                          Example Model
─────────────────────────────────────────────────────────────
<BOS> / <s>    Beginning of sequence             LLaMA, Mistral
<EOS> / </s>   End of sequence                   Most models
<PAD>          Padding for batching              BERT, T5
[CLS]          Classification token              BERT
[SEP]          Segment separator                 BERT
[MASK]         Masked token (for MLM)            BERT
<|endoftext|>  End of document                   GPT-2/3
<|im_start|>   Start of message (chat format)    ChatGPT
<|im_end|>     End of message (chat format)      ChatGPT
```

### Chat Templates (How Messages Become Tokens)

```
User message: "What is 2+2?"

ChatML format (GPT):                       LLaMA format:
<|im_start|>system                         <s>[INST] <<SYS>>
You are a helpful assistant.               You are a helpful assistant.
<|im_end|>                                 <</SYS>>
<|im_start|>user
What is 2+2?                               What is 2+2? [/INST]
<|im_end|>
<|im_start|>assistant
```

The exact template must match what the model was trained with — using the wrong template degrades performance significantly.

---

## Interview Questions

### Conceptual

1. **Explain BPE. How does it handle out-of-vocabulary words?**
   <details>
   <summary>Answer</summary>
   BPE starts with a character-level (or byte-level) vocabulary and iteratively merges the most frequent pair of adjacent tokens into a new token, building up the vocabulary to a target size. It handles OOV words by decomposing them into known subword units. For example, "unhappiness" might become ["un", "happiness"] or ["un", "happ", "iness"]. Since the base vocabulary includes all individual bytes (or characters), any string can be encoded — there is truly no OOV problem. The tradeoff is that rare words use more tokens, which consumes more of the context window.
   </details>

2. **Why does GPT-4 use a 100K vocabulary while LLaMA uses 32K? What are the tradeoffs?**
   <details>
   <summary>Answer</summary>
   Larger vocabulary (100K): (1) More words/subwords become single tokens → shorter sequences → more efficient context usage; (2) The embedding matrix and output projection are larger (100K × d_model parameters), increasing model size; (3) Better for multilingual use (more tokens for non-English scripts). Smaller vocabulary (32K): (1) Smaller embedding/output matrices → fewer parameters; (2) Sequences are longer for the same text → uses more compute and context; (3) Simpler, faster softmax over vocabulary. The optimal size depends on the training data distribution, languages supported, and compute budget. Modern trend is toward larger vocabularies (GPT-4: 100K, Gemini: 256K) as the parameter cost is small relative to total model size.
   </details>

3. **What is the difference between static and contextual embeddings? Why did contextual embeddings supersede static ones?**
   <details>
   <summary>Answer</summary>
   Static embeddings (Word2Vec, GloVe) assign one fixed vector per word regardless of context. "Bank" gets the same embedding in "river bank" and "bank account." Contextual embeddings (from BERT, GPT, etc.) produce different vectors for each token based on its surrounding context, computed by passing through transformer layers. Contextual embeddings superseded static ones because: (1) They handle polysemy naturally; (2) They capture complex dependencies and pragmatics; (3) They're part of an end-to-end model that can be fine-tuned for downstream tasks; (4) They dramatically outperform static embeddings on virtually all NLP benchmarks.
   </details>

4. **A user reports that your LLM-powered app uses 3× more tokens for Japanese input than English input of similar semantic content. Diagnose the problem and propose solutions.**
   <details>
   <summary>Answer</summary>
   The tokenizer was likely trained predominantly on English text, so Japanese characters are not well-represented in the vocabulary. Each Japanese character may be split into 2-3 byte-level tokens, while English words are often single tokens. Solutions: (1) Use a tokenizer with better multilingual coverage (larger vocab with explicit Japanese tokens) — e.g., retrain BPE on a balanced multilingual corpus; (2) Use a model designed for multilingual use (e.g., models with 100K+ vocab trained on multilingual data); (3) If using an API, implement prompt compression/summarization for Japanese inputs; (4) For custom models, extend the tokenizer vocabulary with common Japanese tokens and continue pre-training. This is a well-known issue called "tokenizer fertility" — the ratio of tokens to characters varies drastically across languages.
   </details>

### Coding

5. **Implement a simple BPE tokenizer from scratch.**
   <details>
   <summary>Solution</summary>

   ```python
   from collections import Counter

   class SimpleBPE:
       def __init__(self, vocab_size=300):
           self.vocab_size = vocab_size
           self.merges = []  # ordered list of merge rules

       def get_pairs(self, tokens_list):
           """Count frequency of adjacent pairs across all words."""
           pairs = Counter()
           for tokens in tokens_list:
               for i in range(len(tokens) - 1):
                   pairs[(tokens[i], tokens[i + 1])] += 1
           return pairs

       def merge_pair(self, tokens_list, pair):
           """Apply a merge rule to all words."""
           merged = pair[0] + pair[1]
           new_tokens_list = []
           for tokens in tokens_list:
               new_tokens = []
               i = 0
               while i < len(tokens):
                   if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                       new_tokens.append(merged)
                       i += 2
                   else:
                       new_tokens.append(tokens[i])
                       i += 1
               new_tokens_list.append(new_tokens)
           return new_tokens_list

       def train(self, corpus):
           """Train BPE on a corpus (list of words)."""
           # Initialize: split each word into characters
           tokens_list = [list(word) + ['</w>'] for word in corpus]
           vocab = set(c for tokens in tokens_list for c in tokens)

           while len(vocab) < self.vocab_size:
               pairs = self.get_pairs(tokens_list)
               if not pairs:
                   break
               best_pair = max(pairs, key=pairs.get)
               tokens_list = self.merge_pair(tokens_list, best_pair)
               self.merges.append(best_pair)
               vocab.add(best_pair[0] + best_pair[1])

           self.vocab = vocab

       def encode(self, word):
           """Encode a single word using learned merges."""
           tokens = list(word) + ['</w>']
           for pair in self.merges:
               tokens_new = []
               i = 0
               while i < len(tokens):
                   if i < len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                       tokens_new.append(pair[0] + pair[1])
                       i += 2
                   else:
                       tokens_new.append(tokens[i])
                       i += 1
               tokens = tokens_new
           return tokens
   ```

   </details>

6. **Implement cosine similarity search over a set of embeddings (used in vector databases).**
   <details>
   <summary>Solution</summary>

   ```python
   import numpy as np

   class EmbeddingIndex:
       def __init__(self, dimension):
           self.dimension = dimension
           self.embeddings = []  # list of (embedding, metadata) tuples
           self._matrix = None   # cached normalized matrix

       def add(self, embedding, metadata=None):
           self.embeddings.append((embedding / np.linalg.norm(embedding), metadata))
           self._matrix = None  # invalidate cache

       def _build_matrix(self):
           if self._matrix is None:
               self._matrix = np.array([e for e, _ in self.embeddings])

       def search(self, query, top_k=5):
           """Find top-k most similar embeddings to query."""
           self._build_matrix()
           query_norm = query / np.linalg.norm(query)

           # Cosine similarity via dot product (vectors are already normalized)
           similarities = self._matrix @ query_norm

           # Get top-k indices
           top_indices = np.argsort(similarities)[-top_k:][::-1]

           results = []
           for idx in top_indices:
               results.append({
                   'score': float(similarities[idx]),
                   'metadata': self.embeddings[idx][1],
                   'index': int(idx)
               })
           return results
   ```

   </details>

### System Design

7. **Design a multilingual embedding pipeline for a global search engine that needs to provide semantic search across 50+ languages with sub-100ms latency at 10K QPS.**
   <details>
   <summary>Answer</summary>

   **Embedding model selection:** Use a multilingual embedding model (e.g., multilingual-e5-large, BGE-M3, or Cohere embed-multilingual-v3). These models map text from any language into a shared vector space, so an English query can match French documents.

   **Tokenizer considerations:** The tokenizer must handle all 50+ languages. SentencePiece with a large vocabulary (128K+) trained on multilingual data ensures no language is over-fragmented. Verify: average tokens/word should be <2 for all supported languages (common failure mode: Chinese/Japanese/Korean sentence-level tokenization).

   **Indexing pipeline:** Documents ingested → language detection (fastText lid.176) → chunking (language-aware sentence splitter, not just character count) → embedding (batch inference on GPU) → store in vector DB (Milvus/Qdrant with HNSW, M=32, efConstruction=200). Separate metadata field for source language, enabling language-specific filtering.

   **Serving architecture at 10K QPS:** Query → embedding service (GPU pool, batched inference, ~20ms) → ANN search on vector DB cluster (~10ms, horizontal sharding by document partition) → optional cross-encoder reranker on top-20 results (~50ms on GPU). Total: <100ms P99. Use embedding cache (Redis) for popular queries.

   **Key challenges:** (1) Cross-lingual quality: regularly benchmark with MIRACL/MTEB multilingual benchmarks per language pair. (2) Script-specific normalization: Arabic diacritics, Chinese simplified vs traditional, Japanese katakana normalization. (3) Query language ≠ document language: ensure the model handles this well (test cross-lingual retrieval explicitly). (4) Low-resource languages: may need synthetic data augmentation via translation.
   </details>

---

## Key Papers

- Mikolov et al. (2013) — "Efficient Estimation of Word Representations in Vector Space" (Word2Vec)
- Pennington et al. (2014) — "GloVe: Global Vectors for Word Representation"
- Sennrich et al. (2015) — "Neural Machine Translation of Rare Words with Subword Units" (BPE)
- Kudo & Richardson (2018) — "SentencePiece: A simple and language independent subword tokenizer"
- Reimers & Gurevych (2019) — "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
