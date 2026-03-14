# Module 6: Retrieval-Augmented Generation (RAG)

> **Prerequisites:** Modules 1–5 (especially embeddings from Module 3 and prompt engineering from Module 5)
> **Estimated Study Time:** 8–10 hours

---

## 6.1 The Problem RAG Solves

LLMs have three fundamental limitations that RAG addresses:

```
┌──────────────────────────────────────────────────────────────────┐
│                     LLM LIMITATIONS                               │
│                                                                   │
│  1. Knowledge Cutoff                                              │
│     Model trained in 2024 doesn't know about 2026 events.        │
│                                                                   │
│  2. Hallucination                                                 │
│     Model confidently states false facts, especially for         │
│     specific numbers, names, and niche information.               │
│                                                                   │
│  3. Context Window Limits                                         │
│     Can't fit 10,000 support documents in context.               │
│     128K tokens ≈ 100K words ≈ one small book.                  │
└──────────────────────────────────────────────────────────────────┘

RAG Solution: At inference time, RETRIEVE relevant documents
              and inject them into the prompt as context.

User Query → [Retrieve top-k docs] → [Docs + Query → LLM] → Answer
```

**When to use RAG vs fine-tuning:**

```
┌─────────────────────┬────────────────────────┬─────────────────────┐
│                     │    RAG                 │   Fine-tuning       │
├─────────────────────┼────────────────────────┼─────────────────────┤
│ Knowledge updates   │ ✓ (re-index docs)      │ ✗ (retrain)        │
│ Source attribution  │ ✓ (return doc refs)    │ ✗ (opaque)         │
│ Hallucination risk  │ Lower                  │ Higher              │
│ Custom behavior     │ ✗ (prompt only)        │ ✓                  │
│ Cost                │ Retrieval + LLM        │ Training + LLM      │
│ Latency             │ Higher (retrieval)     │ Lower               │
│ Best for            │ Dynamic knowledge,     │ Style, format,     │
│                     │ precise factual QA     │ domain behavior     │
└─────────────────────┴────────────────────────┴─────────────────────┘
```

---

## 6.2 RAG System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       RAG ARCHITECTURE                               │
│                                                                      │
│   INDEXING PIPELINE (offline)                                        │
│   ──────────────────────────────────────────────────────────────     │
│   Documents → [Chunking] → [Embedding Model] → [Vector Store]        │
│   (PDFs, URLs,  (split into  (encode chunks      (ANN index +        │
│    DB records)   passages)    into dense          metadata store)     │
│                              vectors ∈ ℝ^d)                          │
│                                                                      │
│   QUERY PIPELINE (online)                                            │
│   ──────────────────────────────────────────────────────────────     │
│                               ┌──── Query Expansion                  │
│   User Query ──→ [Query Proc] ─┤                                     │
│                               └──── Rewrite / HyDE                  │
│         │                                                            │
│         ▼                                                            │
│   [Embedding Model] ──→ Query Vector                                 │
│         │                                                            │
│         ▼                                                            │
│   [Vector Store] ──→ Top-k Chunks                                    │
│         │                                                            │
│         ▼                                                            │
│   [Reranker] ──→ Reranked / Filtered Chunks                          │
│         │                                                            │
│         ▼                                                            │
│   [Prompt Assembly] ──→ [LLM] ──→ [Response + Citations]            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6.3 Chunking Strategies

Chunking is often the highest-impact decision in a RAG system. Too small: lost context. Too large: noisy retrieval.

### Fixed-Size Chunking

```python
def fixed_size_chunks(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Slide a window across the text with optional overlap.
    Overlap preserves context at chunk boundaries.
    """
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks
```

```
Text:  [─────────────────────────────────────────────────]
Chunk1:  [════════════════]
Chunk2:          [════════════════]    ← overlap preserves boundary context
Chunk3:                  [════════════════]
         64-token overlap
```

### Semantic Chunking

Split at natural semantic boundaries (paragraphs, headings, sentences) rather than arbitrary token counts.

**Embedding similarity approach:**

1. Split the document into individual sentences
2. Embed each sentence using an embedding model
3. Compute cosine similarity between consecutive sentence embeddings
4. When similarity drops below a threshold (e.g., < 0.75), insert a chunk boundary — this indicates a topic shift
5. Group consecutive above-threshold sentences into the same chunk

Alternatively, **LLM-based chunking** uses a language model to identify semantically coherent sections, which is more accurate but significantly more expensive.

```
Markdown doc:
  # Introduction          → chunk boundary
  ## Core Concepts        → chunk boundary
  Paragraph 1...          → may split at sentence boundaries based on embedding similarity
  Paragraph 2...

Embedding similarity between consecutive sentences:
  Sent 1 ←→ Sent 2: 0.92  (same topic, keep together)
  Sent 2 ←→ Sent 3: 0.88  (same topic, keep together)
  Sent 3 ←→ Sent 4: 0.61  (topic shift! → split here)
  Sent 4 ←→ Sent 5: 0.85  (same topic, keep together)
```

### Hierarchical Chunking (Parent-Child)

```
Document
├── Section (Parent chunk, ~1000 tokens)
│   ├── Paragraph (Child chunk, ~150 tokens)  ← indexed for retrieval
│   └── Paragraph (Child chunk, ~150 tokens)
└── Section (Parent chunk)
    ├── Paragraph
    └── Paragraph

Retrieval: match on child chunks (high precision)
Context:   return parent chunk to LLM (more context)
```

### Chunking Best Practices

| Strategy           | Chunk Size       | Use Case                        |
| ------------------ | ---------------- | ------------------------------- |
| Fixed with overlap | 256–512 tokens   | General purpose                 |
| Sentence-level     | ~3–5 sentences   | FAQ, dense factual text         |
| Paragraph-level    | ~200–400 tokens  | Narrative, explanatory text     |
| Page-level         | ~800–1500 tokens | Legal, technical docs           |
| Hierarchical       | Variable         | Complex multi-section documents |

---

## 6.4 Embedding Models for Retrieval

Retrieval embeddings differ from LLM token embeddings — they encode an entire passage into a single dense vector for similarity search.

```
Text passage ──→ [Embedding Model] ──→ v ∈ ℝ^1536
                                       (e.g., OpenAI text-embedding-3-large)

Similarity between two passages:
  cosine_similarity(v_query, v_doc) = (v_q · v_d) / (‖v_q‖ · ‖v_d‖)
```

### Embedding Model Comparison

| Model                         | Dimension | MTEB Score | Notes              |
| ----------------------------- | --------- | ---------- | ------------------ |
| OpenAI text-embedding-3-large | 3072      | ~65        | Strong, API        |
| OpenAI text-embedding-3-small | 1536      | ~63        | Cheaper API        |
| Cohere embed-v3               | 1024      | ~64        | Multilingual, API  |
| BGE-M3                        | 1024      | ~65        | Open, multilingual |
| E5-mistral-7b                 | 4096      | ~67        | Open, largest      |
| all-MiniLM-L6-v2              | 384       | ~59        | Tiny, fast, local  |

### Bi-Encoder vs Cross-Encoder

```
Bi-Encoder (fast, for ANN retrieval):
  Query ──→ [Encoder] ──→ q_vec       similarity = dot(q_vec, d_vec)
  Doc   ──→ [Encoder] ──→ d_vec       (can precompute d_vec offline)

Cross-Encoder (slow, for reranking):
  [Query + Doc] ──→ [Encoder] ──→ relevance score
  (attends query and doc jointly — more accurate but can't precompute)
```

**Why cross-encoders are more accurate:** Bi-encoders independently encode query and document into separate embeddings, so they can only capture similarity at the final embedding level — they cannot model token-level interactions between query and document. For example, a bi-encoder can't easily distinguish "dogs chase cats" from "cats chase dogs" when matching against a query. Cross-encoders concatenate `[CLS] query [SEP] document [SEP]` as a single input, allowing every query token to attend to every document token through full self-attention. This captures nuances like negation ("not recommended" vs "recommended"), qualifier matching ("only in cases where..."), and fine-grained semantic alignment that embedding-level comparison misses.

**Production pattern:** Bi-encoder for top-k retrieval (k=50–200), cross-encoder for reranking to top-5.

---

## 6.5 Vector Databases

A vector database stores embeddings and enables efficient approximate nearest neighbor (ANN) search.

```
┌──────────────────────────────────────────────────────────────────┐
│                   VECTOR DATABASE COMPARISON                      │
├──────────────┬──────────────┬───────────┬───────────┬────────────┤
│ Database     │ Type         │ Scale     │ ANN Algo  │ Notes      │
├──────────────┼──────────────┼───────────┼───────────┼────────────┤
│ Pinecone     │ Managed SaaS │ Billions  │ Custom    │ Easiest    │
│ Weaviate     │ OSS + Cloud  │ Hundreds  │ HNSW      │ GraphQL    │
│              │              │ of millions│          │ API         │
│ Qdrant       │ OSS + Cloud  │ Millions  │ HNSW      │ Fast, Rust │
│ Chroma       │ OSS          │ Millions  │ HNSW      │ Dev-friendly│
│ pgvector     │ Postgres ext │ Millions  │ IVFFLAT   │ SQL + vector│
│ FAISS        │ Library      │ Billions  │ IVF, HNSW │ Meta, no   │
│              │              │           │           │ metadata   │
│ Milvus       │ OSS + Cloud  │ Billions  │ HNSW, IVF │ Enterprise │
└──────────────┴──────────────┴───────────┴───────────┴────────────┘
```

### HNSW: The Dominant ANN Algorithm

```
Hierarchical Navigable Small World graphs build a multi-layer graph:

Layer 2 (sparse): ●─────────────────●
Layer 1:          ●────●────────●───●
Layer 0 (dense):  ●──●──●──●──●──●──●──●

Construction algorithm:
  For each new element, assign a random maximum layer:
    l = floor(-ln(uniform(0,1)) × mL)     (mL = 1/ln(M), M = max connections)
  This exponential decay means most elements only appear on Layer 0,
  few reach Layer 1, and very few reach Layer 2+.

  Insertion:
    1. Start at the entry point (top layer)
    2. Greedily navigate to the nearest neighbor at each layer
    3. Descend to the next layer
    4. At each layer ≤ l, add bidirectional edges to the M nearest
       neighbors found so far

  Upper layers act as a "coarse highway" — sparse long-range links
  enable fast traversal across the graph. Lower layers are dense
  with short-range links for final precision.

Query: start at top layer, greedily navigate to nearest neighbor,
       descend to next layer, repeat until Layer 0.

Result: O(log N) approximate search instead of O(N) exact search.
  The logarithmic complexity comes from the hierarchical structure:
  each layer halves the effective search space (analogous to skip lists).
Tradeoff: ~98-99% recall at 10-50× speedup over brute force.
```

---

## 6.6 Retrieval Quality Improvements

### Query Rewriting

The user's query is often a poor retrieval input. Rewrite it first.

```python
rewrite_prompt = """
Given a user question, rewrite it to be an optimal search query for
retrieving relevant documents. Make it more specific and include
synonyms where helpful.

User question: {question}
Optimized search query:"""
```

### HyDE (Hypothetical Document Embeddings)

Instead of embedding the query, generate a hypothetical answer and embed that — it lies in "document space."

```
User query: "What are the side effects of metformin?"

Step 1: LLM generates a hypothetical answer:
  "Metformin commonly causes gastrointestinal side effects including
   nausea, diarrhea, and stomach upset. Lactic acidosis is rare..."

Step 2: Embed the hypothetical answer (not the original query)
Step 3: Search the vector DB with this embedding

Why this works — the distribution gap problem:
  Queries are short and interrogative:   "What are the side effects of metformin?"
  Documents are long and declarative:    "Metformin commonly causes GI side effects..."

  These occupy DIFFERENT regions in embedding space. A query embedding
  and a relevant document embedding may have moderate cosine similarity
  even when the document perfectly answers the question.

  By generating a hypothetical answer, we produce text that is:
    - Declarative (like stored documents)
    - Long-form (like stored documents)
    - In the same linguistic register as stored documents

  The hallucinated content doesn't matter because:
    1. It's only used for retrieval (vector similarity search),
       NOT as the final answer to the user
    2. Even an imperfect hypothetical answer lands closer to the
       correct document cluster than the original query would
    3. The LLM still generates the final answer from real retrieved docs
```

### Hybrid Search: Dense + Sparse

```
Dense retrieval (semantic):
  embed(query) ←→ embed(doc)
  Good for: paraphrases, conceptual similarity

Sparse retrieval (keyword, BM25):
  TF-IDF-style matching on exact tokens
  Good for: product IDs, names, technical terms, rare words

Hybrid search:
  score = α · dense_score + (1-α) · sparse_score

Reciprocal Rank Fusion (RRF): weight-free combination
  score(d) = Σ_i 1 / (k + rank_i(d))    k typically 60
```

### Reranking

```
Query: "How does HNSW ANN indexing work?"

Top-50 dense retrieval results:
  Rank 1: "HNSW graph construction algorithm..."     (relevant ✓)
  Rank 2: "Vector similarity metrics comparison..."   (somewhat relevant)
  Rank 12: "HNSW parameter tuning guide..."           (very relevant ✓)
  Rank 47: "Introduction to graph databases..."      (irrelevant)

Cross-encoder reranker (CohereRerank, BGE-reranker, etc.)
re-scores all 50 docs jointly with the query:

  New Rank 1: "HNSW graph construction algorithm..."  ✓
  New Rank 2: "HNSW parameter tuning guide..."        ✓ (promoted from 12)
  New Rank 3: "Vector similarity metrics..."

Send top-5 to LLM.
```

---

## 6.7 Advanced RAG Patterns

### Naive RAG vs Advanced RAG vs Modular RAG

```
Naive RAG:                          Advanced RAG:
  Query → Retrieve → Generate         Pre-retrieval: query expansion, routing
                                       Retrieval: hybrid, reranking, filtering
                                       Post-retrieval: compression, reordering
                                       Generation: citation grounding

Modular RAG (state-of-the-art):
  Configurable pipeline of modules that can be composed, swapped,
  or skipped based on the query type (e.g., different pipelines
  for different document types or user intents).
```

### Multi-Hop RAG

Some questions require multiple retrieval steps:

```
Q: "What did the CEO of the company that acquired OpenAI in 2026 say
    about AI safety?"

Step 1: "Who acquired OpenAI in 2026?" → retrieve → "Microsoft"
Step 2: "Who is the CEO of Microsoft in 2026?" → retrieve → "Satya Nadella"
Step 3: "What did Satya Nadella say about AI safety?" → retrieve → answer
```

### CRAG (Corrective RAG)

```
       ┌────────────────────────────────────────────────────────┐
       │                   CRAG PIPELINE                        │
       │                                                        │
       │  1. Retrieve documents                                 │
       │  2. Evaluate relevance with a retrieval evaluator       │
       │                                                        │
       │     HIGH confidence → use retrieved docs               │
       │     LOW confidence  → fall back to web search          │
       │     AMBIGUOUS       → combine both sources             │
       │                                                        │
       │  3. Process docs (knowledge stripping)                 │
       │  4. Generate answer with corrected context             │
       └────────────────────────────────────────────────────────┘
```

### RAG-Fusion

```
User query → [LLM generates N query variants] → parallel retrieval
             "What is RAG?"
             "How does retrieval augmented generation work?"
             "RAG architecture for LLMs"
             "Vector database for question answering"

Merge with Reciprocal Rank Fusion → unified ranked result set
```

### Graph RAG

```
Traditional RAG retrieves isolated text chunks.
Graph RAG adds STRUCTURE — connecting entities and relationships.

Traditional RAG:             Graph RAG:
┌──────────┐                ┌──────────┐
│ Chunk 1  │                │  Entity  │──relates_to──▶│ Entity │
│ Chunk 2  │                │  "OpenAI"│               │ "GPT-4"│
│ Chunk 3  │                │          │──founded_by──▶│"Altman"│
│ (flat)   │                │  (graph) │               │        │
└──────────┘                └──────────┘               └────────┘

Why Graph RAG?
  Traditional RAG fails at:
  - Multi-hop questions: "Who founded the company that made GPT-4?"
  - Aggregation: "What are all the products in category X?"
  - Reasoning over relationships: "How are A and B connected?"

  Graph RAG excels because it retrieves SUBGRAPHS, not just chunks.

Graph RAG Pipeline:
  ┌──────────────────────────────────────────────────────────┐
  │  1. INDEXING                                             │
  │     Documents → LLM extracts (entity, relation, entity)  │
  │     "OpenAI released GPT-4" → (OpenAI, released, GPT-4)  │
  │     Store in knowledge graph (Neo4j, NetworkX)           │
  │     Also: community detection (Leiden algorithm)          │
  │                                                           │
  │     Leiden is a community detection algorithm (an improved │
  │     version of Louvain) that partitions the knowledge      │
  │     graph into clusters of densely connected entities.     │
  │     It optimizes modularity: nodes within a community have │
  │     many edges between them, few edges to other communities│
  │                                                           │
  │     → cluster related entities into communities           │
  │     → generate a text summary for each community via LLM  │
  │                                                           │
  │     When a query is too broad for individual chunks        │
  │     (e.g., "What are the main themes in AI safety?"),      │
  │     Graph RAG retrieves community summaries instead of     │
  │     individual chunks, enabling global reasoning over      │
  │     the entire corpus without stuffing all docs in context │
  │     → summarize each community                            │
  │                                                           │
  │  2. RETRIEVAL                                             │
  │     Query → extract key entities                          │
  │     → traverse graph (neighbors, paths, subgraphs)       │
  │     → retrieve community summaries for global questions  │
  │     → combine with traditional vector retrieval           │
  │                                                           │
  │  3. GENERATION                                            │
  │     Context = graph triples + community summaries         │
  │              + retrieved text chunks                      │
  │     LLM generates answer grounded in structured +        │
  │     unstructured knowledge                                │
  └──────────────────────────────────────────────────────────┘

Local vs Global queries:
  LOCAL:  "What is GPT-4's context window?" → entity lookup
  GLOBAL: "What are the main themes across all AI safety papers?"
          → community summaries → map-reduce synthesis

Tools: Microsoft GraphRAG, LlamaIndex Knowledge Graphs,
       LangChain + Neo4j integration
```

---

## 6.8 Evaluation of RAG Systems

### Key Metrics (RAGAS framework)

```
┌───────────────────────────────────────────────────────────────┐
│                    RAG EVALUATION METRICS                      │
├─────────────────────┬─────────────────────────────────────────┤
│ Metric              │ Definition                               │
├─────────────────────┼─────────────────────────────────────────┤
│ Faithfulness        │ Is the answer supported by the context? │
│                     │ (measures hallucination vs grounding)   │
├─────────────────────┼─────────────────────────────────────────┤
│ Answer Relevance    │ Does the answer address the question?   │
│                     │ (measures response quality)             │
├─────────────────────┼─────────────────────────────────────────┤
│ Context Precision   │ Of retrieved docs, what fraction is     │
│                     │ actually relevant?                       │
├─────────────────────┼─────────────────────────────────────────┤
│ Context Recall      │ Of all relevant docs, what fraction was │
│                     │ retrieved?                               │
├─────────────────────┼─────────────────────────────────────────┤
│ Answer Correctness  │ Is the final answer factually correct?  │
└─────────────────────┴─────────────────────────────────────────┘
```

### Retrieval Metrics

```
Precision@k = (relevant docs in top-k) / k
Recall@k    = (relevant docs in top-k) / (total relevant docs)
MRR         = (1/|Q|) Σ 1/rank_first_relevant   (Mean Reciprocal Rank)
NDCG@k      = normalized discounted cumulative gain (graded relevance)
```

---

## 6.9 Production RAG Implementation

### Minimal RAG in Python (LangChain + OpenAI + FAISS)

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Load and chunk documents
loader = DirectoryLoader("./docs/", glob="**/*.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " "],
)
chunks = splitter.split_documents(documents)

# 2. Embed and index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("./index")

# 3. Build RAG chain
retriever = vectorstore.as_retriever(
    search_type="mmr",          # Maximal Marginal Relevance for diversity
    search_kwargs={"k": 5, "fetch_k": 50},
)

RAG_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't know based on the provided documents."
Cite the document source for each claim.

Context:
{context}

Question: {question}

Answer:""")

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": RAG_PROMPT},
    return_source_documents=True,
)

# 4. Query
result = qa_chain.invoke({"query": "What is our refund policy?"})
print(result["result"])
print("Sources:", [d.metadata["source"] for d in result["source_documents"]])
```

### Production Checklist

```
Indexing:
  □ Chunk size tuned to embedding model's optimal window
  □ Metadata stored alongside vectors (source, date, type)
  □ Incremental indexing (don't re-index entire corpus on updates)
  □ Index backup and recovery plan

Retrieval:
  □ Hybrid search (dense + BM25) implemented
  □ Reranker deployed (Cohere, BGE, or fine-tuned)
  □ Query expansion / rewriting in place
  □ Fallback handling (no results → broader search or web)

Generation:
  □ Faithfulness guardrails (answer grounded in context)
  □ Citation extraction and verification
  □ Hallucination detection layer
  □ Response length limits

Evaluation:
  □ RAGAS metric suite running in CI/CD
  □ Human eval sample reviewed weekly
  □ Regression test set with golden Q&A pairs
  □ Latency and cost budgets enforced
```

---

## Interview Questions

### Conceptual

1. **What is the difference between RAG and fine-tuning? When would you choose each?**
   <details>
   <summary>Answer</summary>
   RAG retrieves external documents at inference time and injects them into the prompt, while fine-tuning bakes knowledge into the model weights during training. Choose RAG when: knowledge changes frequently (news, product catalogs, internal documents), you need source citations, you want to avoid forgetting base capabilities, or you need to handle private/sensitive data without it entering training. Choose fine-tuning when: you need to change the model's behavior/style (not just knowledge), the domain vocabulary is very different from pre-training, you want lower latency without retrieval overhead, or you have enough labeled data (>1K–10K examples). In practice, combining both (RAG + fine-tuning) is often best: fine-tune for format/style/behavior, RAG for fresh factual knowledge.
   </details>

2. **Explain the bi-encoder / cross-encoder architecture. Why use both?**
   <details>
   <summary>Answer</summary>
   A bi-encoder encodes the query and each document independently, producing separate embeddings whose similarity is measured by dot product or cosine similarity. This allows documents to be embedded and indexed offline; queries only require a single embedding at query time, enabling sub-millisecond ANN search across millions of docs. However, bi-encoders miss token-level interactions between query and document. A cross-encoder takes the concatenated (query, document) pair as input and produces a relevance score, allowing full cross-attention between query and document tokens. Much more accurate, but can't precompute document scores, so latency scales linearly with corpus size. Production RAG uses both: bi-encoder for fast top-50/100 retrieval, cross-encoder to rerank top-50 down to top-5 for the LLM. This achieves near cross-encoder accuracy with near bi-encoder speed.
   </details>

3. **What are the main failure modes of RAG systems? How do you debug them?**
   <details>
   <summary>Answer</summary>
   (1) Retrieval failure: correct documents exist in the corpus but aren't retrieved. Debug: compute Recall@k; often caused by vocabulary mismatch (fix with hybrid search), semantic gaps (fix with HyDE or query expansion), or chunking that splits context across boundaries (fix with larger chunks or hierarchical chunking). (2) Faithfulness failure: retrieved context is correct, but the LLM generates an answer that contradicts or goes beyond it. Debug: measure faithfulness score (RAGAS); fix with stronger grounding instructions, citation forcing, or smaller context windows. (3) Relevance failure: retrieved documents are technically related but don't contain the specific information needed. Debug: compute Context Precision; fix with reranking, better chunking, or multi-hop retrieval. (4) Answer quality failure: correct information is retrieved and grounded, but the generated answer is poor quality. Debug: this is a prompt engineering problem.
   </details>

4. **What is Maximal Marginal Relevance (MMR) and why is it useful?**
   <details>
   <summary>Answer</summary>
   MMR is a retrieval algorithm that selects documents to maximize both relevance to the query AND diversity among selected documents. Standard top-k retrieval returns the k most similar documents, which often means retrieving 5 nearly identical chunks from the same part of the same document. MMR iteratively selects the document that maximizes: λ · similarity(doc, query) - (1-λ) · max_similarity(doc, already_selected). With λ=0.5, it balances relevance and novelty. Result: instead of 5 repetitive chunks, you get 5 diverse, non-redundant chunks covering different aspects of the topic. This is especially important for long documents where the same information is repeated in different sections. Almost always better than vanilla top-k for the final context sent to the LLM.
   </details>

5. **How would you handle a RAG system that needs to answer questions about very long documents (e.g., entire legal contracts of 200 pages)?**
   <details>
   <summary>Answer</summary>
   Several approaches: (1) Hierarchical indexing: create a two-level hierarchy — section-level summaries (for rough navigation) and paragraph-level chunks (for precise retrieval). Query both levels, use section context to disambiguate paragraph results. (2) Map-reduce: for questions that require synthesizing across the whole document ("summarize all indemnification clauses"), retrieve all relevant chunks, summarize each chunk, then synthesize across summaries. (3) Query routing: classify queries as "local" (answerable from one section) vs "global" (requires full document), and route to different retrieval strategies. (4) Long-context LLMs: if using Gemini 2.0 Pro (1M tokens) or similar, stuff the full document in context for high-stakes documents where precision matters more than cost. (5) Document-specific embeddings: fine-tune the embedding model on legal terminology to improve retrieval quality.
   </details>

### Coding

6. **Implement a simple BM25 + dense hybrid search with RRF fusion.**
   <details>
   <summary>Solution</summary>

   ```python
   import numpy as np
   from rank_bm25 import BM25Okapi
   from sentence_transformers import SentenceTransformer

   class HybridRetriever:
       def __init__(self, documents: list[str], model_name: str = "all-MiniLM-L6-v2"):
           self.documents = documents
           self.model = SentenceTransformer(model_name)

           # BM25 index
           tokenized = [doc.lower().split() for doc in documents]
           self.bm25 = BM25Okapi(tokenized)

           # Dense index
           self.doc_embeddings = self.model.encode(documents, normalize_embeddings=True)

       def bm25_retrieve(self, query: str, k: int) -> list[tuple[int, float]]:
           scores = self.bm25.get_scores(query.lower().split())
           top_indices = np.argsort(scores)[::-1][:k]
           return [(int(i), float(scores[i])) for i in top_indices]

       def dense_retrieve(self, query: str, k: int) -> list[tuple[int, float]]:
           query_emb = self.model.encode([query], normalize_embeddings=True)[0]
           scores = self.doc_embeddings @ query_emb
           top_indices = np.argsort(scores)[::-1][:k]
           return [(int(i), float(scores[i])) for i in top_indices]

       def rrf_fusion(
           self,
           ranked_lists: list[list[tuple[int, float]]],
           k_rrf: int = 60,
       ) -> list[tuple[int, float]]:
           """Reciprocal Rank Fusion."""
           scores: dict[int, float] = {}
           for ranked in ranked_lists:
               for rank, (doc_id, _) in enumerate(ranked, start=1):
                   scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k_rrf + rank)
           return sorted(scores.items(), key=lambda x: x[1], reverse=True)

       def retrieve(self, query: str, k: int = 5, fetch_k: int = 50) -> list[str]:
           bm25_results = self.bm25_retrieve(query, fetch_k)
           dense_results = self.dense_retrieve(query, fetch_k)
           fused = self.rrf_fusion([bm25_results, dense_results])
           top_doc_ids = [doc_id for doc_id, _ in fused[:k]]
           return [self.documents[i] for i in top_doc_ids]
   ```

   </details>

7. **Implement a RAGAS-style faithfulness check (does the answer stay within the retrieved context).**
   <details>
   <summary>Solution</summary>

   ```python
   from openai import OpenAI

   client = OpenAI()

   def check_faithfulness(answer: str, context: str, model: str = "gpt-4o") -> dict:
       """
       Check if each statement in the answer is supported by the context.
       Returns faithfulness score (0–1) and per-statement verdicts.
       """
       # Step 1: Extract statements from the answer
       extraction_response = client.chat.completions.create(
           model=model,
           messages=[{
               "role": "user",
               "content": f"""Extract all distinct factual statements from this answer.
   Return a JSON list of strings, one per statement.

   Answer: {answer}"""
           }],
           temperature=0,
       )
       import json
       statements = json.loads(extraction_response.choices[0].message.content)

       # Step 2: Verify each statement against context
       verdicts = []
       for stmt in statements:
           verification = client.chat.completions.create(
               model=model,
               messages=[{
                   "role": "user",
                   "content": f"""Is this statement supported by the context?
   Reply with exactly "yes" or "no".

   Context: {context}

   Statement: {stmt}"""
               }],
               temperature=0,
           )
           supported = verification.choices[0].message.content.strip().lower() == "yes"
           verdicts.append({"statement": stmt, "supported": supported})

       score = sum(v["supported"] for v in verdicts) / len(verdicts) if verdicts else 0
       return {"faithfulness_score": score, "verdicts": verdicts, "num_statements": len(verdicts)}
   ```

   </details>

### System Design

8. **Design a RAG system for a company's internal knowledge base (10M documents, 10K QPS, <500ms P99 latency). Cover indexing, retrieval, generation, and caching.**
   <details>
   <summary>Answer</summary>

   **Indexing:** Use chunking (512 tokens, 64-token overlap) with semantic boundary detection. Embed with text-embedding-3-large via OpenAI API or a self-hosted BGE-M3. Use Milvus or Qdrant for vector storage (support for 10M+ vectors). Maintain a parallel Elasticsearch index for BM25 hybrid search. Store metadata (document_id, source, timestamp, access_controls) alongside vectors. Incremental indexing via a Kafka queue for new/updated documents; only re-embed changed chunks.

   **Retrieval at Scale:** Serve queries with an ANN index partitioned across multiple nodes (Milvus cluster). Use HNSW with ef=200 for ~99% recall. Run hybrid search (FAISS dense + Elasticsearch BM25) with RRF fusion. For reranking at 10K QPS, cache cross-encoder reranker results by (query_hash, doc_hash); deploy cross-encoder as a dedicated microservice with GPU batching.

   **Caching:** Semantic cache: embed each incoming query; if cosine similarity to a cached query > 0.97, return cached result (hits ~30-40% of traffic for common questions). TTL-based document cache for top-1000 documents by access frequency.

   **Generation:** Route to gpt-4o-mini for simple factual queries, gpt-4o for complex synthesis. Apply prompt stuffing with top-5 reranked chunks. Target < 200ms for retrieval, < 300ms for generation (streaming).

   **Access Control:** Filter vector search results to documents the user has permissions to access (metadata filter on user_id/group). Never inject unauthorized documents into prompts.

   **Monitoring:** Track: cache hit rate, retrieval precision (sampled), answer faithfulness (sampled), P50/P95/P99 latency, cost per query.
   </details>

---

## Key Papers

- Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Karpukhin et al. (2020) — "Dense Passage Retrieval for Open-Domain Question Answering" (DPR)
- Gao et al. (2023) — "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE)
- Ma et al. (2023) — "Query Rewriting for Retrieval-Augmented Large Language Models"
- Es et al. (2023) — "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
- Yan et al. (2024) — "Corrective Retrieval Augmented Generation" (CRAG)
- Shi et al. (2023) — "REPLUG: Retrieval-Augmented Language Model Pre-Training"
- Khattab et al. (2020) — "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"
