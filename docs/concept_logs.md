# 📚 DocuMind — Concept Log

---

## Day 1 — RAG Fundamentals + Transformer Attention

**Date: 11/03/2026** Day 1 of 21

---

### 🔍 What is RAG & Why It Exists

**The problem:**
LLMs are frozen in time. They have no access to private or recently updated documents. Pasting docs into prompts doesn't scale — context windows overflow, costs explode, hallucinations persist.

**The solution — RAG (Retrieval Augmented Generation):**
Instead of baking knowledge into weights, retrieve relevant context at query time and inject it into the prompt. LLM answers from evidence, not memory.

**The 7-step RAG pipeline:**

```
Ingestion → Chunking → Embedding → Storage
                                      ↓
Question → Embed question → Retrieve top-K chunks → Augment prompt → Generate answer
```

**Where naive RAG breaks:**

| Failure         | Why                               |
| --------------- | --------------------------------- |
| Bad chunking    | Cuts mid-sentence, loses context  |
| Wrong retrieval | Close in words, wrong in meaning  |
| No reranking    | First pass is never perfect       |
| Tables & images | Embeddings can't handle structure |
| No evaluation   | Never know if hallucinating       |

---

### 🧮 What is an Embedding

A word, sentence or paragraph converted into a list of numbers that captures its **meaning.**

```
"The cat sat on the mat"  →  [0.21, -0.54, 0.89, ...]
"A kitten rested on rug"  →  [0.19, -0.51, 0.91, ...]  ← similar numbers = similar meaning
"Quantum physics paper"   →  [-0.87, 0.33, -0.21, ...]  ← very different
```

Semantic similarity = geometric closeness in vector space.
RAG retrieval = find chunks whose vectors are closest to the question vector.

---

### 🆚 RAG vs Fine-tuning

|               | RAG                          | Fine-tuning                   |
| ------------- | ---------------------------- | ----------------------------- |
| What          | External docs at query time  | Bakes knowledge into weights  |
| When          | Knowledge changes frequently | Behavior/style changes needed |
| Cost          | Low                          | High — GPU retraining         |
| Updatable     | Yes — just add docs          | No — full retrain             |
| Hallucination | Lower — grounded             | Still possible                |

**Rule:** Knowledge changes → RAG. Behavior changes → Fine-tune.

---

### 🎯 Attention — Query, Key, Value

**Why attention exists:**
RNNs forgot long-range context (word 1 lost by word 50).
Attention lets every word look at every other word **simultaneously.**

**The library analogy:**

```
Query  = your sticky note (what am I looking for?)
Key    = book spine label (what do I represent?)
Value  = actual book content (what do I give if selected?)
```

**The mechanism:**

```
1. Every word gets Q, K, V vectors via learned weight matrices W_Q, W_K, W_V
2. Q · K = relevance scores for every word pair
3. Divide by √dimension → numerical stability
4. Softmax → attention weights (sum to 1)
5. Weighted sum of V vectors → updated word representation
```

**Critical insight:**
Q, K, V are NOT the word itself — they are **learned projections.**
Same word, 3 different jobs, 3 different representations.

---

### 🏗️ Attention + MLP Stack (The "Deep" in Deep Learning)

**The backpack analogy:**
Every word carries a backpack (its vector).

* **Attention** = everyone opens each other's backpacks and copies what's relevant into their own
* **MLP** = each word goes back to their desk and processes what's now in their backpack — alone

**Why context is preserved without MLP communication:**
Context doesn't travel DURING MLP — it already traveled DURING attention.
By the time a vector hits MLP, it already contains other words' information.

**The floor-by-floor rhythm:**

```
Attention → "everyone share backpacks"
MLP       → "think alone about what you collected"
Attention → "share again — richer now"
MLP       → "think alone — deeper"
... × 32 floors in Mistral 7B ...
```

**Why the LAST vector predicts the next word:**
After 32 floors of attention, the last vector has absorbed context from every other word.
It started as just "in" — it ends as a rich summary of the entire input.

---

### 🔗 Connection — Attention inside LLM vs RAG outside LLM

| Attention (inside LLM) | RAG (system level)             |
| ---------------------- | ------------------------------ |
| Query vector           | User question embedding        |
| Key vectors            | Chunk embeddings in vector DB  |
| Dot product            | Cosine similarity search       |
| Softmax weights        | Relevance scores               |
| Value aggregation      | Retrieved chunks passed to LLM |

RAG is attention at the **system level.**
Both ask: *"Given this query, what information is most relevant?"*


