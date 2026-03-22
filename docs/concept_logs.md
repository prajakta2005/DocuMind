# 📚 DocuMind — Concept Log
## Day 1 — RAG Fundamentals + Transformer Attention
**Date:** Day 1 of 21

---

### 🔍 What is RAG & Why It Exists

**The problem:**
LLMs are frozen in time. They have no access to private or recently
updated documents. Pasting docs into prompts doesn't scale —
context windows overflow, costs explode, hallucinations persist.

**The solution — RAG (Retrieval Augmented Generation):**
Instead of baking knowledge into weights, retrieve relevant context
at query time and inject it into the prompt. LLM answers from
evidence, not memory.

**The 7-step RAG pipeline:**
```
OFFLINE:
Ingestion → Chunking → Embedding → Storage

ONLINE (per query):
User Question → Embed Question → Retrieve top-K chunks → Augment prompt → Generate answer
```

**Where naive RAG breaks:**
| Failure | Why |
|---------|-----|
| Bad chunking | Cuts mid-sentence, loses context |
| Wrong retrieval | Close in words, wrong in meaning |
| No reranking | First pass is never perfect |
| Tables & images | Embeddings can't handle structure |
| No evaluation | Never know if hallucinating |

---

### 🧮 What is an Embedding

A word, sentence or paragraph converted into a list of numbers
that captures its meaning.

```
"The cat sat on the mat"  →  [0.21, -0.54, 0.89, ...]
"A kitten rested on rug"  →  [0.19, -0.51, 0.91, ...]  ← similar meaning
"Quantum physics paper"   →  [-0.87, 0.33, -0.21, ...]  ← very different
```

Semantic similarity = geometric closeness in vector space.
RAG retrieval = find chunks whose vectors are closest to the question vector.

---

### 🆚 RAG vs Fine-tuning

| | RAG | Fine-tuning |
|--|-----|-------------|
| What | External docs at query time | Bakes knowledge into weights |
| When | Knowledge changes frequently | Behavior/style changes needed |
| Cost | Low | High — GPU retraining |
| Updatable | Yes — just add docs | No — full retrain |
| Hallucination | Lower — grounded | Still possible |

**Rule:** Knowledge changes → RAG. Behavior changes → Fine-tune.

---

### 🎯 Attention — Query, Key, Value

**Why attention exists:**
RNNs forgot long-range context (word 1 lost by word 50).
Attention lets every word look at every other word simultaneously.

**The library analogy:**
```
Query  = your sticky note  (what am I looking for?)
Key    = book spine label  (what do I represent?)
Value  = actual book content (what do I give if selected?)
```

**The full mechanism:**
```
Step 1: Every word gets Q, K, V vectors via learned matrices W_Q, W_K, W_V
Step 2: Q · K = raw relevance scores for every word pair
Step 3: Divide by √d = scale down so scores don't explode with dimensions
Step 4: Softmax = convert scores to probabilities that sum to 1
Step 5: Weighted sum of V vectors = updated word representation
```

**What √d does:**
Without it — scores explode as vector dimensions grow.
Large scores → softmax gives one word 97% attention → model stops learning.
Dividing by √d keeps scores in manageable range regardless of dimensions.

**What Softmax does:**
Converts raw scores into probabilities.
```
Raw:           [8.4,  7.9,  7.2]
After softmax: [0.72, 0.21, 0.07]  ← sum = 1.0
```
Not standardization — probability conversion.

**Critical insight:**
Q, K, V are NOT the word itself — they are learned projections.
Same word, 3 different jobs, 3 different representations.

---

### 🏗️ Attention + MLP Stack

**The backpack analogy:**
Every word carries a backpack (its vector).

```
Attention = everyone opens each other's backpacks,
            copies what's relevant into their own
MLP       = each word goes to their desk alone,
            processes what's now in their backpack
```

**Why context is preserved without MLP communication:**
Context doesn't travel DURING MLP.
It already traveled DURING attention.
By the time a vector hits MLP, it already contains other words' information.

**The floor-by-floor rhythm:**
```
Attention → "everyone share backpacks"
MLP       → "think alone about what you collected"
Attention → "share again — richer now"
MLP       → "think alone — deeper this time"
... × 32 floors in Mistral 7B ...
```

**Why the LAST vector predicts the next word:**
After 32 floors of attention, the last vector has absorbed
context from every other word in the sequence.
It started as just "in" — ends as a rich summary of entire input.

---

### 🔗 Connection — Attention vs RAG

| Attention (inside LLM) | RAG (system level) |
|------------------------|-------------------|
| Query vector | User question embedding |
| Key vectors | Chunk embeddings in vector DB |
| Dot product | Cosine similarity search |
| Softmax weights | Relevance scores |
| Value aggregation | Retrieved chunks passed to LLM |

**RAG is attention at the system level.**
Both ask: "Given this query, what information is most relevant?"

---

## Day 2 — Vector Databases + ChromaDB
**Date:** Day 2 of 21

---

### 🗄️ Why Vector Databases Exist

Normal databases store exact values.
They can do exact match, greater than, less than.
But they have NO concept of similarity between vectors.
1 million chunks = 1 million brute force comparisons = too slow.

Vector databases are built for one job:
"Given this vector, find me the most similar vectors — fast."

**The magical library analogy:**
Normal library = organised by category + alphabetically.
Vector DB = every book has a GPS coordinate in meaning-space.
Similar books are physically close to each other.
Query = get a GPS coordinate → walk to it → grab nearest books.

---

### ⚙️ What's Inside a Vector DB

```
ID      Vector                    Metadata
----    ------                    --------
001     [0.21, -0.54, 0.89...]    {source: pdf1, page: 3}
002     [0.19, -0.51, 0.91...]    {source: pdf2, page: 7}
003     [-0.87, 0.33, -0.21...]   {source: pdf1, page: 12}

+ INDEX (the magic fast-lookup layer)
```

---

### 🗺️ HNSW — How Fast Lookup Works

HNSW = Hierarchical Navigable Small World

Like Google Maps navigation — works in layers:
```
Layer 3 (highways)      → get close to destination fast
Layer 2 (main roads)    → narrow down the area
Layer 1 (local streets) → find exact location
```

HNSW does the same with vectors:
```
Layer 3 (few nodes, long jumps)   → get into right neighbourhood
Layer 2 (more nodes)              → narrow down
Layer 1 (all nodes, small steps)  → find exact nearest neighbours
```

Result: checks ~50-100 vectors instead of 1 million. Still finds closest match.

---

### 🆚 ChromaDB vs Pinecone

| | ChromaDB | Pinecone |
|--|----------|---------|
| Runs | Local laptop | Cloud |
| Cost | Free | Free tier + paid |
| Setup | pip install | API key needed |
| Speed | Fast for small data | Fast for millions |
| DocuMind usage | Weeks 1 & 2 | Week 3 deployment |

---

### 🆚 Vector DB vs Regular DB

| | Regular DB | Vector DB |
|--|-----------|----------|
| Stores | Rows, exact values | Vectors + metadata |
| Queries | Exact match, range | Similarity search |
| Index type | B-tree | HNSW |
| Use case | Transactions, user data | Semantic search, RAG |

---

### 🧪 ChromaDB Experiment Results

**Test 1:** "Where is the Eiffel Tower?"
```
Rank 1: Eiffel Tower chunk ✅
Rank 2: Paris capital chunk ✅
Rank 3: Louvre chunk — both Paris landmarks in vector space
```

**Test 2:** "Which city has the famous iron structure?"
```
Zero shared words with correct answer.
Rank 1: Eiffel Tower chunk ✅ — meaning matched despite no keywords
```

**Key insight:** Vector search is semantic, not syntactic.
Same meaning = geometrically close vectors = retrieved correctly.
This is why RAG works on real documents.

**Sentence Transformer used:** all-MiniLM-L6-v2
```
all    = trained on all sentence types
MiniLM = small, fast version of larger model
L6     = 6 transformer layers
v2     = version 2
Output = 384 dimensions per sentence
Size   = ~80MB, runs locally, no GPU needed
```

---

## Day 3 — Chunking Strategies
**Date:** Day 3 of 21

---

### ✂️ Why Chunking Exists

**Problem 1 — Retrieval precision:**
One giant vector = one giant meaning = entire doc retrieved every time.
No precision. No relevance. Just noise.

**Problem 2 — Context window limits:**
```
50 pages ≈ 35,000 tokens
Mistral 7B context window = 8,000 tokens
It literally cannot fit. Breaks completely.
```

**Problem 3 — LLM focus:**
Too much irrelevant text = LLM loses focus = worse answers.
Needle in a haystack problem.

**The pizza analogy:**
```
Chunks too large  → retrieves too much, LLM confused
Chunks too small  → loses context, incomplete meaning
Just right        → precise retrieval, complete thought preserved
```

---

### 📦 The 4 Chunking Strategies

**Strategy 1 — Fixed Size:**
Splits every N characters. No respect for sentence boundaries.
```
CRIME SCENE from experiment:
Chunk 1 ended:   "Today it i"
Chunk 2 started: "s the most visited..."
→ "it is" split into "it i" + "s the" — mid word violation 💀

Chunk 3 ended:   "and Scik"
Chunk 4 started: "it-learn."
→ "Scikit-learn" destroyed 💀
```
Verdict: Never use in production. Educational only.

---

**Strategy 2 — Recursive Character Splitting:**
Tries to split at natural boundaries in priority order:
```
\n\n (paragraph) → \n (line) → ". " (sentence) → " " (word) → character
```
```
Experiment result:
Chunk 1: Entire Eiffel Tower + ML section (497 chars)
Chunk 2: Python section (175 chars)
→ Respected paragraph boundaries ✅
→ No broken words or sentences ✅
```
Verdict: Gold standard default for 80% of use cases.

---

**Strategy 3 — Sliding Window:**
Fixed size chunks with intentional overlap.
```
Chunk size = 500, Overlap = 100
Chunk 1: words 1-500
Chunk 2: words 401-900  ← 100 word overlap
Chunk 3: words 801-1300 ← 100 word overlap
```
Why overlap? Answer might sit at chunk boundary.
Without overlap → critical fact split across two chunks, neither retrieved fully.
With overlap → boundary content appears in both chunks → always retrievable.
Verdict: Best for legal, medical, contracts where boundary info is critical.

---

**Strategy 4 — Semantic Chunking:**
Splits when cosine similarity between consecutive sentences drops below threshold.
```
Threshold experiment results:
0.1 → 3 chunks  — only split on MASSIVE meaning shifts ✅
0.3 → 5 chunks  — good balance, sweet spot for most docs ✅
0.6 → 11 chunks — over-splits, every sentence separate ❌
```

**What threshold means:**
```
Low (0.1)  → "only split if topics are VERY different"
             → stays together more → fewer chunks
High (0.6) → "split even if slightly different"
             → splits aggressively → too many chunks
```

Verdict: Best for long mixed-topic docs. Expensive. Tune threshold carefully.

---

### 🆚 Strategy Comparison

| Strategy | Splits on | Overlap | Best for | Never for |
|----------|-----------|---------|----------|-----------|
| Fixed Size | N chars | No | Nothing in prod | Everything |
| Recursive | Natural boundaries | Optional | General docs | Tables, code |
| Sliding Window | Fixed + overlap | Yes | Legal, medical | Storage-sensitive |
| Semantic | Meaning shift | No | Long mixed docs | Short docs, speed |

---

### 🎯 Interview Answer — "How do you chunk?"

> "It depends on document type. Recursive character splitting
> as the default for general documents, sliding window for
> boundary-critical legal and medical docs, semantic chunking
> for long mixed-topic research papers. I tune chunk size and
> overlap based on retrieval precision experiments."

---

*"Never write code before you can explain what it does."*

*The 5 Laws:*
*1. WHY before HOW*
*2. Explain before Execute*
*3. Learn it → Log it → Lock it*
*4. No Blind Copy-Paste. Ever.*
*5. Ship Ugly, Refine Later*

## Day 4 — PDF & Table Ingestion

**Why ingestion exists:**
PDF is a visual format, not text. Raw extraction destroys structure.
Ingestion = converting visual format into structured text Python can use.

**Why tables need special treatment:**
Raw text extraction: "Q1 Q2 Q3 Q4 Revenue 100 200 150 300"
→ structure destroyed, relationships lost
Markdown extraction: | | Q1 | Q2 | → structure preserved ✅

**Why Markdown over JSON for tables:**
More token-efficient, LLMs trained on Markdown, human readable.

**PyMuPDF (fitz):** Low level, fast, clean PDFs, find_tables() built in
**Unstructured.io:** High level, handles messy real-world docs

**Metadata matters:** page_number + source = citation ability later
"Answer found on page 3 of contract.pdf" requires this metadata.

**Pipeline position:**
PDF → pdf_loader.py → table_extractor.py → chunker.py → embedder.py


## Day 5 — Image Handling

**Why images break naive RAG:**
Sentence transformers only understand text.
Images are invisible to the entire RAG pipeline without handling.

**Two scenarios:**
1. Image with text → OCR (pytesseract + Tesseract engine)
   Reads text pixels → returns string
   
2. Image without text → BLIP captioning
   Vision encoder reads image → text decoder generates description
   "a line chart showing revenue growth from 2019 to 2023"

**Why load BLIP at module level:**
Loading takes ~5 seconds. Module level = loads once, reuses forever.
Inside function = reloads every call = too slow.

**Why skip images < 100x100px:**
Tiny images are icons/decorations — not real content.
Filtering them saves processing time and noise.

**Auto-detection logic:**
Run quick OCR → if > 20 chars detected → use OCR
Otherwise → use BLIP captioning

**Full pipeline now:**
PDF → pdf_loader + table_extractor + image_handler → chunker → embedder
Text + Tables + Images all converted to text → all searchable in ChromaDB


## Tesseract vs pytesseract

### Tesseract
- The actual OCR engine  
- Built by Google  
- Written in C++  
- Does the real work of converting pixels → text  
- Installed as a system program (like installing Chrome)  

---

### pytesseract
- A Python wrapper around Tesseract  
- Written in Python  
- Does **NOT** perform OCR itself  
- Simply sends the image to Tesseract and returns the extracted text  
- Installed via `pip`  

## Tokenization Example

**Original Sentence:**  
"The cat sat on the mat"

**After Tokenization:**  
[CLS] The cat sat on the mat [SEP]

## Vision-Language Model Architecture

### Part 1 — Vision Encoder (ViT — Vision Transformer)
- Looks at the image  
- Breaks the image into patches (like chunks but for images)  
- Converts each patch into a vector  
- Builds a visual understanding of the image  

---

### Part 2 — Text Decoder (BERT-based)
- Takes the visual vectors  
- Generates natural language descriptions word by word  
- Example:  
  "a bar chart showing..." → "revenue" → "growth" → ...


  ## ViT and BERT Explained

### 🔹 ViT — Vision Transformer
**Full Form:** Vision Transformer  

- A model used to **understand images**  
- Uses the **Transformer architecture** (instead of CNNs)  
- Splits an image into small patches (e.g., 16×16 blocks)  
- Treats each patch like a “token” (similar to words in a sentence)  
- Uses **self-attention** to understand relationships between patches  

**Simple Explanation:**  
ViT = "reads images like a sentence of patches"  

---

### 🔹 BERT — Bidirectional Encoder Representations from Transformers

- A model used to **understand text**  
- Developed by Google  
- Reads text **in both directions (left and right context)**  
- Strong at tasks like:
  - Text understanding  
  - Question answering  
  - Sentence similarity  

**Simple Explanation:**  
BERT = "deep understanding of language context"  

---

## 🔥 Key Differences

| Feature | ViT | BERT |
|--------|-----|------|
| Input | Image patches | Words / tokens |
| Domain | Computer Vision | Natural Language Processing |
| Role | Understand images | Understand text |

---

## 🧠 How They Work Together

- **ViT** → understands what’s in the image  
- **BERT (or similar decoder)** → converts that understanding into human language  

**Day 5 Experiment Results:**
PDF: Student assignment with code screenshots

- Small images (338x22, 160x55) correctly skipped — decorations ✅
- Large images (678x940, 700x908) processed ✅
- OCR correctly extracted Python code from screenshots ✅
- OCR extracted ML results: "Logistic Regression Accuracy: 0.7467" ✅

Key insight: Data trapped in images is now searchable.
User can ask "what was the accuracy?" and DocuMind finds it
even though it was inside an image, not text.

CLS token = classification token, added at start, represents whole sequence
SEP token = separator token, added at end, separates sequences
skip_special_tokens=True removes these from output captions

Tesseract = OCR engine (C++, does actual work)
pytesseract = Python wrapper (talks to Tesseract)
BLIP = Vision-Language model, 129M image-text pairs training
     = Vision Encoder (ViT) + Text Decoder (BERT-based)