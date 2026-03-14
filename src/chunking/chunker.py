# ============================================
# DocuMind v2 - Chunking Module
# ============================================
# Job: Take raw text → return clean chunks
# Position in pipeline: After ingestion, before embedding
# ============================================

import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

# -----------------------------------------------
# Load embedding model once (important for speed)
# -----------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------------
# Strategy 1 — Fixed Size (naive, educational only)
# -----------------------------------------------
def fixed_size_chunk(text: str, chunk_size: int = 200) -> list[str]:
    """
    Splits text every N characters.
    Does NOT respect sentence boundaries.
    Demonstrates why naive chunking is bad.
    """

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=0
    )

    return splitter.split_text(text)


# -----------------------------------------------
# Strategy 2 — Recursive (smart default)
# -----------------------------------------------
def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Attempts to split at natural boundaries:
    paragraph → sentence → word.

    This is the default chunking strategy for DocuMind.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_text(text)


# -----------------------------------------------
# Strategy 3 — Sliding Window
# -----------------------------------------------
def sliding_window_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Fixed-size chunks but with intentional overlap.
    Useful for legal/medical documents where
    context continuity matters.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    return splitter.split_text(text)


# -----------------------------------------------
# Strategy 4 — Semantic Chunking
# -----------------------------------------------
def semantic_chunk(text: str, threshold: float = 0.3) -> list[str]:
    """
    Splits text when semantic meaning changes.

    Method:
    1. Split text into sentences
    2. Convert sentences → embeddings
    3. Measure cosine similarity
    4. If similarity drops below threshold → new chunk
    """

    sentences = [s.strip() for s in text.split('.') if s.strip()]

    if len(sentences) <= 1:
        return sentences

    embeddings = embedding_model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):

        sim = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) *
            np.linalg.norm(embeddings[i-1])
        )

        if sim < threshold:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append(". ".join(current_chunk) + ".")

    return chunks


# -----------------------------------------------
# Strategy Comparison (for experiments)
# -----------------------------------------------
def compare_strategies(text: str) -> None:
    """
    Runs all chunking strategies on the same text
    so you can visually compare them.

    Used in evaluation notebooks.
    """

    print("=" * 60)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 60)

    strategies = {
        "Fixed Size": fixed_size_chunk(text),
        "Recursive": recursive_chunk(text),
        "Sliding Window": sliding_window_chunk(text),
        "Semantic": semantic_chunk(text)
    }

    for name, chunks in strategies.items():

        print(f"\n📦 {name}: {len(chunks)} chunks")
        print("-" * 40)

        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1} ({len(chunk)} chars):")
            print(chunk[:100] + "...")
            print()
