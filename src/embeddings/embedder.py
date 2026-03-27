from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path
import uuid
print("🤖 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded")

def get_chroma_client(persist_dir: str = "./chroma_db") -> chromadb.PersistentClient:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str = "documind"
) -> chromadb.Collection:
    
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity
        # Why cosine? Measures angle between vectors = semantic similarity
        # Dot product measures magnitude too = not what we want
    )
def embed_and_store(
    chunks: list[dict],
    collection: chromadb.Collection,
    batch_size: int = 32
) -> int:
    
    if not chunks:
        print("⚠️ No chunks to embed")
        return 0

    total_stored = 0

    # Process in batches
    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i + batch_size]

        texts = [chunk["content"] for chunk in batch]

        vectors = embedding_model.encode(
            texts,
            show_progress_bar=False
        ).tolist()

        ids = []
        metadatas = []

        for chunk in batch:
            unique_id = str(uuid.uuid4())
            ids.append(unique_id)

            metadatas.append({
                "source":      chunk.get("source", "unknown"),
                "page":        str(chunk.get("page_number", 0)),
                "type":        chunk.get("type", "text"),
                "char_count":  str(len(chunk["content"]))
            })

        collection.add(
            documents=texts,
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas
        )

        total_stored += len(batch)
        print(f"  ✅ Batch {i//batch_size + 1}: stored {len(batch)} chunks")

    return total_stored
def query_collection(
    question: str,
    collection: chromadb.Collection,
    n_results: int = 5,
    filter_type: str = None
) -> list[dict]:
    
    # Embed the question
    question_vector = embedding_model.encode([question]).tolist()
    where = {"type": filter_type} if filter_type else None

    # Query ChromaDB
    results = collection.query(
        query_embeddings=question_vector,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    # Format results cleanly
    formatted = []
    for i in range(len(results["documents"][0])):
        formatted.append({
            "content":  results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": round(results["distances"][0][i], 4)
        })

    return formatted

def ingest_pdf(pdf_path: str, collection: chromadb.Collection) -> dict:

    import sys
    sys.path.append('../')

    from src.ingestion.pdf_loader import load_pdf
    from src.ingestion.table_extractor import extract_all_tables
    from src.chunking.chunker import recursive_chunk

    print(f"\n{'='*50}")
    print(f"📄 Ingesting: {pdf_path}")
    print(f"{'='*50}")

    all_chunks = []
    print("\n📝 Step 1: Extracting text...")
    pages = load_pdf(pdf_path)
    for page in pages:
        # Chunk each page's text
        chunks = recursive_chunk(page["text"])
        for chunk in chunks:
            all_chunks.append({
                "content":     chunk,
                "source":      page["source"],
                "page_number": page["page_number"],
                "type":        "text"
            })
    print(f"  → {len(all_chunks)} text chunks")

    print("\n📊 Step 2: Extracting tables...")
    tables = extract_all_tables(pdf_path)
    for table in tables:
        all_chunks.append({
            "content":     table["content"],
            "source":      table["source"],
            "page_number": table["page_number"],
            "type":        "table"
        })
    print(f"  → {len(tables)} table chunks")

    print(f"\n🔢 Step 3: Embedding {len(all_chunks)} total chunks...")
    stored = embed_and_store(all_chunks, collection)

    summary = {
        "pdf":          pdf_path,
        "text_chunks":  len(pages),
        "table_chunks": len(tables),
        "total_stored": stored
    }

    print(f"\n Ingestion complete!")
    print(f"   Text chunks:  {len(pages)}")
    print(f"   Table chunks: {len(tables)}")
    print(f"   Total stored: {stored}")

    return summary