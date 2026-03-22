import fitz  
import os
from pathlib import Path


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Loads a PDF and extracts text page by page.
    
    Returns a list of dicts, one per page:
    {
        "page_number": int,
        "text": str,
        "char_count": int,
        "source": str  ← filename for metadata
    }
    
    Why return dicts not just strings?
    We need metadata (page number, source) for citations later.
    "Answer found on page 3 of contract.pdf" requires this.
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if not pdf_path.endswith('.pdf'):
        raise ValueError(f"Expected .pdf file, got: {pdf_path}")
    
    pages = []
    
    with fitz.open(pdf_path) as doc:
        
        print(f"📄 Loading: {pdf_path}")
        print(f"📝 Total pages: {len(doc)}")
        
        for page_num in range(len(doc)):
            
            page = doc[page_num]
            text = page.get_text("text")
            
            text = text.strip()
            
            if not text:
                print(f"  ⚠️ Page {page_num + 1}: empty, skipping")
                continue
            
            pages.append({
                "page_number": page_num + 1,  
                "text": text,
                "char_count": len(text),
                "source": Path(pdf_path).name 
            })
            
            print(f"  ✅ Page {page_num + 1}: {len(text)} chars extracted")
    
    print(f"\n✅ Loaded {len(pages)} pages from {Path(pdf_path).name}")
    return pages


def load_multiple_pdfs(pdf_folder: str) -> list[dict]:
    """
    Loads all PDFs from a folder.
    Useful when ingesting an entire document collection.
    """
    all_pages = []
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDFs found in {pdf_folder}")
        return []
    
    print(f"📁 Found {len(pdf_files)} PDFs in {pdf_folder}")
    
    for pdf_file in pdf_files:
        pages = load_pdf(str(pdf_file))
        all_pages.extend(pages)
    
    print(f"\n✅ Total pages loaded: {len(all_pages)}")
    return all_pages


def get_pdf_metadata(pdf_path: str) -> dict:
    """
    Extracts PDF metadata without loading full text.
    Useful for quick inspection before full ingestion.
    """
    with fitz.open(pdf_path) as doc:
        metadata = doc.metadata
        return {
            "title":       metadata.get("title", "Unknown"),
            "author":      metadata.get("author", "Unknown"),
            "page_count":  len(doc),
            "file_size_kb": round(os.path.getsize(pdf_path) / 1024, 2),
            "source":      Path(pdf_path).name
        }