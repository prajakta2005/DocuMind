import fitz
import pytesseract
import io
from PIL import Image
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


print("🤖 Loading BLIP captioning model...")
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
print("✅ BLIP model loaded")


def has_text(image: Image.Image, min_chars: int = 20) -> bool:
    """
    Runs quick OCR to check if image contains real text.
    
    Why min_chars = 20?
    OCR always returns something — even on blank images it 
    finds noise. 20 chars filters out false positives.
    """
    try:
        text = pytesseract.image_to_string(image).strip()
        return len(text) >= min_chars
    except Exception:
        return False


def extract_text_from_image(image: Image.Image) -> str:
    """
    Extracts text from an image using OCR.
    Use when: image contains text (scanned docs, screenshots)
    
    pytesseract.image_to_string():
    - Sends image to Tesseract OCR engine
    - Returns all detected text as a string
    - Works on printed text, struggles with handwriting
    """
    try:
        # Convert to RGB — tesseract works best with RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        text = pytesseract.image_to_string(image)
        return text.strip()
    
    except Exception as e:
        print(f"  ⚠️ OCR failed: {e}")
        return ""

def caption_image(image: Image.Image) -> str:
    """
    How BLIP works:
    - Vision encoder: reads image → creates visual embeddings
    - Text decoder: converts visual embeddings → natural language
    - Trained on 129M image-text pairs from the web
    
    Result: "a line chart showing revenue growth from 2019 to 2023"
    """
    try:
        # Convert to RGB — BLIP expects RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # processor = converts PIL image → tensors BLIP understands
        inputs = blip_processor(image, return_tensors="pt")
        
        # generate() = runs the model → produces token IDs
        # max_new_tokens = max length of generated caption
        with torch.no_grad():  
            output = blip_model.generate(
                **inputs,
                max_new_tokens=100
            )
        
        caption = blip_processor.decode(
            output[0],
            skip_special_tokens=True 
        )
        
        return caption
    
    except Exception as e:
        print(f"  ⚠️ Captioning failed: {e}")
        return ""

def extract_images_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts all images from a PDF.
    Auto-detects whether to use OCR or captioning.
    
    Returns list of dicts:
    {
        "page_number": int,
        "image_index": int,
        "content": str,      ← OCR text OR BLIP caption
        "method": str,       ← "ocr" or "caption"
        "source": str,
        "type": "image"
    }
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    all_images = []
    
    with fitz.open(pdf_path) as doc:
        print(f"🔍 Scanning for images in: {Path(pdf_path).name}")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
        
            image_list = page.get_images(full=True)
            
            if not image_list:
                continue
            
            print(f"  📄 Page {page_num+1}: {len(image_list)} image(s) found")
            
            for img_idx, img_ref in enumerate(image_list):
                
                xref = img_ref[0]
                
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                image = Image.open(io.BytesIO(image_bytes))
                
                width, height = image.size
                if width < 100 or height < 100:
                    print(f"    ⚠️ Image {img_idx+1}: too small ({width}x{height}), skipping")
                    continue
                
                # Auto-detect: OCR or Caption?
                print(f"    🔍 Image {img_idx+1} ({width}x{height}): ", end="")
                
                if has_text(image):
                    # Image contains text → use OCR
                    print("text detected → OCR")
                    content = extract_text_from_image(image)
                    method = "ocr"
                else:
                    # Visual only → use BLIP captioning
                    print("visual only → captioning")
                    content = caption_image(image)
                    method = "caption"
                
                if not content:
                    print(f"    ⚠️ No content extracted, skipping")
                    continue
                
                all_images.append({
                    "page_number": page_num + 1,
                    "image_index": img_idx,
                    "content":     content,
                    "method":      method,
                    "source":      Path(pdf_path).name,
                    "type":        "image",
                    "dimensions":  f"{width}x{height}"
                })
                
                print(f"    ✅ Extracted: {content[:80]}...")
    
    print(f"\n✅ Total images processed: {len(all_images)}")
    return all_images