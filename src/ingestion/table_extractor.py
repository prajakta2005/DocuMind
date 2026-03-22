import fitz
from pathlib import Path


def extract_tables_from_page(page) -> list[str]:
    """
    Extracts tables from a single PDF page.
    Converts each table to Markdown format.
    
    Why Markdown?
    - Preserves header-to-value relationships
    - LLMs trained heavily on Markdown
    - More token-efficient than JSON
    - Human readable for debugging
    """
    markdown_tables = []

    tables = page.find_tables()
    
    if not tables.tables:
        return []
    
    for table_idx, table in enumerate(tables.tables):
        data = table.extract()
        
        if not data or len(data) < 2:
            continue
        
        markdown = convert_to_markdown(data)
        markdown_tables.append(markdown)
        
    return markdown_tables


def convert_to_markdown(table_data: list[list]) -> str:
    
    if not table_data:
        return ""
    
    markdown_rows = []
    
    for row_idx, row in enumerate(table_data):
        
        cleaned_cells = []
        for cell in row:
            if cell is None:
                cleaned_cells.append("")
            else:
                cleaned_cells.append(str(cell).strip())
        
        markdown_row = "| " + " | ".join(cleaned_cells) + " |"
        markdown_rows.append(markdown_row)
        
        if row_idx == 0:
            separator = "| " + " | ".join(["--"] * len(cleaned_cells)) + " |"
            markdown_rows.append(separator)
    
    return "\n".join(markdown_rows)


def extract_all_tables(pdf_path: str) -> list[dict]:
    
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    all_tables = []
    
    with fitz.open(pdf_path) as doc:
        print(f"🔍 Scanning for tables in: {Path(pdf_path).name}")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            tables = extract_tables_from_page(page)
            
            for table_idx, table_markdown in enumerate(tables):
                all_tables.append({
                    "page_number":  page_num + 1,
                    "table_index":  table_idx,
                    "content":      table_markdown,
                    "source":       Path(pdf_path).name,
                    "type":         "table"
                })
                print(f"  Page {page_num+1}, Table {table_idx+1}: extracted")
    
    print(f"\n Total tables found: {len(all_tables)}")
    return all_tables


def preview_table(table_dict: dict) -> None:
    print(f"\n📊 Table from {table_dict['source']}")
    print(f"   Page {table_dict['page_number']}, Table {table_dict['table_index']+1}")
    print("-" * 50)
    print(table_dict['content'])
    print("-" * 50)