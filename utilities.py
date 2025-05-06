import os
import sys
import re
import json
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import argparse
import fitz
from docx import Document as DocxDocument
import pandas as pd
from pptx import Presentation
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")     

# Candidate labels (you can customize these)
labels = [
    "Case Study", "Brochure", "Whitepaper", "Technical Guide", 
    "Datasheet", "Manual", "Bill of Material", "Proposal", "Guide", "Solution",
    "Reference", "Catalog", "Flyer", "Report", "Specification", "Document", "Spreadsheet",
    "Handbook", "Ebook", "Press Releases", "Application Note", "Webinar", "Integration", "Certification"
]

def get_text_hash(text):
    """Generate a SHA256 hash of the full text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_document_type(filename: str) -> str:
    extension_map = {
        '.pdf': 'PDF',
        '.doc': 'Word',
        '.docx': 'Word',
        '.xls': 'Excel',
        '.xlsx': 'Excel',
        '.ppt': 'PowerPoint',
        '.pptx': 'PowerPoint',
        '.txt': 'Text',
        '.csv': 'CSV',
        '.rtf': 'Rich Text Format',
        '.odt': 'OpenDocument Text',
        '.ods': 'OpenDocument Spreadsheet',
        '.odp': 'OpenDocument Presentation',
        '.html': 'HTML',
        '.htm': 'HTML',
        '.json': 'JSON',
        '.xml': 'XML'
    }

    extension = Path(filename).suffix.lower()
    return extension_map.get(extension, 'Unknown Document Type')   

def infer_document_type(text: str, threshold: float = 0) -> str:
    if not isinstance(text, str) or not text.strip():
        return "Unknown"
    # Ensure the text is wrapped in a list for the classifier
    result = classifier(text[:1000], labels)  # Pass the text in a list format
    if result['scores'][0] >= threshold:
        return result['labels'][0]
    return "Unknown"     

def extract_text_with_page_numbers(pdf_path) -> list[dict]:
    results = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():  # Avoid empty pages
                results.append({
                    "page": i,
                    "text": text
                })
    except Exception as e:
        print(f"❌ Error reading PDF {pdf_path}: {e}")
    return results    

    return cleaned_bom  # structured dictionary of BoM data per sheet    

def process_file(file=None):
    if os.path.exists(file):
        ext = Path(file).suffix.lower()
        filename = Path(file).name
        file_type = get_document_type(filename)

        if ext == ".pdf":
            pages = extract_text_with_page_numbers(file)
            doc_type = infer_document_type(pages[0]['text'])
            cleaned_pages = [
                {"text": page["text"], "page_number": page["page"]} 
                for page in pages if page["text"].strip()
            ]
        elif ext in [".doc", ".docx"]:
            from docx import Document
            doc = Document(file)
            cleaned_pages = "\n".join(para.text for para in doc.paragraphs)
            return filename, file_type, doc_type, cleaned_pages   
        elif ext in [".xls", ".xlsx"]:
            xls = pd.read_excel(file, sheet_name=None)  # Load all sheets
            cleaned_pages = []

            for sheet_name, df in xls.items():
                df = df.dropna(how="all").reset_index(drop=True)

                # Collect each row as a "key: value" string
                rows = []
                for _, row in df.iterrows():
                    row_str = "; ".join(f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]))
                    if row_str.strip():
                        rows.append(f"- {row_str}")

                # Add structured content per sheet
                if rows:
                    cleaned_pages.append({
                        "text": f"Sheet: {sheet_name}\n" + "\n".join(rows),
                        "page_number": sheet_name
                    })
                    
                doc_type = infer_document_type(cleaned_pages[0]['text'])

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        return filename, file_type, doc_type, cleaned_pages
    else:
        #print("File does not exist!")
        return None, None, None, None


if __name__ == "__main__":
    print("Started...")

    filename, file_type, doc_type, cleaned_pages = process_file(r"C:\Users\samue\Desktop\OVCirrus-AIOps\repositories\pdfs\data networking\agdsn-short-case-study-en.pdf")
    
    if cleaned_pages:
        print(f"Filename: {filename}, Type:{file_type}, Document: {doc_type}")

    print("Completed!")
