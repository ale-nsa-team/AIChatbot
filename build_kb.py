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

import torch
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # or use `langchain_core.documents.Document` if using v0.2+
from langchain.text_splitter import RecursiveCharacterTextSplitter
from hash_registry import load_processed_hashes, is_already_processed, mark_as_processed
from utilities import process_file, get_text_hash


def add_documents_to_vector_db(
    file=None, 
    model='thenlper/gte-small', 
    device='cpu', 
    vectorstore=None, 
    index_path='faiss_index',
    embeddings=None,
    ):

    print(f"Processing ...")
    
    filename = file_type = ""
    
    if file is None:
        raise ValueError("Argument file cannot be None.")   

    # Check if file exist
    if not os.path.exists(file):
        raise ValueError("File does not exist.")        
    
    # Generate file hash
    file_hash = get_text_hash(file)
    processed_hashes = load_processed_hashes()

    # Check if file has been processed before.
    if vectorstore:
        if is_already_processed(file_hash, processed_hashes):
            print(f"⏩ Skipped: File already processed.")
            return vectorstore
        else:
            documents = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            filename, file_type, document_type, page_chunks = process_file(file)
            if page_chunks:
                if file_type in ["PDF", "Excel"]:
                    for chunk in page_chunks:
                        splits = splitter.create_documents([chunk["text"]])
                        for doc in splits:
                            doc.metadata["source"] = filename                                   # ✅ Add File name
                            doc.metadata["created"] = datetime.now(timezone.utc).isoformat()    # ✅ Add UTC timestamp
                            doc.metadata["document_type"] = document_type                       # ✅ Add Document Type
                            doc.metadata["file_type"] = file_type                               # ✅ Add File Type
                            if chunk["page_number"] is not None:
                                doc.metadata["page"] = chunk["page_number"]                     # ✅ Add Page Number
                            documents.append(doc)
                
                print(f"Adding {filename} to vector database...")                    
                vectorstore.add_documents(documents)

                # After successful processing
                mark_as_processed(file_hash, processed_hashes)
                vectorstore.save_local(index_path)
                
            return vectorstore

    else:
        documents = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        filename, file_type, document_type, page_chunks = process_file(file)
        if page_chunks:
            if file_type in ["PDF", "Excel"]: 
                for chunk in page_chunks:
                    splits = splitter.create_documents([chunk["text"]])
                    for doc in splits:
                        doc.metadata["source"] = filename                                   # ✅ Add File name
                        doc.metadata["created"] = datetime.now(timezone.utc).isoformat()    # ✅ Add UTC timestamp
                        doc.metadata["document_type"] = document_type                       # ✅ Add Document Type
                        doc.metadata["file_type"] = file_type                               # ✅ Add File Type
                        if chunk["page_number"] is not None:
                            doc.metadata["page"] = chunk["page_number"]                    
                        documents.append(doc)    
                
            vectorstore = FAISS.from_documents(documents, embeddings)
            # After successful processing
            mark_as_processed(file_hash, processed_hashes)        
            print(f"Creating vector database...")
            print(f"Adding {filename} to vector database...")
        else:
            # Failed to process doc
            return vectorstore

    # Step 7: Save the vector database
    vectorstore.save_local(index_path)
    return vectorstore

def build_vector_store(folders: list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name='thenlper/gte-small', model_kwargs={'device': 'cpu'})
    index_path = 'faiss_index'

    if os.path.exists(index_path):              
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("Vectorstore loaded...")
    else:
        print("Vectorstore not found! Building a new vectorstore...")
        vectorstore = None

    supported_exts = [".pdf", ".docx", ".txt", ".xls", ".xlsx", ".html", ".htm"]

    # Iterate over each folder (full path provided in the list)
    for folder_path in folders:
        folder_path = Path(folder_path).resolve()  # Convert string path to Path object and resolve full path
        if not folder_path.exists():
            print(f"⚠️ Folder does not exist: {folder_path}")
            continue

        print(f"📁 Scanning folder: {folder_path}")
        
        # Use rglob to search for files with supported extensions
        for file in folder_path.rglob("*"):  # Recursively search all files
            if file.suffix.lower() in supported_exts:
                vectorstore = add_documents_to_vector_db(
                    file=rf"{file.parent}/{file.name}",
                    vectorstore=vectorstore,
                    index_path=index_path,
                    embeddings=embeddings
                    )
  

if __name__ == "__main__":
    print("Building knowledgebase vectorstore...")

    parser = argparse.ArgumentParser(description="Build a vector KB from documents.")
    parser.add_argument("--folders", required=True, nargs='+', help="Folders to scan.")
    args = parser.parse_args()

    build_vector_store(args.folders)    

    print("Job Completed!")

        
