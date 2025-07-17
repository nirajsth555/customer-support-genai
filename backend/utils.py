import os
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings

def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def get_embedding_function():
    return OpenAIEmbeddings()
