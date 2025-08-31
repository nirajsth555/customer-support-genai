import os
import shutil
from fastapi import UploadFile
from utils import load_pdf_text, get_embedding_function
from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

async def ingest_pdf(file: UploadFile):
    os.makedirs(f"uploaded_files", exist_ok=True)
    file_path = f"uploaded_files/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = load_pdf_text(file_path)
    # splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=500,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = splitter.split_text(text)

    embeddings = get_embedding_function()
    persist_directory = f"vector_store"
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="default"
    )
    vectorstore.persist()

    return {"status": "success", "message": f"{file.filename} ingested successfully."}
