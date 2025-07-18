import os
import shutil
from fastapi import UploadFile
from utils import load_pdf_text, get_embedding_function
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

async def ingest_pdf(file: UploadFile, doc_id: str):
    os.makedirs(f"uploaded_files/{doc_id}", exist_ok=True)
    file_path = f"uploaded_files/{doc_id}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = load_pdf_text(file_path)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(text)

    embeddings = get_embedding_function()
    persist_directory = f"vector_store/chroma_{doc_id}"

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=f"collection_{doc_id}"
    )
    vectorstore.persist()

    return {"status": "success", "message": f"{file.filename} ingested successfully."}
