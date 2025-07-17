from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ingestion import ingest_pdf
from qa import answer_question

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), doc_id: str = Form(...)):
    return await ingest_pdf(file, doc_id)

@app.post("/ask/")
async def ask(question: str = Form(...), doc_id: str = Form(...)):
    return await answer_question(question, doc_id)
