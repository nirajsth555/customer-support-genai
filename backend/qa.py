import os
from typing import List, Union
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from string import Template
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from utils import get_embedding_function 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = "vector_store"

# Module-level cache so we don't rebuild BM25 on every request
_HYBRID_RETRIEVER: Union[EnsembleRetriever, None] = None

def load_vectorstore(persist_directory: str = PERSIST_DIR) -> Chroma:
    embeddings = get_embedding_function()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="default"
    )

def _documents_from_chroma(vectorstore: Chroma) -> List[Document]:
    """
    Read raw documents + metadata from Chroma and convert to langchain Document objects.
    """
    data = vectorstore.get(include=["documents", "metadatas"])
    if not isinstance(data, dict) or "documents" not in data:
        raise RuntimeError(
            "Chroma.get(...) did not return expected dict with 'documents'. "
            "Make sure you have ingested docs and persisted the vector store."
        )

    docs = data["documents"]
    metas = data.get("metadatas", [None] * len(docs))

    # Ensure metadata length matches documents length (pad with None if missing)
    if len(metas) < len(docs):
        # pad metas if necessary
        metas = metas + [None] * (len(docs) - len(metas))

    # Convert to LangChain Document objects (pair each doc with its metadata)
    documents = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(docs, metas)
    ]
    return documents

def build_hybrid_retriever(vectorstore: Chroma, use_cache: bool = True) -> EnsembleRetriever:
    """
    Build (or return cached) hybrid retriever composed of:
     - dense retriever (Chroma.as_retriever)
     - sparse BM25 retriever built from the raw texts in Chroma
    """
    global _HYBRID_RETRIEVER
    if use_cache and _HYBRID_RETRIEVER is not None:
        return _HYBRID_RETRIEVER

    # Step 1: Create dense retriever from Chroma (semantic similarity search)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # Step 2: Reconstruct raw documents for sparse (BM25) retriever
    documents = _documents_from_chroma(vectorstore)
    if not documents:
        raise RuntimeError("No documents found in vectorstore. Did you run the ingestion step?")

    # Step 3: Build sparse retriever using BM25 (traditional keyword-based search)
    sparse_retriever = BM25Retriever.from_documents(documents)

    # Step 4: Combine dense + sparse retrievers into a hybrid retriever
    # weights define importance of each retriever in scoring
    hybrid = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4] # 60% dense, 40% sparse
    )

    _HYBRID_RETRIEVER = hybrid
    return hybrid

def retrieve_context(
    vectorstore_or_hybrid: Union[Chroma, EnsembleRetriever],
    question: str,
    k: int = 3
) -> List[Document]:
    if isinstance(vectorstore_or_hybrid, EnsembleRetriever):
        hybrid = vectorstore_or_hybrid
    else:
        # assume Chroma instance
        hybrid = build_hybrid_retriever(vectorstore_or_hybrid)

    docs = hybrid.get_relevant_documents(question)
    return docs[:k]

# def retrieve_context(vectorstore: Chroma, question: str, k: int = 3) -> List[Document]:
#     return vectorstore.similarity_search(question, k=k)
def format_prompt(context: List[Document], question: str) -> str:
    context_text = "\n".join([doc.page_content for doc in context])
    prompt_template = Template("""
You are a professional AI assistant trained to help customers by answering their questions based strictly on the provided context.

To answer:
1. Carefully read the context.
2. Identify the most relevant parts that relate to the customer question.
3. Reason step-by-step using only the context (don’t use outside knowledge).
4. Provide a final, clear, and concise answer.

Your response should:
- Be clear, concise, and helpful.
- Rely only on the context provided below.
- Politely state if the answer is not available in the context.

Avoid fabricating answers.

---

Context:
$context

---

Customer Question: "$question"

Final Answer:
""")
    return prompt_template.substitute(context=context_text, question=question)

def ask_question(question: str, k: int = 3) -> str:
    vs = load_vectorstore()
    hybrid = build_hybrid_retriever(vs)
    docs = retrieve_context(hybrid, question, k=k)
    prompt = format_prompt(docs, question)

    # call the chat model (keeps your invoke() style)
    chat = ChatOpenAI(model_name="gpt-4.1-nano", openai_api_key=OPENAI_API_KEY)
    response = chat.invoke([{"role": "user", "content": prompt}])
    return response.content
