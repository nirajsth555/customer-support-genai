import os
from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from string import Template

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_vectorstore(doc_id: str) -> Chroma:
    persist_directory = f"vector_store/chroma_{doc_id}"
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY),
        collection_name=f"collection_{doc_id}"
    )

def retrieve_context(vectorstore: Chroma, question: str, k: int = 3) -> List[Document]:
    return vectorstore.similarity_search(question, k=k)

def format_prompt(context: List[Document], question: str) -> str:
    context_text = "\n".join([doc.page_content for doc in context])
    prompt_template = Template("""
You are a professional AI assistant trained to help customers by answering their questions based strictly on the provided context.

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

Your Answer:
""")
    return prompt_template.substitute(context=context_text, question=question)

def ask_question(doc_id: str, question: str) -> str:
    vectorstore = load_vectorstore(doc_id)
    docs = retrieve_context(vectorstore, question)
    prompt = format_prompt(docs, question)

    chat = ChatOpenAI(model_name="gpt-4.1-nano", openai_api_key=OPENAI_API_KEY)
    response = chat.invoke([{"role": "user", "content": prompt}])
    return response.content
