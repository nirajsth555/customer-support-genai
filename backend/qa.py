import os
from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from string import Template

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_vectorstore() -> Chroma:
    persist_directory = f"vector_store"
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY),
        collection_name="default"
    )

def retrieve_context(vectorstore: Chroma, question: str, k: int = 3) -> List[Document]:
    return vectorstore.similarity_search(question, k=k)

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
                            
Step-by-Step Reasoning (internal only, not shown to customer):
(Think through the answer here before giving the final response)

Final Answer (customer-facing):
""")
    return prompt_template.substitute(context=context_text, question=question)

def ask_question(question: str) -> str:
    vectorstore = load_vectorstore()
    docs = retrieve_context(vectorstore, question)
    prompt = format_prompt(docs, question)

    chat = ChatOpenAI(model_name="gpt-4.1-nano", openai_api_key=OPENAI_API_KEY)
    response = chat.invoke([{"role": "user", "content": prompt}])
    return response.content
