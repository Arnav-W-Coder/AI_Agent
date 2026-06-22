# running RAG with LangChain, ChromaDB, and Ollama LLM
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.agents import create_agent
from langchain_core.tools import tool

import mimetypes
mimetypes.add_type("application/pdf", ".pdf")

# 1. Load documents (swap this loader for .txt, .docx, web pages, etc.)
loader = DirectoryLoader("./docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# 2. Chunk them — this is the most underrated tuning knob in RAG
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,     # characters per chunk
    chunk_overlap=150,  # overlap so context isn't cut mid-thought
)
chunks = splitter.split_documents(documents)

# 3. Embed + store in ChromaDB (persists to disk)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

print(f"Indexed {len(chunks)} chunks.")

# --------------------------------------------------------------------------------------------

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # top-4 chunks

llm = ChatOllama(model="llama3.2", num_ctx=8192)  # bump context window for RAG

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below. If the answer isn't
in the context, say you don't know — don't make it up.

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What does the document say about the westernization of Japan?"))

# ---------------------------------------------------------------------------------------------

@tool
def search_documents(query: str) -> str:
    """Search the knowledge base for relevant information."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '12 * 4 + 1'."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

agent = create_agent(
    model=llm,
    tools=[search_documents, calculator],
    system_prompt="You are a helpful assistant. Use search_documents when the "
           "question needs info from the knowledge base. Use calculator for math.",
)

response = agent.invoke({"messages": [{"role": "user", "content": "Summarize the document's content."}]})
print(response["messages"][-1].content)