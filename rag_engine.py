import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from nemoguardrails import LLMRails, RailsConfig
import ssl, certifi
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["SSL_CERT_FILE"] = certifi.where()

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
print("Initializing RAG Engine...")

# 1. Load Document
loader = PyPDFLoader("Environmental_Conservation_Essay.pdf")
docs = loader.load()

# 2. Chunk Document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Create Vector Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 4. Store Chunks in Chroma Vector Store
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# 5. Connect to Local Ollama Model
llm = ChatOllama(model="llama3.2", temperature=0)

# 6. Define Prompt Template
system_prompt = (
    "You are a strict document analysis assistant.\n"
    "Your ONLY source of knowledge is the provided context below.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Answer the user query strictly using only the provided context.\n"
    "2. If the user query is a general greeting (e.g., 'hi', 'hello', 'good evening'), respond politely.\n"
    "3. If the user asks a question whose answer is NOT present in the provided context, respond EXACTLY with:\n"
    "   'I cannot answer this question because the information is not present in the uploaded document.'\n"
    "4. NEVER use general knowledge, speculation, or external facts outside the context.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. Construct RAG Chain using LCEL
rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Load NeMo Guardrails
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

async def ask_rag(query: str):
    return rag_chain.invoke(query)

rails.register_action(ask_rag, name="ask_rag")

print("RAG Engine successfully initialized!")

# ---------------------------------------------------------
# Exported Function
# ---------------------------------------------------------
def get_rag_response(user_query: str) -> str:
    """Invokes the RAG chain and returns the generated answer."""
    return rag_chain.invoke(user_query)