import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

# Modern, robust imports for prompt and output handling
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# NeMo Guardrails imports
from nemoguardrails import LLMRails, RailsConfig

# ---------------------------------------------------------
# Setup Base RAG Pipeline
# ---------------------------------------------------------

# 1. Load the Document
loader = PyPDFLoader("Environmental_Conservation_Essay.pdf")
docs = loader.load()

# 2. Chunk the Document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Create Free Local Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 4. Store Chunks in Chroma Vector Store
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 5})

# 5. Connect to Local Ollama Model
llm = ChatOllama(model="llama3.2", temperature=0)

# 6. Define the Prompt Template
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

# Helper function to join retrieved document chunks into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. Construct RAG Chain using LCEL (No deprecated chain modules required)
rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# # ---------------------------------------------------------
# # Load External Guardrails Configuration
# # ---------------------------------------------------------
# 1. Automatically loads config.yaml and rails.colang from the ./config folder
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# Register RAG chain action directly into NeMo Guardrails
async def ask_rag(query: str):
    return rag_chain.invoke(query)

rails.register_action(ask_rag, name="ask_rag")

# ---------------------------------------------------------
# Interactive Execution Loop
# ---------------------------------------------------------
# 1. Interactive Question-Answering Loop
print("\n--- PDF Chatbot Ready with Modular NeMo Guardrails! Type 'exit' to quit. ---\n")

while True:
    user_query = input("You: ").strip()
    
    if user_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    if not user_query.strip():
        continue

    print("\nThinking...")

    rag_response = rag_chain.invoke(user_query)
    print(f"\nAssistant: {rag_response}")
    print("-" * 50 + "\n")