import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# File paths
DATA_PATH = "data"
CHROMA_DB_PATH = "chroma_db"

# API Key
load_dotenv()
print("API Key loaded:", os.getenv("OPENAI_API_KEY") is not None)

# Embedding
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n'],
    chunk_size = 1200,
    chunk_overlap = 200
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

documents = []
for file in os.listdir(DATA_PATH):
    if file.endswith('.txt'):
        file_path = os.path.join(DATA_PATH, file)
        with open(file=file_path, mode='r', encoding='utf-8') as f:
            transcript = f.read()
    

        summary_docs = text_splitter.create_documents(
            [transcript],
            metadatas=[{'source': file}]
        )
        
        documents.extend(summary_docs)
        print(summary_docs)
print(f"\n✅ Total summarized chunks ready for embedding: {len(documents)}")

# Save embeddings in ChromaDB
db = Chroma.from_documents(documents=documents, embedding=embeddings, collection_name='meeting_summaries', persist_directory=CHROMA_DB_PATH)
db.persist()

print(f"Chroma DB created and saved at '{CHROMA_DB_PATH}'")