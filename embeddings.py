import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# File paths
DATA_PATH = "data"
CHROMA_DB_PATH = "chroma_db"

documents = []
for file in os.listdir(DATA_PATH):
    if file.endswith('.txt'):
        file_path = os.path.join(DATA_PATH, file)
        with open(file=file_path, mode='r', encoding='utf-8') as x:
            transcript = x.read()
        documents.append(Document(page_content=transcript, metadata={"source":file}))

print(f"Loaded {len(documents)} meeting transcripts")

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n'],
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = text_splitter.split_documents(documents=documents)

print(f"The length of chunks is {len(chunks)}")