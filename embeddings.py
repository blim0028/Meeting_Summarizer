import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

# File paths
DATA_PATH = "data"
CHROMA_DB_PATH = "chroma_db"

load_dotenv()
print("API Key loaded:", os.getenv("OPENAI_API_KEY") is not None)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n'],
    chunk_size = 1200,
    chunk_overlap = 200
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

all_summaries = []
for file in os.listdir(DATA_PATH):
    if file.endswith('.txt'):
        file_path = os.path.join(DATA_PATH, file)
        with open(file=file_path, mode='r', encoding='utf-8') as f:
            transcript = f.read()
        
        doc = Document(page_content=transcript, metadata={"source": file})
        summary = summarize_chain.invoke([doc])
        print(f"✅ Summary complete for {file}")

        summary_docs = text_splitter.create_documents(
            [summary],
            metadatas=[{'source': file}]
        )
        
        all_summaries.extend(summary_docs)
        
print(f"\n✅ Total summarized chunks ready for embedding: {len(all_summaries)}")

db = Chroma.from_documents(documents=all_summaries, embedding=embeddings, collection_name='meeting_summaries', persist_directory=CHROMA_DB_PATH)

print(f"Chroma DB created and saved at '{CHROMA_DB_PATH}'")