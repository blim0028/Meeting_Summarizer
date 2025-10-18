import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.agents import initialize_agent, Tool

load_dotenv()

CHROMA_DB_PATH = "chroma_db"

# Main page
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k":3})

st.title("Meeting Summarizer")

query = st.text_input("Ask a question or set a task for past meetings")

response = st.button("Enter")

if response and query:
    st.write("HIHI")
    

    # retriever = db.as_retriever(search_kwargs={"k": 3})

    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    # answer = qa_chain.run(query)

    # st.subheader("💡 Answer")
    # st.write(answer)