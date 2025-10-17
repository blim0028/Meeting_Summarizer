import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains.summarize.chain import load_summarize_chain

# Main page
st.title("Meeting Summarizer")

query = st.text_input("Ask a question or set a task for past meetings")

response = st.button("Enter")

if response and query:
    st.write("HIHI")