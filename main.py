import os
from dotenv import load_dotenv
import streamlit as st
from graph import app
import asyncio

st.title("Meeting Summarizer")
st.subheader("Ask questions about meetings or create Trello tasks automatically.")

query = st.text_input(
    "Ask a question or set a task for past meetings",
    placeholder= "e.g. What tasks were discussed in meeting 3?"
)

response = st.button("Enter")

if response and query:
    with st.spinner("Processing...."):
        result = asyncio.run(app.ainvoke({"query": query}))
    st.success("Done")
    st.subheader("Response / Tasks")
    st.text(result["final_response"])
