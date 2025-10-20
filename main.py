import os
import streamlit as st
from graph import app
import asyncio

# File path
DATA_PATH = "data"

st.title("Meeting Summarizer")
st.subheader("Ask questions about meetings or create Trello tasks automatically.")

# All documents
st.sidebar.title("Documents stored:")
for file in os.listdir(DATA_PATH):
    if file.endswith('.txt'):
        st.sidebar.write(file)

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
    st.markdown(result["final_response"])
    for task in result["task_created"]:
        st.success(task)
