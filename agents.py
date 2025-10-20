import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.prompts import ChatPromptTemplate
import requests

load_dotenv()

# File path
CHROMA_DB_PATH = "chroma_db"
TRELLO_KEY = os.getenv("TRELLO_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
TRELLO_BOARD_ID = os.getenv("TRELLO_BOARD_ID")

# LLM-RAG chain
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings, collection_name="meeting_summaries")
retriever = db.as_retriever(search_kwargs={"k": 4})

system_prompt = (
    "You are an AI meeting assistant. Use the context below to summarize discussions or answer questions.\n"
    "Include any clear action items if they exist. If unsure, say 'I don't know.'\n"
    "Keep responses clear, concise, and context-based.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

rag_chain = (
    RunnableParallel({
        "context": retriever,
        "input": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

# Trello task creation
def create_trello_task(name, desc):
    # Get lists in board
    list_url = f"https://api.trello.com/1/boards/{TRELLO_BOARD_ID}/lists"
    list_query = {
        "key" : TRELLO_KEY,
        "token" : TRELLO_TOKEN
    }
    lists_response = requests.get(list_url, params=list_query)
    lists = lists_response.json()

    if not lists:
        raise Exception("No lists found in this Trello board")
    
    todo_list = None
    for lst in lists:
        if lst["name"].lower() == "to do":
            todo_list = lst
            break
    if todo_list is None:
        todo_list = lists[0]

    # Create new Trello card in "To Do" list
    card_url = "https://api.trello.com/1/cards"
    card_query = {
        "idList": todo_list["id"],
        "name": name,
        "desc": desc,
        "key": TRELLO_KEY,
        "token": TRELLO_TOKEN,
    }

    response = requests.post(card_url, params= card_query)

    if response.status_code == 200:
        print(f"Task : {name} created successfully")
        return f"Task : {name} created successfully"
    else:
        print(f"Failed to create task : {response.text}")
        return f"Failed to create task : {response.text}"
