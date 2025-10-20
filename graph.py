from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents import llm, rag_chain, create_trello_task
import json
import asyncio

class GraphState(TypedDict):
    query: str
    rag_response: str
    final_response: str
    task_created: list[str]

async def rag_node(state: GraphState):
    query = state["query"]
    result = await rag_chain.ainvoke(query)
    print("RAG NODE")
    return {"rag_response": result}

async def task_decision_node(state: GraphState):
    rag_response = state["rag_response"]

    decision_prompt = (
        f"Analyze this assistant response:\n\n'{rag_response}'\n\n"
        "Does this response include any clear action items or tasks "
        "that should be created in Trello? Answer strictly 'yes' or 'no'."
    )
    decision = (await llm.ainvoke(decision_prompt)).content.lower()

    if "yes" in decision:
        print("Create task in task detection node")
        return {"decision": "create_task"}  
    else:
        print("No task in task detection node")
        return {"decision": "no_task", "task_created": []}

async def trello_node(state: GraphState):
    extract_prompt = (
       f"Extract all action items from this text:\n{state['rag_response']}\n\n"
        "Return them as a JSON list in this format: "
        '[{"name": "Task title", "desc": "Task details"}]. '
        "Return [] if no tasks found."
    )
    tasks_json = (await llm.ainvoke(extract_prompt)).content
    print(tasks_json)

    cleaned = tasks_json.strip().strip("`")
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        tasks = json.loads(cleaned)
    except:
        tasks = []

    results = []
    if tasks:
        for t in tasks:
            result = await asyncio.to_thread(create_trello_task, t["name"], t["desc"])
            results.append(result)

        print("Trello task creation node")

        return {
            "task_created": results,
            "final_response": f"{state['rag_response']}"
        }
    else:
        print("No tasks node")
        return {
            "task_created": [],
            "final_response": state["rag_response"]
        } 

def end_node(state: GraphState):
    print("END NODE")
    return {
            "final_response": state["rag_response"],
            "task_created" : state.get("task_created", [])
        }

# Build workflow
workflow = StateGraph(GraphState)
workflow.add_node("RAG", rag_node)
workflow.add_node("Decide", task_decision_node)
workflow.add_node("CreateTask", trello_node)
workflow.add_node("End", end_node)

# Start and end point of graph
workflow.set_entry_point("RAG")
workflow.add_edge("RAG", "Decide")
workflow.add_conditional_edges(
    "Decide",
    lambda state: state["decision"] if "decision" in state else "no_task",
    {"create_task": "CreateTask", "no_task": "End"}
)
workflow.add_edge("CreateTask", "End")
workflow.add_edge("End", END)

app = workflow.compile()