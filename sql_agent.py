from http.client import HTTPException
import os
import sqlite3
from typing import Annotated, Literal, Sequence

from dotenv import load_dotenv  # NEW: Load environment variables from .env

from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict
import uvicorn

# --- 1. Load environment variables ---
load_dotenv()  # Loads .env file into os.environ

# Optional: Verify the key is loaded (remove in production)
# if not os.getenv("GEMINI_API_KEY"):
#     raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")

# --- 2. Set up Google Gemini LLM ---
# Uses GOOGLE_API_KEY from environment (loaded via dotenv)
#model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)
# model = ChatOllama(model="llama3", max_retries=3)
model = ChatOllama(
    model="llama3.2:latest",   # Recommended: fast + good tool calling
    temperature=0,             # For deterministic SQL generation
)
# --- 3. Set up the database ---
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print("Database dialect:", db.dialect)
print("Available tables:", db.get_usable_table_names())

# --- 4. Create tools ---
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# --- 5. Define agent state ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- 6. Prompt template ---
system_prompt = """
You are a data analyst expert using a SQL database.
Answer the user's question by generating and executing SQL queries when needed.

Guidelines:
- Use the provided tools to get schema, list tables, check queries, and execute queries.
- Always check your query with the query checker tool before executing.
- Only execute queries that are necessary to answer the question.
- Be concise in your final answer.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

# Bind tools to the model
llm_with_tools = model.bind_tools(tools)

# --- 7. Agent node ---
def agent(state: AgentState, config: RunnableConfig):
    chain = prompt | llm_with_tools
    response = chain.invoke(state, config)
    return {"messages": [response]}

# --- 8. Tool node ---
tool_node = ToolNode(tools)

# --- 9. Routing logic ---
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"

# --- 10. Build the graph ---
graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue)
graph_builder.add_edge("tools", "agent")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

#---- 11 . Fast API Integration ----#

app = FastAPI(
    title="Chinook SQL Agent API",
    description="Ask natural language questions about the Chinook music database",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str
    thread_id: str = "default"  # Optional: for conversation history

class QueryResponse(BaseModel):
    answer: str
    thread_id: str

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        final_answer = None
        # Stream through the graph to get the final response
        for event in graph.stream(
            {"messages": [HumanMessage(content=request.question)]},
            config,
            stream_mode="values"
        ):
            messages = event.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                    final_answer = last_msg.content

        if final_answer is None:
            final_answer = "Sorry, I couldn't generate a response."

        return QueryResponse(answer=final_answer, thread_id=request.thread_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Chinook SQL Agent API is running. POST to /query with {'question': 'your question'}"}

# --- Run with: uvicorn main:app --reload ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    
# # --- 11. Run the agent ---
# def run_query(question: str, thread_id: str = "default"):
#     print(f"Question: {question}\n")
#     config = {"configurable": {"thread_id": thread_id}}
#     events = graph.stream(
#         {"messages": [HumanMessage(content=question)]},
#         config,
#         stream_mode="values"
#     )
#     for event in events:
#         if "messages" in event:
#             event["messages"][-1].pretty_print()

# if __name__ == "__main__":
#     print("SQL Agent with Google Gemini ready! (Type 'quit' to exit)\n")
#     while True:
#         user_input = input("Enter a question: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break
#         run_query(user_input)
