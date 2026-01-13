from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from datetime import date

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

# ─── Load environment ────────────────────────────────────────────────────────
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# ─── LLM ─────────────────────────────────────────────────────────────────────
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.1
)

# ─── Database ────────────────────────────────────────────────────────────────
db = SQLDatabase.from_uri(
    "postgresql://migr_user:POST##2026@192.168.0.101:5432/pgsql_mig"
)

print("Database dialect:", db.dialect)
print("Available tables:", db.get_usable_table_names())

# ─── Tools ───────────────────────────────────────────────────────────────────
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# ─── State ───────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ─── Base system prompt (without date) ───────────────────────────────────────
base_system_prompt = """\
You are a senior financial analyst preparing clear, concise executive summaries 
for SysArc Infomatix / VT360 senior management.

Your answers must always be professional, calm, business-oriented and easy to read.

─────────────────────────────  CORE RULES ──────────────────────────────────────

1. Be concise — aim for 4–8 lines maximum when possible (table + summary)
2. Use everyday business language — never technical jargon
3. NEVER mention in the final answer:
   • table names
   • column names
   • SQL
   • database
   • query
   • records / rows / data source
   • phrases like "according to the data", "query shows", "no records found in"
4. Use ₹ symbol for all monetary values
5. Round amounts sensibly (nearest thousand / lakh / crore when appropriate)
6. Always include Client Name when relevant (sourced from spms_po_master)

─────────────────────────────  REPORT TYPES & EXPECTED PATTERNS ─────────────────

A. Standard AR Report / Outstanding Invoices / Client-wise AR
   When user asks: "AR Report", "outstanding", "client wise AR", "receivables", etc.
   
   Show detailed list with these columns (preferred order):
   PO Code | Invoice No | Invoice Date | Invoice Value (₹) | Cost Code | Milestone Name | Client Name | Outstanding (₹)

   Conditions (internally):
   - Invoice has been issued (status NOT in 'Collected', 'Draft', 'Cancelled', etc.)
   - Outstanding amount > 0 (invoice_value > collected_value)

   Always add short summary:
   • Total outstanding amount
   • Number of pending invoices
   • Oldest invoice age

B. Age-wise AR / AR Ageing Report
   When user asks: "age wise AR", "AR age wise", "ageing report", "AR aging", "aged receivables", etc.
   
   Show grouped summary table with exactly these columns:
   Client Name | below 30 | above 30 | above 60 | above 90 | above 120 | above 180 | above 365

   Ageing bucket rules (very important – follow exactly):
   • Age = CURRENT_DATE - invoice_date   (direct date subtraction → number of days)
   • below 30    = 0–30 days
   • above 30    = 31–60 days
   • above 60    = 61–90 days
   • above 90    = 91–120 days
   • above 120   = 121–180 days
   • above 180   = 181–365 days
   • above 365   = >365 days

   Requirements:
   - Only positive outstanding amounts
   - Include **Grand Total** row
   - After table: 1–2 line summary (total outstanding, number of clients with overdue, largest bucket)

C. AFDA / Allowance for Doubtful Accounts
   When user asks: "AFDA", "doubtful accounts", "bad debts", "allowance for doubtful", "uncollectible", etc.
   
   Current company guideline (Jan 2026):
   Invoice is considered potentially doubtful when:
   • Invoice status indicates issued
   • Age ≥ 365 days (CURRENT_DATE - invoice_date >= 365)
   • AND still has outstanding balance (collected_value < invoice_value)

   Output style:
   - First: summary count + total doubtful value
   - Then: list if reasonable number (Invoice Date, Age, Value, Outstanding, PO Code, Client)
   - Or simple statement if nothing found:
     • "No AFDA items currently"
     • "No invoices older than 1 year remain outstanding"

─────────────────────────────  WHEN NOTHING IS FOUND ─────────────────────────────

Use natural, calm statements:
• "No Data Found at this time. Please verify the criteria or try again with different parameters."

─────────────────────────────  FINAL NOTES ─────────────────────────────────────

• Keep language executive-friendly and focused on business impact
• Never explain filtering logic, calculations or data sources in the answer
"""

# ─── Prompt template with placeholder ────────────────────────────────────────
prompt_template = ChatPromptTemplate.from_messages([
    ("system", base_system_prompt + "\n\nCurrent date for all calculations and reports: {current_date}"),
    MessagesPlaceholder(variable_name="messages"),
])

llm_with_tools = model.bind_tools(tools)

# ─── Agent node ──────────────────────────────────────────────────────────────
def agent(state: AgentState, config: RunnableConfig):
    # Get current date each time the agent is called
    today_str = date.today().strftime("%Y-%m-%d")  # or "%d %B %Y"

    chain = prompt_template | llm_with_tools
    response = chain.invoke({
        "messages": state["messages"],
        "current_date": today_str
    })
    return {"messages": [response]}

# ─── Tool node & router ──────────────────────────────────────────────────────
tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"

# ─── Build graph ─────────────────────────────────────────────────────────────
graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue)
graph_builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# ─── FastAPI ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VT360 / SysArc Financial SQL Agent",
    description="Natural language questions about financial & project data",
    version="1.2.0"
)

class QueryRequest(BaseModel):
    question: str
    thread_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    thread_id: str

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    config = {"configurable": {"thread_id": request.thread_id},"recursion_limit": 50}

    try:
        final_answer = None

        for event in graph.stream(
            {"messages": [HumanMessage(content=request.question)]},
            config,
            stream_mode="values"
        ):
            if "messages" not in event:
                continue

            last_msg = event["messages"][-1]
            if isinstance(last_msg, ToolMessage):
                print(f"Last message was a tool call.{last_msg.pretty_print()}")

            if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                if isinstance(last_msg.content, str):
                    final_answer = last_msg.content.strip()
                elif isinstance(last_msg.content, list):
                    texts = []
                    for item in last_msg.content:
                        if isinstance(item, str):
                            texts.append(item)
                        elif isinstance(item, dict) and item.get("type") == "text":
                            texts.append(item.get("text", ""))
                    final_answer = " ".join(filter(None, texts)).strip()
                else:
                    final_answer = str(last_msg.content).strip()

                if final_answer:
                    print("Final answer:", final_answer[:120], "...")
                    break

        if final_answer is None:
            final_answer = "Sorry, I couldn't generate a proper final answer."

        return QueryResponse(answer=final_answer, thread_id=request.thread_id)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Financial SQL Agent API is running. POST to /query"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)