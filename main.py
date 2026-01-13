from fastapi import FastAPI, HTTPException
import os
import sqlite3
from typing import Annotated, Literal, Sequence
from datetime import datetime

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage,ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict
import uvicorn

# ─── 1. Environment ───────────────────────────────────────────────────────────
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# ─── 2. LLM ───────────────────────────────────────────────────────────────────
model = init_chat_model(
    "gemini-2.5-flash",          # or gemini-1.5-pro / gemini-2.0-flash...
    model_provider="google_genai",
    temperature=0.1
)

# ─── 3. Database ──────────────────────────────────────────────────────────────
# Choose ONE connection - comment out the one you're not using

# For development / testing
# db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Production / your real database
db = SQLDatabase.from_uri(
    "postgresql://migr_user:POST##2026@192.168.0.101:5432/pgsql_mig"
)

print("Database dialect:", db.dialect)
print("Available tables:", db.get_usable_table_names())

# ─── 4. Tools ─────────────────────────────────────────────────────────────────
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# ─── 5. State ─────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ─── 6. System Prompt ─────────────────────────────────────────────────────────
system_prompt = """\
You are a senior financial data analyst for sysarc infomatix company, specialized in project accounting, milestone billing, collections and receivables analysis.

Important Accounting Context - AFDA:
In our company, "AFDA" stands for **Allowance for Doubtful Accounts** (also known as Allowance for Uncollectible Accounts or Bad Debt Reserve).
It is a contra-asset account used to estimate the portion of receivables that are likely to become bad debts.

Important Accounting Context - Accounts Receivable:
In our company, "AR" stands for **Accounts Receivable**.


Current company practice for identifying potentially doubtful invoices (as of January 2026):
→ An invoice is considered **potentially doubtful** if BOTH conditions are met:
  1. spm_invoice_status indicates the invoice was issued ('invoiced', 'issued', 'sent', etc.)
  2. Invoice is older than 365 days
     → CURRENT_DATE - spm_invoice_date >= 365 days AND spm_collected_value == 0 


Most important table for this analysis:
• spms_po_master
  Key columns for AFDA / receivables and AR analysis:
  - spm_cost_code
  - spm_client_name
  - spm_po_number

• spms_po_milestones
  Key columns for AFDA / receivables and AR analysis:
  - spm_invoice_status
  - spm_invoice_date
  - spm_invoice_value
  - spm_collected_value
  - spm_deducted_value
  - spm_total_value
  - spm_realization_date
  - spm_dispute_flag
  - spm_po_code
  - spm_cost_code
  - spm_milestone_name
  - spm_invoice_no

Guidelines:
1. When user asks about "AFDA", "doubtful accounts", "bad debts", "allowance for doubtful", "uncollectible" → they most likely want old outstanding invoices (>365 days)
2. Preferred output structure:
   - First: summary statistics (count, total outstanding value, maybe aging buckets)
   - Then (if reasonable number): list of relevant records
   - Show: invoice date, age in days, invoice value, collected so far, outstanding, PO/cost code
3. Be very careful with date calculations — use CURRENT_DATE / NOW() correctly
4. Round monetary values sensibly
5. Be professional and use correct financial terminology

Answer concisely. Use tables or clear formatting when showing multiple records.

You are preparing short executive summaries for VT360 senior management.

For AR / Account Receivables report:
- Only include invoices that are issued but not fully collected
- Report only business meaning — never explain the filtering logic

Rules for every answer:

1. Be concise — maximum 4–6 lines when possible
2. Use everyday business language only
3. Never mention: tables, columns, SQL, database, query, records, rows
4. Never write sentences like:
   - "According to the spms_po_milestones table..."
   - "The query returned..."
   - "No records were found in..."
5. When no doubtful accounts exist, prefer these styles:
   - "No AFDA items currently"
   - "No invoices older than 1 year remain outstanding"
   - "All older receivables have been collected"
6. When showing receivables / AR / outstanding items, prefer clean table format with these columns only:
   - PO Code
   - Invoice No
   - Invoice Date
   - Invoice Value (₹)
   - Cost Code
   - Client Name

7. When showing Accounts Receivables / AR / outstanding items,Always add a very short summary line at the bottom:
   • Total outstanding amount
   • Number of pending invoices
   • Oldest invoice age (in days or approximate month)

8.When nothing is found in Accounts Receivables / AR / outstanding items, use simple statements like:
   - "No outstanding invoices at this time"
   - "All invoiced amounts have been collected"
   - "Currently no pending receivables"

9. Focus on business impact: amounts, counts, status, trends

10.Round money amounts sensibly (nearest thousand / lakh when appropriate)

11.Use ₹ symbol for Indian Rupees

12.Client names are important — always include them when available.
    - use the value from spms_po_master.spm_client_name column where spms_po_milestones.spm_cost_code =  spms_po_master.spm_cost_code

Keep answers professional, calm and to-the-point.

"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

llm_with_tools = model.bind_tools(tools)

# ─── 7. Agent node ────────────────────────────────────────────────────────────
def agent(state: AgentState, config: RunnableConfig):
    chain = prompt | llm_with_tools
    response = chain.invoke(state, config)
    return {"messages": [response]}

# ─── 8. Tool node ─────────────────────────────────────────────────────────────
tool_node = ToolNode(tools)

# ─── 9. Router ────────────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]


    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"

# ─── 10. Graph ────────────────────────────────────────────────────────────────
graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue)
graph_builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# ─── 11. FastAPI ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="VT360 SQL Agent API",
    description="Natural language questions about VT360 financial/project data",
    version="1.1.0"
)

class QueryRequest(BaseModel):
    question: str
    thread_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    thread_id: str

def is_afda_related_question(question: str) -> bool:
    text = question.lower()
    afda_keywords = [
        "afda", "doubtful", "bad debt", "uncollectible", "allowance for doubtful",
        "allowance for uncollectible", "bad debts reserve"
    ]
    return any(kw in text for kw in afda_keywords)

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
# set recursion limit higher for complex graphs
    config = {"configurable": {"thread_id": request.thread_id ,},"recursion_limit": 50,}

    try:
        # Special fast-path for AFDA related questions
        # if is_afda_related_question(request.question):
        #     today = datetime.now().strftime("%Y-%m-%d")
        #     hint = (
        #         f"Current date is {today}. "
        #         "Use spms_po_milestones table. "
        #         "Consider invoice doubtful only if: "
        #         "spm_invoice_status indicates issued invoice AND "
        #         "CURRENT_DATE - spm_invoice_date >= 365 days. "
        #         "Focus on outstanding amount (invoice_value - collected_value)."
        #     )
        #     user_question = f"{request.question}\n\n{hint}"
        # else:
        #     user_question = request.question
        user_question = request.question
        final_answer = None

        for event in graph.stream(
            {"messages": [HumanMessage(content=user_question)]},
            config,
            stream_mode="values"
        ):
            if "messages" not in event:
                continue

            last_msg = event["messages"][-1]

            if isinstance(last_msg, ToolMessage):
                print(f"Last message is a ToolMessage with tool calls. {last_msg.pretty_print()}")

            if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                # ── Modern langchain-google-genai behavior (2025+) ──
                if isinstance(last_msg.content, str):
                    final_answer = last_msg.content.strip()
                elif isinstance(last_msg.content, list):
                    # Multimodal / mixed content fallback
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
                    print("Final answer extracted:", final_answer[:], "...")
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
    return {"message": "VT360 SQL Agent API is running. POST to /query"}

if __name__ == "__main__":
    # For development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # For production consider using:
    # serve(app, host="0.0.0.0", port=8000)