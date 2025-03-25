from typing import TypedDict, Annotated
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tavily import TavilyClient
from IPython.display import Image
import operator
import os

# Load environment variables
load_dotenv()

# Define LangChain agent state
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[AgentAction], operator.add]

# Set up Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=tavily_api_key)

# Define web_search tool
@tool("web_search")
def web_search(query: str) -> str:
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )
    return "\n---\n".join([
        "\n".join([res["title"], res["content"], res["url"]])
        for res in response["results"]
    ])

# Define final_answer tool
@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str
) -> str:
    if isinstance(research_steps, list):
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if isinstance(sources, list):
        sources = "\n".join([f"- {s}" for s in sources])
    return f"""## Introduction
{introduction}

## Research Steps
{research_steps}

## Main Body
{main_body}

## Conclusion
{conclusion}

## Sources
{sources}
"""

# Set up the model
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0
)

# Prompt template
system_prompt = """You are the oracle, the great AI decision maker.
... [shortened here for clarity, keep the full one in your file]
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])

tools = [web_search, final_answer]

def create_scratchpad(intermediate_steps: list[AgentAction]):
    return "\n---\n".join([
        f"Tool: {a.tool}, input: {a.tool_input}\nOutput: {a.log}"
        for a in intermediate_steps if a.log != "TBD"
    ])

oracle = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(x["intermediate_steps"])
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)

# Router & executor
def run_oracle(state: AgentState):
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    return {
        "intermediate_steps": [AgentAction(tool=tool_name, tool_input=tool_args, log="TBD")]
    }

def router(state: AgentState):
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    return "final_answer"

tool_str_to_func = {
    "web_search": web_search,
    "final_answer": final_answer
}

def run_tool(state: AgentState):
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    out = tool_str_to_func[tool_name].invoke(tool_args)
    return {
        "intermediate_steps": [
            AgentAction(tool=tool_name, tool_input=tool_args, log=str(out))
        ]
    }

# Build LangGraph
graph = StateGraph(AgentState)

graph.add_node("oracle", run_oracle)
graph.add_node("web_search", run_tool)
graph.add_node("final_answer", run_tool)

graph.set_entry_point("oracle")

graph.add_conditional_edges("oracle", path=router)

for tool in tools:
    if tool.name != "final_answer":
        graph.add_edge(tool.name, "oracle")

graph.add_edge("final_answer", END)

runnable = graph.compile()

# Optional: visualize
Image(runnable.get_graph().draw_png())

# Test it
state = {
    "input": "What is the impact of NVIDIA's Q1 2025 earnings on the AI industry?",
    "chat_history": [],
    "intermediate_steps": []
}

result = runnable.invoke(state)
print("\nFINAL RESULT:\n", result)
