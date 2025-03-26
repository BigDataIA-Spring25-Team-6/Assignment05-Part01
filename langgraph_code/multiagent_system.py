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
import re
from data_prep.pinecone_rag import QUARTER_REGEX, YEAR_REGEX, embed_texts, get_or_create_index

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
    """
    Search for general knowledge using Tavily API and return the top 5 results.
    """
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )
    return "\n---\n".join([
        "\n".join([res["title"], res["content"], res["url"]])
        for res in response["results"]
    ])

def format_rag_contents(matches: list):
    contexts = []
    for x in matches:
        text=(
            f"Source: {x['metadata']['source']}\n"
            f"Quarter: {x['metadata']['quarter']}\n"
            f"Year: {x['metadata']['year']}\n"
        )
        contexts.append(text)
    return "\n---\n".join(contexts)


@tool("rag_search_filter")
def rag_search_filter(query: str, filters: dict):
    """Finds information from our database using a natural language query
    and specific metadata filters."""
    index = get_or_create_index()
    query_embedding = embed_texts([query])[0]
    
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        filter=filters  
    )
    threshold = 0.4
    filtered_matches = [match for match in results.get("matches", []) if match.get("score", 0) >= threshold]
    print(f"results in rag_search_filter: {filtered_matches}")
    return format_rag_contents(filtered_matches)

@tool("rag_search")
def rag_search(query: str):
    """Finds information from our database using a natural language query."""
    index = get_or_create_index()
    query_embedding = embed_texts([query])[0]
    
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True
    )
    threshold = 0.4
    filtered_matches = [match for match in results.get("matches", []) if match.get("score", 0) >= threshold]
    print(f"results in rag_search: {filtered_matches}")

    return format_rag_contents(filtered_matches)

def construct_filters_from_query(query:str):
    """Construct metadata filters dynamically based on user query."""
    filters = []
    quarter_match = re.search(QUARTER_REGEX, query)
    year_match = re.search(YEAR_REGEX, query)
    
    if quarter_match:
        quarter = quarter_match.group(0).upper()
        filters.append({"quarter": {"$eq": quarter}})

    if year_match:
        year = year_match.group(0)
        filters.append({"year": {"$eq": year}})
    
    if len(filters)>1:
        print({"$and": filters})
        return {"$and": filters}
    elif len(filters)==1:
        print(filters[0])
        return filters[0]
    else:
        return {}

    



# Define final_answer tool
@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str
) -> str:
    """
    Formats a detailed research report with introduction, steps, findings, conclusion, and sources.
    """
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
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

# Prompt template
system_prompt = """You are the oracle, the great AI decision maker.
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

If you see that a tool has been used (in the scratchpad) with a particular
query, do NOT use that same tool with the same query again. Also, do NOT use
any tool more than twice (ie, if the tool appears in the scratchpad twice, do
not use it again).

You should aim to collect information from a diverse range of sources before
providing the answer to the user. Once you have collected plenty of information
to answer the user's question (stored in the scratchpad) use the final_answer
tool."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])

tools = [rag_search, rag_search_filter, web_search, final_answer]

def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            # this was the ToolExecution
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)

oracle = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(x["intermediate_steps"]),
        "filters": lambda x: construct_filters_from_query(x["input"])
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)

# Router & executor
def run_oracle(state: list):
    print("run_oracle")
    print(f"intermediate_steps: {state['intermediate_steps']}")
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )
    return {
        "intermediate_steps": [action_out]
    }

def router(state: list):
    # return the tool name to use
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    else:
        # if we output bad format go to final answer
        print("Router invalid format")
        return "final_answer"

tool_str_to_func = {
    "web_search": web_search,
    "rag_search": rag_search,
    "rag_search_filter": rag_search_filter,
    "final_answer": final_answer
}

def run_tool(state: list):
    # use this as helper function so we repeat less code
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input

    if tool_name == "rag_search_filter":
        tool_args["filters"] = construct_filters_from_query(state["input"])

    print(f"{tool_name}.invoke(input={tool_args})")
    # run tool
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )
    return {"intermediate_steps": [action_out]}

# Build LangGraph
graph = StateGraph(AgentState)

graph.add_node("oracle", run_oracle)
graph.add_node("web_search", run_tool)
graph.add_node("rag_search", run_tool)
graph.add_node("rag_search_filter", run_tool)
graph.add_node("final_answer", run_tool)

graph.set_entry_point("oracle")

graph.add_conditional_edges("oracle", path=router)

for tool in tools:
    if tool.name != "final_answer":
        graph.add_edge(tool.name, "oracle")

graph.add_edge("final_answer", END)

runnable = graph.compile()

# Optional: visualize
#Image(runnable.get_graph().draw_png())

# Test it
# state = {
#     "input": "What is the revenue details of NVIDIA?",
#     "chat_history": []
# }

state = {
    "input": "What NBA team is Stephen Curry in?",
    "chat_history": []
}

# state = {
#     "input": "What is the operating lease expense in Q1 2023 of NVIDIA?",
#     "chat_history": []
# }

result = runnable.invoke(state)
#print("\nFINAL RESULT:\n", result)
def build_report(output: dict):
    research_steps = output["research_steps"]
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    sources = output["sources"]
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    return f"""
INTRODUCTION
------------
{output["introduction"]}

RESEARCH STEPS
--------------
{research_steps}

REPORT
------
{output["main_body"]}

CONCLUSION
----------
{output["conclusion"]}

SOURCES
-------
{sources}
"""

print(build_report(output=result["intermediate_steps"][-1].tool_input))