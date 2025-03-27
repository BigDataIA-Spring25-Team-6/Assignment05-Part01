import operator
import os
from typing import TypedDict, Annotated
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from tavily import TavilyClient
from IPython.display import Image
from matplotlib import pyplot as plt
import pandas as pd
import snowflake.connector
import re
from data_prep.pinecone_rag import QUARTER_REGEX, YEAR_REGEX, embed_texts, get_or_create_index
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define LangChain agent state
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# Define web_search tool
tavily_api_key = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=tavily_api_key)

@tool("web_search")
def web_search(query: str):
    """Finds general knowledge information using Google search. Can also be used
    to augment more 'general' knowledge to a previous specialist query."""
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )
    results = response["results"]
    contexts = "\n---\n".join(
        ["\n".join([x["title"], x["content"], x["url"]]) for x in results]
    )
    return contexts

def format_rag_contents(matches: list):
    """Formats the RAG search results into a readable format."""
    contexts = []
    for x in matches:
        text=(
            f"Source: {x['metadata']['text']}\n"
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
 
# Define snowflake_agent tool
@tool("snowflake_agent")
def snowflake_agent(query: str, year: str = None, quarter: str = None) -> str:
    """
    Retrieves NVIDIA valuation data from Snowflake using natural language + filters.
    Returns a GPT-generated summary and a saved chart path.
    """
    import json
    import uuid

    # Step 1: Use GPT to generate SQL in structured format
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    system_prompt = """
    You are a Snowflake SQL expert. Given a natural language question about NVIDIA valuation metrics,
    return a JSON object with two fields:
    - "sql": a valid SQL SELECT query on table 'quarterly_valuation_metrics'
    - "explanation": a short explanation of what this query does

    Supported columns include:
    period_end_date, trailing_pe, forward_pe, peg_ratio, market_cap,
    enterprise_value, price_to_sales, price_to_book, ev_to_ebitda,
    scraped_at, symbol

    Always filter to SYMBOL = 'NVDA'.
    NEVER use backticks or double quotes around identifiers.
    Return raw JSON only, without markdown or code block formatting.
    """

    # Map quarter name to number (Q1 = Jan-Mar)
    quarter_mapping = {
        "Q1": (1, 3),
        "Q2": (4, 6),
        "Q3": (7, 9),
        "Q4": (10, 12)
    }

    user_input = query
    if year:
        user_input += f" Filter by year {year}."
    if quarter:
        q = quarter.upper().strip()
        if q in quarter_mapping:
            start_month, end_month = quarter_mapping[q]
            user_input += f" Filter for months between {start_month} and {end_month}."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    response = llm.invoke(messages).content.strip()

    # Handle possible code block formatting from GPT
    if response.startswith("```json"):
        response = response.strip("` \n")
        response = response.replace("json", "", 1).strip()

    try:
        parsed = json.loads(response)
        sql = parsed.get("sql", "").strip()
        explanation = parsed.get("explanation", "")
        print("GPT Explanation:", explanation)
        print("Generated SQL:\n", sql)
    except Exception as e:
        return f"Failed to parse GPT response as JSON.\n\nRaw output:\n{response}"

    # Step 2: Run the SQL on Snowflake
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        cursor.close()
    except Exception as e:
        return f"SQL execution failed: {str(e)}\n\n SQL:\n{sql}"

    if df.empty or len(df.columns) < 2:
        return "No data returned. Try refining the filters or query."

    x_col = df.columns[0]
    y_col = df.columns[1]
    df[x_col] = pd.to_datetime(df[x_col])

    # Step 3: Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[x_col], df[y_col], marker='o')
    ax.set_title(f"NVIDIA {y_col.replace('_', ' ').title()} Trend")
    ax.set_xlabel("Quarter")
    ax.set_ylabel(y_col)
    ax.grid(True)

    chart_filename = f"chart_{uuid.uuid4().hex}.png"
    plt.tight_layout()
    plt.savefig(chart_filename, format="png")
    plt.close()

    # Step 4: Summary using GPT
    summary_prompt = f"""
    Provide a concise but insightful summary of the following NVIDIA valuation data.
    Focus on trends, peaks, and patterns.

    Column: {y_col.replace('_', ' ').title()}
    Data:
    {df[[x_col, y_col]].to_string(index=False)}
    """

    summary = llm.invoke(summary_prompt).content.strip()

    return summary + f"\n\n Chart saved to: {chart_filename}"

# Define final_answer tool
@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str
):
    """Returns a natural language response to the user in the form of a research
    report. There are several sections to this report, those are:
    - `introduction`: a short paragraph introducing the user's question and the
    topic we are researching.
    - `research_steps`: a few bullet points explaining the steps that were taken
    to research your report.
    - `main_body`: this is where the bulk of high quality and concise
    information that answers the user's question belongs. It is 3-4 paragraphs
    long in length.
    - `conclusion`: this is a short single paragraph conclusion providing a
    concise but sophisticated view on what was found.
    - `sources`: a bulletpoint list provided detailed sources for all information
    referenced during the research process
    """
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    return ""

# Nodes for the graph

# System prompt for the oracle
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

# Runnable pipline for the oracle
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0
)

tools=[
    rag_search,
    rag_search_filter,
    snowflake_agent,
    web_search,
    final_answer
]

# define a function to transform intermediate_steps from list
# of AgentAction to scratchpad string
def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            output = str(action.log)
            if "data:image/png;base64" in output:
                output = output.split("data:image/png;base64")[0] + "[chart image omitted]"
            elif len(output) > 2000:
                output = output[:2000] + "... [truncated]"
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
        "scratchpad": lambda x: create_scratchpad(
            intermediate_steps=x["intermediate_steps"]
        ),
        "filters": lambda x: construct_filters_from_query(x["input"])
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)

# Define nodes for the graph
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
    "snowflake_agent": snowflake_agent,
    "rag_search": rag_search,
    "rag_search_filter": rag_search_filter,
    "web_search": web_search,
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

# Define the graph
graph = StateGraph(AgentState)

graph.add_node("oracle", run_oracle)
graph.add_node("snowflake_agent", run_tool)
graph.add_node("rag_search", run_tool)
graph.add_node("rag_search_filter", run_tool)
graph.add_node("web_search", run_tool)
graph.add_node("final_answer", run_tool)

graph.set_entry_point("oracle")

graph.add_conditional_edges(
    source="oracle",  # where in graph to start
    path=router,  # function to determine which node is called
)

# create edges from each tool back to the oracle
for tool_obj in tools:
    if tool_obj.name != "final_answer":
        graph.add_edge(tool_obj.name, "oracle")

# if anything goes to final answer, it must then move to END
graph.add_edge("final_answer", END)

runnable = graph.compile()

#Image(runnable.get_graph().draw_png())

# state = {
#     "input": "What is the market cap of NVIDIA?",
#     "chat_history": [],
# }

# out = runnable.invoke(state)

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

#print(build_report(output=out["intermediate_steps"][-1].tool_input))