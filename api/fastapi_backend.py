from fastapi import FastAPI, HTTPException, Query
from typing import Optional, Dict
from langgraph.graph import StateGraph
from langchain_core.tools import tool
from langgraph_code.multi_agent_system import runnable, tool_str_to_func, construct_filters_from_query, build_report

# initialize the FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multiagent API!"}


@app.post("/use-agent/")
async def use_agent(
    agent_name: str = Query(..., description="Name of the agent to use (e.g., rag_search, rag_search_filter, web_search, snowflake_agent)"),
    query: str = Query(..., description="User query for the agent"),
    year: Optional[int] = Query(None, description="Year filter for agents like rag_search_filter"),
    quarter: Optional[str] = Query(None, description="Quarter filter for agents like rag_search_filter")
):
    """
    Endpoint to use a specific agent directly without going through LangGraph.
    """
    # Validate agent name
    if agent_name not in tool_str_to_func:
        raise HTTPException(status_code=400, detail=f"Agent '{agent_name}' not found. Available agents: {list(tool_str_to_func.keys())}")

    # Construct tool input dynamically based on agent type
    tool_args = {"query": query}

    if agent_name == "rag_search_filter":
        # Dynamically construct filters for rag_search_filter
        filters = construct_filters_from_query(query)
        tool_args["filters"] = filters

    try:
        # Directly invoke the selected tool
        result = tool_str_to_func[agent_name].invoke(input=tool_args)
        if len(result) == 0:
            result = "No relevant information found to answer the query."

        # Format the output using build_report
        final_output = {
            "introduction": f"Query about {query}",
            "research_steps": [f"Tool: {agent_name}, input: {tool_args}"],
            "main_body": result
        }

        return {"agent": agent_name, "query": query, "result": final_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking agent '{agent_name}': {str(e)}")


@app.post("/use-all-agents/")
async def use_all_agents(query: str):
    """
    Endpoint to use all agents combined via the multiagent graph.
    """
    state = {
        "input": query,
        "chat_history": [],
        "intermediate_steps": []
    }

    try:
        result = runnable.invoke(state)
        final_output = build_report(output=result["intermediate_steps"][-1].tool_input)
        return {"query": query, "result": final_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking multiagent system: {str(e)}")
    

