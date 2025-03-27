import streamlit as st
import requests

st.set_page_config(page_title="NVIDIA Research Assistant", layout="wide")
st.title("🧾 Assignment 5.1 Team 6 - NVIDIA Research Assistant")

# --- Sidebar Filters ---
with st.sidebar:
    st.header("🔧 Customize Filters")
    selected_years = st.multiselect("📅 Select Year(s)", ["2020", "2021", "2022", "2023", "2024", "2025"])
    selected_quarters = st.multiselect("📆 Select Quarter(s)", ["Q1", "Q2", "Q3", "Q4"])
    agent_mode = st.radio("🧠 Agent Mode", [
        "Default",
        "Snowflake Agent",
        "RAG Search Agent",
        "Web Search Agent",
    ])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores tuples of (role, message)

backend_base_url = "http://localhost:8000"
agent_map = {
    "Snowflake Agent": "snowflake_agent",
    "RAG Search Agent": "rag_search",
    "Web Search Agent": "web_search",
}

# --- Query Input ---
query = st.chat_input("Ask your financial research question...")

if query:
    # Save user message first
    st.session_state.chat_history.append(("user", query))

    # Render user bubble
    with st.chat_message("user"):
        st.markdown(query)

    year = selected_years[0] if selected_years else None
    quarter = selected_quarters[0] if selected_quarters else None

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call appropriate agent endpoint
                if agent_mode == "Default":
                    res = requests.post(f"{backend_base_url}/use-all-agents/", params={"query": query})
                    result = res.json()["result"]
                else:
                    agent_name = agent_map[agent_mode]
                    res = requests.post(
                        f"{backend_base_url}/use-agent/",
                        params={
                            "agent_name": agent_name,
                            "query": query,
                            "year": year,
                            "quarter": quarter,
                        }
                    )
                    # snowflake returns structured dict
                    result = res.json()["result"]
                    if isinstance(result, dict) and "main_body" in result:
                        result = result["main_body"]

                # Save assistant message
                st.session_state.chat_history.append(("assistant", result))

                # Display inside assistant bubble
                st.markdown(
                    f"<div style='background-color: #262730; padding: 1rem; border-radius: 10px; font-size: 0.95rem;'>{result}</div>",
                    unsafe_allow_html=True
                )

                # Download button
                filename = f"{query.strip().replace(' ', '_').lower()}.txt"
                st.download_button(
                    label="📄 Download Report",
                    data=result,
                    file_name=filename,
                    mime="text/plain"
                )

            except Exception as e:
                error_text = f"❌ Error: {e}"
                st.session_state.chat_history.append(("assistant", error_text))
                st.error(error_text)

# --- Show history (except the current round just rendered) ---
# Prevent duplicate by skipping last two messages (already rendered above)
for role, msg in st.session_state.chat_history[:-2]:
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(
                f"<div style='background-color: #262730; padding: 1rem; border-radius: 10px; font-size: 0.95rem;'>{msg}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(msg)