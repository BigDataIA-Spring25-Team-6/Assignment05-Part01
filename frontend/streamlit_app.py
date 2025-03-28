import html
import streamlit as st
import requests
import base64
from io import BytesIO

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

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

backend_base_url = "http://localhost:8000"
agent_map = {
    "Snowflake Agent": "snowflake_agent",
    "RAG Search Agent": "rag_search",
    "Web Search Agent": "web_search",
}

# --- Query Input ---
query = st.chat_input("Ask your financial research question...")

if query:

    year = selected_years[0] if selected_years else None
    quarter = selected_quarters[0] if selected_quarters else None


    with st.spinner("Thinking..."):
        try:
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
                result = res.json()["result"]
                if isinstance(result, dict) and "main_body" in result:
                    result = result["main_body"]

            # Append both user and assistant messages to chat history
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("assistant", result))

        except Exception as e:
            error_text = f"❌ Error: {e}"
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("assistant", error_text))

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(
                f"<div style='background-color: #262730; padding: 1rem; border-radius: 10px; font-size: 0.95rem;'>{msg}</div>",
                unsafe_allow_html=True
            )

            plain_msg = html.unescape(msg)
            plain_msg = plain_msg.replace("<", "&lt;").replace(">", "&gt;")

            filename = f"{msg[:20].strip().replace(' ', '_').lower()}.txt"
            st.download_button(
                label="📄 Download Report",
                data=plain_msg,
                file_name=filename,
                mime="text/plain",
                key=f"download_{hash(msg)}"
            )
        else:
            st.markdown(msg)