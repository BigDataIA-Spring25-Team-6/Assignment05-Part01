# Creating an Agentic Multi-Agent System with LangGraph

## Team Members
- **Aditi Ashutosh Deodhar**  002279575  
- **Lenin Kumar Gorle**       002803806  
- **Poorvika Girish Babu**    002801388

## Project Overview
### Problem Statement

In today’s data-driven world, business analysts and researchers are often overwhelmed by the fragmented nature of information sources—ranging from structured financial databases to unstructured reports and dynamic, real-time web content. Conducting a comprehensive analysis of a company like NVIDIA requires synthesizing historical performance, financial metrics, and the latest industry trends, which traditionally involves time-consuming manual research and tool-hopping.
This project addresses the challenge of automating and integrating these diverse data sources into a unified, intelligent research assistant. The goal is to design a multi-agent system that can:
Retrieve and analyze historical data from quarterly reports in Pinecone.
Query and summarize structured valuation measures from Snowflake.
Fetch and incorporate the latest insights from the web.

### Methodology

This project addresses the challenge of automating and integrating these diverse data sources into a unified, intelligent research assistant. The goal is to design a multi-agent system that can:
- Retrieve and analyze historical data from quarterly reports in Pinecone.
- Query and summarize structured valuation measures from Snowflake.
- Fetch and incorporate the latest insights from the web.

### Scope
```
The desired outcome is a streamlined application that empowers users to generate consolidated research reports using natural language queries, filter by specific timeframes (Year/Quarter), and receive actionable insights in real-time.

Key requirements and constraints include:

- Enabling metadata filtering in Pinecone for time-based queries.
- Maintaining accurate and secure connectivity to external platforms like Snowflake and Tavily.
- Orchestrating agent collaboration using LangGraph.
- Ensuring portability through Docker-based deployment.

By solving this problem, the system not only enhances productivity but also brings transparency and depth to financial and market research.
```

## Technologies Used
```
FastAPI
Streamlit
Pinecone Database
Docker
AWS S3
LangGraph

```

## Architecture Diagram
![mermaid-diagram-2025-03-28-045518](https://github.com/user-attachments/assets/2ffbdf4d-55da-4ad5-bce7-b2a1ba6a2c2c)

## LangGraph Architecture
![image](https://github.com/user-attachments/assets/ecdfc8ea-b4c9-4101-b04f-49f991ad5f18)

## Codelabs Documentation
https://codelabs-preview.appspot.com/?file_id=1j6ZJwPm5CLaXTn8r8ah-3pHVCoUhSG7Y9n3v9m87rxk#0

## Demo
https://shorturl.at/shNyk

## Hosted Applications links 

- Frontend : https://frontend-325254329458.us-central1.run.app
- Backend : [https://backend-487006321216.us-central1.run.app](https://backend-325254329458.us-central1.run.app)

## Prerequisites
```
-Python 3.10+
-Docker installed and running
-Docker Compose installed
-AWS S3 bucket with credentials
-OpenAI API key
-Pinecone API key
-Streamlit installed
-FastAPI framework
-Tavilly API Key
```

## Set Up the Environment
```sh
# Clone the repository
git clone https://github.com/BigDataIA-Spring25-Team-6/Assignment05-Part01.git
cd DAMG7245-Assignment05-Part-01.git

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with your AWS, Pinecone, and OpenAI credentials

# Run FastAPI backend (inside /api folder)
cd api
uvicorn fastapi_backend:app --host 0.0.0.0 --port 8000 --reload

# Run Streamlit frontend (in a new terminal, inside /frontend folder)
cd ../frontend
streamlit run streamlit_app.py

# Optional: Run using Docker Compose from root directory
docker-compose up --build

```

## Project Structure

```

ASSIGNMENT05-PART-01/

├── api/                 # FastAPI backend
 ├── Dockerfile          # Docker file for backend to build and deploy
 ├── .dockerignore       # Docker ignore file to ignore the unnecessary files          
 ├── requirements.txt    # backend dependencies

├── data_prep/           # Data processing scripts (chunking, RAG)

├── frontend/            # Streamlit frontend
 ├── Dockerfile          # Docker file for frontend to build and deploy
 ├── .dockerignore       # Docker ignore file to ignore the unnecessary files
 ├── requirements.txt    # frontend dependencies

├── .dockerignore        # Docker ignore file

├── .gitignore           # Git ignore file

├── docker-compose.yaml  # Docker file to locally deploy

├── requirements.txt     # Dependencies file

```







