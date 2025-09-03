# Deep Research

## Overview

A **multi-agent AI research assistant** built with [Google ADK](https://github.com/google/adk-python) that automatically searches [ArXiv](https://arxiv.org/) papers and the web to answer your questions. It creates **research plans**, gathers information from multiple sources (web search and ArXiv papers), and provides **cited responses**.

## Workflow

The system processes user queries through four main phases:

1. **Classification** - Determines if the request is a valid research question, needs more information, or is a general query
2. **Planning** - Creates a targeted research plan with up to 3 steps, combining ArXiv searches for academic foundations and web searches for current developments
3. **Research Execution** - Runs the plan in parallel, with specialized agents handling ArXiv paper retrieval/analysis and web search via [Tavily MCP server](https://docs.tavily.com/documentation/mcp)
4. **Answer Synthesis** - Aggregates all findings and generates a comprehensive, well-cited response using the collected research data

## Architecture

*Architecture diagram will be added here*

## Getting Started

### Prerequisites

- Python 3.11+
- API keys for supported AI providers:
  - Google AI API key (for Gemini models)
  - OpenAI API key (for embeddings, Gemini embeddings not yet implemented)
  - Tavily API key (for web search)
- Docker (for local Qdrant setup) or access to a cloud vector database

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd deep-research
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Configure environment variables:**
   
   Copy the example environment file:
   ```bash
   cp multi_tool_agent/.env.example multi_tool_agent/.env
   ```
   
   Edit the `.env` file to set your API keys:
   ```bash
   # Google AI Configuration
   GOOGLE_API_KEY=your_google_api_key_here
   
   # OpenAI and Search API Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   
   # Vector Store Configuration (default: Qdrant)
   VECTOR_STORE_TYPE=qdrant
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_GRPC_PORT=6334
   QDRANT_PREFER_GRPC=true
   
   # Logging Configuration
   LOG_LEVEL=INFO
   ```

4. **Set up vector database:**
   
   For local development with Docker:
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```
   
   Alternatively, you can use a cloud provider (Qdrant Cloud, etc.) or implement your own vector database by extending the base classes in `multi_tool_agent/data/vector_stores/`.

5. **Execute the research system:**
   
   ```bash
   uv run adk run multi_tool_agent
   ```
