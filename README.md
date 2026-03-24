# OpenRAG by Meritis

Welcome to **OpenRAG**, an open source, user friendly RAG benchmark tool !

The goal of OpenRAG is to provide an intuitive tool to help users decide which RAG method, amongst the large number of existing techniques, is most appropriated to ist own use case and data.

In OpenRAG, 15 RAG methods are implemented and more will be added with time. Each RAG can be customized to better fit each user's need:

- **Naive RAG** - Basic retrieval-augmented generation
- **Advanced RAG** - Enhanced RAG with advanced features
- **Graph RAG** - Graph-based retrieval using entity relationships
- **HyDE RAG** - Hypothetical Document Embedding
- **Self RAG** - Self-reflective RAG with retrieval feedback
- **Corrective RAG** - Corrective retrieval with web crawling fallback
- **Contextual Retrieval RAG** - Context-aware document retrieval
- **Agentic RAG** - Agent-based RAG with tool usage
- **Merger RAG** - Combines multiple RAG methods
- **Query Reformulation** - Query optimization and expansion
- **Semantic Chunking RAG** - Semantic-based document chunking
- **Reranker RAG** - RAG with result reranking
- **Query-based RAG** - Query-focused retrieval
- **Naive Chatbot** - Simple chatbot without retrieval
- **Agentic RAG Router** - Intelligent routing between RAG methods

It can be used with a supported API key. Available LLM providers are:

- **OpenAI** : requires an API key
- **Custom (OpenAI-compatible)** : for other OpenAI-compatible providers or local LLM deployments like vLLM, SGLang, Ollama, LM Studio

## YouTube Tutorial

You can see our tutorial on youtube to how to install and use OpenRAG to start your first benchmark

[![Watch the video](https://img.youtube.com/vi/i9Xyarj-fFQ/maxresdefault.jpg)](https://youtu.be/i9Xyarj-fFQ)

## Launch commands

### Prerequisites
- Docker and Docker Compose installed
- Elasticsearch instance (local or remote)
- API keys for LLM providers

### Start Elasticsearch (if not already installed)

If you don't have Elasticsearch installed, you can start a Docker instance:

```bash
docker compose -f docker-compose.elasticsearch.yml up -d
```

**Default credentials:**
- URL: `http://localhost:9200`
- User: `elastic`
- Password: `openrag`

> **Important:** Change the default password by modifying `ELASTIC_PASSWORD` in `docker-compose.elasticsearch.yml`.

To check if Elasticsearch is running:
```bash
curl http://localhost:9200
```

To stop Elasticsearch:
```bash
docker compose -f docker-compose.elasticsearch.yml down
```

To stop and remove data:
```bash
docker compose -f docker-compose.elasticsearch.yml down -v
```

### Start the application

1. **Start the backend**:
   ```bash
   cd backend
   docker compose up -d
   ```

2. **Start the frontend**:
   ```bash
   cd frontend
   docker compose up -d
   ```

3. **Access the application**: [http://localhost:8502/](http://localhost:8502/)

### Configuration

On first use, go to the **Configuration** page to set up:
- **Elasticsearch**: URL and credentials
- **LLM providers**: API keys
- **Embedding and Reranker models**

## App functionalities

Once the app is up and running, you can now:

- Upload your own data
![](frontend/streamlit_/images/screen_db.png)
- Chat with your favorite RAG method, indexed on your database to roughly asses performances
![](frontend/streamlit_/images/screen_chat.png)
- Customize each RAG method (parameters, embedding models, chunking strategies, etc.)
![](frontend/streamlit_/images/screen_rag_maker.png)
- Merge multiple RAG methods to create hybrid approaches and compare their performance
- Benchmark selected methods and retrieved a quantitative report on their performances, their answering time, their energy consumption, greenhouse gas emissions and token consumption
![](frontend/streamlit_/images/screen_report.png)

### Contacts

For any question concerning the application, feel free to contact the developers at <https://meritis.fr/expertise/innovation-ia/#block-form>
