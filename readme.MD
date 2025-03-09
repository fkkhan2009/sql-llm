# LLM Query & Analysis System

A modular and extensible LLM-powered application for document processing, conversational AI, and database integration with workspace management.

![System Architecture](https://via.placeholder.com/800x400?text=LLM+System+Architecture)

## Features

### Core Capabilities
- **Advanced Document Processing**: Process text, CSV, JSON, and Excel files with automatic chunking and embedding
- **Conversational Memory**: Maintain conversation context with automatic summarization and retrieval
- **SQL Generation**: Generate valid Snowflake SQL queries from natural language
- **Vector Search**: ChromaDB integration for semantic search of documents and conversation history
- **Database Integration**: Secure Snowflake connectivity with multiple authentication methods

### User Experience
- **Workspace Management**: Organize documents and conversations in isolated workspaces
- **Multi-Collection Support**: Create multiple document collections within workspaces
- **Conversation Management**: Create, list, and manage conversations with full history
- **Contextual Responses**: LLM responses that incorporate document, conversation, and database context

### Development Features
- **Modular Architecture**: Clean separation of concerns with well-defined components
- **RESTful API**: Comprehensive API endpoints for all functionality
- **Dependency Injection**: Efficient component management with proper dependency injection
- **Extensible Design**: Easily add new capabilities or integrations

## Architecture

The application follows a modular architecture with these key components:

1. **Document Processing Layer**
   - `DocumentLoader`: Handles loading various file formats
   - `DocumentProcessor`: Manages chunking and preprocessing
   - `VectorStore`: Interfaces with ChromaDB for embedding storage and retrieval

2. **LLM Orchestration Layer**
   - `OllamaClient`: Manages interaction with local LLM
   - `ContextBuilder`: Assembles context from multiple sources
   - `PromptTemplates`: Handles prompt engineering for different query types
   - `LLMOrchestrator`: Coordinates the overall LLM interaction flow

3. **Database Layer**
   - `SnowflakeConnector`: Handles secure connection to Snowflake
   - `SchemaUtils`: Extracts and formats databasecd  schema information

4. **Workspace & Conversation Management**
   - `WorkspaceManager`: Manages isolation of resources by workspace
   - `ConversationMemory`: Tracks conversation history with context management
   - `ConversationManager`: Orchestrates multiple conversations

5. **API Layer**
   - `BaseRouter`: Core query and document endpoints
   - `WorkspaceRouter`: Workspace management endpoints
   - `ConversationRouter`: Conversation management endpoints

## Installation

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/download) for local LLM inference
- Snowflake account (optional)

### Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/llm-qa-app.git
   cd llm-qa-app
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**
   Create a `.env` file in the project root:
   ```
   # Core settings
   DEBUG=True
   CHROMA_PERSIST_DIRECTORY=./chroma_db
   
   # LLM settings
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama2
   EMBEDDING_MODEL=ollama:llama2
   
   # Snowflake settings (optional)
   SNOWFLAKE_ACCOUNT=your_account
   SNOWFLAKE_USER=your_username
   
   # Choose one of these authentication methods (in order of preference):
   SNOWFLAKE_PRIVATE_KEY_FILE=/path/to/rsa_key.p8
   SNOWFLAKE_PRIVATE_KEY_PASSPHRASE=your_passphrase
   # or
   SNOWFLAKE_PASSWORD_FILE=/path/to/password_file.txt
   # or
   SNOWFLAKE_PASSWORD=your_password
   
   # Additional Snowflake settings
   SNOWFLAKE_DATABASE=your_database
   SNOWFLAKE_SCHEMA=your_schema
   SNOWFLAKE_WAREHOUSE=your_warehouse
   SNOWFLAKE_ROLE=your_role
   ```

5. **Pull Required Ollama Models**
   ```bash
   ollama pull llama2
   ```

## Running the Application

Start the API server:
```bash
python run.py
```

The API will be available at `http://localhost:8000`.

## API Documentation

Once the server is running, interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Key Endpoints

### Core Endpoints
- `GET /` - Health check
- `POST /api/query` - Submit a query for the LLM to answer
- `POST /api/upload` - Upload a document (legacy endpoint)
- `POST /api/generate-sql` - Generate SQL from natural language
- `GET /api/tables` - List available database tables
- `GET /api/models` - List available Ollama models

### Workspace Endpoints
- `POST /api/workspaces` - Create a new workspace
- `GET /api/workspaces` - List all workspaces
- `GET /api/workspaces/{workspace_id}` - Get workspace details
- `DELETE /api/workspaces/{workspace_id}` - Delete a workspace
- `POST /api/workspaces/{workspace_id}/upload` - Upload a document to a workspace

### Conversation Endpoints
- `POST /api/workspaces/{workspace_id}/conversations` - Create a new conversation
- `GET /api/workspaces/{workspace_id}/conversations` - List all conversations in a workspace
- `POST /api/workspaces/{workspace_id}/conversations/{conversation_id}/messages` - Send a message and get a response
- `GET /api/workspaces/{workspace_id}/conversations/{conversation_id}/messages` - Get conversation history

## Usage Examples

### Creating a Workspace
```bash
curl -X POST "http://localhost:8000/api/workspaces" \
  -H "Content-Type: application/json" \
  -d '{"name": "Research Project", "description": "Financial analysis research"}'
```

### Uploading a Document
```bash
curl -X POST "http://localhost:8000/api/workspaces/{workspace_id}/upload" \
  -F "file=@/path/to/document.csv" \
  -F "collection_name=financial_data"
```

### Creating a Conversation
```bash
curl -X POST "http://localhost:8000/api/workspaces/{workspace_id}/conversations" \
  -H "Content-Type: application/json" \
  -d '{"name": "Financial Analysis"}'
```

### Sending a Message
```bash
curl -X POST "http://localhost:8000/api/workspaces/{workspace_id}/conversations/{conversation_id}/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "What trends do you see in our quarterly revenue data?"}'
```

### Querying with Workspace Context
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize our sales performance", 
    "workspace_id": "{workspace_id}", 
    "include_docs": true
  }'
```

### Generating SQL
```bash
curl -X POST "http://localhost:8000/api/generate-sql" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me total sales by region for January 2023"}'
```

## Project Structure

```
llm_qa_app/
├── app/                        # Main application package
│   ├── config.py               # Application configuration
│   ├── dependencies.py         # Dependency injection system
│   ├── main.py                 # FastAPI application entry point
│   ├── conversation/           # Conversation management
│   │   ├── manager.py          # Multi-conversation orchestration
│   │   └── memory.py           # Conversation memory management
│   ├── database/               # Database connectivity
│   │   ├── schema_utils.py     # Schema metadata utilities
│   │   └── snowflake_connector.py # Snowflake integration
│   ├── document_processing/    # Document handling
│   │   ├── loader.py           # Document loading utilities
│   │   ├── processor.py        # Document chunking and preprocessing
│   │   └── vector_store.py     # ChromaDB integration
│   ├── llm/                    # LLM integration
│   │   ├── context_builder.py  # Context assembly
│   │   ├── ollama_client.py    # Ollama API client
│   │   ├── orchestrator.py     # LLM orchestration
│   │   └── prompt_templates.py # Prompt engineering
│   ├── models/                 # Pydantic models
│   │   ├── base.py             # Core models
│   │   ├── conversation.py     # Conversation models
│   │   └── workspace.py        # Workspace models
│   ├── routers/                # API endpoints
│   │   ├── base.py             # Core endpoints
│   │   ├── conversation.py     # Conversation endpoints
│   │   └── workspace.py        # Workspace endpoints
│   └── workspace/              # Workspace management
│       └── manager.py          # Workspace orchestration
├── .env.example                # Example environment variables
├── README.md                   # This documentation
├── requirements.txt            # Python dependencies
└── run.py                      # Application runner
```

## Extending the Application

### Adding a New Document Type
1. Update `DocumentLoader` in `app/document_processing/loader.py`
2. Add a new method for your document type (e.g., `_load_pdf`)
3. Update the file extension handling in the `load_document` method

### Adding a New Database Connector
1. Create a new connector class in `app/database/`
2. Implement the necessary methods for connection and query execution
3. Update `app/dependencies.py` to include your new connector
4. Add new router endpoints for database-specific functionality

### Creating a Custom Vector Store
1. Create a new class that implements the same interface as `VectorStore`
2. Update `app/dependencies.py` to use your vector store implementation
3. Modify `WorkspaceManager` if needed to support your vector store

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.