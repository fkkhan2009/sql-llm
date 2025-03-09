"""
Base router for core functionalities
"""
import os
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form

from app.models.base import (
    QueryRequest, QueryResponse,
    DocumentUploadResponse,
    SQLGenerationRequest, SQLGenerationResponse
)

from app.document_processing.loader import DocumentLoader
from app.document_processing.processor import DocumentProcessor
from app.llm.context_builder import ContextBuilder
from app.llm.orchestrator import LLMOrchestrator
from app.workspace.manager import WorkspaceManager
from app.dependencies import get_base_components

# Create router
router = APIRouter(tags=["Core"])


@router.get("/")
async def root():
    """Root endpoint for health check"""
    from app.config import settings
    return {"status": "ok", "message": f"Welcome to {settings.APP_NAME}"}


@router.post("/api/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    components: Dict[str, Any] = Depends(get_base_components)
):
    """
    Generate a response to a user query
    """
    document_processor = components["document_processor"]
    vector_store = components["vector_store"]
    snowflake_connector = components["snowflake_connector"]
    ollama_client = components["ollama_client"]
    llm_orchestrator = components["llm_orchestrator"]
    workspace_manager = components["workspace_manager"]
    
    # If workspace_id is provided, use the workspace-specific vector store
    if request.workspace_id:
        workspace = workspace_manager.get_workspace(request.workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
            
        workspace_vector_store = workspace_manager.get_vector_store(request.workspace_id)
        if not workspace_vector_store:
            raise HTTPException(status_code=404, detail="Vector store not found for workspace")
            
        # Create a workspace-specific context builder and orchestrator
        context_builder = ContextBuilder(
            vector_store=workspace_vector_store,
            snowflake_connector=snowflake_connector
        )
        
        workspace_orchestrator = LLMOrchestrator(
            ollama_client=ollama_client,
            context_builder=context_builder,
            vector_store=workspace_vector_store,
            snowflake_connector=snowflake_connector
        )
        
        response = workspace_orchestrator.generate_response(
            query=request.query,
            include_docs=request.include_docs,
            include_db_schema=request.include_db_schema,
            temperature=request.temperature
        )
    else:
        # Use the default global orchestrator
        response = llm_orchestrator.generate_response(
            query=request.query,
            include_docs=request.include_docs,
            include_db_schema=request.include_db_schema,
            temperature=request.temperature
        )
    
    return response


@router.post("/api/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    components: Dict[str, Any] = Depends(get_base_components)
):
    """
    Upload and process a document to the default vector store
    
    Note: This endpoint is for backward compatibility. 
    Use workspace-specific document upload for new applications.
    """
    document_processor = components["document_processor"]
    vector_store = components["vector_store"]
    
    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # Write the uploaded file content to the temporary file
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        try:
            # Load the document
            documents = DocumentLoader.load_document(temp_file_path)
            
            # Process the documents (chunking)
            processed_docs = document_processor.process_documents(documents)
            
            # Add to vector store
            document_ids = vector_store.add_documents(processed_docs)
            
            return {
                "success": True,
                "message": f"Successfully processed document: {file.filename}",
                "document_ids": document_ids
            }
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@router.post("/api/generate-sql", response_model=SQLGenerationResponse)
async def generate_sql(
    request: SQLGenerationRequest,
    components: Dict[str, Any] = Depends(get_base_components)
):
    """
    Generate SQL based on a natural language query
    """
    snowflake_connector = components["snowflake_connector"]
    llm_orchestrator = components["llm_orchestrator"]
    
    if not snowflake_connector:
        raise HTTPException(
            status_code=400,
            detail="Snowflake connection not available"
        )
    
    sql, explanation = llm_orchestrator.generate_sql(
        query=request.query,
        include_db_schema=request.include_db_schema,
        temperature=request.temperature
    )
    
    return {
        "query": request.query,
        "sql": sql,
        "explanation": explanation
    }


@router.get("/api/tables")
async def list_tables(
    components: Dict[str, Any] = Depends(get_base_components)
):
    """
    List available tables in the database
    """
    snowflake_connector = components["snowflake_connector"]
    
    if not snowflake_connector:
        raise HTTPException(
            status_code=400,
            detail="Snowflake connection not available"
        )
    
    tables = snowflake_connector.get_tables()
    return {"tables": tables}


@router.get("/api/models")
async def list_models(
    components: Dict[str, Any] = Depends(get_base_components)
):
    """
    List available Ollama models
    """
    ollama_client = components["ollama_client"]
    
    models = ollama_client.list_models()
    return {"models": models}
