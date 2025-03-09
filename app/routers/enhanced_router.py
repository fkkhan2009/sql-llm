"""
Enhanced router for database query interactions
"""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Path, Query, WebSocket, WebSocketDisconnect
import logging

from app.models.base import (
    QueryRequest, QueryResponse,
    SQLGenerationRequest, SQLGenerationResponse
)
from app.llm.enhanced_orchestrator import EnhancedLLMOrchestrator, DatabaseResponse
from app.database.schema_utils import SchemaUtils
from app.dependencies import get_enhanced_components

# Set up logging
logger = logging.getLogger(__name__)

# Create a new model for enhanced database queries
from pydantic import BaseModel, Field

class EnhancedQueryRequest(BaseModel):
    """Request model for enhanced database queries"""
    question: str
    workspace_id: Optional[str] = None
    include_docs: bool = True
    save_as_view: bool = False
    view_name: Optional[str] = None
    view_description: Optional[str] = None
    temperature: float = 0.3
    use_progressive_building: bool = Field(default=True, description="Whether to use progressive query building")

class EnhancedQueryResponse(BaseModel):
    """Response model for enhanced database queries"""
    question: str
    sql: str
    results: Dict[str, Any]
    explanation: str
    relevant_tables: List[str] = []
    view_metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    timestamp: str

# Create router
router = APIRouter(
    prefix="/api/enhanced",
    tags=["Enhanced Database Queries"]
)

@router.post("/query", response_model=EnhancedQueryResponse)
async def enhanced_query(
    request: EnhancedQueryRequest,
    components: Dict[str, Any] = Depends(get_enhanced_components)
):
    """
    Process a natural language query against the database using multi-step reasoning
    """
    enhanced_orchestrator = components["enhanced_orchestrator"]
    
    try:
        logger.info(f"Processing enhanced query: {request.question}")
        
        # Process the query using the enhanced orchestrator
        response = await enhanced_orchestrator.process_database_query(
            question=request.question,
            workspace_id=request.workspace_id,
            include_docs=request.include_docs,
            save_as_view=request.save_as_view,
            view_name=request.view_name,
            view_description=request.view_description,
            temperature=request.temperature,
            use_progressive_building=request.use_progressive_building
        )
        
        logger.info(f"Query processed successfully. Execution time: {response.execution_time:.2f}s")
        
        # Convert to dictionary for the response
        return response.to_dict()
    except Exception as e:
        logger.error(f"Error processing database query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing database query: {str(e)}"
        )

@router.get("/tables")
async def list_tables(
    components: Dict[str, Any] = Depends(get_enhanced_components)
):
    """
    List available tables in the database with descriptions
    """
    snowflake_connector = components["snowflake_connector"]
    schema_utils = components["schema_utils"]
    
    if not snowflake_connector:
        raise HTTPException(
            status_code=400,
            detail="Database connection not available"
        )
    
    try:
        tables = snowflake_connector.get_tables()
        
        # Get descriptions for each table
        table_info = []
        for table in tables:
            columns = snowflake_connector.get_table_schema(table)
            sample = snowflake_connector.get_table_sample(table, limit=1)
            
            table_info.append({
                "name": table,
                "columns": len(columns),
                "has_sample": not sample.empty,
                "column_info": columns
            })
            
        return {"tables": table_info}
    except Exception as e:
        logger.error(f"Error listing tables: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing tables: {str(e)}"
        )

@router.get("/schemas/{table_name}")
async def get_table_schema(
    table_name: str = Path(..., description="The name of the table"),
    include_sample: bool = Query(False, description="Whether to include sample data"),
    components: Dict[str, Any] = Depends(get_enhanced_components)
):
    """
    Get detailed schema information for a specific table
    """
    snowflake_connector = components["snowflake_connector"]
    
    if not snowflake_connector:
        raise HTTPException(
            status_code=400,
            detail="Database connection not available"
        )
    
    try:
        columns = snowflake_connector.get_table_schema(table_name)
        
        if not columns:
            raise HTTPException(
                status_code=404,
                detail=f"Table {table_name} not found"
            )
            
        result = {
            "table_name": table_name,
            "columns": columns
        }
        
        if include_sample:
            sample = snowflake_connector.get_table_sample(table_name, limit=5)
            if not sample.empty:
                result["sample_data"] = sample.to_dict(orient='records')
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting table schema: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting table schema: {str(e)}"
        )

@router.get("/views")
async def list_views(
    workspace_id: Optional[str] = None,
    components: Dict[str, Any] = Depends(get_enhanced_components)
):
    """
    List saved views, optionally filtered by workspace
    """
    snowflake_connector = components["snowflake_connector"]
    
    if not snowflake_connector:
        raise HTTPException(
            status_code=400,
            detail="Database connection not available"
        )
    
    try:
        # Get all views
        all_views = snowflake_connector.get_views()
        
        # Filter by workspace if provided
        if workspace_id:
            prefix = f"WS_{workspace_id[:8]}_"
            filtered_views = [v for v in all_views if v.get("name", "").startswith(prefix)]
            return {"views": filtered_views}
        else:
            return {"views": all_views}
    except Exception as e:
        logger.error(f"Error listing views: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing views: {str(e)}"
        )

@router.post("/query/test")
async def test_query_components(
    request: EnhancedQueryRequest,
    components: Dict[str, Any] = Depends(get_enhanced_components)
):
    """
    Test the query component generation without executing the full query
    Useful for debugging and understanding the progressive building
    """
    enhanced_orchestrator = components["enhanced_orchestrator"]
    
    try:
        # Get document context
        doc_context = ""
        if request.include_docs and request.workspace_id and enhanced_orchestrator.context_builder:
            doc_context = enhanced_orchestrator.context_builder.get_document_context(request.question)
        
        # Identify relevant tables
        relevant_tables = await enhanced_orchestrator._identify_relevant_tables(
            request.question, doc_context, request.temperature
        )
        
        # Get schema context
        schema_context = enhanced_orchestrator.schema_utils.get_schema_context_for_tables(relevant_tables)
        
        # Generate query components
        select_clause = await enhanced_orchestrator._generate_select_clause(
            request.question, relevant_tables, schema_context, request.temperature
        )
        
        from_clause = await enhanced_orchestrator._generate_from_clause(
            request.question, relevant_tables, schema_context, request.temperature
        )
        
        where_clause = await enhanced_orchestrator._generate_where_clause(
            request.question, relevant_tables, schema_context, request.temperature
        )
        
        additional_clauses = await enhanced_orchestrator._generate_group_order_clause(
            request.question, relevant_tables, schema_context, request.temperature
        )
        
        # Combine components
        query_parts = [select_clause, from_clause]
        if where_clause:
            query_parts.append(where_clause)
        if additional_clauses:
            query_parts.append(additional_clauses)
        
        combined_query = " ".join(query_parts)
        
        # Return the components and combined query
        return {
            "question": request.question,
            "relevant_tables": relevant_tables,
            "components": {
                "select_clause": select_clause,
                "from_clause": from_clause,
                "where_clause": where_clause,
                "additional_clauses": additional_clauses
            },
            "combined_query": combined_query
        }
    except Exception as e:
        logger.error(f"Error testing query components: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error testing query components: {str(e)}"
        )
    
@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    components: Dict[str, Any] = Depends(get_enhanced_components)
):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            data = EnhancedQueryRequest(**data)
            enhanced_orchestrator = components["enhanced_orchestrator"]

            #Process the user query with streaming
            async for step_response in enhanced_orchestrator.process_database_query_stream(
                question=data.question,
                workspace_id=data.workspace_id,
                include_docs=data.include_docs,
                save_as_view=data.save_as_view,
                view_name=data.view_name,
                view_description=data.view_description,
                temperature=data.temperature,
                use_progressive_building=data.use_progressive_building
            ):
                await websocket.send_json(step_response)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in websocket: {str(e)}", exc_info=True)
        await websocket.send_json({"error": str(e)})
                
                
