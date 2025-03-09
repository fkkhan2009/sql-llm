"""
Base models for the application
"""
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel

# Query models
class QueryRequest(BaseModel):
    query: str
    include_docs: bool = True
    include_db_schema: bool = True
    temperature: float = 0.7
    workspace_id: Optional[str] = None  # Optional workspace to use for document context

class SqlResult(BaseModel):
    columns: Optional[List[str]] = None
    data: Optional[List[Dict[str, Any]]] = None
    shape: Optional[Tuple[int, int]] = None
    message: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    response: str
    is_sql_query: bool
    sql_result: Optional[SqlResult] = None

# Document models
class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    document_ids: List[str] = []

# SQL Generation models
class SQLGenerationRequest(BaseModel):
    query: str
    include_db_schema: bool = True
    temperature: float = 0.3

class SQLGenerationResponse(BaseModel):
    query: str
    sql: Optional[str] = None
    explanation: str
