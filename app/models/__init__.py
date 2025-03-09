"""
Package for Pydantic models
"""
from app.models.base import (
    QueryRequest, QueryResponse, SqlResult,
    DocumentUploadResponse,
    SQLGenerationRequest, SQLGenerationResponse
)

from app.models.workspace import (
    WorkspaceCreate, WorkspaceResponse
)

from app.models.conversation import (
    ConversationCreate, ConversationResponse,
    MessageRequest, MessageResponse, MessagesResponse
)

__all__ = [
    'QueryRequest',
    'QueryResponse',
    'SqlResult',
    'DocumentUploadResponse',
    'SQLGenerationRequest',
    'SQLGenerationResponse',
    'WorkspaceCreate',
    'WorkspaceResponse',
    'ConversationCreate',
    'ConversationResponse',
    'MessageRequest',
    'MessageResponse',
    'MessagesResponse'
]
