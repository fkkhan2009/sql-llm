"""
Models related to conversations
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

class ConversationCreate(BaseModel):
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ConversationResponse(BaseModel):
    conversation_id: str
    workspace_id: str
    name: str
    metadata: Dict[str, Any] = {}

class MessageRequest(BaseModel):
    content: str
    role: str = "user"
    metadata: Optional[Dict[str, Any]] = None

class MessageResponse(BaseModel):
    message_id: str
    content: str
    role: str
    timestamp: float
    metadata: Dict[str, Any] = {}

class MessagesResponse(BaseModel):
    messages: List[Dict[str, Any]]
