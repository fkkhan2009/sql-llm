"""
Updated models related to workspaces
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel

class WorkspaceCreate(BaseModel):
    name: str
    description: Optional[str] = None
    owner_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class WorkspaceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class WorkspaceResponse(BaseModel):
    workspace_id: str
    name: str
    description: str
    owner_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
