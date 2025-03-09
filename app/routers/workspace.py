"""
Updated router for workspace-related endpoints with refresh mechanism
"""
import os
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Path, Query

from app.models.workspace import WorkspaceCreate, WorkspaceResponse, WorkspaceUpdate
from app.models.base import DocumentUploadResponse
from app.document_processing.loader import DocumentLoader
from app.dependencies import get_workspace_components

router = APIRouter(
    prefix="/api/workspaces",
    tags=["Workspaces"]
)


@router.post("", response_model=WorkspaceResponse)
async def create_workspace(
    request: WorkspaceCreate,
    components: Dict[str, Any] = Depends(get_workspace_components)
):
    """
    Create a new workspace
    """
    workspace_manager = components["workspace_manager"]
    
    workspace = workspace_manager.create_workspace(
        name=request.name,
        description=request.description,
        owner_id=request.owner_id,
        metadata=request.metadata
    )
    
    return {
        "workspace_id": workspace.workspace_id,
        "name": workspace.name,
        "description": workspace.description,
        "owner_id": workspace.owner_id,
        "metadata": workspace.metadata,
        "created_at": workspace.created_at,
        "updated_at": workspace.updated_at
    }


@router.get("", response_model=List[WorkspaceResponse])
async def list_workspaces(
    owner_id: Optional[str] = None,
    refresh: bool = Query(False, description="Force a refresh from storage"),
    components: Dict[str, Any] = Depends(get_workspace_components)
):
    """
    List all workspaces, optionally filtered by owner
    """
    workspace_manager = components["workspace_manager"]
    
    # Refresh workspaces if requested
    if refresh:
        workspace_manager.refresh_workspaces(force=True)
    
    workspaces = workspace_manager.list_workspaces(owner_id=owner_id)
    
    return [
        {
            "workspace_id": w.workspace_id,
            "name": w.name,
            "description": w.description,
            "owner_id": w.owner_id,
            "metadata": w.metadata,
            "created_at": w.created_at,
            "updated_at": w.updated_at
        } for w in workspaces
    ]


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: str = Path(..., description="The ID of the workspace"),
    refresh: bool = Query(False, description="Force a refresh from storage"),
    components: Dict[str, Any] = Depends(get_workspace_components)
):
    """
    Get a workspace by ID
    """
    workspace_manager = components["workspace_manager"]
    
    # Refresh workspaces if requested
    if refresh:
        workspace_manager.refresh_workspaces(force=True)
    
    workspace = workspace_manager.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    return {
        "workspace_id": workspace.workspace_id,
        "name": workspace.name,
        "description": workspace.description,
        "owner_id": workspace.owner_id,
        "metadata": workspace.metadata,
        "created_at": workspace.created_at,
        "updated_at": workspace.updated_at
    }


@router.patch("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    request: WorkspaceUpdate,
    workspace_id: str = Path(..., description="The ID of the workspace"),
    components: Dict[str, Any] = Depends(get_workspace_components)
):
    """
    Update a workspace
    """
    workspace_manager = components["workspace_manager"]
    
    workspace = workspace_manager.update_workspace(
        workspace_id=workspace_id,
        name=request.name,
        description=request.description,
        metadata=request.metadata
    )
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    return {
        "workspace_id": workspace.workspace_id,
        "name": workspace.name,
        "description": workspace.description,
        "owner_id": workspace.owner_id,
        "metadata": workspace.metadata,
        "created_at": workspace.created_at,
        "updated_at": workspace.updated_at
    }


@router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: str = Path(..., description="The ID of the workspace"),
    components: Dict[str, Any] = Depends(get_workspace_components)
):
    """
    Delete a workspace
    """
    workspace_manager = components["workspace_manager"]
    
    success = workspace_manager.delete_workspace(workspace_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    return {"success": True, "message": "Workspace deleted"}


@router.post("/{workspace_id}/upload", response_model=DocumentUploadResponse)
async def upload_document_to_workspace(
    workspace_id: str = Path(..., description="The ID of the workspace"),
    file: UploadFile = File(...),
    collection_name: str = Form("documents"),
    components: Dict[str, Any] = Depends(get_workspace_components)
):
    """
    Upload and process a document to a specific workspace
    """
    document_processor = components["document_processor"]
    workspace_manager = components["workspace_manager"]
    
    # Check if workspace exists
    workspace = workspace_manager.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # Write the uploaded file content to the temporary file
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        try:
            # Get vector store for the workspace and collection
            workspace_vector_store = workspace_manager.get_vector_store_for_collection(
                workspace_id=workspace_id,
                collection_name=collection_name
            )
            
            if not workspace_vector_store:
                raise ValueError(f"Failed to get vector store for workspace {workspace_id}")
            
            # Load the document
            documents = DocumentLoader.load_document(temp_file_path)
            
            # Add workspace metadata to documents
            for doc in documents:
                doc.metadata["workspace_id"] = workspace_id
                doc.metadata["filename"] = file.filename
            
            # Process the documents (chunking)
            processed_docs = document_processor.process_documents(documents)
            
            # Add to vector store
            document_ids = workspace_vector_store.add_documents(processed_docs)
            
            return {
                "success": True,
                "message": f"Successfully processed document: {file.filename} in workspace: {workspace_id}",
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


@router.post("/{workspace_id}/refresh")
async def refresh_workspace(
    workspace_id: str = Path(..., description="The ID of the workspace"),
    components: Dict[str, Any] = Depends(get_workspace_components)
):
    """
    Refresh a workspace from storage
    """
    workspace_manager = components["workspace_manager"]
    
    # Check if workspace exists
    workspace = workspace_manager.get_workspace(workspace_id)
    if not workspace:
        # Try to refresh in case it was created externally
        workspace_manager.refresh_workspaces(force=True)
        
        # Check again after refresh
        workspace = workspace_manager.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Return the refreshed workspace
    return {
        "workspace_id": workspace.workspace_id,
        "name": workspace.name,
        "description": workspace.description,
        "owner_id": workspace.owner_id,
        "metadata": workspace.metadata,
        "refreshed": True
    }
