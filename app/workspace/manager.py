"""
Enhanced workspace management with better synchronization
"""
from typing import List, Dict, Any, Optional, Callable, Set
import os
import json
import uuid
import shutil
import time
import threading
from langchain.docstore.document import Document
from app.document_processing.vector_store import VectorStore
from app.config import settings

class WorkspaceChangeEvent:
    """Represents a workspace change event"""
    
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    
    def __init__(self, event_type: str, workspace_id: str, workspace_name: str):
        """
        Initialize a workspace change event
        
        Args:
            event_type: Type of event (create, update, delete)
            workspace_id: ID of the affected workspace
            workspace_name: Name of the affected workspace
        """
        self.event_type = event_type
        self.workspace_id = workspace_id
        self.workspace_name = workspace_name
        self.timestamp = time.time()

class Workspace:
    """Represents a single workspace"""
    
    def __init__(
        self,
        workspace_id: str,
        name: str,
        description: Optional[str] = None,
        owner_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None
    ):
        """
        Initialize a workspace
        
        Args:
            workspace_id: Unique workspace identifier
            name: Workspace name
            description: Workspace description
            owner_id: ID of the workspace owner
            metadata: Additional metadata
            created_at: Creation timestamp
            updated_at: Last update timestamp
        """
        self.workspace_id = workspace_id
        self.name = name
        self.description = description or ""
        self.owner_id = owner_id
        self.metadata = metadata or {}
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workspace':
        """Create workspace from dictionary"""
        return cls(
            workspace_id=data["workspace_id"],
            name=data["name"],
            description=data.get("description", ""),
            owner_id=data.get("owner_id"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )

class WorkspaceManager:
    """Manages workspaces and their vector stores"""
    
    def __init__(
        self, 
        base_path: str = settings.CHROMA_PERSIST_DIRECTORY,
        auto_refresh_interval: Optional[int] = None
    ):
        """
        Initialize workspace manager
        
        Args:
            base_path: Base path for storing workspace data
            auto_refresh_interval: Seconds between auto-refreshes (None to disable)
        """
        self.base_path = base_path
        self.workspaces_file = os.path.join(base_path, "workspaces.json")
        self.workspaces: Dict[str, Workspace] = {}
        self.vector_stores: Dict[str, VectorStore] = {}
        self.last_refresh_time = 0
        self.change_listeners: List[Callable[[WorkspaceChangeEvent], None]] = []
        self.auto_refresh_interval = auto_refresh_interval
        self.stop_auto_refresh = threading.Event()
        
        # Create base directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        # Load existing workspaces
        self._load_workspaces()
        
        # Start auto-refresh if enabled
        if auto_refresh_interval:
            self._start_auto_refresh()
    
    def _start_auto_refresh(self):
        """Start background thread for auto-refresh"""
        def refresh_loop():
            while not self.stop_auto_refresh.is_set():
                # Sleep first to prevent immediate double-loading
                self.stop_auto_refresh.wait(self.auto_refresh_interval)
                if not self.stop_auto_refresh.is_set():
                    self.refresh_workspaces()
        
        thread = threading.Thread(target=refresh_loop, daemon=True)
        thread.start()
    
    def stop(self):
        """Stop any background tasks"""
        self.stop_auto_refresh.set()
    
    def add_change_listener(self, listener: Callable[[WorkspaceChangeEvent], None]):
        """
        Add a listener for workspace change events
        
        Args:
            listener: Callback function that takes a WorkspaceChangeEvent
        """
        if listener not in self.change_listeners:
            self.change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[WorkspaceChangeEvent], None]):
        """
        Remove a workspace change listener
        
        Args:
            listener: Callback function to remove
        """
        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
    
    def _notify_listeners(self, event: WorkspaceChangeEvent):
        """
        Notify all listeners of a workspace change
        
        Args:
            event: Workspace change event
        """
        for listener in self.change_listeners:
            try:
                listener(event)
            except Exception as e:
                print(f"Error in workspace change listener: {str(e)}")
    
    def _load_workspaces(self) -> None:
        """Load workspaces from storage"""
        self.workspaces = {}
        
        if os.path.exists(self.workspaces_file):
            try:
                with open(self.workspaces_file, 'r') as f:
                    workspaces_data = json.load(f)
                
                for workspace_data in workspaces_data:
                    workspace = Workspace.from_dict(workspace_data)
                    self.workspaces[workspace.workspace_id] = workspace
                
                self.last_refresh_time = time.time()
            except Exception as e:
                print(f"Error loading workspaces: {str(e)}")
    
    def _save_workspaces(self) -> None:
        """Save workspaces to storage"""
        workspaces_data = [w.to_dict() for w in self.workspaces.values()]
        try:
            # First write to a temporary file
            temp_file = f"{self.workspaces_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(workspaces_data, f, indent=2)
            
            # Then rename it (atomic operation)
            os.replace(temp_file, self.workspaces_file)
        except Exception as e:
            print(f"Error saving workspaces: {str(e)}")
    
    def refresh_workspaces(self, force: bool = False) -> bool:
        """
        Refresh workspaces from storage
        
        Args:
            force: Force refresh even if file hasn't changed
            
        Returns:
            Whether any changes were detected
        """
        # Check if the file has been modified since last refresh
        try:
            if os.path.exists(self.workspaces_file):
                file_mtime = os.path.getmtime(self.workspaces_file)
                if not force and file_mtime <= self.last_refresh_time:
                    return False  # No changes
                    
                # Remember current workspaces for change detection
                old_workspace_ids = set(self.workspaces.keys())
                
                # Load workspaces from file
                self._load_workspaces()
                
                # Detect changes
                new_workspace_ids = set(self.workspaces.keys())
                
                # Find created, deleted workspaces
                created_ids = new_workspace_ids - old_workspace_ids
                deleted_ids = old_workspace_ids - new_workspace_ids
                
                # Clean up vector stores for deleted workspaces
                for workspace_id in deleted_ids:
                    self._cleanup_vector_stores(workspace_id)
                
                return len(created_ids) > 0 or len(deleted_ids) > 0
            return False
        except Exception as e:
            print(f"Error refreshing workspaces: {str(e)}")
            return False
    
    def _cleanup_vector_stores(self, workspace_id: str) -> None:
        """
        Clean up vector stores for a deleted workspace
        
        Args:
            workspace_id: Workspace ID
        """
        # Find all vector stores for this workspace
        stores_to_remove = []
        for key in self.vector_stores:
            if key == workspace_id or key.startswith(f"{workspace_id}_"):
                stores_to_remove.append(key)
        
        # Close and remove vector stores
        for key in stores_to_remove:
            try:
                # The VectorStore class doesn't have a close method,
                # but we remove the reference to allow garbage collection
                del self.vector_stores[key]
            except Exception as e:
                print(f"Error removing vector store {key}: {str(e)}")
    
    def create_workspace(
        self,
        name: str,
        description: Optional[str] = None,
        owner_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Workspace:
        """
        Create a new workspace
        
        Args:
            name: Workspace name
            description: Workspace description
            owner_id: ID of the workspace owner
            metadata: Additional metadata
            
        Returns:
            Created Workspace object
        """
        workspace_id = str(uuid.uuid4())
        now = time.time()
        
        workspace = Workspace(
            workspace_id=workspace_id,
            name=name,
            description=description,
            owner_id=owner_id,
            metadata=metadata or {},
            created_at=now,
            updated_at=now
        )
        
        # Store workspace
        self.workspaces[workspace_id] = workspace
        self._save_workspaces()
        
        # Create directory for workspace
        workspace_dir = os.path.join(self.base_path, workspace_id)
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Notify listeners
        self._notify_listeners(WorkspaceChangeEvent(
            event_type=WorkspaceChangeEvent.CREATE,
            workspace_id=workspace_id,
            workspace_name=name
        ))
        
        return workspace
    
    def update_workspace(
        self,
        workspace_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Workspace]:
        """
        Update a workspace
        
        Args:
            workspace_id: Workspace ID
            name: New name (if changing)
            description: New description (if changing)
            metadata: New metadata (if changing)
            
        Returns:
            Updated Workspace object or None if not found
        """
        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            return None
        
        # Update fields
        if name is not None:
            workspace.name = name
        if description is not None:
            workspace.description = description
        if metadata is not None:
            workspace.metadata = metadata
        
        workspace.updated_at = time.time()
        
        # Save changes
        self._save_workspaces()
        
        # Notify listeners
        self._notify_listeners(WorkspaceChangeEvent(
            event_type=WorkspaceChangeEvent.UPDATE,
            workspace_id=workspace_id,
            workspace_name=workspace.name
        ))
        
        return workspace
    
    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """
        Get a workspace by ID
        
        Args:
            workspace_id: Workspace ID
            
        Returns:
            Workspace object or None if not found
        """
        return self.workspaces.get(workspace_id)
    
    def delete_workspace(self, workspace_id: str) -> bool:
        """
        Delete a workspace and all its data
        
        Args:
            workspace_id: Workspace ID
            
        Returns:
            Whether deletion was successful
        """
        if workspace_id not in self.workspaces:
            return False
        
        # Get workspace name for event
        workspace_name = self.workspaces[workspace_id].name
        
        # Remove from workspaces dictionary
        del self.workspaces[workspace_id]
        
        # Clean up vector stores
        self._cleanup_vector_stores(workspace_id)
        
        # Remove workspace directory
        workspace_dir = os.path.join(self.base_path, workspace_id)
        if os.path.exists(workspace_dir):
            try:
                shutil.rmtree(workspace_dir)
            except Exception as e:
                print(f"Error removing workspace directory: {str(e)}")
                return False
        
        # Save updated workspaces
        self._save_workspaces()
        
        # Notify listeners
        self._notify_listeners(WorkspaceChangeEvent(
            event_type=WorkspaceChangeEvent.DELETE,
            workspace_id=workspace_id,
            workspace_name=workspace_name
        ))
        
        return True
    
    def list_workspaces(self, owner_id: Optional[str] = None) -> List[Workspace]:
        """
        List all workspaces, optionally filtered by owner
        
        Args:
            owner_id: Optional owner ID to filter by
            
        Returns:
            List of Workspace objects
        """
        if owner_id:
            return [w for w in self.workspaces.values() if w.owner_id == owner_id]
        return list(self.workspaces.values())
    
    def get_vector_store(self, workspace_id: str) -> Optional[VectorStore]:
        """
        Get vector store for a workspace
        
        Args:
            workspace_id: Workspace ID
            
        Returns:
            VectorStore object or None if workspace doesn't exist
        """
        if workspace_id not in self.workspaces:
            # Try to refresh workspaces in case it was created externally
            if self.refresh_workspaces() and workspace_id in self.workspaces:
                # Workspace now exists after refresh
                pass
            else:
                return None
            
        # Create vector store if not already created
        if workspace_id not in self.vector_stores:
            workspace_dir = os.path.join(self.base_path, workspace_id)
            os.makedirs(workspace_dir, exist_ok=True)
            
            self.vector_stores[workspace_id] = VectorStore(
                persist_directory=workspace_dir,
                embedding_model_name=settings.EMBEDDING_MODEL,
                collection_name="documents"  # Default collection name
            )
            
        return self.vector_stores[workspace_id]
    
    def create_vector_store_for_collection(
        self,
        workspace_id: str,
        collection_name: str
    ) -> Optional[VectorStore]:
        """
        Create a vector store for a specific collection in a workspace
        
        Args:
            workspace_id: Workspace ID
            collection_name: Collection name
            
        Returns:
            VectorStore object or None if workspace doesn't exist
        """
        if workspace_id not in self.workspaces:
            # Try to refresh workspaces in case it was created externally
            if self.refresh_workspaces() and workspace_id in self.workspaces:
                # Workspace now exists after refresh
                pass
            else:
                return None
            
        workspace_dir = os.path.join(self.base_path, workspace_id)
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Create a unique key for this workspace+collection
        store_key = f"{workspace_id}_{collection_name}"
        
        self.vector_stores[store_key] = VectorStore(
            persist_directory=workspace_dir,
            embedding_model_name=settings.EMBEDDING_MODEL,
            collection_name=collection_name
        )
            
        return self.vector_stores[store_key]
    
    def get_vector_store_for_collection(
        self,
        workspace_id: str,
        collection_name: str
    ) -> Optional[VectorStore]:
        """
        Get vector store for a specific collection in a workspace
        
        Args:
            workspace_id: Workspace ID
            collection_name: Collection name
            
        Returns:
            VectorStore object or None if workspace doesn't exist
        """
        if workspace_id not in self.workspaces:
            # Try to refresh workspaces in case it was created externally
            if self.refresh_workspaces() and workspace_id in self.workspaces:
                # Workspace now exists after refresh
                pass
            else:
                return None
            
        # Create a unique key for this workspace+collection
        store_key = f"{workspace_id}_{collection_name}"
        
        # Create vector store if not already created
        if store_key not in self.vector_stores:
            return self.create_vector_store_for_collection(workspace_id, collection_name)
            
        return self.vector_stores[store_key]
