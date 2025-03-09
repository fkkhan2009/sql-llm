"""
Conversation management for handling multiple conversations
"""
from typing import Dict, List, Any, Optional
import uuid
from app.conversation.memory import ConversationMemory
from app.workspace.manager import WorkspaceManager
from app.llm.ollama_client import OllamaClient

class ConversationManager:
    """Manages multiple conversations across workspaces"""
    
    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        llm_client: OllamaClient
    ):
        """
        Initialize conversation manager
        
        Args:
            workspace_manager: Workspace manager for accessing vector stores
            llm_client: LLM client for summarization
        """
        self.workspace_manager = workspace_manager
        self.llm_client = llm_client
        self.active_conversations: Dict[str, ConversationMemory] = {}
    
    def get_conversation_key(self, workspace_id: str, conversation_id: str) -> str:
        """Generate a unique key for a conversation"""
        return f"{workspace_id}_{conversation_id}"
    
    def create_conversation(
        self,
        workspace_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversation in a workspace
        
        Args:
            workspace_id: Workspace ID
            name: Optional name for the conversation
            metadata: Additional metadata
            
        Returns:
            Conversation ID
            
        Raises:
            ValueError: If workspace doesn't exist
        """
        # Check if workspace exists
        if not self.workspace_manager.get_workspace(workspace_id):
            raise ValueError(f"Workspace {workspace_id} does not exist")
        
        # Generate conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Get vector store for conversations
        vector_store = self.workspace_manager.get_vector_store_for_collection(
            workspace_id=workspace_id,
            collection_name="conversations"
        )
        
        if not vector_store:
            raise ValueError(f"Failed to create vector store for workspace {workspace_id}")
        
        # Create conversation memory
        conversation = ConversationMemory(
            vector_store=vector_store,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            llm_client=self.llm_client
        )
        
        # Add system message with conversation start
        initial_metadata = metadata or {}
        if name:
            initial_metadata["name"] = name
            
        conversation.add_message(
            role="system",
            content=f"Conversation started: {name or 'New conversation'}",
            metadata=initial_metadata
        )
        
        # Add to active conversations
        key = self.get_conversation_key(workspace_id, conversation_id)
        self.active_conversations[key] = conversation
        
        return conversation_id
    
    def get_conversation(
        self,
        workspace_id: str,
        conversation_id: str
    ) -> Optional[ConversationMemory]:
        """
        Get a conversation memory object
        
        Args:
            workspace_id: Workspace ID
            conversation_id: Conversation ID
            
        Returns:
            ConversationMemory object or None if not found
        """
        # Check if conversation is already active
        key = self.get_conversation_key(workspace_id, conversation_id)
        if key in self.active_conversations:
            return self.active_conversations[key]
        
        # Check if workspace exists
        if not self.workspace_manager.get_workspace(workspace_id):
            return None
        
        # Get vector store for conversations
        vector_store = self.workspace_manager.get_vector_store_for_collection(
            workspace_id=workspace_id,
            collection_name="conversations"
        )
        
        if not vector_store:
            return None
        
        # Try to find conversation messages
        filter_dict = {
            "conversation_id": conversation_id,
            "workspace_id": workspace_id
        }
        
        try:
            # Check if any messages exist for this conversation
            docs = vector_store.search_documents(
                query="",
                filter_dict=filter_dict,
                top_k=1
            )
            
            if not docs:
                return None  # Conversation doesn't exist
                
            # Create conversation memory
            conversation = ConversationMemory(
                vector_store=vector_store,
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                llm_client=self.llm_client
            )
            
            # Add to active conversations
            self.active_conversations[key] = conversation
            return conversation
            
        except Exception as e:
            print(f"Error loading conversation: {str(e)}")
            return None
    
    def add_message(
        self,
        workspace_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to a conversation
        
        Args:
            workspace_id: Workspace ID
            conversation_id: Conversation ID
            role: Message role
            content: Message content
            metadata: Additional metadata
            
        Returns:
            Whether message was added successfully
        """
        conversation = self.get_conversation(workspace_id, conversation_id)
        if not conversation:
            return False
            
        try:
            conversation.add_message(role=role, content=content, metadata=metadata)
            return True
        except Exception as e:
            print(f"Error adding message: {str(e)}")
            return False
    
    def list_conversations(
        self,
        workspace_id: str
    ) -> List[Dict[str, Any]]:
        """
        List all conversations in a workspace
        
        Args:
            workspace_id: Workspace ID
            
        Returns:
            List of conversation metadata
        """
        # Get vector store for conversations
        vector_store = self.workspace_manager.get_vector_store_for_collection(
            workspace_id=workspace_id,
            collection_name="conversations"
        )
        
        if not vector_store:
            return []
        
        try:
            # Use the raw Chroma client to query for unique conversation IDs
            # This is a simplified approach - in a real implementation, you might want
            # to store conversation metadata separately for more efficient retrieval
            filter_dict = {"workspace_id": workspace_id}
            docs = vector_store.search_documents(
                query="",  # Empty query to get all
                filter_dict=filter_dict,
                top_k=1000  # Get a large number to cover multiple conversations
            )
            
            # Extract unique conversation IDs and metadata
            conversations = {}
            for doc in docs:
                conversation_id = doc.metadata.get("conversation_id")
                if not conversation_id or conversation_id in conversations:
                    continue
                    
                # Basic metadata from first message
                conversations[conversation_id] = {
                    "conversation_id": conversation_id,
                    "workspace_id": workspace_id,
                    "name": doc.metadata.get("name", "Unnamed conversation"),
                    "timestamp": doc.metadata.get("timestamp", 0)
                }
            
            # Sort by timestamp (newest first)
            return sorted(
                conversations.values(),
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
        except Exception as e:
            print(f"Error listing conversations: {str(e)}")
            return []
