"""
Conversation memory management
"""
from typing import List, Dict, Any, Optional, Tuple
import uuid
import time
import json
from datetime import datetime
from langchain.docstore.document import Document
from app.document_processing.vector_store import VectorStore
from app.llm.client import LLMClient

class Message:
    """Represents a single message in a conversation"""
    
    def __init__(
        self,
        role: str,  # "user", "assistant", "system"
        content: str,
        message_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a message
        
        Args:
            role: Role of the message sender
            content: Message content
            message_id: Unique message ID (generated if not provided)
            timestamp: Message timestamp (current time if not provided)
            metadata: Additional metadata
        """
        self.role = role
        self.content = content
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "role": self.role,
            "content": self.content,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            role=data["role"],
            content=data["content"],
            message_id=data.get("message_id"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {})
        )
    
    def to_document(self) -> Document:
        """Convert to LangChain Document for vector storage"""
        formatted_time = datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        return Document(
            page_content=self.content,
            metadata={
                "role": self.role,
                "message_id": self.message_id,
                "timestamp": self.timestamp,
                "formatted_time": formatted_time,
                **self.metadata
            }
        )

class ConversationMemory:
    """Manages conversation history and context building"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        conversation_id: str,
        workspace_id: str,
        max_context_messages: int = 10,
        summarization_threshold: int = 20,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize conversation memory
        
        Args:
            vector_store: Vector store for persistence
            conversation_id: Unique conversation identifier
            workspace_id: Workspace this conversation belongs to
            max_context_messages: Maximum messages to include in immediate context
            summarization_threshold: When to generate a summary (# of messages)
            llm_client: LLM client for summarization (required if using summarization)
        """
        self.vector_store = vector_store
        self.conversation_id = conversation_id
        self.workspace_id = workspace_id
        self.max_context_messages = max_context_messages
        self.summarization_threshold = summarization_threshold
        self.llm_client = llm_client
        
        # In-memory message cache
        self.recent_messages: List[Message] = []
        self.summary: Optional[str] = None
        
        # Collection name for this conversation
        self.collection_name = f"conversation_{workspace_id}_{conversation_id}"
        
        # Load existing messages if any
        self._load_recent_messages()
    
    def _load_recent_messages(self) -> None:
        """Load the most recent messages from storage"""
        # Query for messages with this conversation_id
        filter_dict = {
            "conversation_id": self.conversation_id,
            "workspace_id": self.workspace_id
        }
        
        # Try to get messages from vector store
        try:
            docs = self.vector_store.search_documents(
                query="",  # Empty query to get all
                filter_dict=filter_dict,
                top_k=self.max_context_messages
            )
            
            # Convert to Message objects and sort by timestamp
            messages = []
            for doc in docs:
                messages.append(Message(
                    role=doc.metadata.get("role", "unknown"),
                    content=doc.page_content,
                    message_id=doc.metadata.get("message_id"),
                    timestamp=doc.metadata.get("timestamp"),
                    metadata={k: v for k, v in doc.metadata.items() 
                              if k not in ["role", "message_id", "timestamp"]}
                ))
            
            # Sort by timestamp
            self.recent_messages = sorted(messages, key=lambda m: m.timestamp)
            
        except Exception as e:
            print(f"Error loading conversation messages: {str(e)}")
            self.recent_messages = []
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a new message to the conversation
        
        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
            metadata: Additional metadata
            
        Returns:
            The created Message object
        """
        # Create base metadata
        base_metadata = {
            "conversation_id": self.conversation_id,
            "workspace_id": self.workspace_id
        }
        
        # Merge with provided metadata
        if metadata:
            base_metadata.update(metadata)
        
        # Create new message
        message = Message(role=role, content=content, metadata=base_metadata)
        
        # Add to in-memory cache
        self.recent_messages.append(message)
        
        # Store in vector store
        doc = message.to_document()
        try:
            self.vector_store.add_documents([doc])
        except Exception as e:
            print(f"Error storing message in vector store: {str(e)}")
        
        # Check if we need to summarize
        if len(self.recent_messages) >= self.summarization_threshold and self.llm_client:
            self._generate_summary()
        
        return message
    
    def _generate_summary(self) -> None:
        """Generate a summary of older messages"""
        if not self.llm_client:
            return  # Cannot summarize without LLM
            
        # Keep the most recent messages untouched
        messages_to_summarize = self.recent_messages[:-self.max_context_messages]
        if not messages_to_summarize:
            return
            
        # Format messages for summarization
        conversation_text = "\n".join([
            f"{msg.role.capitalize()}: {msg.content}" for msg in messages_to_summarize
        ])
        
        # Generate summary prompt
        prompt = f"""Summarize the following conversation concisely, capturing key points, 
questions asked, and information provided. Maintain the essential context needed for 
continuing the conversation naturally:

{conversation_text}

Concise summary:"""
        
        # Generate summary
        try:
            summary = self.llm_client.generate(prompt=prompt, temperature=0.3)
            self.summary = summary
            
            # Replace older messages with summary in memory
            self.recent_messages = self.recent_messages[-self.max_context_messages:]
            
            # Create a system message with the summary
            summary_msg = Message(
                role="system",
                content=f"Previous conversation summary: {summary}",
                metadata={
                    "conversation_id": self.conversation_id,
                    "workspace_id": self.workspace_id,
                    "is_summary": True,
                    "summarized_messages": len(messages_to_summarize)
                }
            )
            
            # Store summary message
            doc = summary_msg.to_document()
            self.vector_store.add_documents([doc])
            
        except Exception as e:
            print(f"Error generating conversation summary: {str(e)}")
    
    def get_context_for_llm(self) -> str:
        """
        Get the conversation context formatted for the LLM
        
        Returns:
            Formatted conversation context
        """
        context_parts = []
        
        # Add summary if available
        if self.summary:
            context_parts.append(f"Previous conversation summary: {self.summary}\n")
        
        # Add recent messages
        context_parts.append("Recent conversation:")
        for msg in self.recent_messages[-self.max_context_messages:]:
            context_parts.append(f"{msg.role.capitalize()}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages in standard format
        
        Returns:
            List of message dictionaries
        """
        return [msg.to_dict() for msg in self.recent_messages]
    
    def search_conversation(self, query: str, top_k: int = 5) -> List[Message]:
        """
        Search for relevant messages in the conversation history
        
        Args:
            query: Search query
            top_k: Number of messages to retrieve
            
        Returns:
            List of relevant messages
        """
        filter_dict = {
            "conversation_id": self.conversation_id,
            "workspace_id": self.workspace_id
        }
        
        docs = self.vector_store.search_documents(
            query=query,
            filter_dict=filter_dict,
            top_k=top_k
        )
        
        messages = []
        for doc in docs:
            messages.append(Message(
                role=doc.metadata.get("role", "unknown"),
                content=doc.page_content,
                message_id=doc.metadata.get("message_id"),
                timestamp=doc.metadata.get("timestamp"),
                metadata={k: v for k, v in doc.metadata.items() 
                          if k not in ["role", "message_id", "timestamp"]}
            ))
        
        return messages
