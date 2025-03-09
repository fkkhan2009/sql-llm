"""
Router for conversation-related endpoints
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Path

from app.models.conversation import (
    ConversationCreate, ConversationResponse,
    MessageRequest, MessageResponse
)
from app.llm.context_builder import ContextBuilder
from app.llm.prompt_templates import PromptTemplates
from app.dependencies import get_conversation_components

router = APIRouter(
    prefix="/api/workspaces/{workspace_id}/conversations",
    tags=["Conversations"]
)


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationCreate,
    workspace_id: str = Path(..., description="The ID of the workspace"),
    components: Dict[str, Any] = Depends(get_conversation_components)
):
    """
    Create a new conversation in a workspace
    """
    workspace_manager = components["workspace_manager"]
    conversation_manager = components["conversation_manager"]
    
    # Check if workspace exists
    if not workspace_manager.get_workspace(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    try:
        conversation_id = conversation_manager.create_conversation(
            workspace_id=workspace_id,
            name=request.name,
            metadata=request.metadata
        )
        
        # Get conversation name
        name = request.name or "New conversation"
        
        return {
            "conversation_id": conversation_id,
            "workspace_id": workspace_id,
            "name": name,
            "metadata": request.metadata or {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("", response_model=List[Dict[str, Any]])
async def list_conversations(
    workspace_id: str = Path(..., description="The ID of the workspace"),
    components: Dict[str, Any] = Depends(get_conversation_components)
):
    """
    List all conversations in a workspace
    """
    workspace_manager = components["workspace_manager"]
    conversation_manager = components["conversation_manager"]
    
    # Check if workspace exists
    workspace = workspace_manager.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    conversations = conversation_manager.list_conversations(workspace_id)
    return conversations


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def add_message(
    message: MessageRequest,
    workspace_id: str = Path(..., description="The ID of the workspace"),
    conversation_id: str = Path(..., description="The ID of the conversation"),
    components: Dict[str, Any] = Depends(get_conversation_components)
):
    """
    Add a message to a conversation and get a response
    """
    workspace_manager = components["workspace_manager"]
    conversation_manager = components["conversation_manager"]
    snowflake_connector = components["snowflake_connector"]
    ollama_client = components["ollama_client"]
    
    # Get the conversation
    conversation = conversation_manager.get_conversation(workspace_id, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Add user message
    conversation.add_message(
        role=message.role,
        content=message.content,
        metadata=message.metadata
    )
    
    # Generate response using the LLM
    if message.role == "user":
        # Get conversation context
        conversation_context = conversation.get_context_for_llm()
        
        # Get document and database context
        context_builder = ContextBuilder(
            vector_store=workspace_manager.get_vector_store(workspace_id),
            snowflake_connector=snowflake_connector
        )
        doc_db_context = context_builder.build_context(
            query=message.content,
            include_docs=True,
            include_db_schema=True
        )
        
        # Combine contexts
        combined_context = f"{conversation_context}\n\n{doc_db_context}" if doc_db_context else conversation_context
        
        # Use the LLM to generate a response
        is_sql_query = context_builder.is_sql_query(message.content)
        
        prompt = PromptTemplates.combined_prompt(
            query=message.content,
            context=combined_context,
            is_sql_query=is_sql_query
        )
        
        system_prompt = PromptTemplates.system_prompt()
        response_text = ollama_client.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        # Add assistant message
        assistant_message = conversation.add_message(
            role="assistant",
            content=response_text
        )
        
        # Return the assistant message
        return {
            "message_id": assistant_message.message_id,
            "content": assistant_message.content,
            "role": assistant_message.role,
            "timestamp": assistant_message.timestamp,
            "metadata": assistant_message.metadata
        }
    else:
        # For non-user messages, just return the added message
        last_message = conversation.recent_messages[-1]
        return {
            "message_id": last_message.message_id,
            "content": last_message.content,
            "role": last_message.role,
            "timestamp": last_message.timestamp,
            "metadata": last_message.metadata
        }


@router.get("/{conversation_id}/messages", response_model=List[Dict[str, Any]])
async def get_messages(
    workspace_id: str = Path(..., description="The ID of the workspace"),
    conversation_id: str = Path(..., description="The ID of the conversation"),
    components: Dict[str, Any] = Depends(get_conversation_components)
):
    """
    Get all messages in a conversation
    """
    conversation_manager = components["conversation_manager"]
    
    conversation = conversation_manager.get_conversation(workspace_id, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation.get_messages()
