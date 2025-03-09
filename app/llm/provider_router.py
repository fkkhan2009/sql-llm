"""
Router for LLM provider selection and management
"""
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException

from app.llm.provider_factory import LLMProviderFactory, ProviderType
from app.llm.client import LLMClient
from app.dependencies import get_llm_client
from app.config import settings

# Pydantic models for requests and responses
from pydantic import BaseModel, Field

class ProviderConfig(BaseModel):
    """Configuration for an LLM provider"""
    provider_type: str
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Additional provider-specific fields can go here
    
    class Config:
        protected_namespaces = ()  # Allows field names that are Python reserved words

class ProviderInfo(BaseModel):
    """Information about the current LLM provider"""
    provider_name: str
    model_name: str
    model_type: str
    embedding_model: str
    context_window: int
    available_models: List[Dict[str, Any]] = []

# Create router
router = APIRouter(
    prefix="/api/llm",
    tags=["LLM Providers"]
)

@router.get("/providers")
async def list_providers():
    """
    List all available LLM providers
    """
    providers = [
        {"type": ProviderType.OLLAMA, "name": "Ollama (Local)"},
        {"type": ProviderType.OPENAI, "name": "OpenAI"},
        {"type": ProviderType.ANTHROPIC, "name": "Anthropic (Claude)"},
        {"type": ProviderType.AZURE_OPENAI, "name": "Azure OpenAI"}
    ]
    
    return {"providers": providers}

@router.get("/current", response_model=ProviderInfo)
async def get_current_provider(
    llm_client: LLMClient = Depends(get_llm_client)
):
    """
    Get information about the current LLM provider
    """
    if not llm_client:
        raise HTTPException(status_code=500, detail="LLM client not initialized")
    
    provider_info = llm_client.provider_info
    
    # Get available models if possible
    try:
        models = llm_client.list_models()
    except:
        models = []
    
    return {
        "provider_name": provider_info["provider_name"],
        "model_name": provider_info["model_name"],
        "model_type": provider_info["model_type"],
        "embedding_model": provider_info["embedding_model"],
        "context_window": provider_info["context_window"],
        "available_models": models
    }

@router.post("/select")
async def select_provider(
    config: ProviderConfig,
    llm_client: LLMClient = Depends(get_llm_client)
):
    """
    Select and configure an LLM provider
    
    Note: This is a temporary switch for the current session.
    For a permanent change, update the environment variables or .env file.
    """
    if not llm_client:
        raise HTTPException(status_code=500, detail="LLM client not initialized")
    
    try:
        # Remove None values from config
        provider_config = {k: v for k, v in config.dict().items() if v is not None}
        
        # Create new provider
        provider = LLMProviderFactory.create_provider(
            provider_type=config.provider_type,
            config=provider_config
        )
        
        # Replace current provider in LLM client
        llm_client.provider = provider
        
        # Get updated provider info
        provider_info = llm_client.provider_info
        
        return {
            "success": True,
            "message": f"Successfully switched to {provider_info['provider_name']} provider",
            "provider": provider_info
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to select provider: {str(e)}"
        )

@router.post("/test")
async def test_provider(
    message: str = "Hello, this is a test message. Please respond briefly.",
    llm_client: LLMClient = Depends(get_llm_client)
):
    """
    Test the current LLM provider with a simple message
    """
    if not llm_client:
        raise HTTPException(status_code=500, detail="LLM client not initialized")
    
    try:
        # Generate with details for timing information
        response = llm_client.generate(
            prompt=message,
            temperature=0.7,
            max_tokens=100,
            with_details=True
        )
        
        return {
            "success": True,
            "response": response
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Provider test failed: {str(e)}"
        )
