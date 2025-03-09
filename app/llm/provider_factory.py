"""
Factory for creating LLM providers
"""
from typing import Dict, Any, Optional, Type
import os
from enum import Enum

from app.llm.provider_interface import LLMProviderInterface
from app.llm.provider_implementations import (
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    AzureOpenAIProvider
)


class ProviderType(str, Enum):
    """Supported LLM provider types"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(
        provider_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> LLMProviderInterface:
        """
        Create an LLM provider instance
        
        Args:
            provider_type: Type of provider to create
            config: Configuration options for the provider
            
        Returns:
            LLM provider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        config = config or {}
        
        if provider_type.lower() == ProviderType.OLLAMA:
            return OllamaProvider(
                base_url=config.get("base_url", "http://localhost:11434"),
                model=config.get("model", "llama2"),
                embedding_model=config.get("embedding_model")
            )
        elif provider_type.lower() == ProviderType.OPENAI:
            return OpenAIProvider(
                api_key=config.get("api_key"),
                model=config.get("model", "gpt-4o"),
                embedding_model=config.get("embedding_model", "text-embedding-3-small"),
                org_id=config.get("org_id"),
                base_url=config.get("base_url")
            )
        elif provider_type.lower() == ProviderType.ANTHROPIC:
            return AnthropicProvider(
                api_key=config.get("api_key"),
                model=config.get("model", "claude-3-opus-20240229"),
                embedding_model=config.get("embedding_model"),
                base_url=config.get("base_url", "https://api.anthropic.com")
            )
        elif provider_type.lower() == ProviderType.AZURE_OPENAI:
            return AzureOpenAIProvider(
                api_key=config.get("api_key"),
                endpoint=config.get("endpoint"),
                deployment_name=config.get("deployment_name"),
                api_version=config.get("api_version", "2023-05-15"),
                embedding_deployment=config.get("embedding_deployment")
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def create_from_environment() -> LLMProviderInterface:
        """
        Create an LLM provider based on environment variables
        
        Environment variables:
        - LLM_PROVIDER: Provider type (ollama, openai, anthropic, azure_openai)
        - Various provider-specific env vars (see provider classes)
        
        Returns:
            LLM provider instance
            
        Raises:
            ValueError: If provider type is not specified or not supported
        """
        provider_type = os.environ.get("LLM_PROVIDER", "").lower()
        
        if not provider_type:
            # Default to Ollama if available
            try:
                provider = OllamaProvider()
                # Test connection
                provider.list_models()
                return provider
            except:
                # Fallback to OpenAI if API key is available
                if os.environ.get("OPENAI_API_KEY"):
                    return OpenAIProvider()
                else:
                    raise ValueError("No LLM provider specified and no default provider available")
        
        # Create provider based on type
        config = {}
        
        # Common config options
        if provider_type == ProviderType.OLLAMA:
            config = {
                "base_url": os.environ.get("OLLAMA_BASE_URL"),
                "model": os.environ.get("OLLAMA_MODEL"),
                "embedding_model": os.environ.get("OLLAMA_EMBEDDING_MODEL")
            }
        elif provider_type == ProviderType.OPENAI:
            config = {
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "model": os.environ.get("OPENAI_MODEL"),
                "embedding_model": os.environ.get("OPENAI_EMBEDDING_MODEL"),
                "org_id": os.environ.get("OPENAI_ORG_ID"),
                "base_url": os.environ.get("OPENAI_BASE_URL")
            }
        elif provider_type == ProviderType.ANTHROPIC:
            config = {
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                "model": os.environ.get("ANTHROPIC_MODEL"),
                "embedding_model": os.environ.get("ANTHROPIC_EMBEDDING_MODEL"),
                "base_url": os.environ.get("ANTHROPIC_BASE_URL")
            }
        elif provider_type == ProviderType.AZURE_OPENAI:
            config = {
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                "deployment_name": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
                "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
                "embedding_deployment": os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            }
        
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        return LLMProviderFactory.create_provider(provider_type, config)
