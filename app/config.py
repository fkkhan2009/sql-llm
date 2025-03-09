"""
Updated application settings with LLM provider configuration
"""
import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from enum import Enum

# Load environment variables from .env file
load_dotenv()

class LLMProviderEnum(str, Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    APP_NAME: str = "LLM QA Application"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Document processing settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Vector database settings
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # LLM provider settings
    LLM_PROVIDER: LLMProviderEnum = LLMProviderEnum(os.getenv("LLM_PROVIDER", "ollama").lower())
    LLM_MODEL: Optional[str] = os.getenv("LLM_MODEL")  # Model-specific name based on provider
    
    # Embedding model settings
    EMBEDDING_PROVIDER: Optional[str] = os.getenv("EMBEDDING_PROVIDER")  # Defaults to LLM_PROVIDER if not set
    EMBEDDING_MODEL: Optional[str] = os.getenv("EMBEDDING_MODEL")  # Model-specific name based on provider
    
    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama2")
    OLLAMA_EMBEDDING_MODEL: Optional[str] = os.getenv("OLLAMA_EMBEDDING_MODEL")
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_ORG_ID: Optional[str] = os.getenv("OPENAI_ORG_ID")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
    
    # Anthropic settings
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    
    # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Optional[str] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    
    # Snowflake settings
    CONNECT_TO_DB: bool = os.getenv("CONNECT_TO_DB", "False").lower() == "true"
    SNOWFLAKE_ACCOUNT: Optional[str] = os.getenv("SNOWFLAKE_ACCOUNT")
    SNOWFLAKE_USER: Optional[str] = os.getenv("SNOWFLAKE_USER")
    SNOWFLAKE_PASSWORD: Optional[str] = os.getenv("SNOWFLAKE_PASSWORD")
    SNOWFLAKE_PASSWORD_FILE: Optional[str] = os.getenv("SNOWFLAKE_PASSWORD_FILE")
    SNOWFLAKE_PRIVATE_KEY_FILE: Optional[str] = os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE") 
    SNOWFLAKE_PRIVATE_KEY_PASSPHRASE: Optional[str] = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    SNOWFLAKE_DATABASE: Optional[str] = os.getenv("SNOWFLAKE_DATABASE")
    SNOWFLAKE_SCHEMA: Optional[str] = os.getenv("SNOWFLAKE_SCHEMA")
    SNOWFLAKE_WAREHOUSE: Optional[str] = os.getenv("SNOWFLAKE_WAREHOUSE")
    SNOWFLAKE_ROLE: Optional[str] = os.getenv("SNOWFLAKE_ROLE")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get configuration for the selected LLM provider
        
        Returns:
            Dictionary with provider-specific configuration
        """
        if self.LLM_PROVIDER == LLMProviderEnum.OLLAMA:
            return {
                "base_url": self.OLLAMA_BASE_URL,
                "model": self.LLM_MODEL or self.OLLAMA_MODEL,
                "embedding_model": self.OLLAMA_EMBEDDING_MODEL
            }
        elif self.LLM_PROVIDER == LLMProviderEnum.OPENAI:
            return {
                "api_key": self.OPENAI_API_KEY,
                "model": self.LLM_MODEL or self.OPENAI_MODEL,
                "embedding_model": self.EMBEDDING_MODEL or self.OPENAI_EMBEDDING_MODEL,
                "org_id": self.OPENAI_ORG_ID,
                "base_url": self.OPENAI_BASE_URL
            }
        elif self.LLM_PROVIDER == LLMProviderEnum.ANTHROPIC:
            return {
                "api_key": self.ANTHROPIC_API_KEY,
                "model": self.LLM_MODEL or self.ANTHROPIC_MODEL,
                "embedding_model": self.EMBEDDING_MODEL,  # Use OpenAI embeddings by default
                "base_url": self.ANTHROPIC_BASE_URL
            }
        elif self.LLM_PROVIDER == LLMProviderEnum.AZURE_OPENAI:
            return {
                "api_key": self.AZURE_OPENAI_API_KEY,
                "endpoint": self.AZURE_OPENAI_ENDPOINT,
                "deployment_name": self.LLM_MODEL or self.AZURE_OPENAI_DEPLOYMENT,
                "api_version": self.AZURE_OPENAI_API_VERSION,
                "embedding_deployment": self.EMBEDDING_MODEL or self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            }
        
        # Default empty config
        return {}
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get configuration for the embedding provider
        
        Returns:
            Dictionary with provider-specific configuration for embeddings
        """
        # If no specific embedding provider is set, use the LLM provider
        embedding_provider = self.EMBEDDING_PROVIDER or self.LLM_PROVIDER
        
        if embedding_provider == LLMProviderEnum.OLLAMA:
            return {
                "base_url": self.OLLAMA_BASE_URL,
                "model": self.EMBEDDING_MODEL or self.OLLAMA_MODEL
            }
        elif embedding_provider == LLMProviderEnum.OPENAI:
            return {
                "api_key": self.OPENAI_API_KEY,
                "model": self.EMBEDDING_MODEL or self.OPENAI_EMBEDDING_MODEL,
                "org_id": self.OPENAI_ORG_ID,
                "base_url": self.OPENAI_BASE_URL
            }
        elif embedding_provider == LLMProviderEnum.AZURE_OPENAI:
            return {
                "api_key": self.AZURE_OPENAI_API_KEY,
                "endpoint": self.AZURE_OPENAI_ENDPOINT,
                "deployment_name": self.EMBEDDING_MODEL or self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                "api_version": self.AZURE_OPENAI_API_VERSION
            }
        
        # Default: if Anthropic is selected but no embedding provider,
        # fall back to OpenAI for embeddings
        if self.LLM_PROVIDER == LLMProviderEnum.ANTHROPIC and self.OPENAI_API_KEY:
            return {
                "api_key": self.OPENAI_API_KEY,
                "model": self.EMBEDDING_MODEL or self.OPENAI_EMBEDDING_MODEL,
                "org_id": self.OPENAI_ORG_ID,
                "base_url": self.OPENAI_BASE_URL
            }
        
        # If no specific configuration is found, return empty config
        # and let the factory handle defaults
        return {}
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()
