"""
Abstract interface for LLM providers
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncGenerator
import os
from enum import Enum


class ModelType(str, Enum):
    """Types of models supported by the provider"""
    COMPLETION = "completion"  # Text completion models
    CHAT = "chat"              # Chat-optimized models
    EMBEDDING = "embedding"    # Embedding models


class LLMProviderInterface(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate text using the LLM
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (for chat models)
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional sequences to stop generation
            stream: Whether to stream the response
            
        Returns:
            Generated text or response object
        """
        pass
    
    @abstractmethod
    def get_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Get embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    @abstractmethod
    def get_context_window(self) -> int:
        """
        Get the context window size of the model
        
        Returns:
            Context window size in tokens
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the provider name
        
        Returns:
            Provider name (e.g., "OpenAI", "Anthropic", "Ollama")
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the model name
        
        Returns:
            Model name (e.g., "gpt-4", "claude-2", "llama2")
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """
        Get the model type
        
        Returns:
            Model type (completion, chat, embedding)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_model_name(self) -> str:
        """
        Get the embedding model name
        
        Returns:
            Embedding model name
        """
        pass
    
    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from this provider
        
        Returns:
            List of model information dictionaries
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using the LLM with streaming support
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (for chat models)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional sequences to stop generation
            
        Yields:
            Tokens/chunks of generated text as they become available
        """
        pass
