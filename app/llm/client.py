"""
Updated LLM client with provider abstraction
"""
from typing import Dict, List, Any, Optional, Union
import logging
import time

from app.llm.provider_interface import LLMProviderInterface
from app.llm.provider_factory import LLMProviderFactory, ProviderType

# Set up logging
logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with LLMs through various providers"""
    
    def __init__(
        self,
        provider: Optional[LLMProviderInterface] = None,
        provider_type: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LLM client
        
        Args:
            provider: LLM provider instance (if already created)
            provider_type: Type of provider to create (if provider not provided)
            provider_config: Configuration for provider creation
        """
        if provider:
            self.provider = provider
        elif provider_type:
            self.provider = LLMProviderFactory.create_provider(
                provider_type=provider_type,
                config=provider_config
            )
        else:
            # Create from environment
            self.provider = LLMProviderFactory.create_from_environment()
            
        # Log the initialized provider
        logger.info(
            f"Initialized LLM client with provider: {self.provider.provider_name}, "
            f"model: {self.provider.model_name}"
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        with_details: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate text using the LLM
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional sequences to stop generation
            stream: Whether to stream the response
            with_details: Whether to return detailed info including timing
            
        Returns:
            Generated text or response object with details
        """
        start_time = time.time()
        token_count = self.provider.get_token_count(prompt)
        
        if system_prompt:
            system_token_count = self.provider.get_token_count(system_prompt)
            token_count += system_token_count
            
        context_window = self.provider.get_context_window()
        
        # Check if we're approaching the context window limit
        if token_count > 0.8 * context_window:
            logger.warning(
                f"Prompt token count ({token_count}) is approaching "
                f"context window limit ({context_window})"
            )
        
        # Generate text
        try:
            response = self.provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                stream=stream
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            if with_details:
                if isinstance(response, str):
                    response_token_count = self.provider.get_token_count(response)
                    
                    return {
                        "text": response,
                        "provider": self.provider.provider_name,
                        "model": self.provider.model_name,
                        "input_tokens": token_count,
                        "output_tokens": response_token_count,
                        "total_tokens": token_count + response_token_count,
                        "generation_time": generation_time
                    }
                return response
            
            return response
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            if with_details:
                return {
                    "text": f"Error generating text: {str(e)}",
                    "provider": self.provider.provider_name,
                    "model": self.provider.model_name,
                    "input_tokens": token_count,
                    "output_tokens": 0,
                    "total_tokens": token_count,
                    "generation_time": time.time() - start_time,
                    "error": str(e)
                }
            return f"Error generating text: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.provider.get_embeddings(texts)
    
    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return self.provider.get_token_count(text)
    
    def get_context_window(self) -> int:
        """
        Get the context window size of the model
        
        Returns:
            Context window size in tokens
        """
        return self.provider.get_context_window()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from the provider
        
        Returns:
            List of model information dictionaries
        """
        return self.provider.list_models()
    
    @property
    def provider_info(self) -> Dict[str, Any]:
        """
        Get provider information
        
        Returns:
            Dictionary with provider information
        """
        return {
            "provider_name": self.provider.provider_name,
            "model_name": self.provider.model_name,
            "model_type": self.provider.model_type,
            "embedding_model": self.provider.embedding_model_name,
            "context_window": self.get_context_window()
        }
