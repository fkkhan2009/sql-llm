"""
Implementations of LLM providers
"""
import os
import json
import requests
import tiktoken
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncGenerator
import logging

from app.llm.provider_interface import LLMProviderInterface, ModelType

# Set up logging
logger = logging.getLogger(__name__)


class OllamaProvider(LLMProviderInterface):
    """Provider for Ollama LLMs"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        embedding_model: Optional[str] = None
    ):
        """
        Initialize Ollama provider
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use for generation
            embedding_model: Model to use for embeddings (defaults to generation model)
        """
        self.base_url = base_url.rstrip("/")
        self._model_name = model
        self._embedding_model_name = embedding_model or model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Generate text using Ollama"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": stream
        }
        
        # Add optional parameters if provided
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["num_predict"] = max_tokens
        if stop_sequences:
            payload["stop"] = stop_sequences
            
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            if stream:
                return response
            else:
                result = response.json()
                return result.get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting embeddings from Azure OpenAI: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * 1536 for _ in texts]  # Default size for OpenAI embeddings
    
    def get_token_count(self, text: str) -> int:
        """Count tokens using tiktoken if available"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.error(f"Error counting tokens: {str(e)}")
        
        # Fallback to character-based estimate
        return len(text) // 4
    
    def get_context_window(self) -> int:
        """Get the context window size of the model"""
        # We don't have a direct way to query this from Azure, so we use common values
        # Based on deployment name, try to guess the model
        if "gpt-4" in self.deployment_name.lower():
            if "32k" in self.deployment_name.lower():
                return 32768
            elif "turbo" in self.deployment_name.lower() or "vision" in self.deployment_name.lower():
                return 128000
            else:
                return 8192
        elif "gpt-35" in self.deployment_name.lower() or "gpt-3.5" in self.deployment_name.lower():
            if "16k" in self.deployment_name.lower():
                return 16384
            else:
                return 4096
        
        # Default context window size
        return 4096
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available deployments from Azure OpenAI"""
        # Azure OpenAI doesn't have a straightforward way to list deployments via API
        # Return the configured deployment
        return [
            {
                "id": self.deployment_name,
                "name": self.deployment_name,
                "embedding_model": self.embedding_deployment or "None"
            }
        ]
    
    @property
    def provider_name(self) -> str:
        return "Azure OpenAI"
    
    @property
    def model_name(self) -> str:
        return self.deployment_name
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.CHAT  # Azure OpenAI typically uses chat models
    
    @property
    def embedding_model_name(self) -> str:
        return self.embedding_deployment or "Unknown"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Ollama"""
        url = f"{self.base_url}/api/embeddings"
        
        all_embeddings = []
        for text in texts:
            payload = {
                "model": self._embedding_model_name,
                "prompt": text
            }
            
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                if "embedding" in result:
                    all_embeddings.append(result["embedding"])
                else:
                    logger.error(f"No embedding in response for text: {text[:50]}...")
                    # Return a zero vector as fallback
                    all_embeddings.append([0.0] * 768)  # Default size
            except requests.exceptions.RequestException as e:
                logger.error(f"Error getting embedding from Ollama: {str(e)}")
                # Return a zero vector as fallback
                all_embeddings.append([0.0] * 768)  # Default size
        
        return all_embeddings
    
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for Ollama models
        Note: This is a rough estimate as Ollama doesn't provide a tokenizer
        """
        # Simple heuristic: ~4 characters per token for most models
        return len(text) // 4
    
    def get_context_window(self) -> int:
        """Get the context window size of the model"""
        try:
            url = f"{self.base_url}/api/show"
            response = requests.post(url, json={"name": self._model_name})
            response.raise_for_status()
            
            model_info = response.json()
            # Context window size might be in different fields depending on Ollama version
            context_window = (
                model_info.get("parameters", {}).get("context_length") or 
                model_info.get("parameters", {}).get("num_ctx") or 
                4096  # Default if not found
            )
            return int(context_window)
        except Exception as e:
            logger.error(f"Error getting context window size: {str(e)}")
            # Default to a safe value if we can't determine
            return 4096
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            return models
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    @property
    def provider_name(self) -> str:
        return "Ollama"
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.COMPLETION
    
    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model_name

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text using Ollama with streaming"""
        import aiohttp
        
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": True  # Enable streaming
        }
        
        # Add optional parameters
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["num_predict"] = max_tokens
        if stop_sequences:
            payload["stop"] = stop_sequences
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    
                    # Ollama streams JSON objects, one per line
                    async for line in response.content:
                        if not line:
                            continue
                            
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Error in Ollama streaming: {str(e)}")
            raise


class OpenAIProvider(LLMProviderInterface):
    """Provider for OpenAI LLMs"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        org_id: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize OpenAI provider
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use for generation
            embedding_model: Model to use for embeddings
            org_id: Optional organization ID
            base_url: Optional API base URL for custom endpoints
        """
        # Get API key either from params or environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required (set OPENAI_API_KEY or pass api_key)")
        
        self._model_name = model
        self._embedding_model_name = embedding_model
        self.org_id = org_id or os.environ.get("OPENAI_ORG_ID")
        self.base_url = base_url or "https://api.openai.com/v1"
        
        # Model-specific configurations
        self.chat_models = [
            "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-4-32k", 
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
        ]
        self.completion_models = ["davinci", "curie", "babbage", "ada"]
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except:
            # Fallback to cl100k_base for newer models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Generate text using OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.org_id:
            headers["OpenAI-Organization"] = self.org_id
        
        # Check if using a chat model
        is_chat_model = any(self._model_name.startswith(model) for model in self.chat_models)
        
        if is_chat_model:
            # Chat completion endpoint
            url = f"{self.base_url}/chat/completions"
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self._model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": stream
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if stop_sequences:
                payload["stop"] = stop_sequences
        else:
            # Text completion endpoint
            url = f"{self.base_url}/completions"
            
            # If system prompt is provided, prepend it to the prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            payload = {
                "model": self._model_name,
                "prompt": full_prompt,
                "temperature": temperature,
                "stream": stream
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if stop_sequences:
                payload["stop"] = stop_sequences
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            if stream:
                return response
            else:
                result = response.json()
                
                # Handle different response formats
                if is_chat_model:
                    return result["choices"][0]["message"]["content"]
                else:
                    return result["choices"][0]["text"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return f"Error calling OpenAI API: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI"""
        url = f"{self.base_url}/embeddings"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.org_id:
            headers["OpenAI-Organization"] = self.org_id
        
        payload = {
            "model": self._embedding_model_name,
            "input": texts
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract embeddings in the same order as input texts
            embeddings = []
            for i in range(len(texts)):
                embedding_data = next(
                    (item for item in result["data"] if item["index"] == i), 
                    None
                )
                if embedding_data:
                    embeddings.append(embedding_data["embedding"])
                else:
                    logger.error(f"No embedding found for text at index {i}")
                    # Return a zero vector as fallback
                    embeddings.append([0.0] * 1536)  # Default size for OpenAI embeddings
            
            return embeddings
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting embeddings from OpenAI: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * 1536 for _ in texts]  # Default size for OpenAI embeddings
    
    def get_token_count(self, text: str) -> int:
        """Count tokens using tiktoken"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            # Fallback to character-based estimate
            return len(text) // 4
    
    def get_context_window(self) -> int:
        """Get the context window size of the model"""
        # Context window sizes for different models
        model_to_context_length = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384
        }
        
        # Check for exact matches
        if self._model_name in model_to_context_length:
            return model_to_context_length[self._model_name]
        
        # Check for model name prefixes
        for model_prefix, context_length in model_to_context_length.items():
            if self._model_name.startswith(model_prefix):
                return context_length
                
        # Default context window size
        return 4096
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models from OpenAI"""
        url = f"{self.base_url}/models"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.org_id:
            headers["OpenAI-Organization"] = self.org_id
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            return result.get("data", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models from OpenAI: {str(e)}")
            return []
    
    @property
    def provider_name(self) -> str:
        return "OpenAI"
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def model_type(self) -> ModelType:
        if any(self._model_name.startswith(model) for model in self.chat_models):
            return ModelType.CHAT
        return ModelType.COMPLETION
    
    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model_name

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation from OpenAI"""
        from openai import AsyncOpenAI
        
        # Initialize async client
        client = AsyncOpenAI(
            api_key=self.api_key,
            organization=self.org_id,
            base_url=self.base_url
        )
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            stream = await client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in OpenAI streaming: {str(e)}")
            raise


class AnthropicProvider(LLMProviderInterface):
    """Provider for Anthropic LLMs (Claude)"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        embedding_model: Optional[str] = None,  # Anthropic doesn't have embedding models yet
        base_url: str = "https://api.anthropic.com"
    ):
        """
        Initialize Anthropic provider
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name to use (e.g., "claude-3-opus-20240229")
            embedding_model: Not used for Anthropic (will use OpenAI embeddings)
            base_url: Anthropic API base URL
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required (set ANTHROPIC_API_KEY or pass api_key)")
        
        self._model_name = model
        self._embedding_model_name = embedding_model or "text-embedding-3-small"  # Default to OpenAI
        self.base_url = base_url.rstrip("/")
        
        # Context window sizes for Claude models
        self.model_context_lengths = {
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000, 
            "claude-instant-1.2": 100000,
            "claude-instant-1.1": 100000,
            "claude-instant-1": 100000,
            "claude-1.3": 100000,
            "claude-1.2": 100000,
            "claude-1.1": 100000,
            "claude-1.0": 100000
        }
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Generate text using Anthropic API"""
        url = f"{self.base_url}/v1/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            if stream:
                return response
            else:
                result = response.json()
                return result["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            return f"Error calling Anthropic API: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings (Anthropic doesn't provide an embeddings API yet, so this is a fallback)
        This method should be overridden with a different embedding provider
        """
        logger.warning("Anthropic doesn't provide embeddings. Using OpenAI embedding API instead.")
        
        # Try to use OpenAI embeddings if API key is available
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            openai_provider = OpenAIProvider(
                api_key=openai_api_key,
                embedding_model=self._embedding_model_name
            )
            return openai_provider.get_embeddings(texts)
        
        # Fallback: return zero vectors
        logger.error("No OpenAI API key available for embeddings. Returning zero vectors.")
        return [[0.0] * 1536 for _ in texts]  # Default size for OpenAI embeddings
    
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for Claude models
        Note: This is a rough estimate as Anthropic doesn't provide a public tokenizer
        """
        # Claude uses approximately 4 characters per token
        return len(text) // 4
    
    def get_context_window(self) -> int:
        """Get the context window size of the model"""
        # Check for exact model name match
        if self._model_name in self.model_context_lengths:
            return self.model_context_lengths[self._model_name]
        
        # Check for model name prefixes
        for model_prefix, context_length in self.model_context_lengths.items():
            if self._model_name.startswith(model_prefix):
                return context_length
        
        # Default context window size for Claude
        return 100000
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models from Anthropic"""
        # Anthropic doesn't have a models listing endpoint, so we return known models
        claude_models = [
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "context_length": 200000},
            {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet", "context_length": 200000},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "context_length": 200000},
            {"id": "claude-2.1", "name": "Claude 2.1", "context_length": 200000},
            {"id": "claude-2.0", "name": "Claude 2.0", "context_length": 100000},
            {"id": "claude-instant-1.2", "name": "Claude Instant 1.2", "context_length": 100000}
        ]
        return claude_models
    
    @property
    def provider_name(self) -> str:
        return "Anthropic"
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.CHAT
    
    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model_name

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation from Anthropic"""
        import anthropic
        
        # Initialize async client
        client = anthropic.AsyncAnthropic(
            api_key=self.api_key
        )
        
        try:
            # Combine system prompt and user prompt if both provided
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            stream = await client.messages.create(
                model=self._model_name,
                max_tokens=max_tokens or 4096,
                temperature=temperature,
                messages=[{"role": "user", "content": full_prompt}],
                stream=True
            )
            
            async for chunk in stream:
                if chunk.delta.text:
                    yield chunk.delta.text
                    
        except Exception as e:
            logger.error(f"Error in Anthropic streaming: {str(e)}")
            raise


class AzureOpenAIProvider(LLMProviderInterface):
    """Provider for Azure OpenAI LLMs"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: str = "2023-05-15",
        embedding_deployment: Optional[str] = None
    ):
        """
        Initialize Azure OpenAI provider
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_API_KEY env var)
            endpoint: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT env var)
            deployment_name: Azure OpenAI deployment name (defaults to AZURE_OPENAI_DEPLOYMENT env var)
            api_version: Azure OpenAI API version
            embedding_deployment: Azure OpenAI embedding deployment name 
                                  (defaults to AZURE_OPENAI_EMBEDDING_DEPLOYMENT env var)
        """
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required")
            
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint is required")
        
        self.deployment_name = deployment_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not self.deployment_name:
            raise ValueError("Azure OpenAI deployment name is required")
            
        self.embedding_deployment = (
            embedding_deployment or 
            os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )
            
        self.api_version = api_version
        self.endpoint = self.endpoint.rstrip("/")
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Common for GPT models
        except:
            self.tokenizer = None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Generate text using Azure OpenAI API"""
        # Azure OpenAI uses a different URL structure
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            if stream:
                return response
            else:
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Azure OpenAI API: {str(e)}")
            return f"Error calling Azure OpenAI API: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Azure OpenAI API"""
        if not self.embedding_deployment:
            logger.error("Azure OpenAI embedding deployment name is not set")
            return [[0.0] * 1536 for _ in texts]  # Default size for OpenAI embeddings
        
        url = f"{self.endpoint}/openai/deployments/{self.embedding_deployment}/embeddings?api-version={self.api_version}"
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": texts
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract embeddings in the same order as input texts
            embeddings = []
            for i in range(len(texts)):
                embedding_data = next(
                    (item for item in result["data"] if item["index"] == i), 
                    None
                )
                if embedding_data:
                    embeddings.append(embedding_data["embedding"])
                else:
                    logger.error(f"No embedding found for text at index {i}")
                    # Return a zero vector as fallback
                    embeddings.append([0.0] * 1536)  # Default size for OpenAI embeddings
            
            return embeddings
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting embeddings from Azure OpenAI: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * 1536 for _ in texts]  # Default size for OpenAI embeddings
    
    def get_token_count(self, text: str) -> int:
        """Count tokens using tiktoken"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            # Fallback to character-based estimate
            return len(text) // 4
    
    def get_context_window(self) -> int:
        """Get the context window size of the model"""
        # Context window sizes for different models
        model_to_context_length = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384
        }
        
        # Check for exact matches
        if self.deployment_name in model_to_context_length:
            return model_to_context_length[self.deployment_name]
        
        # Check for model name prefixes
        for model_prefix, context_length in model_to_context_length.items():
            if self.deployment_name.startswith(model_prefix):
                return context_length
                
        # Default context window size
        return 4096
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models from Azure OpenAI"""
        url = f"{self.endpoint}/openai/deployments"
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            return result.get("data", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models from Azure OpenAI: {str(e)}")
            return []
    
    @property
    def provider_name(self) -> str:
        return "Azure OpenAI"
    
    @property
    def model_name(self) -> str:
        return self.deployment_name
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.CHAT  # Azure OpenAI typically uses chat models
    
    @property
    def embedding_model_name(self) -> str:
        return self.embedding_deployment or "Unknown"

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation from Azure OpenAI"""
        from openai import AsyncAzureOpenAI
        from openai import APIError, APIConnectionError, RateLimitError
        
        # Initialize async client
        client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            stream = await client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except RateLimitError as e:
            logger.error(f"Azure OpenAI rate limit exceeded: {str(e)}")
            raise
        except APIConnectionError as e:
            logger.error(f"Connection error to Azure OpenAI: {str(e)}")
            raise
        except APIError as e:
            logger.error(f"Azure OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in Azure OpenAI streaming: {str(e)}")
            raise