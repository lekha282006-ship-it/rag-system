"""
Local LLM Handler for Ollama integration.
Provides a production-ready interface to Ollama's local LLM API with error handling,
logging, and optional token counting.
"""

import requests
import logging
from typing import Optional, Dict, Any
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLMHandler:
    """Handler for Ollama local LLM interactions with error handling and logging.
    
    This class provides a clean interface to Ollama's local LLM API, handling:
    - Ollama API communication (localhost:11434)
    - Error handling and connection checking
    - Token counting (estimation)
    - Request/response logging
    - Model management
    """
    
    def __init__(self, model: str = "mistral", 
                 api_url: str = "http://localhost:11434",
                 timeout: int = 300):
        """Initialize the LocalLLMHandler.
        
        Args:
            model: Ollama model to use (default: mistral)
            api_url: Ollama API URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 300)
            
        Raises:
            ConnectionError: If Ollama is not running
        """
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        
        # Usage tracking (free, but track for monitoring)
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Check if Ollama is running
        if not self._is_ollama_running():
            raise ConnectionError(
                f"Ollama is not running at {api_url}. "
                "Please start Ollama with: ollama run mistral"
            )
        
        logger.info(f"LocalLLMHandler initialized with model: {model} at {api_url}")
    
    def _is_ollama_running(self) -> bool:
        """Check if Ollama API is running and accessible.
        
        Returns:
            True if Ollama is running, False otherwise
        """
        try:
            response = requests.get(
                f"{self.api_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except Exception as e:
            logger.warning(f"Error checking Ollama status: {e}")
            return False
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Note: This is an approximation. Actual token count may vary.
        Approximation: 1 token ≈ 4 characters for English text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters
        estimated_tokens = max(1, len(text) // 4)
        return estimated_tokens
    
    def generate_response(self, query: str, context: str, 
                         system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate a response using Ollama local LLM.
        
        Args:
            query: User's question or prompt
            context: Context information (e.g., retrieved documents)
            system_prompt: Optional system prompt to guide model's behavior
            
        Returns:
            Dictionary containing:
                - response: Generated text response
                - input_tokens: Estimated number of input tokens
                - output_tokens: Estimated number of output tokens
                - model: Model used
                - timestamp: Request timestamp
                
        Raises:
            ValueError: If query or context is empty
            requests.exceptions.RequestException: If API call fails
        """
        if not query or not context:
            raise ValueError("Query and context cannot be empty")
        
        # Build the prompt
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant that answers questions based on provided context. 
Be accurate, concise, and cite your sources when possible. If you don't know the answer based on the context, 
admit it rather than making up information."""
        
        # Combine system prompt, context, and query
        full_prompt = f"""{system_prompt}

Context: {context}

Question: {query}

Answer: """
        
        start_time = datetime.now()
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise requests.exceptions.RequestException(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
            
            # Extract response
            result = response.json()
            response_text = result.get("response", "")
            
            # Estimate tokens
            input_tokens = self.count_tokens(full_prompt)
            output_tokens = self.count_tokens(response_text)
            
            # Update usage tracking
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1
            
            # Log the API call
            logger.info(
                f"Ollama API call - Model: {self.model}, "
                f"Input tokens: {input_tokens}, Output tokens: {output_tokens}"
            )
            
            return {
                "response": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": self.model,
                "timestamp": start_time.isoformat(),
                "cost": 0.0  # Free! Local LLM
            }
            
        except requests.exceptions.Timeout:
            logger.error(f"Ollama API timeout (>{self.timeout}s)")
            raise
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running? (ollama run mistral)")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API call: {str(e)}")
            raise
    
    def stream_response(self, query: str, context: str) -> str:
        """Generate a streaming response using Ollama.
        
        Args:
            query: User's question or prompt
            context: Context information
            
        Yields:
            Text chunks as they are generated
        """
        full_prompt = f"""You are a helpful AI assistant.

Context: {context}

Question: {query}

Answer: """
        
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": True
                },
                timeout=self.timeout,
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = line.decode('utf-8')
                    result = eval(data)  # Parse JSON
                    yield result.get("response", "")
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "cost": 0.0,  # Always free with local Ollama
            "average_tokens_per_request": (
                (self.total_input_tokens + self.total_output_tokens) / self.total_requests 
                if self.total_requests > 0 else 0
            ),
            "model": self.model,
            "status": "✅ Running locally (FREE)" if self._is_ollama_running() else "❌ Offline"
        }
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics to zero."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        logger.info("Usage statistics reset")