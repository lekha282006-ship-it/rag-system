"""
Cloud LLM Handler for Groq API.
Fast cloud inference with usage tracking and dynamic context/conversational fallbacks.
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudLLMHandler:
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile"
    ):
        self.model = model

        # Securely fetch API Key from the .env file
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env file. Please check your setup."
            )

        self.client = Groq(api_key=api_key)

        # Usage and Token Metric Monitoring
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        logger.info(f"CloudLLMHandler initialized with model: {model}")

    def count_tokens(self, text: str) -> int:
        """Fallback character-based heuristic tool to approximate token counts if API fields drop."""
        return max(1, len(text) // 4)

    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:

        if not query:
            raise ValueError("Query cannot be empty")

        # Clean/strip context safely if provided as an empty string or None
        cleaned_context = context.strip() if context else ""

        # DYNAMIC SYSTEM PROMPT ROUTER:
        # If a custom prompt is provided by rag_core, use it.
        # Otherwise, dynamically pick between strict RAG mode and General Chat mode!
        if system_prompt is None:
            if cleaned_context:
                # Document context is active -> enforce strict boundary limits
                system_prompt = """You are a helpful AI assistant.
Answer ONLY using the provided context.
If the answer is not in the context, clearly say:
'I could not find this information in the provided sources.'"""
            else:
                # No documents loaded -> Allow natural general-knowledge chat behaviors
                system_prompt = """You are a helpful, versatile AI assistant. 
Answer the user's questions clearly, naturally, and accurately using your broad, general knowledge space."""

        # Package the prompt structure intelligently based on context state
        if cleaned_context:
            prompt = f"""Context:
{cleaned_context}

Question:
{query}"""
        else:
            prompt = query

        start_time = datetime.now()

        try:
            # Dispatch structural payload to Groq API infrastructure
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = response.choices[0].message.content

            # Track payload stats via native API tracking parameters
            input_tokens = getattr(
                response.usage,
                "prompt_tokens",
                self.count_tokens(prompt)
            )

            output_tokens = getattr(
                response.usage,
                "completion_tokens",
                self.count_tokens(response_text)
            )

            # Commit local increments to active handler tracking registers
            self.total_requests += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            return {
                "response": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": self.model,
                "timestamp": start_time.isoformat(),
                "cost": 0.0
            }

        except Exception as e:
            logger.error(f"Groq API error encountered: {str(e)}")
            raise

    def get_usage_stats(self) -> Dict[str, Any]:
        """Returns usage analysis figures for the application's sidebar system state layout monitor."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "average_tokens_per_request": (
                (self.total_input_tokens + self.total_output_tokens)
                / self.total_requests
                if self.total_requests > 0
                else 0
            ),
            "model": self.model,
            "status": "✅ Groq Cloud Connected"
        }

    def reset_usage_stats(self):
        """Resets structural usage logs back to 0."""
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0