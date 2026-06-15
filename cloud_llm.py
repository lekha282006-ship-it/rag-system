"""
Cloud LLM Handler for Groq API.
Fast cloud inference with usage tracking.
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

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env file"
            )

        self.client = Groq(api_key=api_key)

        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        logger.info(f"CloudLLMHandler initialized with model: {model}")

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def generate_response(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:

        if not query:
            raise ValueError("Query cannot be empty")

        if system_prompt is None:
            system_prompt = """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not in the context, clearly say:
'I could not find this information in the provided sources.'
"""

        prompt = f"""
Context:
{context}

Question:
{query}
"""

        start_time = datetime.now()

        try:

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
            logger.error(f"Groq API error: {str(e)}")
            raise

    def get_usage_stats(self) -> Dict[str, Any]:

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
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0