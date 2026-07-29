"""OpenAI / OpenRouter judge that expects JSON responses."""

import json
import math
from openai import AsyncOpenAI
from .config import setup_credentials, OPENROUTER_BASE_URL

# Set up credentials
config = setup_credentials()

_client: AsyncOpenAI | None = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if config.use_openrouter:
            
            _client = AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=config.openrouter_api_key,
            )
        else:
            _client = AsyncOpenAI()
    return _client


class OpenAiJsonJudge:
    """Judge that expects structured JSON responses from the LLM.

    Used for pairwise comparison and other structured evaluation tasks where
    the response needs to be parsed as JSON rather than single-token logprobs.
    """

    def __init__(
        self,
        model: str,
        prompt_template: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        """Initialize JSON judge.

        Args:
            model:           OpenAI / OpenRouter model ID (e.g., "openai/gpt-4o-mini")
            prompt_template: Prompt template with {placeholders} for format()
            max_new_tokens:  Maximum tokens in the judge's response (default 256)
            temperature:     Sampling temperature (default 0.0 for deterministic output)
        """
        self.model = model
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    async def judge(self, **kwargs) -> dict:
        """Run judge and parse JSON response.

        Args:
            **kwargs: Values to fill into prompt_template via format()

        Returns:
            Parsed JSON dict from the model's response, or {} on error
        """
        messages = [
            {
                "role": "user",
                "content": self.prompt_template.format(**kwargs),
            }
        ]

        try:
            completion = await _get_client().chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                seed=0,
                response_format={"type": "json_object"},  # Force JSON mode
            )
            response_text = completion.choices[0].message.content
            if not response_text:
                return {}

            # Parse JSON
            result = json.loads(response_text)
            return result

        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract from markdown code block
            import re
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            return {"error": f"JSON decode failed: {e}"}

        except Exception as e:
            return {"error": f"API call failed: {e}"}

    async def __call__(self, **kwargs) -> dict:
        """Alias for judge()."""
        return await self.judge(**kwargs)
