from groq import Groq

from config import settings


class GroqClient:
    def __init__(self):
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        self.client = Groq(api_key=settings.groq_api_key, max_retries=2, timeout=60)
        self.last_usage: dict[str, int] = {}

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = settings.llm_temperature) -> str:
        response = self.client.chat.completions.create(
            model=settings.groq_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = response.usage
        self.last_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        } if usage else {}
        return (response.choices[0].message.content or "").strip()
