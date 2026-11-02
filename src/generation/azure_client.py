from config import settings


class AzureClient:
    """Azure OpenAI chat client with the same `complete()` interface as GroqClient.

    It is the backup provider: the router only reaches it when Groq has failed.
    """

    def __init__(self):
        if not settings.azure_enabled:
            raise RuntimeError(
                "Azure OpenAI is not configured. Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT "
                "and AZURE_OPENAI_DEPLOYMENT in .env to enable the backup provider."
            )
        from openai import AzureOpenAI  # imported here so the package is only needed when Azure is used
        self.client = AzureOpenAI(
            api_key=settings.azure_api_key,
            azure_endpoint=settings.azure_endpoint,
            api_version=settings.azure_api_version,
            max_retries=1,
            timeout=settings.llm_timeout_s,
        )
        self.last_usage: dict[str, int] = {}

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = settings.llm_temperature) -> str:
        response = self.client.chat.completions.create(
            model=settings.azure_deployment,   # on Azure, "model" is the name of YOUR deployment
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
