from src.generation.groq_client import GroqClient
from src.generation.prompts import GENERAL_KNOWLEDGE_PROMPT

class RAGPipeline:
    def __init__(self):
        self.groq_client = GroqClient()

    def answer(self,question:str) -> str:
        return self.groq_client.complete(GENERAL_KNOWLEDGE_PROMPT,question)