import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    def __init__(self):
        self.groq_api_key=os.getenv("GROQ_API_KEY","")
        self.groq_model=os.getenv("GROQ_MODEL","openai/gpt-oss-20b")

settings = Settings()