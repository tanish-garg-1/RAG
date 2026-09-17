import typer

from src.generation.groq_client import GroqClient
from src.pipeline.rag_pipeline import RAGPipeline

app = typer.Typer()
groq_client = GroqClient()
pipeline = RAGPipeline()

@app.callback()
def callback():
    """PDF RAG CLI."""
    pass

@app.command()
def ask(question: str):
    """Answer a question."""
    print(pipeline.answer(question))


if __name__ == "__main__":
    app()