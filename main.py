import typer

from src.pipeline.rag_pipeline import RAGPipeline

app = typer.Typer()
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