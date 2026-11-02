FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: this layer is rebuilt only when requirements.txt changes, not on every code edit.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config.py main.py ./
COPY src ./src
COPY app ./app
COPY ml ./ml
COPY models ./models

# data/ (PDFs, vector store, downloaded models) and logs/ are mounted as volumes by docker-compose.
EXPOSE 8000 8501

# Default process: the API. docker-compose starts a second container for the UI with a different command.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
