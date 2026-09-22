from src.models import RetrievedChunk

GENERAL_KNOWLEDGE_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly "
    "using your own knowledge."
)

IN_PDF_PROMPT = (
    "You answer questions using ONLY the provided excerpts from a PDF document.\n"
    "Rules:\n"
    "- Base every statement on the excerpts. Do not add outside knowledge.\n"
    "- The excerpts are the document's own text. If the question mentions the author, the book or "
    "'the document', the excerpts ARE that source: first-person statements are the author's views.\n"
    "- Cite the page for each claim like (p. 12).\n"
    "- If the excerpts only partly answer the question, answer that part and say what is missing.\n"
    "- Be concise and direct."
)

OUT_OF_PDF_PROMPT = (
    "The user asked a question that the provided PDF document does not cover. "
    "Answer it from your own general knowledge, clearly and concisely. "
    "Do not claim or imply that the answer comes from the document. "
    "If you are unsure or the facts may be outdated, say so."
)

GROUNDING_CHECK_PROMPT = (
    "You judge whether document excerpts contain enough information to answer a question.\n"
    "The excerpts are the document's own text; first-person statements are the author's.\n"
    "Reply with exactly one word: YES if the excerpts answer the question (fully or substantially), "
    "NO if they do not."
)


REWRITE_QUERY_PROMPT = (
    "You rewrite questions into search queries for finding passages in a document.\n"
    "The first search found nothing relevant, so try different wording: use synonyms, the likely "
    "terms the document itself would use, and the key concepts. Drop filler words and names of "
    "people who may simply be the author.\n"
    "Reply with only the search query on one line, no explanation."
)


def rewrite_user_prompt(question: str, previous_queries: list[str]) -> str:
    tried = "".join(f"\nAlready tried: {q}" for q in previous_queries)
    return f"Question: {question}{tried}"


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[Excerpt {i} | {rc.chunk.source} | page {rc.chunk.page}]\n{rc.chunk.text}"
        for i, rc in enumerate(chunks, start=1)
    )


def in_pdf_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    # This note lives in the user turn on purpose: smaller models follow it far more
    # reliably here than in the system prompt (tested: "What does Thiel say..." on his own book).
    return (
        "The excerpts below come from the document the user is asking about. It is written in the "
        "author's own voice, so if the question names the author, the author's views are the ones "
        "expressed here.\n\n"
        f"Excerpts:\n{build_context(chunks)}\n\nQuestion: {question}"
    )


def grounding_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"Excerpts:\n{build_context(chunks)}\n\nQuestion: {question}\n\nAnswer YES or NO."
