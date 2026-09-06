import os


os.environ.setdefault("RAG_LOAD_IN_4BIT", "true")

from backend.main import app  # noqa: E402
