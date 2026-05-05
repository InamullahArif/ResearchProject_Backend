"""
ASGI entrypoint for uvicorn: `uvicorn app:app --reload`.
Application implementation is in api.main (modular routes and services).
"""
from api.main import app  # noqa: F401
