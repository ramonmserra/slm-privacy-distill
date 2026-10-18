"""Entrypoint para `uvicorn main:app`. O FastAPI de verdade vive em app/main.py."""

from app.main import app

__all__ = ["app"]
