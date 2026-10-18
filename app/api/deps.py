"""Dependencias compartilhadas pelos endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.core.auth import Principal, autenticar

Credencial = Annotated[Principal, Depends(autenticar)]
