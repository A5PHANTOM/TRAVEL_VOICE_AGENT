from pydantic import BaseModel
from typing import Any


class Lead(BaseModel):
    id: int | None = None
    package: str
    lead: dict[str, Any] | None = None
    timestamp: str | None = None
