from typing import Optional
from pydantic import BaseModel


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    abstract: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    included: bool = True