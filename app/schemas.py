from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PaperBase(BaseModel):
    title: str
    abstract: str
    authors: List[str] = []
    url: Optional[str] = None
    year: int = 2024
    field: str = "cs.AI"
    citations: int = 0

class PaperUpload(PaperBase):
    pass

class Paper(PaperBase):
    id: str

class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
    rerank: bool = False

class SearchResult(PaperBase):
    id: str
    score: float
    metadata: Dict[str, Any] = {}

class BatchUpload(BaseModel):
    papers: List[PaperUpload]

class Stats(BaseModel):
    total_papers: int
    dimension: int
    index_name: str
