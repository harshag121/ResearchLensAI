from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas import PaperUpload, BatchUpload, SearchQuery, SearchResult, Stats
from app.services import ingest_paper, ingest_batch, search_papers, get_db_stats

router = APIRouter()

@router.post("/ingest", response_model=dict[str, str])
async def ingest(paper: PaperUpload):
    """Ingest a single paper."""
    try:
        paper_id = await ingest_paper(paper)
        return {"id": paper_id, "status": "indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/batch", response_model=dict[str, int])
async def ingest_batch_papers(batch: BatchUpload):
    """Ingest multiple papers."""
    try:
        count = await ingest_batch(batch.papers)
        return {"count": count, "status": "indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=List[SearchResult])
async def search(query: SearchQuery):
    """Search for papers."""
    return await search_papers(query)

@router.get("/stats", response_model=Stats)
async def stats():
    """Get index stats."""
    return await get_db_stats()

@router.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}
