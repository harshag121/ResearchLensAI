import logging
from typing import List, Dict, Any
from core.endee_client import get_client, get_index, create_index
from core.paper_processor import process_paper, process_papers_batch
from core.embeddings import encode_text
from app.schemas import SearchQuery, SearchResult, PaperUpload, Stats

logger = logging.getLogger(__name__)

async def initialize_db():
    """Ensure index exists on startup."""
    try:
        create_index()
    except Exception as e:
        logger.error(f"Failed to initialize index: {e}")

async def ingest_paper(paper: PaperUpload) -> str:
    """Ingest a single paper into Endee."""
    client = get_client()
    index = get_index()
    
    # Process paper into vector object
    paper_dict = paper.model_dump()
    vector_obj = process_paper(paper_dict)
    
    # Upsert to Endee
    try:
        index.upsert([vector_obj])
        return vector_obj["id"]
    except Exception as e:
        logger.error(f"Error ingesting paper: {e}")
        raise

async def ingest_batch(papers: List[PaperUpload]) -> int:
    """Ingest multiple papers in batch."""
    client = get_client()
    index = get_index()
    
    # Convert to dicts
    paper_dicts = [p.model_dump() for p in papers]
    
    # Process batch
    vector_objs = process_papers_batch(paper_dicts)
    
    # Upsert in chunks if needed (Endee limit might be 1000)
    chunk_size = 100
    total = 0
    
    for i in range(0, len(vector_objs), chunk_size):
        chunk = vector_objs[i:i + chunk_size]
        try:
            index.upsert(chunk)
            total += len(chunk)
        except Exception as e:
            logger.error(f"Error ingesting batch chunk {i}: {e}")
            
    return total

async def search_papers(query: SearchQuery) -> List[SearchResult]:
    """Search papers using hybrid search."""
    index = get_index()
    
    # Encode query
    query_vector = encode_text(query.query)
    
    try:
        # Execute search
        results = index.search(
            vector=query_vector.tolist(),
            limit=query.limit,
            filter=query.filters,
            search_params={"ef": 100}  # HNSW search parameter
        )
        
        # Format results
        search_results = []
        for res in results:
            meta = res.metadata if hasattr(res, 'metadata') else {}
            search_results.append(SearchResult(
                id=str(res.id),
                score=res.score,
                title=meta.get("title", "Untitled"),
                abstract=meta.get("abstract", ""),
                authors=meta.get("authors", []),
                url=meta.get("url", ""),
                year=meta.get("year", 2024),
                field=meta.get("field", "cs.AI"),
                citations=meta.get("citation_count", 0),
                metadata=meta
            ))
            
        return search_results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

async def get_db_stats() -> Stats:
    """Get database statistics."""
    index = get_index()
    try:
        info = index.describe()
        return Stats(
            total_papers=info.get("count", 0),
            dimension=info.get("dimension", 384),
            index_name=info.get("name", "academic_papers")
        )
    except Exception:
        return Stats(total_papers=0, dimension=384, index_name="academic_papers")
