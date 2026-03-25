"""Endee vector database client manager."""

import os
import logging
from typing import Optional, Dict, Any
from endee import Endee, Precision

logger = logging.getLogger(__name__)

_client_cache: Optional[Endee] = None
_index_cache: Optional[Any] = None


def get_client() -> Endee:
    """Get singleton Endee client with connection pooling."""
    global _client_cache
    if _client_cache is None:
        endee_host = os.getenv("ENDEE_HOST", "http://localhost:8080")
        endee_token = os.getenv("ENDEE_TOKEN", "default-token")

        logger.info(f"Initializing Endee client at {endee_host}")
        _client_cache = Endee(token=endee_token)
        _client_cache.set_base_url(f"{endee_host}/api/v1")
        logger.info("Endee client initialized successfully")

    return _client_cache


def create_index(index_name: str = "academic_papers", dimension: int = 384) -> None:
    """Create Endee index with hybrid search config if it doesn't exist."""
    client = get_client()

    try:
        # Check if index exists
        try:
            index = client.get_index(index_name)
            logger.info(f"Index '{index_name}' already exists")
            return
        except Exception:
            pass

        # Create index with hybrid search config
        logger.info(f"Creating index '{index_name}'")
        client.create_index(
            name=index_name,
            dimension=dimension,
            space_type="cosine",
            precision=Precision.INT8,
            sparse_model="default",
            M=32,
            ef_con=200,
        )
        logger.info(f"Index '{index_name}' created successfully")

    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise


def get_index(index_name: str = "academic_papers") -> Any:
    """Get index reference for operations."""
    client = get_client()
    return client.get_index(index_name)


def check_health() -> bool:
    """Verify Endee connection status."""
    try:
        client = get_client()
        # Try to get indices to verify connection
        client.get_index_list()
        logger.info("Endee health check passed")
        return True
    except Exception as e:
        logger.error(f"Endee health check failed: {e}")
        return False


def get_index_stats(index_name: str = "academic_papers") -> Dict[str, Any]:
    """Get index statistics."""
    try:
        index = get_index(index_name)
        stats = index.describe()
        return stats
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        return {}
