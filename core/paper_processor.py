"""Process academic papers for ingestion into Endee."""

import logging
from typing import Dict, Any, List, Tuple
from collections import Counter
import re
import numpy as np
from .embeddings import encode_text, encode_batch

logger = logging.getLogger(__name__)


def process_paper(
    paper: Dict[str, Any],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """Process a single paper and return Endee-ready vector object."""
    # Extract metadata
    metadata = extract_metadata(paper)

    # Generate dense embedding from abstract
    text_to_embed = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    dense_vector = encode_text(text_to_embed, model_name)

    # Generate sparse vectors (BM25-style)
    sparse_indices, sparse_values = generate_sparse_vector(text_to_embed)

    # Construct vector object for Endee
    vector_obj = {
        "id": metadata["id"],
        "vector": dense_vector.tolist(),
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "meta": {
            "title": metadata["title"],
            "abstract": paper.get("abstract", ""),
            "authors": metadata["authors"],
            "url": paper.get("url", ""),
        },
        "filter": {
            "year": metadata["year"],
            "field": metadata["field"],
            "citation_count": metadata["citations"],
        },
    }

    return vector_obj


def extract_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured metadata from paper."""
    return {
        "id": paper.get("id", f"paper_{id(paper)}"),
        "title": paper.get("title", "Untitled"),
        "authors": paper.get("authors", []),
        "year": int(paper.get("year", 2024)),
        "field": paper.get("field", "cs.AI"),
        "citations": int(paper.get("citations", 0)),
    }


def generate_sparse_vector(text: str, vocab_size: int = 1000) -> Tuple[List[int], List[float]]:
    """Generate BM25-style sparse representation of text."""
    # Simple term frequency approach
    # Tokenize text
    tokens = re.findall(r"\b\w+\b", text.lower())

    if not tokens:
        return [], []

    # Count term frequencies
    term_counts = Counter(tokens)

    # Sort by frequency and take top terms
    top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:100]

    if not top_terms:
        return [], []

    # Create indices based on term hash
    sparse_indices = []
    sparse_values = []

    for term, count in top_terms:
        # Hash term to get index (0-999)
        term_idx = hash(term) % vocab_size
        sparse_indices.append(term_idx)
        # Normalize frequency as weight
        weight = count / len(tokens)
        sparse_values.append(min(weight, 1.0))

    return sparse_indices, sparse_values


def process_papers_batch(
    papers: List[Dict[str, Any]],
    batch_size: int = 32,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[Dict[str, Any]]:
    """Process multiple papers efficiently with batch embedding."""
    processed_papers = []
    
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        
        # Extract texts for batch embedding
        texts_to_embed = []
        for paper in batch:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            texts_to_embed.append(text)
            
        # Batch encode
        dense_vectors = encode_batch(texts_to_embed, batch_size=len(batch), model_name=model_name)
        
        # Process each paper with its pre-computed vector
        for j, paper in enumerate(batch):
            metadata = extract_metadata(paper)
            text_to_embed = texts_to_embed[j]
            
            # Generate sparse vectors
            sparse_indices, sparse_values = generate_sparse_vector(text_to_embed)
            
            vector_obj = {
                "id": str(metadata["id"]),
                "values": dense_vectors[j].tolist(),
                "sparse_indices": sparse_indices,
                "sparse_values": sparse_values,
                "metadata": {
                    "title": metadata["title"],
                    "abstract": paper.get("abstract", ""),
                    "authors": metadata["authors"],
                    "url": paper.get("url", ""),
                    "year": metadata["year"],
                    "field": metadata["field"],
                    "citation_count": metadata["citations"],
                }
            }
            processed_papers.append(vector_obj)
            
    return processed_papers
    if not papers:
        return []

    # Prepare texts for batch encoding
    texts = [
        f"{p.get('title', '')} {p.get('abstract', '')}"
        for p in papers
    ]

    # Batch encode
    embeddings = encode_batch(texts, batch_size, model_name)

    # Process each paper with its embedding
    vector_objects = []
    for i, paper in enumerate(papers):
        metadata = extract_metadata(paper)
        text = texts[i]
        sparse_indices, sparse_values = generate_sparse_vector(text)

        vector_obj = {
            "id": metadata["id"],
            "vector": embeddings[i].tolist(),
            "sparse_indices": sparse_indices,
            "sparse_values": sparse_values,
            "meta": {
                "title": metadata["title"],
                "abstract": paper.get("abstract", ""),
                "authors": metadata["authors"],
                "url": paper.get("url", ""),
            },
            "filter": {
                "year": metadata["year"],
                "field": metadata["field"],
                "citation_count": metadata["citations"],
            },
        }
        vector_objects.append(vector_obj)

    return vector_objects
