import numpy as np
from core import paper_processor as pp


def test_generate_sparse_vector_empty_text():
    idx, vals = pp.generate_sparse_vector("")
    assert idx == []
    assert vals == []


def test_extract_metadata_defaults_and_casting():
    paper = {"title": "T", "year": "2025", "citations": "7"}
    meta = pp.extract_metadata(paper)
    assert meta["title"] == "T"
    assert meta["year"] == 2025
    assert meta["citations"] == 7
    assert isinstance(meta["authors"], list)


def test_process_papers_batch_schema(monkeypatch):
    def fake_encode_batch(texts, batch_size=32, model_name="x"):
        return np.array([[0.1, 0.2, 0.3] for _ in texts])

    monkeypatch.setattr(pp, "encode_batch", fake_encode_batch)

    papers = [
        {
            "id": "p1",
            "title": "Vision Transformers",
            "abstract": "Paper about ViT",
            "authors": ["A"],
            "year": 2024,
            "field": "cs.CV",
            "citations": 10,
        },
        {
            "id": "p2",
            "title": "RAG Systems",
            "abstract": "Paper about retrieval",
            "authors": ["B"],
            "year": 2023,
            "field": "cs.AI",
            "citations": 5,
        },
    ]

    out = pp.process_papers_batch(papers, batch_size=1)
    assert len(out) == 2
    for item in out:
        assert "id" in item
        assert "vector" in item
        assert "sparse_indices" in item
        assert "sparse_values" in item
        assert "meta" in item
        assert "filter" in item
        assert isinstance(item["vector"], list)
