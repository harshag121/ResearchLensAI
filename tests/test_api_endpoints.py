from fastapi.testclient import TestClient
from app.main import app
import app.api.endpoints as endpoints


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stats_endpoint(monkeypatch):
    async def fake_stats():
        return endpoints.Stats(total_papers=10, dimension=384, index_name="academic_papers")

    monkeypatch.setattr(endpoints, "get_db_stats", fake_stats)

    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["total_papers"] == 10
    assert body["dimension"] == 384


def test_search_endpoint(monkeypatch):
    async def fake_search(query):
        return [
            {
                "id": "paper_001",
                "title": "Attention Is All You Need",
                "abstract": "Transformer paper",
                "authors": ["Vaswani"],
                "url": "https://arxiv.org/abs/1706.03762",
                "year": 2017,
                "field": "cs.CL",
                "citations": 85000,
                "score": 0.98,
                "metadata": {"year": 2017, "field": "cs.CL"},
            }
        ]

    monkeypatch.setattr(endpoints, "search_papers", fake_search)

    response = client.post("/api/v1/search", json={"query": "transformer", "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "paper_001"


def test_ingest_batch_endpoint(monkeypatch):
    async def fake_ingest_batch(papers):
        return len(papers)

    monkeypatch.setattr(endpoints, "ingest_batch", fake_ingest_batch)

    payload = {
        "papers": [
            {
                "title": "Test",
                "abstract": "A test paper",
                "authors": ["A"],
                "year": 2024,
                "field": "cs.AI",
                "citations": 0,
                "url": "",
            }
        ]
    }

    response = client.post("/api/v1/ingest/batch", json=payload)
    assert response.status_code == 200
    assert response.json()["count"] == 1
