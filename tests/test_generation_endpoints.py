# tests assume FastAPI TestClient; stub out generator and search to avoid heavy model use.
from fastapi.testclient import TestClient
from app.main import app

def test_caption_endpoint_no_input():
    client = TestClient(app)
    # missing file and image_id -> 400
    resp = client.post("/generate/caption_from_image", json={"k": 5})
    assert resp.status_code == 400
