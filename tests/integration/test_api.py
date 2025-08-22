def test_healthz(client):
    r = client.get("/healthz"); assert r.json()["ok"] is True
