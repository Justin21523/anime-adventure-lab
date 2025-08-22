def test_basic_flow(client):
    assert client.get("/healthz").status_code==200
