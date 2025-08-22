def test_retrieve(client):
    r = client.post("/retrieve", params={"query":"q"})
    assert r.status_code==200
