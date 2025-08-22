def test_turn(client):
    r = client.post("/turn", json={"user":"hi"})
    assert r.status_code==200
