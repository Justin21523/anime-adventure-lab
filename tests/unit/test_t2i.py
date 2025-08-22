def test_gen_image(client):
    r = client.post("/gen_image", json={"prompt":"p"})
    assert r.status_code==200
