from fastapi.testclient import TestClient

from source.app import app

testClient = TestClient(app)


def test_healthcheck():
    res = testClient.get("/")  # rajouter un /healthcheck avec un vrai check
    assert res.status_code == 200
