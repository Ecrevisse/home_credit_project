from fastapi.testclient import TestClient

from source.app import app

testClient = TestClient(app)


def test_healthcheck():
    res = testClient.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_predict_granted():
    res = testClient.post(
        "/predict",
        json={
            "client_id": 208550,
        },
    )
    assert res.status_code == 200
    assert res.json() == [0, 0.07389793125440654, 0.15]


def test_predict_denied():
    res = testClient.post(
        "/predict",
        json={
            "client_id": 144092,
        },
    )
    assert res.status_code == 200
    assert res.json() == [1, 0.27139346407148074, 0.15]


def test_client_info():
    res = testClient.post(
        "/client_info",
        json={
            "client_id": 208550,
            "client_infos": ["EXT_SOURCE_3", "PAYMENT_RATE"],
        },
    )
    assert res.status_code == 200
    assert res.json() == [[0.0, 0.0425209367449743]]
