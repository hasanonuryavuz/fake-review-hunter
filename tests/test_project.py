import pytest
from fastapi.testclient import TestClient
from app.dataset import make_dataset
from app.train import train, split_data
from app.main import create_app

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    output = tmp_path_factory.mktemp("model")
    train(output)
    with TestClient(create_app(output / "model.joblib")) as test_client:
        yield test_client

def test_dataset_and_split():
    df = make_dataset()
    assert df.height == df["text"].n_unique() == 10000
    assert df.filter(df["label"] == 1).height == 5000
    left, right = split_data(df)
    assert set(left["template_id"]).isdisjoint(right["template_id"])
    assert left.height == 8000 and right.height == 2000

def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}

@pytest.mark.parametrize("text,label", [
    ("Ürün gayet güzel, kargolama hızlıydı.", "gercek"),
    ("Harika harika harika kesinlikle al 10 numara!", "sahte")])
def test_predictions(client, text, label):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == label
    assert 0 <= payload["fake_probability"] <= 1
    assert payload["fake_percentage"] == round(payload["fake_probability"] * 100, 2)

@pytest.mark.parametrize("payload", [{}, {"text": "   "}, {"text": "1234"},
    {"text": "x" * 5001}, {"text": "qzxv qzxv"}])
def test_invalid_or_unknown(client, payload):
    assert client.post("/predict", json=payload).status_code == 422

def test_missing_model(tmp_path):
    with pytest.raises(RuntimeError, match="python -m app.train"):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass
