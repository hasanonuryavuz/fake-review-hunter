"""Bir kez yüklenen modelle canlı yorum sınıflandırması."""
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from app.dataset import ROOT

class ReviewRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000, examples=["Ürün gayet güzel, kargolama hızlıydı."])

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value: str):
        value = value.strip()
        if len(value) < 3 or not any(char.isalpha() for char in value):
            raise ValueError("En az 3 karakter ve bir harf içeren yorum gönderin.")
        return value

class Prediction(BaseModel):
    label: str
    fake_probability: float
    fake_percentage: float
    threshold: float
    warning: str

def create_app(model_path: Path | None = None) -> FastAPI:
    path = model_path or ROOT / "artifacts" / "model.joblib"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not path.is_file():
            raise RuntimeError("Model bulunamadı. Önce python -m app.train çalıştırın.")
        # Yalnız bu projede üretilen güvenilir model dosyasını yükleyin.
        app.state.model = joblib.load(path)
        yield

    api = FastAPI(title="Sahte Yorum Avcısı", version="1.0.0", lifespan=lifespan,
                  description="Sentetik veri üzerinde TF-IDF + Multinomial Naive Bayes eğitim projesi.")

    @api.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True}

    @api.post("/predict", response_model=Prediction)
    def predict(request: ReviewRequest):
        model = api.state.model
        vector = model.named_steps["tfidf"].transform([request.text])
        if vector.nnz == 0:
            raise HTTPException(status_code=422, detail="Model sözlüğünde eşleşen kelime yok; yorum değerlendirilemiyor.")
        classifier = model.named_steps["nb"]
        fake_index = list(classifier.classes_).index(1)
        probability = float(classifier.predict_proba(vector)[0, fake_index])
        return Prediction(label="sahte" if probability >= 0.5 else "gercek",
            fake_probability=probability, fake_percentage=round(probability * 100, 2),
            threshold=0.5, warning="Sentetik eğitim modelinin kalibre edilmemiş skoru; sahtecilik kanıtı değildir.")

    return api

app = create_app()
