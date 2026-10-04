"""Şablon bazında ayır, yalnız eğitim verisinde TF-IDF öğren, değerlendir."""
import json
import platform
from pathlib import Path
import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import polars as pl
from app.dataset import ROOT, make_dataset

STOP_WORDS = ["ve", "bir", "bu", "şu", "ile", "için", "da", "de"]

def split_data(df: pl.DataFrame):
    # Aynı şablonun ürün/sayı varyasyonları iki tarafa da düşmemeli.
    groups = df.select("template_id", "label").unique().sort("template_id")
    train_ids, test_ids = train_test_split(groups["template_id"].to_list(),
        test_size=0.2, random_state=42, stratify=groups["label"].to_list())
    return (df.filter(pl.col("template_id").is_in(train_ids)),
            df.filter(pl.col("template_id").is_in(test_ids)))

def train(output_dir: Path | None = None):
    output_dir = output_dir or ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = make_dataset()
    train_df, test_df = split_data(df)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words=STOP_WORDS,
                                 min_df=2, sublinear_tf=True)),
        ("nb", MultinomialNB(alpha=1.0)),
    ])
    model.fit(train_df["text"].to_list(), train_df["label"].to_list())
    expected = test_df["label"].to_list()
    predicted = model.predict(test_df["text"].to_list())
    metrics = {
        "dataset": "synthetic_template_generated", "seed": 42,
        "split": "stratified_template_holdout", "train_rows": train_df.height,
        "test_rows": test_df.height,
        "train_templates": sorted(train_df["template_id"].unique().to_list()),
        "test_templates": sorted(test_df["template_id"].unique().to_list()),
        "accuracy": accuracy_score(expected, predicted),
        "classification_report": classification_report(expected, predicted,
            labels=[0, 1], target_names=["gercek", "sahte"], output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=[0, 1]).tolist(),
        "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "limitation": "Sentetik test başarısı gerçek dünyadaki sahtecilik başarısını ölçmez.",
    }
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return model, metrics

if __name__ == "__main__":
    ROOT.joinpath("data").mkdir(exist_ok=True)
    make_dataset().write_csv(ROOT / "data" / "reviews.csv")
    _, metrics = train()
    ROOT.joinpath("reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Eğitim: {metrics['train_rows']}; test: {metrics['test_rows']}; doğruluk: {metrics['accuracy']:.4f}")
