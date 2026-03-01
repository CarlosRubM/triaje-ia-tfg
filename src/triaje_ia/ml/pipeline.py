# Crear src/triaje_ia/ml/pipeline.py
$content = @'
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
import joblib
from loguru import logger
from pathlib import Path

VITALES_COLS = ["edad", "presion_sistolica", "presion_diastolica",
                "frecuencia_cardiaca", "frecuencia_respiratoria",
                "saturacion_oxigeno", "temperatura", "nivel_dolor"]

MODELS_DIR = Path("models")


class SintomasBinarizer(BaseEstimator, TransformerMixin):
    """Transforma listas de síntomas en matriz binaria."""
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
    
    def fit(self, X, y=None):
        self.mlb.fit(X)
        return self
    
    def transform(self, X):
        return self.mlb.transform(X)


def crear_pipeline() -> Pipeline:
    """Crea el pipeline completo de Scikit-Learn."""
    
    vitales_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("vitales", vitales_transformer, VITALES_COLS),
        ],
        remainder="drop"
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return pipeline


def guardar_modelo(pipeline: Pipeline, nombre: str = "modelo_triaje"):
    MODELS_DIR.mkdir(exist_ok=True)
    ruta = MODELS_DIR / f"{nombre}.pkl"
    joblib.dump(pipeline, ruta)
    logger.success(f"Modelo guardado en: {ruta}")


def cargar_modelo(nombre: str = "modelo_triaje") -> Pipeline:
    ruta = MODELS_DIR / f"{nombre}.pkl"
    logger.info(f"Cargando modelo desde: {ruta}")
    return joblib.load(ruta)
'@

$content | Out-File -FilePath "src\triaje_ia\ml\pipeline.py" -Encoding UTF8

# Git commit
git add src/triaje_ia/ml/pipeline.py
git commit -m "feat(ml): pipeline XGBoost con preprocesamiento vitales"
