import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def fila_minima():
    """DataFrame de una fila con los campos basicos para probar el pipeline."""
    return pd.DataFrame([{
        "stay_id": 1,
        "age": 45,
        "gender": "M",
        "race": "WHITE",
        "arrival_transport": "WALK IN",
        "heartrate": 80.0,
        "resprate": 16.0,
        "o2sat": 98.0,
        "sbp": 120.0,
        "dbp": 80.0,
        "temperature": 98.6,
        "pain": 3,
        "chiefcomplaint": "chest pain",
        "acuity": 2,
    }])
