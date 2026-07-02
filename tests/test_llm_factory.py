import pytest

from triaje_ia.llm.extractor import extraer_vector_clinico
from triaje_ia.llm.factory import crear_extractor


def test_factory_devuelve_extractor_local():
    assert crear_extractor("ollama") is extraer_vector_clinico


def test_factory_rechaza_backend_no_local():
    with pytest.raises(ValueError, match="Usa 'ollama'"):
        crear_extractor("api")
