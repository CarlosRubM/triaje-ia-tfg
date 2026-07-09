from streamlit.testing.v1 import AppTest


def test_app_principal_renderiza_fase_1_sin_excepciones():
    app = AppTest.from_file("src/triaje_ia/ui/app.py")

    app.run(timeout=15)

    assert len(app.exception) == 0
