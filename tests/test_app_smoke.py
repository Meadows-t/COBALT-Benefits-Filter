from streamlit.testing.v1 import AppTest


def test_app_starts_without_errors():
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    assert "COBALT" in at.title[0].value


def test_tabs_prompt_for_upload():
    at = AppTest.from_file("app.py", default_timeout=30).run()
    messages = [i.value for i in at.info]
    assert any("Upload a Flow CSV" in m for m in messages)
    assert any("Upload a Benefits CSV" in m for m in messages)
