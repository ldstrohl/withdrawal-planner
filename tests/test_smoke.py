from streamlit.testing.v1 import AppTest


def test_app_renders_clean():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
