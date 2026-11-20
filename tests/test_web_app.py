from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def app():
    return AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=20).run()


def submit(at, body):
    at.text_area(key="body").set_value(body)
    return at.button(key="FormSubmitter:email_form-Kiểm tra email").click().run()


def test_initial_app_has_no_prediction_and_missing_models_are_clear():
    at = app()
    assert not at.exception
    assert "prediction" not in at.session_state
    assert "Chưa bàn giao" in " ".join(t.value for t in at.sidebar.text)
    assert len(at.radio[0].options) == 2


def test_submit_explain_and_clear():
    at = app()
    submit(at, "Please review the Enron report.")
    assert not at.exception
    assert at.session_state["prediction"]["results"][0]["label"] == 0
    assert not at.get("doc_string")  # Streamlit must not render internal return objects.
    assert any("ham" in t.value for t in at.success)
    assert at.session_state["prediction"]["results"][0]["model_id"] == "nb"
    assert any("Xác suất spam" in t.value for t in at.markdown)
    at.button(key="clear").click().run()
    assert at.text_area(key="body").value == ""
    assert "prediction" not in at.session_state


def test_new_invalid_submit_clears_old_result():
    at = app()
    submit(at, "Enron meeting")
    assert "prediction" in at.session_state
    submit(at, " ")
    assert not at.exception
    assert "prediction" not in at.session_state
    assert any("Vui lòng nhập" in e.value for e in at.error)


def test_examples_mode_and_result_snapshot():
    at = app()
    at.button(key="example_work").click().run()
    assert at.text_input(key="subject").value == "Project meeting"
    submit(at, "Enron report")
    before = at.session_state["prediction"]["request_id"]
    at.text_area(key="body").set_value("New unsent body").run()
    assert at.session_state["prediction"]["request_id"] == before
    at.radio(key="mode").set_value("So sánh mô hình sẵn sàng").run()
    assert "prediction" not in at.session_state
    submit(at, "Enron report")
    assert not at.exception
    assert at.session_state["prediction"]["mode"] == "compare"
    assert [row["model_id"] for row in at.session_state["prediction"]["results"]] == ["tree", "nb"]


def test_bad_config_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("SPAM_WEB_CONFIG", str(tmp_path / "missing.yaml"))
    at = app()
    assert not at.exception
    assert any("Không đọc được cấu hình" in e.value for e in at.error)


def test_nb_default_shows_contributing_words(monkeypatch, tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    from nb_fixture import make_nb_config

    monkeypatch.setenv("SPAM_WEB_CONFIG", str(make_nb_config(tmp_path)))
    at = app()
    submit(at, "win free money now")
    assert not at.exception
    assert at.session_state["prediction"]["results"][0]["model_id"] == "nb"
    assert any("Từ đóng góp" in str(t.value) for t in at.markdown) or any(
        "Từ đóng góp" in str(table.value) for table in at.table)
