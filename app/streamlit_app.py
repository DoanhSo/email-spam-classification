"""Local academic email checker. Run via streamlit; no model training in UI."""

import json
import os
from pathlib import Path

import streamlit as st

from spam_detector.web_service import PredictionService, load_web_config, registry_fingerprint

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "work": ("Project meeting", "Hi team,\nPlease review the attached budget report before our meeting tomorrow.\nThanks."),
    "offer": ("A special offer", "Congratulations! Claim your free prize today. Visit https://example.com to learn more."),
    "enron": ("Enron meeting report", "Please review the attached Enron gas report and meter readings before the meeting. Thanks."),
}

st.set_page_config(page_title="Kiểm tra email spam", page_icon=None, layout="wide")


@st.cache_resource(scope="session", show_spinner=False)
def load_service(config: dict, fingerprint: str) -> PredictionService:
    return PredictionService(config)


def reset_result():
    st.session_state.pop("prediction", None)


def set_example(name: str):
    st.session_state["subject"], st.session_state["body"] = EXAMPLES[name]
    reset_result()


def clear_email():
    st.session_state["subject"], st.session_state["body"] = "", ""
    reset_result()


def render_result(result: dict):
    st.subheader("Kết quả kiểm tra")
    st.caption(f"Lần gửi gần nhất · {result['request_id']} · {result['characters']:,} ký tự. Gửi lại sau khi sửa email để cập nhật.")
    for error in result["errors"]:
        st.error(error["message"])
    rows = result["results"]
    if not rows:
        return
    if result["ensemble"]:
        final = result["ensemble"]
        if final["label"]:
            st.warning("Ensemble dự đoán: Spam")
        else:
            st.success("Ensemble dự đoán: Thư thường (ham)")
        st.caption(f"Điểm bỏ phiếu: {final['score_value']:.3f} · Không phải xác suất đã hiệu chuẩn.")
    elif len(rows) == 1:
        row = rows[0]
        if row["label"]:
            st.warning("Mô hình dự đoán: Spam")
        else:
            st.success("Mô hình dự đoán: Thư thường (ham)")
        st.caption(f"{row['model_name']} · {row['release_id']} · {row['elapsed_ms']:.1f} ms sau tải mô hình")
    elif len({row["label"] for row in rows}) > 1:
        st.info("Các mô hình cho kết quả khác nhau. Xem dự đoán của từng mô hình bên dưới.")
    if result["mode"] == "compare":
        score_names = {"leaf_fraction": "Tỷ lệ spam tại lá", "margin": "Margin", "probability_uncalibrated": "Xác suất chưa hiệu chuẩn", "vote_score": "Điểm bỏ phiếu"}
        st.table([{"Mô hình": row["model_name"], "Dự đoán": "Spam" if row["label"] else "Thư thường",
                   "Loại điểm": score_names[row["score_kind"]], "Điểm": f"{row['score_value']:.3f}",
                   "Thời gian (ms)": f"{row['elapsed_ms']:.1f}"} for row in rows])
        st.caption("Điểm số có ý nghĩa khác nhau giữa các mô hình; không dùng độ lớn để so mức chắc chắn.")
    if result["mode"] == "ensemble" and result["ensemble"] is None:
        st.error("Không thể kết luận ensemble vì một mô hình chưa xử lý được email. Không thay đổi trọng số.")
    for row in rows:
        if row["warning"]:
            st.info(row["warning"])
        if row["model_id"] == "tree":
            with st.expander("Vì sao Decision Tree có dự đoán này?", expanded=False):
                st.write(f"Tỷ lệ spam tại lá: **{row['score_value']:.1%}** · Lá có **{row['leaf_training_samples']}** mẫu huấn luyện.")
                st.caption("Đây là tỷ lệ lớp trong mẫu ở lá, không phải độ tin cậy đã hiệu chuẩn cho email mới.")
                st.table([{"Bước": i, "Từ được kiểm tra": item["word"], "Xuất hiện": "Có" if item["present"] else "Không"}
                          for i, item in enumerate(row["path"], start=1)])
                st.caption(f"Khớp {row['matched_features']} đặc trưng trong {row['unique_tokens']} token duy nhất sau xử lý.")


def main():
    try:
        config_path = Path(os.environ.get("SPAM_WEB_CONFIG", str(ROOT / "configs/web.yaml")))
        config = load_web_config(config_path, ROOT)
        fingerprint = registry_fingerprint(config)
        service = load_service(config, fingerprint)
    except Exception:
        st.error("Không đọc được cấu hình web. Vui lòng kiểm tra configs/web.yaml và dependencies.")
        st.stop()
    if st.session_state.get("registry_version") != fingerprint:
        reset_result()
        st.session_state["registry_version"] = fingerprint
    st.caption("KHAI PHÁ DỮ LIỆU  /  EMAIL CLASSIFICATION")
    st.title("Kiểm tra email spam")
    st.write("Dán email tiếng Anh để xem dự đoán và cách mô hình đưa ra kết luận.")
    st.caption("Bản demo học thuật. Kết quả có thể sai, đặc biệt với dữ liệu khác nguồn huấn luyện.")
    with st.sidebar:
        st.subheader("Mô hình đang sử dụng")
        for status in service.statuses.values():
            label = {"ready": "Sẵn sàng", "not_configured": "Chưa bàn giao", "load_error": "Không tải được release"}[status["status"]]
            st.text(f"{status['name']}\n{label}")
        st.divider()
        st.caption("Nội dung email chỉ được xử lý trong phiên hiện tại; ứng dụng không ghi body vào log hoặc lịch sử chung.")
    modes = {"Một mô hình": "single", "So sánh mô hình sẵn sàng": "compare"}
    if service.ensemble_available():
        modes["Ensemble (4 mô hình)"] = "ensemble"
    mode_label = st.sidebar.radio("Cách kiểm tra", list(modes), key="mode", on_change=reset_result)
    mode = modes[mode_label]
    ready = service.ready_ids
    if not service.ensemble_available():
        st.sidebar.caption("Ensemble chưa sẵn sàng: cần đủ 4 mô hình và trọng số validation khớp release.")
    selected = config["default_model"] if config["default_model"] in ready else (ready[0] if ready else None)
    if mode == "single" and ready:
        selected = st.sidebar.selectbox("Mô hình", ready, index=ready.index(selected),
                                format_func=lambda key: service.statuses[key]["name"], key="selected_model", on_change=reset_result)
    if not ready:
        st.error("Chưa có mô hình sẵn sàng. Kiểm tra đường dẫn và checksum release trong cấu hình.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Nội dung email")
        st.caption(f"Dán toàn bộ thư vào nội dung, hoặc nhập Subject riêng. Tối đa {config['max_characters']:,} ký tự.")
        a, b, c = st.columns(3)
        a.button("Ví dụ công việc", on_click=set_example, args=("work",), key="example_work")
        b.button("Ví dụ quảng cáo", on_click=set_example, args=("offer",), key="example_offer")
        c.button("Ví dụ Enron", on_click=set_example, args=("enron",), key="example_enron")
        st.caption("Các ví dụ minh họa có thể bị dự đoán sai; không phải nhãn kiểm chứng.")
        with st.form("email_form"):
            subject = st.text_input("Subject (không bắt buộc)", key="subject", placeholder="Tiêu đề email")
            body = st.text_area("Nội dung email", key="body", height=240, placeholder="Nhập hoặc dán email tiếng Anh tại đây…")
            submitted = st.form_submit_button("Kiểm tra email", type="primary", disabled=not ready, width="stretch")
        st.button("Xóa nội dung", key="clear", on_click=clear_email)
        if submitted:
            reset_result()
            try:
                with st.spinner("Đang phân tích email…"):
                    st.session_state["prediction"] = service.predict(subject, body, mode, selected)
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                st.error("Không thể xử lý lần kiểm tra này. Vui lòng thử lại.")
    with right:
        if "prediction" in st.session_state:
            render_result(st.session_state["prediction"])
        else:
            st.subheader("Từ nội dung đến dự đoán")
            st.write("**01 · Nhập email**")
            st.caption("Nội dung tiếng Anh, có thể kèm Subject và xuống dòng.")
            st.write("**02 · Mô hình phân tích**")
            st.caption("Dùng đúng bộ biến đổi và release đã huấn luyện.")
            st.write("**03 · Xem kết quả và lý do**")
            st.caption("Đối chiếu nhãn dự đoán với các từ dẫn đến quyết định.")
    st.divider()
    with st.expander("Chất lượng mô hình và giới hạn sử dụng"):
        st.write("Decision Tree học trên Enron, có thể nhận nhầm email từ nguồn khác. Mô hình hiện tại chưa phù hợp để tự động chặn thư thật.")
        for model_id in ready:
            release = Path(config["models"][model_id]["release"])
            for filename, name in (("metrics.test.json", "Test Enron"), ("metrics.external_test.json", "SpamAssassin ngoài nguồn")):
                try:
                    metrics = json.loads((release / filename).read_text(encoding="utf-8"))["metrics"]
                    st.write(f"**{service.statuses[model_id]['name']} · {name}:** F1 {metrics['f1_spam']:.2%}, Precision {metrics['precision_spam']:.2%}, Recall {metrics['recall_spam']:.2%}.")
                except (OSError, ValueError, KeyError):
                    st.caption(f"{name}: chưa có số liệu trong release.")
        st.link_button("Đọc báo cáo thực nghiệm", "https://github.com/DoanhSo/email-spam-classification/blob/main/reports/enron-v1/REPORT.md")
    st.caption("Nhóm Nguyễn Quốc Doanh · Lê Hữu Hiệu · Thân Đức Minh · Nguyễn Thái Dương")


main()
