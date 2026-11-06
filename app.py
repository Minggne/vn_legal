"""
Streamlit chat UI cho hệ thống hỏi đáp pháp luật.
Đặt app.py cùng thư mục với answer.py, search.py, embedder.py và reranker.py.

Chạy:
    streamlit run app.py
"""

import contextlib
import io
import time
from types import SimpleNamespace

import streamlit as st
import retrieval.answer as qa


# ---------------------------------------------------------------------
# Cấu hình cố định
# ---------------------------------------------------------------------
MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434"

POOL = 15
RERANK_POOL = 15
RERANK_BATCH = 4
RERANK_MAXLEN = 1024
RRF_K = None

MAX_DIEU = 3
MAX_CHARS = 3000
MAX_TOKENS = 500
MAX_WORDS = 180
NUM_CTX = 8192
THINK_OFF = "auto"
SHOW_STEPS = 10


st.set_page_config(
    page_title="Hỏi đáp pháp luật Việt Nam",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Chỉ giữ thông tin model trong sidebar.
with st.sidebar:
    st.header("Cấu hình")
    st.markdown(f"**LLM:** `{MODEL}`")


st.title("⚖️ Hỏi đáp pháp luật Việt Nam")


def run_rag(user_question: str):
    """Chạy pipeline hỏi đáp và trả về câu trả lời cùng nguồn."""
    started = time.perf_counter()

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        res = qa.retrieve(
            user_question,
            pool=POOL,
            rerank_pool=RERANK_POOL,
            use_rerank=True,
            rrf_k=RRF_K,
            rerank_batch=RERANK_BATCH,
            rerank_maxlen=RERANK_MAXLEN,
        )
        qa.in_cac_buoc(res, show=SHOW_STEPS)
        qa.in_dieu_cuoi(res, n=MAX_DIEU)

    retrieval_log = log_buffer.getvalue()
    sources = qa.chon_nguon(res, MAX_DIEU, MAX_CHARS)

    if not sources:
        return {
            "answer": qa.TU_CHOI + ".",
            "sources": [],
            "retrieval_log": retrieval_log,
            "errors": [],
            "warnings": [],
            "stats": {
                "retrieve": res.get("t_retrieve"),
                "rerank": res.get("t_rerank"),
                "llm": 0,
                "total": time.perf_counter() - started,
            },
            "truncated": False,
            "thinking_cut": False,
            "llm_stats": [],
        }

    context = qa.build_context(sources)
    user_prompt = f"NGUỒN:\n{context}\n\nCÂU HỎI: {user_question}"
    system_prompt = qa.he_thong(MAX_WORDS)

    # Giải phóng VRAM cho Ollama, phù hợp GPU có VRAM hạn chế.
    retrieval.embedder.release()
    retrieval.reranker.release()

    llm_args = SimpleNamespace(
        model=MODEL,
        think_off=THINK_OFF,
        url=OLLAMA_URL,
        num_ctx=NUM_CTX,
        max_tokens=MAX_TOKENS,
    )

    llm_started = time.perf_counter()
    answer, truncated, thinking_cut, llm_stats = qa.goi_ollama(
        llm_args, system_prompt, user_prompt
    )
    llm_elapsed = time.perf_counter() - llm_started

    if not answer:
        answer = "(Model chưa tạo được câu trả lời hoàn chỉnh.)"

    errors, warnings = (
        ([], [])
        if (truncated or thinking_cut)
        else qa.kiem_tra(answer, sources)
    )

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_log": retrieval_log,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "retrieve": res.get("t_retrieve"),
            "rerank": res.get("t_rerank"),
            "llm": llm_elapsed,
            "total": time.perf_counter() - started,
        },
        "truncated": truncated,
        "thinking_cut": thinking_cut,
        "llm_stats": llm_stats,
    }


# Lưu lịch sử hội thoại trong session hiện tại.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn cũ theo dạng chat.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        result = message.get("result")
        if message["role"] == "assistant" and result:
            stats = result["stats"]
            st.caption(
                "Thời gian: "
                f"truy xuất {stats['retrieve']:.1f}s · "
                f"rerank {stats['rerank']:.1f}s · "
                f"LLM {stats['llm']:.1f}s · "
                f"tổng {stats['total']:.1f}s"
                if stats["retrieve"] is not None and stats["rerank"] is not None
                else f"Tổng thời gian: {stats['total']:.1f}s"
            )

            if result["truncated"] or result["thinking_cut"]:
                st.warning(
                    "Câu trả lời có thể chưa hoàn chỉnh do model hết token "
                    "hoặc phần thinking bị cắt."
                )

            if result["errors"]:
                with st.expander("Lỗi kiểm chứng trích dẫn"):
                    for item in result["errors"]:
                        st.write(f"- {item}")

            if result["warnings"]:
                with st.expander("Lưu ý khi kiểm chứng"):
                    for item in result["warnings"]:
                        st.write(f"- {item}")

            if result["sources"]:
                with st.expander("Văn bản pháp luật được tham khảo"):
                    for index, source in enumerate(result["sources"], 1):
                        chunk = source["c"]
                        score = (
                            f"rerank {chunk['rerank_score']:.4f}"
                            if chunk.get("rerank_score") is not None
                            else f"hybrid {chunk['hybrid_score']:.4f}"
                        )
                        st.markdown(
                            f"**[{index}] {source['duong_dan']}**  \n"
                            f"Tình trạng: {source['tinh_trang']} · Điểm: {score}"
                        )
                        st.text(source["noi_dung"])

            with st.expander("Thông tin xử lý"):
                st.write("**Các bước truy xuất**")
                st.code(
                    result["retrieval_log"] or "Không có log truy xuất.",
                    language="text",
                )
                if result["llm_stats"]:
                    st.write("**Thống kê Ollama**")
                    for line in result["llm_stats"]:
                        st.write(line)


# st.chat_input được Streamlit cố định ở đáy trang như giao diện ChatGPT.
question = st.chat_input("Nhập câu hỏi pháp luật của bạn...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu văn bản pháp luật..."):
            try:
                result = run_rag(question.strip())
                st.markdown(result["answer"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "result": result,
                    }
                )
                st.rerun()
            except SystemExit as exc:
                error_message = f"Không thể gọi Ollama: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
            except Exception as exc:
                error_message = f"Đã xảy ra lỗi: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
