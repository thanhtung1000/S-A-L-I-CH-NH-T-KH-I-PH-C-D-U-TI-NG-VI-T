import os
import sys
import time

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Add src/ to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from evaluate_correction import ProductionMultiStagePipeline
from safety_gate import AlignmentAndSafetyGate
import streamlit as st

def run_cli_demo():
    print("=========================================================")
    print("DEMO HỆ THỐNG SỬA LỖI CHÍNH TẢ & KHÔI PHỤC DẤU TIẾNG VIỆT")
    print("=========================================================")
    pipeline = ProductionMultiStagePipeline("outputs/models/best_model")
    samples = [
        "Hoc deep learning tai Dai hoc Nam Can Tho",
        "Điển ình là Sony, IBM vq̀ một số công tt công nghệ."
    ]
    for sample in samples:
        print("\n[Văn bản đầu vào]:", sample)
        corr = pipeline.process(sample)
        print(" -> Kết quả sau sửa:", corr)

def render_diff_html(original_text: str, corrected_text: str) -> str:
    import difflib
    orig_words = original_text.split()
    corr_words = corrected_text.split()
    
    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
    opcodes = matcher.get_opcodes()
    
    html_parts = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for i in range(i1, i2):
                html_parts.append(orig_words[i])
        elif tag == 'replace':
            orig_part = " ".join(orig_words[i1:i2])
            corr_part = " ".join(corr_words[j1:j2])
            html_parts.append(f'<span style="background-color: rgba(239, 68, 68, 0.2); color: #fca5a5; padding: 2px 6px; border-radius: 4px; text-decoration: line-through; font-weight: 600; margin: 0 2px;">{orig_part}</span>')
            html_parts.append(f'<span style="background-color: rgba(34, 197, 94, 0.2); color: #86efac; padding: 2px 6px; border-radius: 4px; font-weight: 600; margin: 0 2px; box-shadow: 0 0 8px rgba(34, 197, 94, 0.2);">{corr_part}</span>')
        elif tag == 'delete':
            orig_part = " ".join(orig_words[i1:i2])
            html_parts.append(f'<span style="background-color: rgba(239, 68, 68, 0.2); color: #fca5a5; padding: 2px 6px; border-radius: 4px; text-decoration: line-through; font-weight: 600; margin: 0 2px;">{orig_part}</span>')
        elif tag == 'insert':
            corr_part = " ".join(corr_words[j1:j2])
            html_parts.append(f'<span style="background-color: rgba(34, 197, 94, 0.2); color: #86efac; padding: 2px 6px; border-radius: 4px; font-weight: 600; margin: 0 2px; box-shadow: 0 0 8px rgba(34, 197, 94, 0.2);">{corr_part}</span>')
            
    return " ".join(html_parts)

@st.cache_resource
def get_pipeline():
    return ProductionMultiStagePipeline("outputs/models/best_model")

@st.cache_data
def get_cached_prediction(text_input):
    if not text_input.strip():
        return "", ""
    pipeline = get_pipeline()
    corr_text = pipeline.process(text_input)
    highlighted_html = render_diff_html(text_input, corr_text)
    return corr_text, highlighted_html

def run_streamlit_app():
    st.set_page_config(
        page_title="DeepL-Grade ViT5 Spell AI",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # World-Class UI CSS Architecture
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #0b0f19;
            color: #f1f5f9;
        }
        
        /* Top Navigation Bar */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.2rem 2rem;
            background: rgba(15, 23, 42, 0.8);
            border-bottom: 1px solid rgba(51, 65, 85, 0.5);
            backdrop-filter: blur(12px);
            margin-bottom: 2rem;
            border-radius: 0 0 16px 16px;
        }
        
        .brand-logo {
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .status-pill {
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #38bdf8;
            padding: 6px 16px;
            border-radius: 30px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        .card-header-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        
        /* Diff Inspector Container */
        .diff-inspector {
            background-color: #090d16;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 20px;
            font-size: 1.1rem;
            line-height: 1.8;
            color: #e2e8f0;
            margin-top: 16px;
        }
        
        del {
            background-color: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            text-decoration: line-through;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            margin: 0 2px;
        }
        
        ins, u {
            background-color: rgba(34, 197, 94, 0.2);
            color: #86efac;
            text-decoration: none;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            margin: 0 2px;
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.2);
        }
        
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            height: 46px;
            transition: all 0.2s ease;
        }
        
        /* Sleek Code Output Box Customization */
        .stCodeBlock {
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            background-color: #1e293b !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Top Navbar
    st.markdown("""
        <div class="navbar">
            <div class="brand-logo">🎓 ĐỀ TÀI 10: SỬA LỖI CHÍNH TẢ & KHÔI PHỤC DẤU TIẾNG VIỆT</div>
            <div class="status-pill">● Bài Tập Lớn NLP • CSC4005 / CSC4007</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Kích hoạt tải mô hình lần đầu tiên (nếu chưa có trong cache)
    with st.spinner("Đang kết nối AI Engine (Chỉ tải một lần duy nhất)..."):
        get_pipeline()

    default_sample = "Hoc deep learning tai Dai hoc Nam Can Tho. Lớp em đi chuyenthamquan ở daihoc vq̀ đang nghiencuu khoa hoc."
    
    if "user_input_text" not in st.session_state:
        st.session_state["user_input_text"] = default_sample

    # DeepL Studio Layout
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.markdown('<div class="card-header-title">📝 Văn Bản Gốc (Input)</div>', unsafe_allow_html=True)
        
        text_input = st.text_area(
            label="Input Area",
            height=220,
            label_visibility="collapsed",
            placeholder="Nhập hoặc dán văn bản cần sửa lỗi chính tả / tách dính từ tại đây...",
            key="user_input_text"
        )
        
        act_col1, act_col2 = st.columns([3, 1])
        with act_col1:
            submit_btn = st.button("✨ Phân Tích & Sửa Lỗi AI", type="primary", use_container_width=True)
        with act_col2:
            def clear_text():
                st.session_state["user_input_text"] = ""
            st.button("🗑️ Xóa sạch", use_container_width=True, on_click=clear_text)

    # Process AI Correction Pipeline
    corr_text = ""
    highlighted_html = ""
    if text_input.strip():
        try:
            corr_text, highlighted_html = get_cached_prediction(text_input)
        except Exception as e:
            st.error(f"Lỗi hệ thống: {e}")

    with col_right:
        st.markdown('<div class="card-header-title">✨ Văn Bản Đã Tối Ưu (Result)</div>', unsafe_allow_html=True)
        
        # 1. Ô kết quả chính (có nút Copy) luôn hiển thị cố định ở trên
        if corr_text:
            st.code(corr_text, language=None)
        else:
            st.text_area(
                label="Result Area",
                value="",
                height=220,
                label_visibility="collapsed",
                placeholder="Kết quả chuẩn xác sẽ xuất hiện tại đây..."
            )
            
        # 2. Bộ chọn 2 nút Radio nằm ngang dưới ô kết quả chính
        view_mode = st.radio(
            "Chế độ đối chiếu",
            options=["Văn bản", "Đối chiếu tô màu trực quan"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
        
        # 3. Khung hiển thị thứ hai (ô to ở dưới cùng) hiển thị dựa trên chế độ chọn
        if corr_text:
            if view_mode == "Văn bản":
                st.markdown('<div class="diff-inspector">' + corr_text + '</div>', unsafe_allow_html=True)
            elif view_mode == "Đối chiếu tô màu trực quan":
                if highlighted_html:
                    st.markdown('<div class="diff-inspector">' + highlighted_html + '</div>', unsafe_allow_html=True)
                else:
                    st.info("Không phát hiện thay đổi nào so với câu gốc.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli_demo()
    else:
        try:
            import streamlit as st
            run_streamlit_app()
        except Exception as e:
            print("[Demo Fallback] Running CLI demo due to:", e)
            run_cli_demo()
