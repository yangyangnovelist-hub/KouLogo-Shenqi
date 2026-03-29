import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from sklearn.cluster import MiniBatchKMeans
import vtracer
import tempfile
from streamlit_image_coordinates import streamlit_image_coordinates

# --- Constants & Config ---
st.set_page_config(
    page_title="KouLogo AI 精修台",
    layout="wide",
    page_icon="🪄",
    initial_sidebar_state="collapsed"
)

# --- Memory & Cache ---

@st.cache_resource
def get_rembg_session(model_name):
    from rembg import new_session
    return new_session(model_name)

@st.cache_data
def get_checkerboard(width, height, size=18):
    """Generate a constant checkerboard background image."""
    c1, c2 = (255, 255, 255), (230, 230, 230)
    tile = np.zeros((size*2, size*2, 3), dtype=np.uint8)
    tile[:size, :size] = c1; tile[size:, size:] = c1
    tile[:size, size:] = c2; tile[size:, :size] = c2
    repeat_x = (width // (size*2)) + 1
    repeat_y = (height // (size*2)) + 1
    board = np.tile(tile, (repeat_y, repeat_x, 1))
    return board[:height, :width]

# --- Core Processing Logic ---

def apply_color_purify(cv_image_rgba, n_colors, mask=None):
    bgr = cv_image_rgba[:, :, :3].copy()
    alpha = cv_image_rgba[:, :, 3]
    visible = (alpha > 0) & (mask > 0) if mask is not None else alpha > 0
    pixels = bgr[visible].astype(np.float32)
    if len(pixels) < 2: return cv_image_rgba
    model = MiniBatchKMeans(n_clusters=max(1, min(n_colors, len(pixels))), random_state=42, batch_size=2048, n_init=3)
    labels = model.fit_predict(pixels)
    centers = np.clip(model.cluster_centers_, 0, 255).astype(np.uint8)
    out = cv_image_rgba.copy()
    out[visible, :3] = centers[labels]
    return out

def apply_edge_preserving_denoise(cv_image_rgba, mask=None):
    bgr = cv_image_rgba[:, :, :3]
    denoised = cv2.bilateralFilter(bgr, d=7, sigmaColor=35, sigmaSpace=35)
    out = cv_image_rgba.copy()
    if mask is not None: out[mask > 0, :3] = denoised[mask > 0]
    else: out[:, :, :3] = denoised
    return out

def apply_black_cleanup(cv_image_rgba, black_thresh, mask=None):
    bgr = cv_image_rgba[:, :, :3]
    alpha = cv_image_rgba[:, :, 3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = ((gray < black_thresh) & (alpha > 0)).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    clean_mask = np.zeros_like(dark_mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 8: clean_mask[labels == i] = 255
    out = cv_image_rgba.copy()
    eff = clean_mask > 0
    if mask is not None: eff = eff & (mask > 0)
    out[eff, :3] = (0, 0, 0)
    return out

def apply_edge_smoothing(cv_image_rgba, strength):
    alpha = cv_image_rgba[:, :, 3]
    obj_mask = (alpha > 0).astype(np.uint8) * 255
    if np.count_nonzero(obj_mask) == 0: return cv_image_rgba
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(obj_mask, k1, iterations=1)
    ksize = max(3, int(round(strength * 2)) * 2 + 1)
    blurred = cv2.GaussianBlur(dilated, (ksize, ksize), sigmaX=max(0.8, strength))
    smoothed = cv2.threshold(blurred, 64, 255, cv2.THRESH_BINARY)[1]
    out = cv_image_rgba.copy()
    out[:, :, 3] = smoothed
    return out

def apply_defringe(cv_image_rgba, radius, mask=None):
    alpha = cv_image_rgba[:, :, 3]
    obj_mask = (alpha > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    interior = cv2.erode(obj_mask, k, iterations=1)
    if np.count_nonzero(interior) == 0: interior = obj_mask.copy()
    edge_zone = (obj_mask > 0) & (interior == 0)
    bgr = cv_image_rgba[:, :, :3].astype(np.float32)
    interior_color = bgr.copy(); interior_color[interior == 0] = 0
    weight = (interior > 0).astype(np.float32)
    blur_k = max(3, radius * 6 + 1) | 1
    spread_color = cv2.GaussianBlur(interior_color, (blur_k, blur_k), radius * 2.0)
    spread_weight = cv2.GaussianBlur(weight, (blur_k, blur_k), radius * 2.0)
    safe_w = np.where(spread_weight > 1e-6, spread_weight, 1.0)
    clean_rgb = np.clip(spread_color / safe_w[:, :, None], 0, 255)
    out = cv_image_rgba.copy()
    zone = edge_zone if mask is None else edge_zone & (mask > 0)
    out[zone, :3] = clean_rgb[zone].astype(np.uint8)
    out[alpha == 0, :3] = clean_rgb[alpha == 0].astype(np.uint8)
    return out

def apply_rembg_logic(cv_image_rgba):
    from rembg import remove
    pil_in = Image.fromarray(cv2.cvtColor(cv_image_rgba, cv2.COLOR_BGRA2RGBA))
    session = get_rembg_session("u2net")
    pil_out = remove(pil_in, session=session)
    out = cv_image_rgba.copy()
    out[:, :, 3] = np.array(pil_out)[:, :, 3]
    return out

# --- Magic Wand ---

def find_opaque_seed(cv_image_rgba, x, y):
    h, w = cv_image_rgba.shape[:2]
    if 0 <= x < w and 0 <= y < h and cv_image_rgba[y, x, 3] >= 10: return x, y
    for r in range(1, 9):
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if abs(dx)!=r and abs(dy)!=r: continue
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h and cv_image_rgba[ny, nx, 3] >= 10: return nx, ny
    return None

def compute_selection_mask(cv_image_rgba, x, y, tolerance, contiguous):
    seed = find_opaque_seed(cv_image_rgba, x, y)
    if not seed: return None
    x, y = seed
    blr = cv2.GaussianBlur(cv_image_rgba[:, :, :3], (3, 3), 0)
    if contiguous:
        fm = np.zeros((cv_image_rgba.shape[0]+2, cv_image_rgba.shape[1]+2), np.uint8)
        cv2.floodFill(blr, fm, (x, y), (0,0,0), (tolerance,)*3, (tolerance,)*3, cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255<<8))
        mask = fm[1:-1, 1:-1]
    else:
        sc = blr[y, x].astype(np.int16)
        mask = cv2.inRange(blr, np.clip(sc-tolerance,0,255).astype(np.uint8), np.clip(sc+tolerance,0,255).astype(np.uint8))
    mask[cv_image_rgba[:, :, 3] < 10] = 0
    return mask

# --- Utils ---

def scale_image_for_ui(cv_img_rgba, max_size=1200):
    h, w = cv_img_rgba.shape[:2]
    if max(h, w) <= max_size: return cv_img_rgba, 1.0
    scale = max_size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(cv_img_rgba, (nw, nh), interpolation=cv2.INTER_AREA), scale

# --- State ---

if "image" not in st.session_state: st.session_state.image = None
if "protected_mask" not in st.session_state: st.session_state.protected_mask = None
if "history" not in st.session_state: st.session_state.history = []
if "last_click" not in st.session_state: st.session_state.last_click = None

def push_history():
    if st.session_state.image is not None:
        st.session_state.history.append((st.session_state.image.copy(), 
                                        st.session_state.protected_mask.copy() if st.session_state.protected_mask is not None else None))
        if len(st.session_state.history) > 15: st.session_state.history.pop(0)

def undo():
    if st.session_state.history:
        st.session_state.image, st.session_state.protected_mask = st.session_state.history.pop()

# --- UI Styles ---

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f8f9fa; }
    .main-container { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }
    .status-badge { background: #eef2f7; color: #4b5563; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid #e2e8f0; }
    .stButton>button { border-radius: 8px; font-weight: 600; transition: all 0.2s; }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    div[data-testid="stExpander"] { border: none !important; background: #f9fafb !important; border-radius: 10px !important; margin-bottom: 0.5rem !important; }
    .stRadio > div { gap: 10px; }
    .stRadio label { background: #fff; border: 1px solid #e2e8f0; padding: 8px 16px; border-radius: 8px; cursor: pointer; }
    
    .empty-state {
        height: 60vh;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        color: #64748b;
        text-align: center;
        padding: 2rem;
    }
    .empty-state-icon { font-size: 4rem; margin-bottom: 1rem; opacity: 0.5; }
    .empty-state-title { font-size: 1.5rem; font-weight: 800; color: #1e293b; margin-bottom: 0.5rem; }
    .empty-state-text { font-size: 1rem; max-width: 400px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# --- App Body ---

col_ctrl, col_main = st.columns([1, 2.8], gap="large")

with col_ctrl:
    st.markdown('<h2 style="margin-top:0; font-weight:800; color:#1a202c;">🪄 KouLogo AI</h2>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
    if file:
        img = cv2.imdecode(np.asarray(bytearray(file.read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA); img[:,:,3]=255
        if st.session_state.image is None or not img.shape == st.session_state.image.shape:
            st.session_state.image = img.copy(); st.session_state.protected_mask = None; st.session_state.history = []; st.session_state.last_click = None

    if st.session_state.image is not None:
        st.markdown("---")
        h1, h2 = st.columns(2)
        if h1.button("↩ 撤销", use_container_width=True, disabled=not st.session_state.history): undo(); st.rerun()
        if h2.button("🗑 清除保护", use_container_width=True): push_history(); st.session_state.protected_mask = None; st.rerun()

        st.subheader("🛠 工具箱")
        tool = st.radio("选择工具", ["魔棒 (擦除)", "护棒 (保护)", "查看"], horizontal=True, label_visibility="collapsed")
        tol = st.slider("点选容差", 1, 150, 20)
        contig = st.checkbox("仅连接区域", value=False)
        
        st.subheader("⚡ 智能处理")
        scope = st.selectbox("处理范围", ["全图", "仅保护区内", "仅非保护区"], label_visibility="collapsed")
        
        def current_mask():
            if scope == "全图": return None
            m = st.session_state.protected_mask if st.session_state.protected_mask is not None else np.zeros(st.session_state.image.shape[:2], np.uint8)
            return m if scope == "仅保护区内" else cv2.bitwise_not(m)

        with st.expander("🤖 AI 强力去背"):
            if st.button("一键 AI 抠图", use_container_width=True):
                with st.spinner("AI 识别中..."):
                    push_history(); st.session_state.image = apply_rembg_logic(st.session_state.image); st.rerun()
        
        with st.expander("🎨 颜色纯化 & 黑线"):
            k = st.number_input("保留颜色数", 2, 32, 8)
            if st.button("🎨 执行纯化", use_container_width=True):
                push_history(); st.session_state.image = apply_color_purify(st.session_state.image, k, current_mask()); st.rerun()
            bt = st.slider("黑线阈值", 20, 180, 80)
            if st.button("🖊 增强黑线", use_container_width=True):
                push_history(); st.session_state.image = apply_black_cleanup(st.session_state.image, bt, current_mask()); st.rerun()

        with st.expander("✨ 边缘 & 去边"):
            sm = st.slider("平滑强度", 0.5, 4.0, 1.0)
            if st.button("✨ 边缘平滑", use_container_width=True):
                push_history(); st.session_state.image = apply_edge_smoothing(st.session_state.image, sm); st.rerun()
            df = st.slider("去边半径", 1, 12, 3)
            if st.button("🧹 去边渗色", use_container_width=True):
                push_history(); st.session_state.image = apply_defringe(st.session_state.image, df, current_mask()); st.rerun()

        st.markdown("---")
        buf = io.BytesIO(); Image.fromarray(cv2.cvtColor(st.session_state.image, cv2.COLOR_BGRA2RGBA)).save(buf, format="PNG")
        st.download_button("💾 导出 PNG", buf.getvalue(), "ready.png", "image/png", use_container_width=True)

with col_main:
    if st.session_state.image is not None:
        ui_img_bgr, ui_scale = scale_image_for_ui(st.session_state.image)
        h, w = ui_img_bgr.shape[:2]; oh, ow = st.session_state.image.shape[:2]
        st.markdown(f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;"><span class="status-badge">📏 {ow}×{oh}</span><span class="status-badge" style="background:#def7ec; color:#03543f; border-color:#84e1bc;">{"🛡 区域保护激活" if st.session_state.protected_mask is not None else "⚪ 全图模式"}</span></div>', unsafe_allow_html=True)
        ui_img_rgb = cv2.cvtColor(ui_img_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        ui_alpha = ui_img_bgr[:, :, 3] / 255.0
        if st.session_state.protected_mask is not None:
            ui_prot = cv2.resize(st.session_state.protected_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            pm = (ui_prot > 0) & (ui_alpha > 0)
            ui_img_rgb[pm] = (ui_img_rgb[pm] * 0.6 + 200 * 0.4).astype(np.uint8)
        board = get_checkerboard(w, h); ui_final = (ui_img_rgb * ui_alpha[:, :, np.newaxis] + board * (1 - ui_alpha[:, :, np.newaxis])).astype(np.uint8)
        res = streamlit_image_coordinates(Image.fromarray(ui_final), key="main_editor")
        if res and res != st.session_state.last_click:
            st.session_state.last_click = res
            ox, oy = int(res["x"] / ui_scale), int(res["y"] / ui_scale)
            with st.spinner("Processing..."):
                mask = compute_selection_mask(st.session_state.image, ox, oy, tol, contig)
                if mask is not None:
                    if "魔棒" in tool:
                        push_history(); eff = mask.copy()
                        if st.session_state.protected_mask is not None: eff[st.session_state.protected_mask > 0] = 0
                        st.session_state.image[eff > 0, 3] = 0; st.rerun()
                    elif "护棒" in tool:
                        push_history()
                        if st.session_state.protected_mask is None: st.session_state.protected_mask = mask
                        else: st.session_state.protected_mask = cv2.bitwise_or(st.session_state.protected_mask, mask)
                        st.rerun()
        st.markdown('<p style="color:#718096; font-size:0.85em; text-align:center;">💡 提示：点击图片进行魔棒擦除。关闭“仅连接区域”可一次性处理封闭区域。</p>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🖼️</div>
            <div class="empty-state-title">开始精修您的 Logo</div>
            <div class="empty-state-text">
                上传您的成品或草图，使用 <b>AI 一键抠图</b>、<b>魔棒精修</b> 和 <b>边缘平滑</b> 工具，打造完美透明背景。
            </div>
            <p style="margin-top: 2rem; font-size: 0.9em; opacity: 0.7;">点击左侧按钮上传图片 📁</p>
        </div>
        """, unsafe_allow_html=True)
