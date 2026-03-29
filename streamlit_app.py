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

# Try to import rembg, handle if not installed
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# --- Core Logic (Adapted from main.py) ---

def apply_color_purify(cv_image_rgba, n_colors, mask=None):
    bgr = cv_image_rgba[:, :, :3]
    alpha = cv_image_rgba[:, :, 3]
    if mask is not None:
        visible = (alpha > 0) & (mask > 0)
    else:
        visible = alpha > 0
    
    pixels = bgr[visible].astype(np.float32)
    if len(pixels) < 2:
        return cv_image_rgba
    
    n = max(1, min(n_colors, len(pixels)))
    model = MiniBatchKMeans(n_clusters=n, random_state=42, batch_size=2048, n_init=5)
    labels = model.fit_predict(pixels)
    centers = np.clip(model.cluster_centers_, 0, 255).astype(np.uint8)
    
    out_rgba = cv_image_rgba.copy()
    out_rgba[visible, :3] = centers[labels]
    return out_rgba

def apply_edge_preserving_denoise(cv_image_rgba, strength, mask=None):
    bgr = cv_image_rgba[:, :, :3]
    if strength <= 0:
        return cv_image_rgba
    d, sc, ss = (None, None, None)
    if strength == 1: d, sc, ss = 5, 30, 30
    elif strength == 2: d, sc, ss = 7, 45, 45
    else: d, sc, ss = 9, 60, 60
    
    denoised = cv2.bilateralFilter(bgr, d=d, sigmaColor=sc, sigmaSpace=ss)
    
    out_rgba = cv_image_rgba.copy()
    if mask is not None:
        out_rgba[mask > 0, :3] = denoised[mask > 0]
    else:
        out_rgba[:, :, :3] = denoised
    return out_rgba

def apply_black_cleanup(cv_image_rgba, black_thresh, min_area, close_iter, mask=None):
    bgr = cv_image_rgba[:, :, :3]
    alpha = cv_image_rgba[:, :, 3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = ((gray < black_thresh) & (alpha > 0)).astype(np.uint8) * 255
    
    # Remove small components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    clean_mask = np.zeros_like(dark_mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255
            
    if close_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, k, iterations=close_iter)
        # Re-filter
        num, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
        new_mask = np.zeros_like(clean_mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                new_mask[labels == i] = 255
        clean_mask = new_mask
        
    out_rgba = cv_image_rgba.copy()
    effect_mask = clean_mask > 0
    if mask is not None:
        effect_mask = effect_mask & (mask > 0)
    out_rgba[effect_mask, :3] = (0, 0, 0)
    return out_rgba

def apply_edge_smoothing(cv_image_rgba, mode, strength, mask=None):
    # This usually applies to the alpha channel (global or local)
    alpha = cv_image_rgba[:, :, 3].copy()
    obj_mask = (alpha > 0).astype(np.uint8) * 255
    if np.count_nonzero(obj_mask) == 0:
        return cv_image_rgba

    if mode == "柔和曲线":
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(obj_mask, k1, iterations=1)
        ksize = max(3, int(round(strength * 2)) * 2 + 1)
        blurred = cv2.GaussianBlur(dilated, (ksize, ksize), sigmaX=max(0.8, strength))
        smoothed = cv2.threshold(blurred, 64, 255, cv2.THRESH_BINARY)[1]
    else:
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        smoothed = np.zeros_like(obj_mask)
        eps = max(0.5, 1.2 * strength)
        for cnt in contours:
            if cv2.contourArea(cnt) < 4: continue
            approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
            cv2.drawContours(smoothed, [approx], -1, 255, thickness=cv2.FILLED)

    out_rgba = cv_image_rgba.copy()
    if mask is not None:
        # Blend smoothed alpha onto current alpha within mask? 
        # For simplicity, we mostly apply smoothing to the whole object boundaries
        out_rgba[:, :, 3] = smoothed
    else:
        out_rgba[:, :, 3] = smoothed
    return out_rgba

def apply_defringe(cv_image_rgba, radius, mask=None):
    alpha = cv_image_rgba[:, :, 3]
    obj_mask = (alpha > 0).astype(np.uint8) * 255
    k_size = radius * 2 + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    interior = cv2.erode(obj_mask, k, iterations=1)
    if np.count_nonzero(interior) == 0: interior = obj_mask.copy()
    edge_zone = (obj_mask > 0) & (interior == 0)

    bgr = cv_image_rgba[:, :, :3].astype(np.float32)
    interior_color = bgr.copy()
    interior_color[interior == 0] = 0
    interior_weight = (interior > 0).astype(np.float32)

    blur_k = max(3, radius * 6 + 1) | 1
    sigma = radius * 2.0
    spread_color = cv2.GaussianBlur(interior_color, (blur_k, blur_k), sigma)
    spread_weight = cv2.GaussianBlur(interior_weight, (blur_k, blur_k), sigma)
    safe_w = np.where(spread_weight > 1e-6, spread_weight, 1.0)
    clean_rgb = np.clip(spread_color / safe_w[:, :, None], 0, 255)

    out = cv_image_rgba.copy()
    for c in range(3):
        # Defringe zone
        zone = edge_zone
        if mask is not None: zone = zone & (mask > 0)
        out[zone, c] = clean_rgb[zone, c].astype(np.uint8)
        # Also clean fully transparent pixels to avoid artifacts when upscaling
        out[alpha == 0, c] = clean_rgb[alpha == 0, c].astype(np.uint8)
    return out

def apply_rembg_logic(cv_image_rgba, model_name):
    if not REMBG_AVAILABLE:
        return cv_image_rgba
    pil_in = Image.fromarray(cv2.cvtColor(cv_image_rgba, cv2.COLOR_BGRA2RGBA))
    session = new_session(model_name)
    pil_out = remove(pil_in, session=session)
    new_alpha = np.array(pil_out)[:, :, 3]
    out = cv_image_rgba.copy()
    out[:, :, 3] = new_alpha
    return out

def convert_to_svg(cv_image_rgba):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        Image.fromarray(cv2.cvtColor(cv_image_rgba, cv2.COLOR_BGRA2RGBA)).save(tmp.name)
        svg_str = vtracer.convert_image_to_svg(tmp.name)
        os.unlink(tmp.name)
    return svg_str

# --- Magic Wand & Local Tools ---

def find_opaque_seed(cv_image_rgba, x, y):
    h, w = cv_image_rgba.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    if cv_image_rgba[y, x, 3] >= 10:
        return x, y
    # Spiral search
    for r in range(1, 9):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and cv_image_rgba[ny, nx, 3] >= 10:
                    return nx, ny
    return None

def compute_selection_mask(cv_image_rgba, x, y, tolerance, contiguous):
    h, w = cv_image_rgba.shape[:2]
    seed = find_opaque_seed(cv_image_rgba, x, y)
    if not seed:
        return None
    x, y = seed
    
    # Pre-blur for smoother selection as in main.py
    blr = cv2.GaussianBlur(cv_image_rgba[:, :, :3], (3, 3), 0)
    
    if contiguous:
        # Create a mask for floodFill
        fm = np.zeros((h + 2, w + 2), np.uint8)
        flags = cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        cv2.floodFill(blr, fm, (x, y), (0, 0, 0), (tolerance,) * 3, (tolerance,) * 3, flags)
        mask = fm[1:-1, 1:-1]
    else:
        seed_color = blr[y, x].astype(np.int16)
        lo = np.clip(seed_color - tolerance, 0, 255).astype(np.uint8)
        hi = np.clip(seed_color + tolerance, 0, 255).astype(np.uint8)
        mask = cv2.inRange(blr, lo, hi)
        
    # Only select opaque/semi-opaque pixels
    alpha = cv_image_rgba[:, :, 3]
    mask[alpha < 10] = 0
    return mask

def apply_magic_wand_erase(cv_image_rgba, selection_mask, protected_mask):
    if selection_mask is None:
        return cv_image_rgba
    
    # Exclude protected area
    effective_mask = selection_mask.copy()
    if protected_mask is not None:
        effective_mask[protected_mask > 0] = 0
        
    out = cv_image_rgba.copy()
    out[effective_mask > 0, 3] = 0
    return out

# --- Session State Initialization ---

if "image" not in st.session_state:
    st.session_state.image = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "protected_mask" not in st.session_state:
    st.session_state.protected_mask = None
if "history" not in st.session_state:
    st.session_state.history = []
if "selection_mask" not in st.session_state:
    st.session_state.selection_mask = None

def push_history():
    if st.session_state.image is not None:
        st.session_state.history.append((st.session_state.image.copy(), 
                                        st.session_state.protected_mask.copy() if st.session_state.protected_mask is not None else None))
        if len(st.session_state.history) > 20:
            st.session_state.history.pop(0)

def undo():
    if st.session_state.history:
        prev_img, prev_mask = st.session_state.history.pop()
        st.session_state.image = prev_img
        st.session_state.protected_mask = prev_mask
        st.session_state.selection_mask = None

def reset_masks():
    st.session_state.protected_mask = None
    st.session_state.selection_mask = None

# --- UI Setup ---

st.set_page_config(page_title="Logo 精修台 v14 (Web)", layout="wide", page_icon="🪄")

# Inject Custom CSS for dark theme and layout consistent with main.py
st.markdown("""
<style>
    .stApp { background-color: #1e1e1e; color: #e0e0e0; }
    .main { background-color: #2b2b2b; }
    header, footer { display: none !important; }
    .stButton>button { color: #fff; background-color: #333; border: 1px solid #555; }
    .stButton>button:hover { background-color: #444; border-color: #777; }
    .stSelectbox div[data-baseweb="select"] { background-color: #333; border: 1px solid #555; }
    .stSlider label, .stCheckbox label { color: #aaa !important; }
    .stExpander { background-color: #252525; border: 1px solid #333 !important; }
    .tool-active { border: 2px solid #007bff !important; }
    h1, h2, h3 { color: #fff; }
    .status-bar { background: #333; padding: 4px 10px; border-radius: 4px; font-family: monospace; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

# Layout
col_ctrl, col_main = st.columns([1, 3])

with col_ctrl:
    st.title("🪄 Logo 精修")
    
    # File handling
    uploaded_file = st.file_uploader("打开图片", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        loaded_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        if loaded_img.ndim == 2:
            loaded_img = cv2.cvtColor(loaded_img, cv2.COLOR_GRAY2BGRA)
        elif loaded_img.shape[2] == 3:
            loaded_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2BGRA)
            loaded_img[:, :, 3] = 255
            
        if st.session_state.original_image is None or not np.array_equal(st.session_state.original_image, loaded_img):
            st.session_state.original_image = loaded_img.copy()
            st.session_state.image = loaded_img.copy()
            st.session_state.protected_mask = None
            st.session_state.selection_mask = None
            st.session_state.history = []

    if st.session_state.image is not None:
        tb1, tb2 = st.columns(2)
        if tb1.button("↩ 撤销", disabled=len(st.session_state.history) == 0):
            undo()
            st.rerun()
        if tb2.button("🗑 清除保护区"):
            push_history()
            st.session_state.protected_mask = None
            st.rerun()

        st.divider()
        
        # Tools
        st.subheader("🛠 交互式工具")
        tool_mode = st.radio("选择工具", ["魔棒 (Erase)", "护棒 (Protect)", "查看 (View)"], horizontal=True)
        
        tol = st.slider("容差 (Tolerance)", 1, 150, 20)
        contig = st.checkbox("仅连接区域 (Contiguous)", value=True)
        
        st.divider()
        
        # Bulk Process
        st.subheader("⚡ 批量处理")
        scope = st.selectbox("处理范围", ["全图", "仅保护区内", "仅非保护区"])
        
        def get_mask_for_scope():
            if scope == "全图": return None
            if st.session_state.protected_mask is None:
                return np.zeros(st.session_state.image.shape[:2], dtype=np.uint8) if scope == "仅保护区内" else np.ones(st.session_state.image.shape[:2], dtype=np.uint8) * 255
            if scope == "仅保护区内": return st.session_state.protected_mask
            return cv2.bitwise_not(st.session_state.protected_mask)

        with st.expander("AI 抠图 & 净化"):
            if REMBG_AVAILABLE:
                rm_model = st.selectbox("AI 模型", ["u2net", "isnet-general-use", "silueta"])
                if st.button("🤖 执行 AI 抠图"):
                    push_history()
                    st.session_state.image = apply_rembg_logic(st.session_state.image, rm_model)
                    st.rerun()
            
            purify_k = st.number_input("目标颜色数", 2, 32, 8)
            if st.button("🎨 纯化颜色"):
                push_history()
                st.session_state.image = apply_color_purify(st.session_state.image, purify_k, get_mask_for_scope())
                st.rerun()

        with st.expander("线条 & 细节"):
            bl_thr = st.slider("黑线阈值", 20, 180, 80)
            if st.button("🖊 黑线净化"):
                push_history()
                st.session_state.image = apply_black_cleanup(st.session_state.image, bl_thr, 8, 1, get_mask_for_scope())
                st.rerun()
                
            ed_mode = st.selectbox("平滑模式", ["柔和曲线", "直线规整"])
            ed_str = st.slider("强度", 0.5, 4.0, 1.0)
            if st.button("✨ 边缘平滑"):
                push_history()
                st.session_state.image = apply_edge_smoothing(st.session_state.image, ed_mode, ed_str, get_mask_for_scope())
                st.rerun()

        with st.expander("去燥 & 去边"):
            dn_str = st.slider("去燥强度", 0, 3, 1)
            if st.button("🔇 保边去噪"):
                push_history()
                st.session_state.image = apply_edge_preserving_denoise(st.session_state.image, dn_str, get_mask_for_scope())
                st.rerun()
                
            df_rad = st.slider("渗色半径", 1, 12, 3)
            if st.button("🧹 去边渗色"):
                push_history()
                st.session_state.image = apply_defringe(st.session_state.image, df_rad, get_mask_for_scope())
                st.rerun()

        st.divider()
        st.subheader("💾 导出项目")
        ex1, ex2 = st.columns(2)
        
        # PNG
        png_io = io.BytesIO()
        Image.fromarray(cv2.cvtColor(st.session_state.image, cv2.COLOR_BGRA2RGBA)).save(png_io, format="PNG")
        ex1.download_button("💾 PNG", png_io.getvalue(), "processed.png", "image/png")
        
        # SVG
        if ex2.button("📐 生成 SVG"):
            with st.spinner("矢量化中..."):
                svg_data = convert_to_svg(st.session_state.image)
                st.download_button("📩 下载 SVG", svg_data, "processed.svg", "image/svg+xml")

with col_main:
    if st.session_state.image is not None:
        h, w = st.session_state.image.shape[:2]
        
        # Display Info
        st.markdown(f'<div class="status-bar">📏 {w} × {h} px | 🛡 保护区域: {"已开启" if st.session_state.protected_mask is not None else "全图"}</div>', unsafe_allow_html=True)
        
        # Image Preview with Overlays
        display_img = st.session_state.image.copy()
        
        # Apply Protection Highlight (Light Gray Overlay)
        if st.session_state.protected_mask is not None:
            gray_ov = np.full((h, w, 3), (200, 200, 200), dtype=np.uint8)
            alpha_mask = (st.session_state.protected_mask > 0) & (display_img[:, :, 3] > 0)
            alpha_val = 0.4
            for c in range(3):
                display_img[alpha_mask, c] = (display_img[alpha_mask, c] * (1-alpha_val) + gray_ov[alpha_mask, c] * alpha_val).astype(np.uint8)

        # Main interactive component
        pil_display = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGRA2RGBA))
        
        # Using streamlit-image-coordinates
        value = streamlit_image_coordinates(pil_display, key="logo_canvas")
        
        if value:
            vx, vy = value["x"], value["y"]
            
            # Re-compute selection for preview/action
            mask = compute_selection_mask(st.session_state.image, vx, vy, tol, contig)
            
            if mask is not None:
                if tool_mode == "魔棒 (Erase)":
                    push_history()
                    st.session_state.image = apply_magic_wand_erase(st.session_state.image, mask, st.session_state.protected_mask)
                    st.toast("已擦除选中区域", icon="🪄")
                    st.rerun()
                elif tool_mode == "护棒 (Protect)":
                    push_history()
                    if st.session_state.protected_mask is None:
                        st.session_state.protected_mask = mask
                    else:
                        st.session_state.protected_mask = cv2.bitwise_or(st.session_state.protected_mask, mask)
                    st.toast("区域已加入保护", icon="🛡")
                    st.rerun()
        
        st.info("💡 点击图片任意位置即可触发「魔棒」或「护棒」效果。")
        
    else:
        st.empty()
        st.info("请先上传 Logo 图片以启动精修台。")
        st.image("https://images.unsplash.com/photo-1626785774573-4b799315345d?q=80&w=1000&auto=format&fit=crop", use_container_width=True)

