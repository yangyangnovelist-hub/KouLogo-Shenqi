import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from sklearn.cluster import MiniBatchKMeans
import vtracer
import tempfile

# Try to import rembg, handle if not installed
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# --- Core Logic (Extracted and Adapted from main.py) ---

def apply_color_purify(cv_image_rgba, n_colors):
    bgr = cv_image_rgba[:, :, :3]
    alpha = cv_image_rgba[:, :, 3]
    visible = alpha > 0
    pixels = bgr[visible].astype(np.float32)
    if len(pixels) < 2:
        return cv_image_rgba
    n = max(1, min(n_colors, len(pixels)))
    model = MiniBatchKMeans(n_clusters=n, random_state=42, batch_size=2048, n_init=5)
    labels = model.fit_predict(pixels)
    centers = np.clip(model.cluster_centers_, 0, 255).astype(np.uint8)
    out_bgr = bgr.copy()
    out_bgr[visible] = centers[labels]
    out_rgba = cv_image_rgba.copy()
    out_rgba[:, :, :3] = out_bgr
    return out_rgba

def apply_edge_preserving_denoise(cv_image_rgba, strength):
    bgr = cv_image_rgba[:, :, :3]
    if strength <= 0:
        return cv_image_rgba
    d, sc, ss = (None, None, None)
    if strength == 1: d, sc, ss = 5, 30, 30
    elif strength == 2: d, sc, ss = 7, 45, 45
    else: d, sc, ss = 9, 60, 60
    
    denoised = cv2.bilateralFilter(bgr, d=d, sigmaColor=sc, sigmaSpace=ss)
    out_rgba = cv_image_rgba.copy()
    out_rgba[:, :, :3] = denoised
    return out_rgba

def apply_black_cleanup(cv_image_rgba, black_thresh, min_area, close_iter):
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
        # Re-filter after closing
        num, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
        final_mask = np.zeros_like(clean_mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                final_mask[labels == i] = 255
        clean_mask = final_mask
        
    out_rgba = cv_image_rgba.copy()
    out_rgba[clean_mask > 0, :3] = (0, 0, 0)
    return out_rgba

def apply_edge_smoothing(cv_image_rgba, mode, strength):
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
    out_rgba[:, :, 3] = smoothed
    return out_rgba

def apply_defringe(cv_image_rgba, radius):
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
        out[edge_zone, c] = clean_rgb[edge_zone, c].astype(np.uint8)
        out[alpha == 0, c] = clean_rgb[alpha == 0, c].astype(np.uint8)
    return out

def apply_rembg_logic(cv_image_rgba, model_name):
    if not REMBG_AVAILABLE:
        return cv_image_rgba
    pil_in = Image.fromarray(cv2.cvtColor(cv_image_rgba, cv2.COLOR_BGRA2RGBA))
    # Note: Streamlit Cloud caches models across runs
    session = new_session(model_name)
    pil_out = remove(pil_in, session=session)
    new_alpha = np.array(pil_out)[:, :, 3]
    out = cv_image_rgba.copy()
    out[:, :, 3] = new_alpha
    return out

def convert_to_svg(cv_image_rgba):
    # Use vtracer for high quality vectorization
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        Image.fromarray(cv2.cvtColor(cv_image_rgba, cv2.COLOR_BGRA2RGBA)).save(tmp.name)
        svg_str = vtracer.convert_image_to_svg(tmp.name)
        os.unlink(tmp.name)
    return svg_str

# --- Streamlit UI ---

st.set_page_config(page_title="抠Logo 神器 | KouLogo-Shenqi", layout="wide", page_icon="🪄")

# Custom CSS for premium look
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #007bff; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #0056b3; }
    .stDownloadButton>button { color: #007bff; border: 1px solid #007bff; background: transparent; }
    .stDownloadButton>button:hover { background-color: #e7f1ff; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1 { color: #1a1a1a; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

st.title("🪄 抠Logo 神器")
st.markdown("快速净化、优化并矢量化你的 Logo")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ 处理参数")
    
    with st.expander("🤖 AI 重新抠图 (Rembg)", expanded=False):
        if REMBG_AVAILABLE:
            do_rembg = st.checkbox("启用 AI 智能抠图")
            rembg_model = st.selectbox("模型", ["u2net", "isnet-general-use", "silueta"], index=0)
        else:
            st.warning("⚠️ Rembg 未安装")
            do_rembg = False

    with st.expander("🎨 颜色与降噪", expanded=True):
        do_purify = st.checkbox("纯化颜色", value=True)
        k_colors = st.slider("目标颜色数", 2, 32, 8)
        denoise_strength = st.slider("保边去噪强度", 0, 3, 1)

    with st.expander("🖊 黑线与边缘", expanded=True):
        do_black_cleanup = st.checkbox("黑线净化", value=True)
        black_thresh = st.slider("黑线阈值", 20, 180, 80)
        min_area = st.slider("最小连通面积", 1, 120, 8)
        
        do_edge_smoothing = st.checkbox("边缘平滑", value=True)
        edge_mode = st.selectbox("平滑模式", ["柔和曲线", "直线规整"])
        edge_strength = st.slider("平滑强度", 0.2, 4.0, 1.0)
        
    with st.expander("🧹 去边渗色", expanded=False):
        do_defringe = st.checkbox("启用去边渗色")
        defringe_radius = st.slider("渗色半径", 1, 12, 3)

    st.divider()
    st.info("💡 提示：调整参数后，右侧会自动更新预览。")

# Main interface
uploaded_file = st.file_uploader("📤 拖入或选取一张 Logo 图片", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image[:, :, 3] = 255
    
    # Process
    processed = image.copy()
    
    with st.status("正在魔力加工中...", expanded=True) as status:
        if do_rembg:
            st.write("🤖 正在执行 AI 抠图...")
            processed = apply_rembg_logic(processed, rembg_model)
        
        if denoise_strength > 0:
            st.write("🔇 正在保边去噪...")
            processed = apply_edge_preserving_denoise(processed, denoise_strength)
            
        if do_purify:
            st.write("🎨 正在纯化颜色...")
            processed = apply_color_purify(processed, k_colors)
            
        if do_black_cleanup:
            st.write("🖊 正在净化黑线...")
            processed = apply_black_cleanup(processed, black_thresh, min_area, 1)
            
        if do_edge_smoothing:
            st.write("✨ 正在平滑边缘...")
            processed = apply_edge_smoothing(processed, edge_mode, edge_strength)
            
        if do_defringe:
            st.write("🧹 正在去边渗色...")
            processed = apply_defringe(processed, defringe_radius)
        
        status.update(label="处理完成！", state="complete", expanded=False)

    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼 原始图片")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA), use_container_width=True)
        
    with col2:
        st.subheader("🚀 优化结果")
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGRA2RGBA), use_container_width=True)
        
    # Download section
    st.divider()
    st.subheader("💾 导出你的杰作")
    d1, d2, d3 = st.columns(3)
    
    # Export PNG
    png_buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGRA2RGBA)).save(png_buf, format="PNG")
    d1.download_button(
        label="📥 下载优化后的 PNG",
        data=png_buf.getvalue(),
        file_name=f"koulogo_processed.png",
        mime="image/png"
    )
    
    # Export SVG
    with st.spinner("矢量化中..."):
        svg_data = convert_to_svg(processed)
    d2.download_button(
        label="📥 下载 SVG 矢量图",
        data=svg_data,
        file_name=f"koulogo_vector.svg",
        mime="image/svg+xml"
    )
    
    st.balloons()

else:
    st.info("请先上传图片开始体验 ✨")
    # Show placeholder or instructions
    st.image("https://images.unsplash.com/photo-1626785774573-4b799315345d?q=80&w=1000&auto=format&fit=crop", caption="准备好让你的 Logo 焕发新生了吗？", use_container_width=True)
