# Logo 精修台

一个用于 Logo 图片精修和抠图的桌面工具，支持魔棒擦除、画笔工具、保护区域、AI 重新抠图、去边渗色、矢量化导出等功能。

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)

---

## 功能

**交互工具**
- 🪄 **魔棒** — 移动鼠标预览，左键点击擦除同色区域
- 🛡 **护棒** — 点击将同色区域加入保护，后续任何操作都不会影响保护区
- ✏ **护笔** — 拖动涂抹任意区域作为保护区
- 🖌 **画笔擦** — 自由涂抹擦除像素
- 💧 **画笔补** — 将擦除的像素还原为原始颜色

**批量处理**
- 🎨 **纯化颜色** — KMeans 量化，减少颜色数量，让色块更干净
- 🔇 **保边去噪** — 双边滤波，去噪同时保留边缘细节
- 🖊 **黑线净化** — 清理杂散黑色像素，加强线条
- ✨ **边缘平滑** — 高斯平滑或多边形拟合，去除锯齿
- 🧹 **去边渗色** — 消除抠图后边缘残留的背景色（泥巴像素）
- 🤖 **AI 重新抠图** — 用 Rembg/U2Net 深度学习重新生成高质量 alpha 遮罩

**导出**
- 💾 **PNG** — 支持放大倍数、DPI 设置、透明边缘模式
- 💾 **SVG** — 支持 vtracer（彩色）、Potrace CLI（黑线最优质）、Inkscape（需安装）

---

## 安装

**基础依赖**

```bash
pip install -r requirements.txt
```

**macOS 注意**：系统自带 Python 没有 tkinter，建议用 Homebrew 安装：

```bash
brew install python-tk
```

**可选：AI 抠图**

```bash
pip install rembg
```

首次使用时会自动下载模型（约 170MB）。

**可选：Potrace SVG 引擎**

```bash
# macOS
brew install potrace

# Windows
# 从 https://potrace.sourceforge.net 下载
```

---

## 运行

```bash
python main.py
```

---

## 使用流程

1. 拖入图片或点击「打开」
2. 用**护棒 / 护笔**标记不想被修改的区域（浅灰色高亮）
3. 用**魔棒**点击要擦除的背景色区域
4. 批量操作：先「去边渗色」→「边缘平滑」→「纯化颜色」（视需要）
5. 觉得边缘还不够干净？点「AI 重新抠图」让神经网络重做 alpha
6. 导出 PNG 或 SVG

**快捷键**

| 键 | 功能 |
|---|---|
| `Ctrl/Cmd + Z` | 撤销 |
| `F` | 适应窗口 |
| `[` / `]` | 减小 / 增大笔刷 |
| `Ctrl/Cmd + S` | 导出 PNG |
| `Ctrl/Cmd + E` | 导出 SVG |

---

## 依赖说明

| 包 | 用途 |
|---|---|
| opencv-python | 图像处理核心 |
| numpy | 数组运算 |
| Pillow | 图像读写、显示 |
| scikit-learn | KMeans 颜色量化 |
| tkinterdnd2 | 拖拽支持 |
| vtracer | 彩色矢量化 |
| rembg *(可选)* | AI 抠图 |

---

## License

MIT
