import os
from PIL import Image, ImageDraw, ImageFont
W, H = 980, 1150
BG = (250, 250, 252)
BOX_FILL = (255, 255, 255)
BOX_LINE = (208, 214, 224)
INK = (28, 32, 40)
SUB = (108, 118, 132)
ACCENT = (214, 122, 58)
ARROW = (150, 160, 175)
def font(size, bold=False):
    for path in (r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
                 r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()
F_TITLE = font(30, True)
F_NODE = font(21, True)
F_ROLE = font(16)
F_TAG = font(14)
F_FOOT = font(13)
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
d.text((60, 42), "LCS 采样期颜色控制流程", font=F_TITLE, fill=INK)
d.text((60, 82), "颜色只占潜在 patch 空间的 3 维子空间，与结构正交", font=F_ROLE, fill=SUB)
STAGES = [
    ("LCS Load Data", "校准：PCA 求颜色基与锚点", "按 VAE 指纹缓存"),
    ("LCS Color Anchor", "防漂移：各步向锚点回拉", "self / reference / smooth"),
    ("LCS Tone Adjust", "定影调：对比度·亮度·饱和·色温", "preset 或手动"),
    ("LCS Color Intervene", "定色相：目标色位移向量", "#RRGGBB  strength 0.2+"),
    ("LCS Sharpness Intervene", "控锐度：锐度子空间定向调整", "-1 柔  ←  0  →  +1 锐"),
    ("KSampler", "去噪：按模型自身加速配方", "步数与调度不改"),
]
X, BW, BH, GAP, TOP = 70, 470, 108, 44, 132
cx = X + BW // 2
for i, (node, role, tag) in enumerate(STAGES):
    y = TOP + i * (BH + GAP)
    d.rounded_rectangle([X, y, X + BW, y + BH], radius=14, fill=BOX_FILL, outline=BOX_LINE, width=2)
    d.rounded_rectangle([X, y, X + 6, y + BH], radius=3, fill=ACCENT)
    d.text((X + 26, y + 22), node, font=F_NODE, fill=INK)
    d.text((X + 26, y + 58), role, font=F_ROLE, fill=SUB)
    d.line([X + BW + 22, y + BH // 2, X + BW + 96, y + BH // 2], fill=BOX_LINE, width=1)
    d.text((X + BW + 106, y + BH // 2 - 10), tag, font=F_TAG, fill=ACCENT)
    if i < len(STAGES) - 1:
        ay = y + BH + 8
        d.line([cx, ay, cx, ay + GAP - 18], fill=ARROW, width=2)
        d.polygon([(cx - 6, ay + GAP - 22), (cx + 6, ay + GAP - 22), (cx, ay + GAP - 12)], fill=ARROW)
d.text((70, H - 108), "顺序：校准 → 锚定 → 影调 → 色相 → 锐度 → 采样", font=F_ROLE, fill=INK)
d.text((70, H - 66),
       "facok/ComfyUI-LCS (MIT) · 官方 ExplainableML/LCS · Pach, Bader, Bouniot, Belongie, ICML 2026",
       font=F_FOOT, fill=SUB)
out = r"E:\UnifiedTrainer\UTrainer\tmp\LCS_flow.png"
img.save(out)
print("saved", out, img.size)