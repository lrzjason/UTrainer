# 透明图层训练改进方案（偏色 + 边缘明显）

> 参考论文：**PixPerfect**（arXiv:2512.03247，判别性像素空间去污）与 **MaskFlow**（arXiv:2608.06929，mask 进概率路径 + Soft-Poisson De-seaming）。
> 针对现状：Qwen-Image 2.1 LoKr 直接输出 RGBA 透明图层（`_T`），合成回参考图 `_R` 时**偏色**、**边缘明显**。
> 日期：2026-10-08；代码基线：`UTrainer/`（含 `losses/pfm.py` 的可微 VAE 解码、`losses/base.py` 的 `reduce_masked`、`losses/guide_flow_matching.py`、缓存的 `_T/_R/_O` 像素副本）。

---

## 0. 先诊断：两个症状的物理来源

| 症状 | 直接原因 | 论文里的对应 |
|---|---|---|
| **偏色** | FM loss 在 64ch latent 上对 RGB 一视同仁；VAE 解码后在透明区/边缘区出现几个灰阶的色相漂移，人眼敏感但 latent MSE 迟钝 | PixPerfect §1：*chromatic shifts*；§A：latent 空间纠缠（只改 mask 内 latent，背景照样漂） |
| **边缘明显** | alpha matte 边缘（1–2px 过渡带）与 `_R` 的边界对不齐，叠加 latent 16× 下采样的空间错位；像素域 L1 在全图均匀加权时边缘信号被淹没 | MaskFlow §1：*visible seams along the mask boundary*；PixPerfect §1：*visible seams arising from content discontinuities* |

两篇论文的共同立场：**这些病在 latent 空间治不了，必须在像素空间动手**——这与我们前一段 composite loss 实验的教训吻合（latent 合成监督被证明无效，见 `md/composite_consistency_loss.md` §2）。

---

## 1. 可借鉴的核心组件（按收益排序）

### A. 判别性像素空间损失（治偏色，PixPerfect §3.1）

**原理**：普通 L1/LPIPS 对几个灰阶的色差不敏感。PixPerfect 构造"放大色差"的监督目标，再做逐样本的多项式色调映射把预测和 GT 投影进判别空间算损失。

**我们的移植版**（不需要训 refiner，直接作为 LoKr 的辅助 loss）：

1. 解码 `x0_hat`（已有 `adapter.decode_latent_differentiable`，pfm.py 同款）→ 得 `T̂_rgb, α̂`；
2. 合成 `comp = α̂·T̂_rgb + (1−α̂)·R`（R 取缓存像素副本，GT 同理）；
3. 在 **mask 内** 构造放大目标 `y_amp = GT + β·(comp − GT)`，β∈[20,40]；
4. 对 comp 与 GT 各做逐样本 3 次多项式回归（5 阶，Moore–Penrose 伪逆）得到色调映射 fθ，把两者投影后进 `L1 + LPIPS`；
5. **L1 权重 ≫ 感知项**（PixPerfect 用 w1=64, w2=5, w3=1——色彩一致性压倒一切）。

> 说明：pixperfect 的 refiner 是独立 GAN，我们不做——只借它的**损失形式**。关键收益是把"几灰阶的偏色"从噪声里抬出来。

### B. 梯度域接缝损失（治边缘，MaskFlow §4.2 的 SPD 思想）

**原理**：MaskFlow 的 SPD 不是像素 L1，而是"梯度对齐 + 锚定"——在 mask 内让生成图与目标的**梯度场**一致、边界处锚定到源图。接缝本质是梯度域的不连续。

**我们的移植版**（SPD-lite，作为 loss 而非迭代求解）：

$$L_{\text{grad}} = \sum_{p \in \text{matte}} \tilde m(p)\,\|\nabla\,comp(p) - \nabla\,GT(p)\|^2$$

其中 `comp = α̂·T̂_rgb + (1−α̂)·R`，`∇` 用 Sobel/前向差分；matte 外锚定到 `∇R`（即惩罚背景区梯度变化）。
- 这与像素 L1 互补：L1 管色值，梯度域管边缘连续性；
- 不需要 Jacobi 迭代（那是 MaskFlow 的推理期精修）；训练期只需梯度匹配；
- `edge_band_drop`/`feather` 可继续用（与本文档正交）。

### C. Mask 自适应归一化（治"小 mask 被淹没"，MaskFlow §4.1 MFL）

**原理**：把 FM 损失按软 mask 面积归一：`HW / Σm̃ · m̃ ⊙ err`，小 mask 样本不再被大 mask 稀释。

**移植版**：在 `flow_matching`/`guide_flow_matching` 的 `reduce_masked` 里加 `area_normalize: true` 选项（我们的 `loss_mask` 已有缓存 `_T_mask_76x52.npz`，改动极小）。当数据集里物体占比跨度大（7%–50%）时，这一项能显著稳定小物体的学习。

### D. 孪生概率路径（背景更快收敛，MaskFlow SPP）

`x_σ = m̃⊙((1−σ)x₀ + σε) + (1−m̃)⊙((1−σ^γ)x_S + σ^γε)`，γ≈1.5。背景路径更快回到源图，给前景生成提供稳定锚点。
**我们的情况**：输出是透明图层而非全图，`_R` 已是输入条件——SPP 的收益主要在"背景漂移"，而我们本来就不需要生成背景（α=0 区域直接透明）。**结论：暂不引入**，列为可选项；若后续出现"非编辑区被误生成"再启用。

### E. 推理期色彩池化（PixPerfect §3.3，纯推理 trick，零训练成本）

对同一输入做 N 次随机 color jitter → 分别生成 → 用 refiner 的输入-输出差作为"接近 GT 程度"的指标选最优。**我们没有 refiner**，可直接用 composite 残差作指标：生成后计算 `|comp(T̂,R) − O|` 的 mask 内均值，多采样几次取最小。可作为 ComfyUI 侧的零成本增益。

---

## 2. 落地优先级与预期收益

**2026-10-10 实测更新（mask test epoch19 vs nomask-19）**:改进版 loss 在 30 组子集 560 步后,
输出**全面模糊**(laplacian_var 42.8 vs 基线 82.8,全局性的、不只边界)。根因判定:

1. **decode_scale=0.5 是主因**:半分辨率像素监督 = 只约束低频。解码前对 latent 做 area 降采样、
   目标也双线性降到一半——匹配降采样后的像素,最省力的解就是模糊。PixPerfect 的像素损失在
   **全分辨率** 1024² 上运行,没有这个偏差。修复:decode_scale → 1.0(decoder 分块 checkpoint
   已启用,VRAM 可控;OOM 则退 0.75)。
2. σ_max=0.35 仍偏宽(0.2–0.35 区间 x0̂ 仍带可观噪声)→ 收紧到 **0.2**。
3. composite weight 0.05 → **0.03**(辅助项进一步让位给主 FM)。
4. 次要因素:560 步 vs 基线 4200 步的欠训练放大了上述偏差。

已应用到 box G 的 v2 配置(decode_scale 1.0 / σ_max 0.2 / weight 0.03)。

| 优先级 | 方案 | 预期治的病 | 成本 | 风险 |
|---|---|---|---|---|
| **P0** | **A. 判别性像素空间损失**（comp 上） | 偏色 | 中（复用 pfm 解码管线；每步 +1 解码） | 中：注意 σ 门控（沿用 `sigma_max≈0.35`），低 σ 才施加 |
| **P0** | **B. 梯度域接缝损失** | 边缘明显 | 低（同上解码，加 Sobel） | 低：与 A 共用一次解码 |
| **P1** | **C. Mask 面积归一化** | 小物体学习不稳 | 极低（改 `reduce_masked` 一处） | 低 |
| **P2** | E. 推理色彩池化 | 偏色残留 | 零训练成本 | 无 |
| **备选** | D. SPP | 背景漂移（当前未观察到） | 高 | 不建议先上 |

**关键原则（从前一轮 composite loss 的失败里学到的，务必遵守）**：

1. **σ 门控必须有**：高 σ 的 x0̂ 本来就是糊的条件均值，此时上像素损失 = 教模型糊（上一轮模糊的根因）；
2. **权重一律从 0.03–0.1 起步**，pixel 域辅助 loss 只做"引导"不做"主导"（主 FM 占绝对权重）；
3. **羽化目标 + 过渡带降权**：边缘目标要可达，过渡带不要满权重锤；
4. **透明区目标 = 纯黑**（数据已满足，本地实测确认），fg 预乘项如需保留要小权重；
5. **遥测先行**：`mae_255`（合成残差）、`iou_50`（matte 形状）、`alpha_mean`——先当指标跑通，再决定权重。

---

## 3. 建议的实验顺序

1. **Step 1（无训练，先出基线数字）**：在现有 nomask-19 的验证图上，把 `comp 残差 / iou_50 / 偏色均值` 三个指标打出来，建立基线；
2. **Step 2（单项消融）**：只加 A（判别性像素损失，weight 0.05，σ≤0.35），训 2 epoch，看偏色是否下降；
3. **Step 3**：加 B（梯度域损失，weight 0.05），看接缝；
4. **Step 4**：A+B+C 组合，对比 nomask-19 与上一轮 composite 的失败案例集；
5. **Step 5**：如果 P0 两项仍压不住，再考虑 E（推理池化）或 D（SPP）。

---

## 4. 配置草案（已按此实现，2026-10-08）

```json
"losses": [
  {"type": "guide_flow_matching", "weight": 1.0,
   "params": {"fixed_guidance_scale": 1.5, "use_weighting": false,
              "area_normalize": true}},
  {"type": "composite_consistency", "weight": 0.05,
   "params": {
     "decode_scale": 0.5, "metric": "l1",
     "matte_weight": 0.25, "fg_weight": 0.05,
     "feather_sigma_px": 1.0, "edge_band_drop": 0.5,
     "sigma_max": 0.35, "every_n_steps": 1,
     "discriminative_space": true,        // A：PixPerfect 色调映射
     "disc_beta": [20, 40],               // A：放大系数范围
     "grad_weight": 0.05                  // B：梯度域接缝损失
   }}
]
```

> C 的 `area_normalize` 落在 guide_flow_matching 的 params 里（走 `base.reduce_masked`，
> 需要 `loss_mask`，即数据集的 mask_configs 缓存）。
> 注：当前 config 文件名（`..._scratch_comp.json`）是历史遗留，内容以 loss 列表为准。

> 注：`composite_consistency` 是我们已有的像素域 loss 骨架（当前已下线回滚）；A/B/C 都是往它里面加开关。若决定重做，直接在原文件上加这三个参数即可，不必新建 loss。

---

## 5. 与前一轮失败的对应关系（防重蹈覆辙）

| 上轮症状 | 根因 | 本方案对应修复 |
|---|---|---|
| 模糊 | 高 σ 像素监督 | **σ 门控**（A/B 均只在 σ≤0.35 施加） |
| 紫噪 | fg 退化 + 目标歧义 | fg 降权 0.05；数据侧目标已是黑透明 |
| 接缝 | 边缘满权重锤击 + 半分辨率错位 | **B 梯度域**（管连续性）+ 羽化 + band_drop 保留 |
| 扩张覆盖 | composite 对 `T̂≈R` 区零惩罚 | **A 判别空间**（把色差放大出来）+ matte 钉住 α̂ |
| 蓝色半透明 | fg/composite 低 σ 偏置 × turbo 6 步 | 权重 0.05 + σ 门控；必要时 E 推理池化兜底 |

---

## 6. 参考公式速查

- **判别性色调映射**：`y_amp = x_gt + β(x_pred − x_gt)`，`fθ` 为逐样本 5 阶多项式回归（Moore–Penrose 伪逆），`L1+LPIPS` 在投影后空间计算。
- **SPD 目标（训练期简化版）**：`min_z ∫ m̃‖∇z−∇y‖² + m̃‖z−y‖² + (1−m̃)‖z−x_S‖²`，边界 `z|∂Ω̃ = x_S|∂Ω̃`；我们只用它的梯度对齐项作为 loss。
- **MFL 归一**：`HW/Σm̃ · m̃ ⊙ ‖v̂−v‖²`，权重 ω_M=2.0 参考值。
- **SPP 路径**：`x_σ = m̃⊙((1−σ)x₀+σε) + (1−m̃)⊙((1−σ^γ)x_S+σ^γε)`，γ=1.5。

## 7. 文献

- Zheng et al., *PixPerfect: Seamless Latent Diffusion Local Editing with Discriminative Pixel-Space Refinement*, NeurIPS 2025（arXiv:2512.03247）
- Xu et al., *MaskFlow: Precise, Consistent and Seamless Regional Image Editing*, 2026（arXiv:2608.06929）
