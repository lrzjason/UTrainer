# 10. 透明像素编辑模型训练：想法与实践（细化版）

> 基线模型：Qwen-Image 2.1（64ch latent，原生 4 通道 RGBA 因果 VAE，16× 空间压缩）
> 微调方式：LoKr full-rank（factor 4），lr 1e-4 常数，adamw8bit + torchao_float8
> 数据形态：`_T`（RGBA 透明目标图层）+ `_R`（RGB 参考图）+ caption
> 相关文档：`tmp/mask_training_improvement.md`（composite loss 设计）、
> `md/08-qwenimage21-training.md` §5.1（RGBA 数据链）、`md/09-corrected-flow-matching-guide.md`

---

## 1. 任务定义

给定参考图 `_R` 与文本指令，模型直接输出 **RGBA 透明编辑图层** `_T`
（透明区 alpha=0、RGB=纯黑的数据约定），下游与 `_R` 合成得到最终编辑结果：

```
O = α·T_rgb + (1−α)·R
```

qwen21 的 VAE 原生 4 通道（`vae_pixel_channels=4`），alpha 逐像素进 latent，
**不存在"RGB latent 表示不了透明"的表示瓶颈**——这是与 RGB-only 模型
（qwen_image_edit / flux）的本质区别，也是"直接输出透明图层"路线成立的前提。

---

## 2. 核心观察（已有实验事实）

### 2.1 数据集维度：混训失败，单练可行

| 组合 | 结果 |
|---|---|
| 颜色 + 形状混训（sample_weight 1.0 样本级混合） | **都不好** |
| 颜色训练集单练 | 还行 |
| 形状训练集单练 | 还行，**可收敛**（编辑区域从一大片逐步收缩到目标小区域） |

结论：两个任务各自**可学**，问题出在任务间干扰，而非任务本身难度。

### 2.2 Loss 维度：除 guide_flow_matching 外，辅助 loss 基本全负收益

历史复盘（`tmp/mask_training_improvement.md` §2，2026-10-10 实测）记录了
composite loss 的失败细节：

| 当时症状 | 根因 | 修复 |
|---|---|---|
| 全局模糊（laplacian_var 42.8 vs 基线 82.8） | `decode_scale=0.5` 半分辨率像素监督只约束低频 | 移除，全分辨率解码 |
| 模糊加重 | σ 门控 0.35 太宽，高噪 x0̂ 本是模糊条件均值 | 收紧 σ_max=0.2 |
| 半透明倾向 | matte 参考用了羽化后的 alpha | matte/dice 改用 RAW alpha |
| 边缘硬悬崖/接缝 | 边缘满权重像素锤击 | 羽化目标 + 过渡带降权 |

**原则：不要一次把所有 loss 都用上；每个辅助 loss 单独过 eval 门槛。**

### 2.3 基模型能力边界

- 基模型本身具备**一定程度**的编辑能力（全图编辑路径可行）；
- 症状 a：**编辑部分区域会连带其他区域**（区域外泄漏）；
- 症状 b：**编辑响应不准确**（位置/范围不对）；
- **"只输出透明图层"做不到**：训练输出透明时容易**漏编辑区域**——
  直接抄输入是 FM loss 的安全局部最优（copy-shortcut）。

### 2.4 形状训练的正向经验

- 自由形状 mask 可收敛（大区域 → 小区域），where-to-edit 是可学的；
- **训练集需要保留一定边缘**，才方便做边缘羽化监督——与
  `composite_consistency` 的 `feather_sigma_px=1.0` + `edge_band_drop=0.5`
  设计互为印证（羽化让边缘目标可达，过渡带降权避免硬锤出接缝）。

---

## 3. 关键设计决策

### 3.1 输出形式：透明图层 vs 完整图片 vs 固定方框

| 方案 | 评估 | 结论 |
|---|---|---|
| **输出 RGBA 透明图层** | VAE 原生支持；与 `_T/_R` 管线一致；下游合成灵活 | **采用（主线）** |
| 输出完整图片 | 基模型能做到，但放弃透明通道的原生优势，且 copy-shortcut 更严重 | 不采用 |
| 固定方框训练 | 可消歧 where-to-edit，但有几何过拟合风险，自由 mask 已被证明可收敛 | 仅作 curriculum 备选，优先级最低 |

### 3.2 数据规范

- 透明区 RGB = **纯黑**（已本地实测确认的必要约定，否则 fg 项与 FM 互相拉扯）；
- alpha 边缘保留 1–2px 过渡，GT 侧再做羽化；
- **两个数据集的 caption 必须具备任务可区分性**（见 §5 假设 2）。

---

## 4. Loss 分层设计

```
L0  主损失：guide_flow_matching（4ch latent FM，w=1.0，fixed_guidance_scale=1.5）
    ── 对应"带透明的训练目标和模型预测做 flow matching loss"

L1  alpha 覆盖惩罚（自研形式，w=0.03 起步）
    语义：底板 B = R − T（编辑区抠空为透明）；X = B + T̂ 必须完全不透明。
    X 残留的透明像素 = 预测未覆盖目标的像素 → 惩罚。
    公式（字面构造 + margin-hinge + 面积归一）：
        B_alpha = 1 − α_T                      （底板：编辑区抠空）
        X_alpha = B_alpha + α̂ = 1 − α_T + α̂    （底板 + 预测）
        short   = relu(1 − X_alpha) = relu(α_T − α̂)   ← 单侧，只罚欠覆盖
        L       = Σ relu(short − m) / Σ α_T             m≈0.05~0.1
    注：以上是"原图−目标+预测"按字面像素算术的精确结果（alpha 相加后裁剪）；
    与严格 over 合成（α_T(1−α̂)，cover_form='residual'）在二值区一致，
    软边缘上后者要求 α̂→1 会把 matte 往外推，故不作默认。
    特性：
    ── 单侧惩罚 = 正对"漏编辑区域"的失败方向；过覆盖/连带留给 FM，
       必要时再加对称项（一次一个）
    ── 零 RGB 监督 → 不继承 composite 的模糊失败模式；
       FM 管"画什么"，本项管"必须画到"
    ── **不羽化**：α_T 直接用数据集原始 alpha（本就保留 1–2px 边缘），
       匹配边缘（α̂=α_T）即得 0，目标可达；羽化只会重新教出半透明
    ── cover_form='residual'（α_T(1−α̂)，严格 over 合成残差）仅作对照：
       软边缘上要求 α̂→1，会把 matte 往外推（bloat），故不作默认
    ── margin 让达标区域精确为零（约束而非拉力）；
       面积归一让小目标不被稀释（MaskFlow MFL 精神）
    ── **σ 门控要比 RGB 监督的 0.2 宽**：实测（30 组子集，第 1 次启动）
       sigma_max=0.2 时 cover 项 18 步里只生效 1 步（6%），hinge≈0.02 ×
       weight 0.03 = 0.0006，对 guide_fm 的 0.009~0.30 只有 ~3% 量级
       → 约占总梯度预算 0.2%，必然假阴性。
       0.2 是"高噪声下 RGB 像素监督导致模糊"定下的规则；cover 只碰 alpha，
       频率低得多，不受此限 → 实测配置改用 sigma_max 0.5 + weight 0.1；
       若仍无效果，先抬 sigma_max 再抬 weight。
    实现：复用 composite_consistency 骨架（VAE decode/σ门控/遥测），
       data_weight=0 关掉 comp 项 + cover_weight/cover_margin/cover_alpha/
       cover_form 参数；遥测 cover_viol（无 margin、面积归一）在 weight=0
       时也上报 → 先跑基线再开权重
    配置：configs/qwen21_transparent_edit_cover.json

L2  区域加权（**已实现，当前唯一激活路线**；零新 loss、零解码）
    ── mask 来源：目标自身的 alpha 通道。`_T` 是 RGBA，而 CacheBuilder._encode_mask
       是 alpha-first（注释里就写着 "e.g. the _T target itself"），用与目标**同一个**
       bucket 裁剪 → area 下采样到 latent 网格（16×）→ 0.5 二值化 → (1,h,w) 0/1
    ── 权重：w = base_weight + edit_weight·mask（示例 1 区外 / 4 区内），
       按 sum(w) 归一 → loss 数值与 plain mean 同尺度（in-region 梯度 ≈3.5× @4% 覆盖）
    ── 代价：零——不解码、不新增损失项；加 mask_configs 后缓存**自动回填** mask 层
    ── val 对齐：validation.loss_types + loss_params 会构造**全新的未加权实例**
       （trainer._resolve_val_loss_modules：loss_params 存在时不再复用训练实例，
        否则 region 权重会污染 val_loss，无法与基线横比）
    ── 判据：loss/guide_flow_matching/mask_active（须 1.0）、mask_region_frac，
       以及**采样图**上的 precision/recall/Dice——不能用 x0̂ 上的 iou_50（会给 0.888 的假象）
    ── 配置：configs/qwen21_mask_test_region.json（由 qwen21_mask_test_fm.json 生成，
       差异仅 8 处：mask 层 ×3、权重 ×2、val_params、save/run 名、注释）

【实证约束，2026-10 实验结论】现有辅助 loss（composite_consistency 全家族）
相对纯 guide_flow_matching 基本负收益。系统性原因：
    ① 像素域 loss 需每步解码 → σ 门控 → 生效步数少且与 FM 梯度对抗
    ② 像素域全图监督压高频 → 模糊（laplacian_var 42.8 vs 82.8 实测）
    ③ 多子项权重在 338 对小数据上互相干扰
→ 新 loss 的准入门槛：单项、可每步运行、梯度方向与 FM 一致；
   优先考虑"折叠进 FM 权重场"而非新增独立 loss（路线 L2）。
```

**缓存边界（"原图−目标"能预缓存吗）**

| 对象 | 是否已缓存 | 位置 / 成本 |
|---|---|---|
| α_T（cover 项唯一需要的目标量） | **已缓存** | `{T}_comp_px.npz` 的 `Ar`（raw alpha，bucket 分辨率 fp16，单通道 1.57 MB @768×1024） |
| 原始 alpha 的羽化版 A / 合成 GT / 预乘 Fg | 已缓存 | 同 npz；cover 默认不羽化 → 用 `Ar` |
| x_T、x_R latent | 已缓存 | latent 缓存（cache index `references.latent_path`），可直接做区域加权 |
| 底板 B = R − T（RGB） | 未缓存，**且不需要** | cover 项不含 RGB 项；补上要多 3 通道（+35% 体积） |
| latent 网格的 α_T 权重图 | 未缓存 | 单通道 fp16，几 KB/样本 → 路线 L2 的原料 |

关键结论：**缓存能省的是目标侧（已省掉），省不了预测侧**。每步真正的开销是
解码 x0̂（64ch → 1024² RGBA，正反向都算），预测每步都变，无法缓存。
现有 `_comp_px.npz` 已是 17–23 MB/样本（11 通道），560 样本约 10–13 GB；
要加就加单通道小对象，别加 RGB 底板。若补 latent 网格权重图，
**另存小 npz**（`{T}_alpha_lat.npz`）而不是改 `_comp_px.npz` 的 key ——
后者要抬 `_PX_CACHE_VERSION`，会触发整个 10 GB 像素缓存重建。

**引入纪律**（从历史失败中学到的）：

1. σ 门控必须有——高 σ 的 x0̂ 上像素监督 = 教模型糊；
2. 辅助 loss 权重一律 0.03–0.1 起步，主 FM 占绝对主导；
3. 遥测先行：`mae_255` / `iou_50` / `alpha_mean` 先当指标跑通，再决定权重；
4. 一次只加一个 loss，过 eval 门槛才保留。

---

## 5. 混训失败的假设与对照实验（当前主攻方向）

单练都行、混训全崩，按嫌疑排序：

| # | 假设 | 机理 | 对照实验 |
|---|---|---|---|
| 1 | **alpha 先验冲突** | 形状任务教"大面积 alpha→0"，颜色任务教"alpha 不动只改 RGB"；相似输入要求相反 alpha 行为，模型学折中 → 半透明/漏编辑 | 给形状任务 caption 加固定任务词，其他不动 |
| 2 | caption 区分度不足 | 同一条件分布上学两个互斥答案 | 审计两数据集 caption 模板；强制任务标记 |
| 3 | 数据约定不一致 | 透明区 RGB / 羽化宽度不一致，fg 与 FM 拉扯 | 脚本审计两数据集透明区 RGB 均值、边缘宽度分布 |
| 4 | 混合方式 | 样本级混合梯度冲突最强 | batch 级交替；颜色集 sample_weight 降权扫描 |

---

## 6. 实验顺序（基线先行）

1. **基线遥测（无训练）**：`composite_consistency` weight=0 跑现有 checkpoint，
   记录 `mae_255 / iou_50 / alpha_mean / laplacian_var` 基线数字——
   没有基线，后续所有"有没有提升"都是体感。
2. **小训练集过拟合**：20 条数据纯 FM，验证能否精确背下（含透明区域）。
   背不下 → 管线/数据约定有问题，先修管线，不碰 loss。
3. **单任务纯 FM 复现**：颜色、形状各自复现"还行"的水平（已有，作为对照组）。
4. **混训消融**：按 §5 顺序逐个证伪（caption 任务标记 → 数据审计 → 混合方式）。
5. **alpha 通道探针（零训练）**：Δα-only 图对过 encoder，看 alpha 是否集中在
   少数 latent 通道——可分离则走 L2 通道加权 FM；纠缠则退回 L1 覆盖惩罚
   （cover only，σ_max 0.2，w=0.03，先 weight=0 拿 cover_viol 基线）。
6. **区域加权**：loss_mask=|Δα| / area_normalize，治连带与漏编辑。

---

## 7. 指标体系

| 指标 | 来源 | 治什么 |
|---|---|---|
| `mae_255` | composite_consistency.last_components | 合成残差（整体一致性） |
| `iou_50` | 同上 | matte 形状准确度 |
| `alpha_mean` | 同上 | 半透明倾向监控 |
| `matte_mae` / `fg_mae_255` | 同上 | alpha 泄漏 / 透明区 RGB 残留 |
| `laplacian_var` | 验证图统计 | 模糊回归检测（历史教训） |
| 区域外泄漏幅度 | 输出 vs `_R` 在 mask 外的差分 | 连带问题 |
| 人工 / VLM 判定成功率 | 固定 eval 集（50–100 条） | 终检 |

---

## 8. 待确认问题

1. 混训失败的具体症状是什么——半透明、漏编辑、还是偏色加重？
   （决定先验证假设 1 还是假设 2/3）
2. 颜色与形状两数据集的 caption 模板现在有明显区分吗？
3. 下游推理真正需要的是 RGBA 图层，还是"全图 + mask"也可接受？
   （影响是否需要在合成一致性上继续加码）
