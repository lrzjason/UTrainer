# 09 — Guidance-Distilled 模型的修正 Flow Matching 目标（guide_flow_matching）

> 状态：as-built（2026-09-21，含真实权重实测）。实现 DC-Gen（arXiv:2509.25180）
> 附录 A 的修正流匹配目标 ℒ_FM^guide（Eq. 10），作为 `UnifiedTrainer` 的可组合
> loss 模块交付；目标场景：对 **guidance-distilled（CFG 蒸馏）检查点**（如
> Qwen-Image 2.1 这类"cfg distilled model"）做 LoRA/LoKr 微调。
> 单元测试 19/19 + 训练器集成测试 14/14 + 真实 Qwen-Image 2.1 权重端到端
> （8 步冒烟、w* 估计、120 步对比实验）均通过，见 §7。

## 1. 为什么需要它（论文动机）

guidance-distilled 模型 v_η 的训练目标是模仿教师的 CFG 输出（Eq. 5）：

```
v^w_θ(z_t, c, t) = (1 + w)·v_θ(z_t, c, t) − w·v_θ(z_t, ĉ, t)        （Eq. 6，CFG）
```

对这类检查点直接套用**标准** flow matching 目标（Eq. 4，
`‖v_η(z_t, c, t, w) − v_t‖²`）是**有偏**的：v_η 的输出本身就是 CFG 合成
速度，不是原始速度 v_θ，拿它去匹配 flow target v_t = ε − x0 训错了对象。
DC-Gen Fig. 9 在 MJHQ-30K 512×512 上量化了差距：CLIP 26.87（ℒ_FM）
vs 27.38（ℒ_FM^guide），视觉质量同样明显更差。

## 2. 修正推导（附录 A，Eq. 7–10）

核心恒等式：空条件时 CFG 合成对任意 w 都塌缩回原始无条件速度——

```
v_η(z_t, ĉ, t, w) ≈ v_θ(z_t, ĉ, t)                                    （Eq. 7）
```

把蒸馏模型自身输出当作 CFG 输出并解出原始速度：

```
v_θ(z_t, c, t) ≈ [ v_η(z_t, c, t, w) + w·v_η(z_t, ĉ, t, w) ] / (1 + w)  （Eq. 9）
```

训练目标（Eq. 10）：

```
ℒ_FM^guide = E_{z0,ε,t,w} ‖ [ v_η(c) + w·v_η(ĉ) ] / (1 + w) − v_t ‖²
```

要点：
- **无条件前向用空 prompt 嵌入**（cached empty embedding），不需要教师模型。
- **梯度同时流过 cond 与 uncond 两个前向**——两者是同一个被训练网络的输出，
  断开任何一支都会改变目标本身。
- w 为 guidance scale，论文按 E 内每样本从 U[w_min, w_max] 采样。

## 3. 实现（本次交付）

| 文件 | 改动 |
|------|------|
| `UnifiedTrainer/losses/guide_flow_matching.py` | **新增**。`@LossRegistry.register("guide_flow_matching")`，实现 Eq. 10 |
| `UnifiedTrainer/losses/base.py` | `LossContext` 新增 `model_pred_uncond` 字段（默认 `None`，放在所有默认值字段区，dataclass 顺序合规） |
| `UnifiedTrainer/engine/trainer.py` | 训练步 + 验证 loss 路径的无条件前向管线（见 §4） |
| `UnifiedTrainer/utils/estimate_guidance_scale.py` | **新增**。w*（有效蒸馏标度）估计工具：扫描 w 找修正速度 MSE 最小值 |
| `UnifiedTrainer/train.py` | 启动前置检查：配置了 guide_flow_matching 但空嵌入缺失时快速失败 |
| `UnifiedTrainer/configs/qwen_image21_guide_fm_lokr.json` | 示例配置（qwen_image21 + LoKr + NF4，w 固定 1.5——实测值） |
| `run_qwen21_guide_fm.ps1` | 一键运行脚本 |
| `.tmp/test_guide_fm.py`、`.tmp/test_guide_fm_trainer.py` | 单元 + 集成冒烟测试 |

### loss 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `guidance_scale_min` / `guidance_scale_max` | 1.0 / 8.0 | w ~ U[min, max]，逐样本采样 |
| `fixed_guidance_scale` | `None` | 设置后 w 恒定（已知单一蒸馏标度时用；等价于退化区间） |
| `use_weighting` | `false` | SD3 式 σ 加权开关；论文 Eq. 10 无加权，默认关 |

配置示例：

```json
"losses": [
    {"type": "guide_flow_matching", "weight": 1.0, "params": {
        "guidance_scale_min": 1.0,
        "guidance_scale_max": 8.0
    }}
]
```

`velocity_sign` 分发与 `losses/flow_matching.py` 同源（standard / data_ward，
未知值直接报错）；telemetry 上报 `guide_flow_matching/guidance_scale_mean`。

### w 怎么选（实务建议）

修正项对 w 的偏差为 `(w − w*)/(1 + w)·v_θ(ĉ)`（w* = 教师蒸馏时的有效标度）：
- **先测 w\***（强烈建议）：`python -m UnifiedTrainer.utils.estimate_guidance_scale
  --config <配置>`——用未调优的 base model 在缓存真实数据上扫描 w，找修正速度
  MSE 的最小值（Qwen-Image 2.1 实测 **w*≈1.5**，见 §7）。
- 对**不接受 guidance 网络输入**的模型（Qwen-Image 2.1、Z-Image-Turbo），w* 是单一
  常数：设 `fixed_guidance_scale: <w*>`（或围绕它的窄区间）。**不要**用宽区间
  U[1,8]——实测宽区间（均值 4.5，远离 w*）会让每步目标带系统性漂移偏差，loss
  不降反升（§7 运行 B vs C）。
- 对**接受 guidance 网络输入**的检查点（FLUX-Krea 风格，adapter config
  `"guidance"`），论文的区间采样成立（模型本就训练成接受一段 w）；同时把该配置
  与修正用的 w 区间设成一致。
- w* 未知且无法实测时再退回论文做法的区间采样，区间应覆盖 w* 的可能取值。

## 4. 训练器管线（`needs_uncond_forward` 钩子）

loss 模块声明类属性 `needs_uncond_forward = True`，trainer 在训练步内、
主前向之后（同一 `accum_ctx` + `autocast_ctx` 内）执行：

1. `_build_uncond_batch(batch, device, dtype)`——浅拷贝 batch，uncond 条件由
   **两个独立部分**组成（krea2 式分离）：
   - **空文本嵌入**：`embeddings` 换成全局缓存 `empty_embedding.{model_suffix}.npz`
     （成熟方案，缓存构建时自动生成）；
   - **参考图**：`batch["latents"]` 原样保留（浅拷贝共享），edit 数据集的
     参考条件不丢。
2. `adapter.prepare_model_input(uncond_batch, noisy_latents, sigmas)` →
   同一 `transformer` 前向 → `unpack_prediction`。
3. 逐目标写入 `LossContext.model_pred_uncond`；数量与 cond 不一致立即报错。

**qwen_image21 的槽位合成**：该模型编辑路径的参考条件是"文本流里的 VLM 槽位
+ latent 行"二元结构，而 transformer 会把 latent 行**写入槽位位置并覆盖其
内容**（`transformer_qwenimage21.py:950`），所以槽位只是布局占位。uncond 批次
带 `_uncond_empty_text` 标记时，adapter 检测"嵌入无槽位但 latents 有参考"，
按 `参考 latent token 数 ÷ 4` **合成槽位布局**插入空文本流——插入点用空嵌入
缓存的 `user_opener_len`（encode_text 构建时记录的 ti2i 模板精确位置；旧缓存
无此标量则回退到流首，删除 `empty_embedding.*.npz` 重启即可增量重建刷新）。
条件路径被标记门控，行为零变化。

**caption dropout 捷径**：若该步 caption dropout 已触发（`_caption_dropped`，
条件前向本身就是无条件前向、嵌入完全相同），直接复用该预测，**不跑第二次
前向**——每步成本回到单前向。

验证 loss 路径（`validate_epoch`）同样支持：整段已在 `torch.no_grad()` 内，
第二次前向只花激活内存、不建图。

## 5. 成本与显存（实测）

- 每步 **两次带梯度前向**（cond + uncond）。RTX 5060 Ti 16GB + NF4 +
  gradient checkpointing 实测：**peak 5.59 GB**（与单前向基线 5.82GB 相当——
  检查点下额外激活很小，主导项是 NF4 权重与优化器状态），**~6 s/step**
  （单前向基线 3.8 s/step，约 1.6×）。
- 与 `caption_dropout > 0` 共存时，触发 dropout 的步自动退回单前向。

## 6. 适用范围与边界

- **适用**：对 guidance-distilled 检查点做微调（LoRA/LoKr/全量）；替代
  配置里的 `flow_matching`（两者不要同时用——目标不同，叠加无意义）。
- **不适用/注意**：
  - 非蒸馏的基础模型（plain CFG 训练范式）用标准 `flow_matching` 即可，
    本目标相对它更贵且无收益（w* 估计器可判定：最小值在 w=0 即非蒸馏）。
  - 多目标序列（qwen_image21 每序列恰一个目标）不受影响；无条件前向沿用
    同一 `noisy_latents` 列表，契约一致。
  - 接受 guidance 网络输入的检查点见 §3 的 w 一致性说明。
  - **edit（多参考图）数据集**：uncond 分支保留参考图（§4 槽位合成），
    与教师 CFG 负分支"空文本 + 相同参考图"一致。真权重端到端尚未在
    edit 数据集上跑过（布局已过真实 transformer 前向 + 严格元数据校验）。
  - `block_swap > 0` 与双前向的组合未经实测（冒烟/对比均用 block_swap=0）；
    第二次前向复用同一模块的前向钩子，预期兼容，但未验证。
  - 空嵌入 npz 缺失时：启动前置检查直接报错（train.py）；重新构建缓存
    （recreate_cache）或修正 text_encoder_path 即可。

## 7. 验证（本机已完成）

**单元测试 `.tmp/test_guide_fm.py`（19/19 PASS，CPU）**：
注册表自动导入命中；Eq. 10 公式逐元素对齐（含 weight、4D/5D 广播、
data_ward 符号分发、use_weighting）；**论文核心论断复现**——构造
v_η(c) = (1+w*)·target − w*·u、v_η(ĉ) = u 的合成蒸馏模型：标准 FM loss
恒有偏（非零），ℒ_FM^guide 在 w = w* 时精确归零、w ≠ w* 时有偏；梯度
同时到达 cond/uncond 两个预测；参数校验（w_min < 0、max < min）、缺失
uncond 时的明确报错、形状不匹配拒绝。

**集成测试 `.tmp/test_guide_fm_trainer.py`（14/14 PASS，CPU）**：
驱动真实 `Trainer.train_epoch`（stub adapter/transformer）——
每步恰好两次前向且第二次用空嵌入、同一 noisy latent；**训练器上报的 loss
与从记录的前向输入手工重算的 Eq. 10 完全一致**（2.008356 vs 2.008356）；
梯度经 hook 捕获确认到达参数；`caption_dropout=1.0` 时每步恰好一次前向；
`validate_epoch` 验证路径同样跑第二次前向且 val loss 有限。

**edit-uncond 槽位合成测试（2026-09-21）**：
`.tmp/test_edit_uncond.py`（28/28 PASS，CPU）——真实 qwen21 adapter 的
`prepare_model_input`：标记 + 空嵌入 + 参考 latent → 参考保留、槽位布局合成，
全部布局不变量成立（img_mask 长度对齐、槽位×4 == latent 行数、块顺序
[参考…，目标]、`build_token_metadata` 严格校验通过）；无标记 → 参考丢弃
（旧行为回归护栏）；opener 精确落位 vs 流首回退；T2I（无参考）零变化；
`_build_uncond_batch` 标记/元数据透传 + 旧缓存回退。
`.tmp/test_edit_uncond_real.py`——**真实 NF4 transformer 前向接受合成布局**
（hidden_states 512 行 = 参考 256 + 目标 256，EHS 76 = 文本 12 + 合成槽位 64，
解包 (1,64,16,16) 有限值，peak 3.7 GB）。空嵌入已刷新（`user_opener_len=3`，
与提取行 9 token 自洽，旧文件 .bak 备份）。

**真实权重端到端（RTX 5060 Ti 16GB，Qwen-Image 2.1 7.1B NF4 + LoKr，
2026-09-21）**：

1. **冒烟（8 步）**：训练 8/8 步无 OOM；loss 0.25–0.38（量级合理）；
   `guidance_scale_mean` 遥测 1.16–7.98 随 U[1,8] 采样变化；LoKr 检查点
   正常保存（224 modules）；`validate_epoch` 验证路径跑通
   （val_loss=0.284235，breakdown `guide_flow_matching=0.284235`）；
   peak VRAM 5.59 GB。既有警告（ComfyUI 副本设备不一致、minimax_h3 适配器
   导入失败）在 flow_matching 基线上同样出现，与本次改动无关。
2. **w\* 估计**（`UnifiedTrainer/utils/estimate_guidance_scale.py`）：
   未调优 base model 在缓存真实样本上扫描 w——**argmin w\* = 1.5**，
   MSE 0.295 vs 标准 FM 0.312（**低 5.5%**），曲线在 w>1.5 后单调上升。
   即 Qwen-Image 2.1 带有弱蒸馏签名、有效标度约 1.5；标准 FM 的偏差在
   真实模型上可测量（论文 Fig. 9 论断的本机复现）。
3. **120 步对比实验**（同图、同 lr 1e-4、caption_dropout=0、
   use_weighting=false，唯一变量是目标函数；单图高方差，看趋势）：

   | 运行 | 目标 | 前10步均值 | 后30步均值 | epoch 均值 0→1 | 最低 |
   |------|------|-----------|-----------|----------------|------|
   | A | flow_matching | 0.3021 | 0.2729 | 0.287→0.267 ↓ | 0.2255 |
   | B | guide, w~U[1,8] | 0.2618 | 0.3019 | 0.281→0.300 ↑ | 0.2189 |
   | C | guide, w=1.5 固定 | 0.2858 | **0.2625** | 0.273→0.269 ↓ | **0.2204** |

   读法：(a) B 起步即低于 A（0.26 vs 0.30）——修正降低了初始残差，与论文
   一致；(b) **w 匹配 w\* 的 C 稳定下降且终态最优**；w 远高于 w\* 的 B 目标
   带漂移偏差、loss 不降反升——w 的选择比"用不用修正"更关键；(c) A 与 C
   终态差距（0.273 vs 0.263）在单步噪声范围内，**方向性一致但非决定性**，
   更大规模/多图/视觉评估待用户实测。
