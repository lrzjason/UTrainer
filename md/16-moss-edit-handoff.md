# MOSS 编辑训练交接（2026-10-23）

> 给接手的 agent：这份文件是截至 2026-10-23 20:30 的完整状态。先读「结论」和「待办」，
> 细节按需查。**核心结论：问题不在模型/损失/采样，而在编辑配对数据本身不满足“局部编辑”。**

---

## 1. 结论先行

| 结论 | 证据 |
|---|---|
| 参考图通路（`r=1` 帧 + t=0 调制）**是通的** | 恒等任务 R=T，20 图 / 900 步，val PSNR 17.7 → **28.2 dB** |
| 编辑任务**只学到一半**（编辑区完成度饱和在 0.5） | `progress` 0.47（纯描述）→ 0.52（指令文本）；epoch 5→9 不再增长 |
| 根因是**配对数据不是局部编辑** | 325 组里只有 **43 组（13.2%）** 的改动落在标注框内；**272 组（83.7%）** 目标图是整图重绘，框外大片真实改变 |
| 由此导致的症状 | 训练时 `locality` 只有 0.30–0.44（到处改）；模型学不到“保持不动”，只能给折中 → progress 卡 0.5 |
| val loss 不能当判据 | 同一 run 里 val_loss 0.39→0.43（几乎不动），PSNR 却 17.7→28.2 dB（大幅提升） |

---

## 2. 数据集审计（最重要，接手请先看这个）

### 2.1 分类规则（脚本 `E:\UnifiedTrainer\.tmp\audit_full.py`）

对每组取 `|T − R| > 12` 的差异掩码，与标注框（`_T.json.backup` 里 `operation ∈ {add,change,delete,move}` 的 bbox，外扩 8px）比较：

- `clean`：差异基本在框内（框外最大连通块 < 400 px 或 框外均值差 ≤ 18）
- `pair_drift`：框外存在 >400 px 的连通块，且框内覆盖率 < 0.6 或框外均值差 > 18
- `huge_edit`：>40% 画面改变
- `bbox_misplaced`：框内几乎没变、框外有大块（本数据 **0 组**，说明标注框位置是对的）

### 2.2 结果（325 组）

```
clean         43  (13.2%)
pair_drift   272  (83.7%)
huge_edit     10  ( 3.1%)
```

| 类别 | 总数 | clean% |
|---|---|---|
| ad-creative | 38 | 0% |
| character | 9 | 0% |
| ui | 43 | 2% |
| poster | 74 | 5% |
| ecommerce | 16 | 6% |
| comparison | 22 | 18% |
| best_quality | 8 | 25% |
| portrait | 115 | 27% |

分布：框外平均差异 median **20.8 px**（p90 26.3，远超压缩噪声 <8）；框外最大块 median **2494 px**、p90 **25830 px**。
明细 CSV：`E:\images\MOSS_edit_filter\_audit_full.csv`

### 2.3 直观例子（可视化脚本 `E:\UnifiedTrainer\.tmp\vis3.py`，输出 `.tmp\audit_vis\`）

- **DRIFT**：`ad-creative/change_1b769b1e_0001`（BOLT→VOLT）
  意图只改顶部 wordmark，标注框也只圈了 VOLT（**框是对的**），但 T 是整张重绘：
  左下 BOLT 徽标消失、耳机角度变、人物细节与光斑全变 → 差异热图覆盖全图。
- **CLEAN**：`best_quality/change_7e7972b6_0001`（额头帽网）差异只有一条带。

结论：这批 R/T 更像是「同一 prompt 的两次采样/中等 denoise 重绘」，而不是「局部编辑后的结果」。

---

## 3. 已完成的工程（不要重做）

### 3.1 模型适配（`UnifiedTrainer/models/moss/`）

- `__init__.py`：`MossAdapter`（`@ModelRegistry.register("moss")`），latent 128ch（32 VAE × 2×2 patchify）、`vae_scale_factor 8`、`bucket_divisibility 16`。
- `forward.py`：vendored 官方 transformer 的 forward 改写 —— `_lin`（量化安全）、`grid_rope`（4 轴 RoPE）、`_block_forward`、`_conditioning`（fp64 正弦）、`training_forward(...)`，支持 `[target | text | refs]` 打包序列；参考帧 `r=1..N`、t=0 调制、整图铺满画布。
- `pack.py`：`prepare_model_input` 打包 token；`sample_timesteps`（MOSS pmf，`flow_shift`/`dynamic_shift`、新增 `timestep_sigma_max` 硬上限）。
- `encoders.py`：Qwen3.5-9B 文本/视觉编码（transformers 5.x 需传 `mm_token_type_ids`）。
- LoKR full-rank 可用；保存时自动同时写 ComfyUI 版（`_comfyui.safetensors`，纯前缀改名，`networks/lokr_convert.py::convert_moss_state_dict`）。

### 3.2 本轮新增/修复

| 项 | 位置 | 说明 |
|---|---|---|
| **fp8 文本编码器** | `train.py::_quantize_text_encoder`（`torchao_float8`/`fp8`/`float8`） | 16GB 卡可用：bf16 16.8GB → fp8 8.4GB；本地缓存编码 131 s/图 → **<1 s/图** |
| **掩码/损失网格不匹配修复** | `losses/base.py::region_weight_map` | 掩码缓存在 VAE latent（148×110），损失网格是 patchify 后（74×55）→ 自动 `avg_pool2d` 整数倍下采样 |
| **编辑评估脚本** | `UnifiedTrainer/scripts/eval_edit.py` | 输出 `progress / PSNR keep / PSNR tgt / PSNR ref / locality`，替代 val loss |
| **数据格式 v4** | `.tmp\reprocess_v4.py` | 生成 `_T.json`（纯描述 items）、`_T.prompt.txt`（官方描述）、`_T.edit.txt`（指令）、`_T.edit.min.txt`（指令+仅被编辑项） |

### 3.3 ComfyUI 侧（`E:\ComfyUI-EditUtils`）

插件已实现与本项目一致的 r 轴方案（`MossImageEditApply_EditUtils`）。**注意**：
- 工作流必须接上 `MossImageEditApply_EditUtils`（只接 `model` 一根线）；
- LoRA 未用 kv_cache 训练 → 必须 `ref_kv_cache = false`；
- 底模要用 `MOSS-Image-Base` bf16（不要 int8 SFT + MergeLoraToModel）；
- 采样对齐：20 步、CFG 3.0、MOSS scheduler、无 SLG、无负向 prompt。

---

## 4. 实验记录（可对比的基线）

评估协议固定：`val_split_ratio 0.03`、`seed 42`、20 步、CFG 3.0、各样本原生 bucket → 不同 run 的验证图**可比**。

| run | 配置要点 | 结果 |
|---|---|---|
| `moss_identity20`（本地 16GB） | R=T，20 图 ×repeat5，rank16 | train loss 0.464→0.053；**val PSNR 17.7→28.2 dB** |
| `moss_identity_full`（4090D 24G） | R=T 全量，full rank | epoch 0 起 loss ~0.07；产出 epoch0/1 checkpoint |
| `moss_edit_r2t_full`（5090 32G） | 参考=完整 `_R`，纯描述 prompt，flow_shift 1.0 | **progress 0.47**，locality 0.30，val loss ~0.25 |
| `moss_edit_r2t_instr`（5090 32G） | + 指令文本 `_T.edit.txt`，flow_shift 0.5？→ 实为 1.0 | **progress 0.52**（epoch5 0.45 → epoch9 0.52），locality 0.44 |
| `moss_edit_r2t_min`（4090D 24G） | `_T.edit.min.txt` + `masked_flow_matching`(edit 10/base 1) + `flow_shift 0.35` + 从 base 直训（无 resume） | epoch 0：`avg_loss 0.465`，val `progress 0.10 / PSNR keep 24.5 / locality 0.30`；**该机已关闭，训练中断**（后续 epoch 可能已在共享盘） |

checkpoint 位置：`/home/waas/MOSS_output/<run>/`（含 `-<epoch>/` 验证图与 `*_epochN.safetensors`）。

### 已验证“无效/有副作用”的尝试（别再重复）

| 尝试 | 结果 |
|---|---|
| 灰色挖洞参考图（`_m1`） | 模型无法理解灰区该放什么，比完整参考更差 |
| 恒等预训练后再训编辑 | 学到“有参考就抄”，成为负资产（`moss_edit_r2t_min` 已改为从 base 直训） |
| 加长训练 | progress 0.45→0.52 后饱和，加 epoch 无效 |
| 调整 token 顺序（`[x, refs, text]`） | 数学上无影响：MOSS 注意力无 mask + RoPE 按 token 坐标，与序列序无关 |
| 纯 img2img 起点 | 只是绕开问题，不改变模型能力（未采纳） |

---

## 5. 待办（推荐顺序）

> **状态更新（2026-10-23 晚）：P0 方案 A 与 P1 已在本地完成**，见 §5.4。

### P0：修数据（当前瓶颈）

**方案 A（推荐）：合成受约束配对**

对每组：
```
T' = R 在标注框外
T' = T 在标注框内（边界羽化 8–16 px）
```
- 保证「T' 与 R 的差异 == 标注框」→ 监督与标注逐像素一致；
- 保留 T 框内的编辑内容；原 `_T.webp` 不动，另存 `_T.Composed.webp` 便于回滚对比；
- 需同步重算：`_k1` 掩码（=标注框）、`_T.json`/`_T.prompt.txt`/`_T.edit.min.txt`（items 与指令不变，只需确保 bbox 与合成区一致）。

**方案 B：用真实差异重算 bbox**（配合 A）：把标注框替换为差异区域外接框，使 where 与目标同源。

**方案 C：只保留 43 组 clean** 作干净核心集/评测集（不足以训练，但必须是评测集的一部分）。

**方案 D：用 inpainting 管线重生成 T**（质量最高，代价最大）。

### P1：保持干净评测集

把 43 组 clean 固化为 `val_clean` 列表，之后所有 run 都在同一批图上报 `eval_edit` 三指标，避免合成数据污染评估。

### P2：训练对比实验

数据修好后，用现成配置改两处即可（`cap ext` 指向新版文本、`image_configs` 指向 `_T.Composed`），跑 2 个 arm 各 2 epochs：

- Arm A：合成配对 + 从 base 直训
- Arm B：合成配对 + 恒等初权（验证“恒等是否为负资产”）

判据：`progress ≥ 0.8`、`PSNR keep ≥ 30 dB`、`locality ≥ 0.8`。

### 5.4 P0/P1 完成记录（2026-10-23 晚，本地）

**两步完成：①合成受约束配对（方案 A）②标注重写（v5，描述一律描述编辑后的目标）。**

**第一步：合成**（脚本 `.tmp\compose_v5.py`，幂等）。对全部 325 组执行方案 A：
`T' = R（标注框+8px 外）+ T（框内，框内 12px 线性羽化，小框自动缩短羽化）`。

**第二步：标注重写 v5**（脚本 `.tmp\reprocess_v5.py`，幂等，从 `_T.json.backup` 重建）。
用户定则：**描述必须描述「编辑成什么样」（学习目标 T 的内容），不得描述 R 或编辑操作**。
v4 的三处违例已全部改掉：

| v4（违例） | v5（现文） |
|---|---|
| 编辑项 caption = "change the giant white geometric **BOLT** wordmark ... to VOLT"（含 R 旧内容） | "the giant white geometric **VOLT** wordmark across the top of the frame in the same heavy sans-serif letterforms and size"（新内容） |
| keep 项 = "X, kept unchanged from the original."（描述与 R 的关系） | "The X."（纯目标内容） |
| `_T.json`/`.prompt.txt` summary = "Replace ... with ... keeping ... unchanged"（操作句） | 结果描述句（"A cream cable-knit sweater with a wide boat neck." / "...， there is a small ivory clutch bag..." / "...， without the steam swirls..."） |

- 指令文本（`.edit.txt`/`.edit.min.txt` 的 "replace the content of region X with PHRASE"）格式不变，
  但内嵌 PHRASE 一律为编辑后描述；`string` 字段（新文字）74 组全部核实无误；
- 重写规则机械转换（replace-with/change-to/into/as/using、add→there is、delete→without、
  CJK `from 旧 to 新` 的旧词替换等），325/325 flags=0；
- 残留扫描：325 份 `_T.json` + 325 份 `_T.prompt.txt` 中 `kept unchanged/keeping/preserving`
  等字样 **0 处**；
- `_T.json.backup` 是唯一真源，**从未改动**，v4/v5 均可随时重建；
- 产物：重写后的 `_T.json`、`_T.prompt.txt`、`_T.edit.txt`、`_T.edit.min.txt`
  + 同内容 `_T.Composed.*` 副本（0 处不同步）；
- 统计/flags 存 `_reprocess_v5_stats.json`；文本侧问题诊断脚本 `.tmp\audit_anno_mismatch.py`。

**合成产物**（第一步）：`<stem>_T.Composed.webp`（质量阶梯 95→lossless，以重解码后框外
无 >12 差异为准；172 组 q95 / 28 q98-100 / 125 无损）、`val_clean.json`（43 组 clean 评测集，
**评测对原始 `_T.webp`**）、`_compose_manifest.csv`、`.tmp\compose_vis\*.jpg`。
验证：325/325 框外差异=0（40 组独立复检）；`_k1` 与合成差异区逐组对齐（k1 未改，本来就
=标注框+8px）；300 个 add/change bbox 与 v4 json items IoU 全匹配；编辑内容保留全部 ≥0.3。

P2 配置改法：`image_configs` 的 `T` suffix 改 `"_T.Composed"`，`K1` 仍 `"_k1"`，
`caption ext` 不变（`.edit.min.txt` 会命中 Composed 副本，内容与 _T 版相同）。

---

## 6. 环境与路径

### 本地

| 内容 | 路径 |
|---|---|
| 代码 | `E:\UnifiedTrainer\UTrainer\UnifiedTrainer\` |
| 配置 | `...\configs\moss_edit_r2t_{full,instr,min}.json`、`moss_identity20.json`、`moss_identity_full.json` |
| 数据 | `E:\images\MOSS_edit_filter\<8 类>\<stem>_{T,R,Diff}.webp`、`_T.json(.backup)`、`_T.prompt.txt`、`_T.edit.txt`、`_T.edit.min.txt`、`_k1..k3.png`、`_m1..m3.webp` |
| 权重 | `C:\hf_models\MOSS-Image-Base`、`E:\hf_models\Qwen3.5-9B`（processor） |
| 生成脚本 | `.tmp\reprocess_v4.py`、`.tmp\audit_full.py`、`.tmp\eval_instr_run.py`、`.tmp\launch_edit_min.py` |
| 审计产物 | `E:\images\MOSS_edit_filter\_audit_full.csv`、`.tmp\audit_vis\*.jpg` |

### 远端（共享盘 `/home/waas`，CephFS 42T）

| 内容 | 路径 |
|---|---|
| 权重 | `/home/waas/hf_models/MOSS-Image-Base`（27G）、`/home/waas/hf_models/Qwen3.5-9B` |
| 数据 | `/home/waas/MOSS_Training/datasets/MOSS_edit_filter/<8 类>/` |
| 配置 | `/home/waas/MOSS_Training/configs/` |
| 输出 | `/home/waas/MOSS_output/<run>/` |
| 代码 | `/root/UTrainer`（**不共享**，换机必须重新同步） |

### 启动模板（`/root/UTrainer/start_<run>.sh`）

```bash
#!/usr/bin/env bash
cd /root/UTrainer
PY=/root/miniconda3/bin/python
NVIDIA_LIBS="$($PY -c 'import nvidia.cuda_nvrtc,os,glob; root=os.path.dirname(os.path.dirname(nvidia.cuda_nvrtc.__file__)); print(":".join(sorted(glob.glob(os.path.join(root,"*","lib")))))' 2>/dev/null)"
[ -n "$NVIDIA_LIBS" ] && export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
export WANDB_BASE_URL=https://api.bandw.top/
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
exec $PY -u UnifiedTrainer/train.py --model moss \
  --config UnifiedTrainer/configs/<run>.json
# 续训只加：--resume /path/to/xxx.safetensors   （LoRA 权重，full resume=false）
```

启动用 `setsid nohup ./start_x.sh > logs/x.out 2>&1 < /dev/null &`（**不要读那个 channel**，会卡）。
杀进程用 `pkill -f 'train.py --model moss'`（不要用含 `UnifiedTrainer.train` 的模式，会杀自己的 shell）。

### 已知机器

| 机器 | 地址 | 状态 |
|---|---|---|
| 4090D 24G（跑过 `moss_edit_r2t_min`） | `e98ca191b2d841b8bc05b7bf9aa40bc2.region1.waas.aigate.cc:41446` | **已关机（端口不通）** |
| 5090 32G（跑过 `_full`/`_instr`） | `2f99feaa18b042eb882b2b1ac1993dc9.region1.waas.aigate.cc:43597` | 已关 |
| 无卡（看数据/审计） | `e8365ce4b5e845b88293aebb5e193e95.region1.waas.aigate.cc:49097` | 最近给的，密码见会话 |
| 无卡（旧） | `70b5e489d1ad42caa439ce5d1ef33551.region1.waas.aigate.cc:43364`、`d613408176e643cd87cdf310fa3850b2.region1.waas.aigate.cc:45378` | 已关 |

> 密码请从会话记录中取，**不要写进 repo**。本地同步脚本：`.tmp\launch_edit_min.py`（含 302 文件 hash 校验 + 启动 + 验证）。

---

## 7. 注意事项（踩过的坑）

1. **`--resume` 是 LoRA-only**（日志会打印 `full=False`）；`--resume-full` 才恢复优化器/step。用户要求：**full resume = false**。
2. **不要用 nf4**（用户明确要求）；远端 bf16，本地可用 fp8 TE。
3. **fp8 TE 必须在 CPU 上量化**再 `.to(cuda)`（`_quantize_text_encoder` 在加载后、上卡前调用）。
4. **ComfyUI 生成的 LoRA 必须 `ref_kv_cache=false`**，且工作流要有 `MossImageEditApply` 节点，否则参考帧根本没进序列（已验证：缺该节点时跑的是纯 T2I）。
5. **val loss 不可比**：它随 `flow_shift`、`edit_weight` 变；要横比请用 `validation.loss_types: ["flow_matching"]` + `loss_params: {use_weighting: false}`。
6. 掩码缓存是 VAE latent 网格，patchified 模型需要 2× 下采样（已修）。
7. wandb 用 `WANDB_BASE_URL=https://api.bandw.top/`，凭据从 `C:\Users\lrzja\_netrc` 读；`save_code=False`。
8. 数据里 `_T.json.backup` 是**唯一真源**（含 `operation/color/ref_index/z_index/string`），`_T.json` 是可重建的派生文件。
