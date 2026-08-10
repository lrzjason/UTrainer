# MiniMax-H3 训练接入 UTrainer — 实施方案（图像对优先 → 视频，本期无音频）

> 状态：**实施完成（代码）/ 运行验收延后**（2026-08-05）。P0–P3 代码与文档
> 交付完成（G1 环境验收 CPU 部分通过；packing 布局、统一媒体管线、适配器、
> 编排器配置与 checkpoint 往返均已无权重验证）；**G2–G7 运行验收延后，需
> 模型权重，本机未运行**。实施顺序按 2026-08-03 修订：**图像对训练优先**，
> **视频训练其次**，**音频训练本期不做**；项目目录 ScheduledTrainer 已更名
> UTrainer，本文档所有路径均指 `UTrainer/` 根）。
> 每阶段小结见 [progress/012-minimaxh3.md](../progress/012-minimaxh3.md)。
> 目标产出：`UnifiedTrainer` 新增 `minimax_h3` 模型适配器，先以**图像对**形式
> （1 帧视频即图像）打通训练闭环，再扩展到**视频**训练，并接入 UTrainer
> 编排器与文档体系。本文档按"设计决定 → 分阶段实施 → 文档更新 → 验收 → 风险"
> 组织，实施前请通读。

---

## 1. 背景与目标

MiniMax-H3 是 MiniMax 发布的 33B 联合视频+音频生成模型（单一去噪过程同时产出
带同步立体声的视频）。目前 HuggingFace diffusers PR
[#14355 "Add MiniMax-H3"](https://github.com/huggingface/diffusers/pull/14355)
提供了完整推理侧组件（Modular Diffusers blocks，无标准 `DiffusionPipeline`），
本地 `E:\diffusers` 已切换到 `pr-14355` 分支（commit `abc5e9b`）。

**目标（按交付顺序）**：
1. **里程碑 1 — 图像对训练（优先）**：把 MiniMax-H3 当作图像模型训练——
   单帧视频（1 像素帧 → 1 个视频 latent 帧）即图像；源图作为**关键帧条件行**，
   目标图作为**目标视频行**，配文本 caption。**数据层一次建成**：媒体感知的
   统一数据管线（schema 媒体字段、统一 5D 缓存格式 (C,T,H,W)、单一媒体分发
   cache_builder、dataset/collate 5D 逻辑）在里程碑 1 全部落地，图像（T=1）
   作为第一个被端到端验证的媒体类型——**视频训练不重建管线，只激活视频解码**；
2. **里程碑 2 — 视频训练**：激活（不重写）P1 已建成的统一管线视频分支
   （PyAV 解码、24fps 抽帧、17n+5 对齐在 P1 已实现，本阶段用真实视频数据
   端到端验证），同一条适配器路径扩展到 T>1 帧；
3. 接入 UTrainer 编排器（任务即 `train.py --task-id`，无需改编排器主体）；
4. 更新相关文档（md/ 索引、设计文档、requirements、AGENTS.md 等）。

**不做**（本期范围外，明确延后）：音频训练（音频行整体省略，见 D6）、
ref2va（omni 参考）训练、CFG 训练、完整微调（全参）、distill 蒸馏、多机并行。

---

## 2. 现状盘点（关键事实，全部经代码核实）

### 2.1 UnifiedTrainer 训练引擎（模型无关层）

- 入口 `UnifiedTrainer/train.py`：`--model` 选适配器、`--config` 或
  `--task-id/--db` 提供配置；支持 LoRA（PEFT `add_adapter`）/ LoKR、NF4/int8
  量化（bnb）、torchao 量化、block swap（`utils/block_swap.py`）、梯度检查点、
  DDP（accelerate）。
- 适配器协议 `models/base.py`（`BaseModelAdapter`）：`load_transformer/vae/
  scheduler/text_encoder/tokenizer`、`latent_channels/vae_scale_factor/patch_size/
  embedding_dim/resolution_config`、`encode_image/encode_text/decode_latent`、
  `prepare_model_input/unpack_prediction/compute_target/sample_timesteps`。
- 训练步（`engine/trainer.py` `train_epoch`）：
  1. `noisy_latents = (1 - σ)·x0 + σ·noise`（标准 FM 插值，逐 target）；
  2. `adapter.prepare_model_input(batch, noisy_latents, sigmas)` → 模型 kwargs；
  3. `model_pred = transformer(**model_input)`；
  4. `unpacked = adapter.unpack_prediction(model_pred)`（多 target 返回 list，
     与 `noises` 顺序一一对应）；
  5. 逐 target 构建 `LossContext`，loss 求和后按 target 数归一。
- **引擎硬编码了标准流匹配约定**：
  - `losses/flow_matching.py`：`target = noise - learning_target`（标准速度
    v_s = noise − x0），未调用 `adapter.compute_target`；
  - 验证生成循环迭代 `self.noise_scheduler.timesteps`、取
    `self.noise_scheduler.sigmas[i]`、调 `self.noise_scheduler.step(v, t, l)`，
    由 `adapter.load_scheduler()` 返回的调度器驱动；
  - 验证/边距 loss 用 `x0_hat = noise - model_pred`（**仅供 loss 模块消费**；
    flow_matching 不使用它——H3 配置只启用 flow_matching 时无影响）。
- 引擎对 5D latents 的兼容性：`sigmas.view(-1,1,1,1)` 与 (B,C,T,H,W) 广播
  合法（尾维全 1）；collate/`_get_target_latents` 的 5D 堆叠需在 P1 核实。
- 数据管线（`data/`）：五层配置（image_configs / target_configs /
  reference_configs / caption_configs / batch_configs），图片对 + 缓存
  （npz latents + npz embeddings + 每样本 JSON + 显式 index 文件），
  bucket 按分辨率分桶。**全部面向 2D 图片**；caption 缓存按 (caption, 关联图)
  编码，H3 的 vision-block 条件文本正好需要这个关联。H3 的媒体统一层（D3）
  在现有 2D 管线上做**兼容扩展**（新增媒体分支，不触碰 krea2 等现有模型路径），
  图像/视频共用同一 5D 缓存与加载逻辑。

### 2.2 MiniMax-H3（diffusers PR #14355）API 面

- **Transformer** `MiniMaxH3Transformer3DModel`（`models/transformers/
  transformer_minimax_h3.py`）：`ModelMixin, ConfigMixin, AttentionMixin,
  PeftAdapterMixin, CacheMixin` → **原生支持 LoRA**。配置：hidden 5376、
  50 层、56 头×128、ffn 14336、视频 in_channels=24、音频 audio_in_channels=32、
  patch=(1,2,2)、text_dim 5120、time_embed_dim 2688。bf16 权重约 **61.7GB**。
- **前向签名**（训练要喂的 kwargs）：
  `hidden_states`(视频行) / `audio_hidden_states`(音频行) /
  `encoder_hidden_states`(文本) / `timestep`(去重后的时间值, [0,1] 不缩放,
  **t=1 为干净**) / `timestep_indices`(每行→timestep 索引) /
  `token_tags`(0=视频,1=文本,2=音频,−1=padding) / `position_ids`((seq,3)) /
  `video_indices` / `audio_indices` / `text_indices`。返回
  `(video_output, audio_output)`（data-ward 速度）。
- **Packed 序列布局**（`modular_pipelines/minimax_h3/packing.py`，
  `build_packed_sequence` 可直接复用）：`[ text(L) | 关键帧条件(C) | 目标音频(A) |
  目标视频(V) ]`。**音频行可整体省略**（`num_audio_latents=0` → 无 A 块；
  transformer 内 `audio_proj_in(空)` + `index_copy` 空索引均为 no-op，
  已逐行核实）——这是本期不做音频训练的代码级依据。
- **视频 VAE 帧公式（已核实）**：`align_num_frames()` 把像素帧数向上取整到
  `17n+5`；`video_latent_num_frames()` → 视频 latent 帧数 `5n+2`
  （17 像素帧/块 → 5 latent 帧/块，丢每块尾 3 帧）。**单帧输入**：因果 3D 卷积
  （时间 padding=k−1 前置零帧）经两级 temporal stride=2 后 1 帧 → 1 latent 帧
  （机制推断，P0 用真实 VAE 实测确认）。
- **`build_row_timesteps` 语义（已核实）**：文本行**继承视频时间步**；条件行
  钉在各自 condition timestep；目标音频行独立时间步（本期不用）。因此每前向
  最多 2 个去重 timestep（目标 t 与关键帧 t），`timestep/timestep_indices`
  构造简单。
- **关键帧条件行（图像对训练核心，已按 PR 源码核实：packing.py L82-84 /
  before_denoise.py L417 / scale_noise docstring）**：关键帧以固定噪声增强
  混噪 = `scale_noise(clean, 0.999)`（PR `MINIMAX_H3_KEYFRAME_NOISE_AUG =
  0.999`）= 0.999·x0 + 0.001·noise（引擎 σ 术语即 σ_cond=0.001），并钉在
  t=0.999（`max(t, 0.999)`，每个去噪步不变）；行锚定 `"first"`/`"last"`；
  文本侧 presentation = 逐字 prompt 前加 `"<Picture i>: "` 标签 + vision block
  （`<|vision_start|>` + 每 vision patch 一个 `<|image_pad|>` + `<|vision_end|>`），
  **无 chat template**，vision block 行 tag=0（视频）而非 1（文本）。
- **调度器** `MiniMaxH3Scheduler`：与 `FlowMatchEulerDiscreteScheduler` 三个
  不兼容点：① 速度方向相反（**data-ward**：x0 = x_t + σ·v，即 v_d = x0 − noise =
  −v_s）；② timesteps 用 t = 1 − σ ∈ [0,1]，t=1 干净，不缩放；③ σ 网格从
  `linspace(1,0,N)` 起（终点 0 计入步数）。视频 shift=12。`step(v,t,l)` 实现
  x0 估计 + r 混合 Euler，正是验证循环需要的语义。
- **VAE**：视频 `AutoencoderKLMiniMaxH3`（3D 因果卷积、时间压缩 4×、ImageNet
  像素归一化 mean=(0.485,0.456,0.406) std=(0.229,0.224,0.225)）；音频
  `AutoencoderKLMiniMaxH3Audio`（**本期不下载、不使用**）。
  **单帧图像编码路径（已核实，非推断）**：PR 的关键帧即单帧图，走
  `vae._encode_clip`（纯空间编码，tiling 开启）→ **1 个 latent 帧**——
  "1 帧视频=图像"是官方路径；视频走 `vae._encode`（17 帧分块 +
  token_drop=3 → 17n+5 → 5n+2）。**解码注意**：`_decode` 分块对 1 个
  latent 帧得 0 chunk（不可用），单帧解码需 `_decode_clip`（1 latent →
  4 像素帧，裁前 3 帧 frame_pre_padding 取有效首帧，或重复补到 5 latent
  → 20 像素帧裁 3 取首帧）。
- **文本编码器**：Qwen3VLForConditionalGeneration（与 Krea2 同族），取
  `text_encoder.model(...)` 输出 **`hidden_states[50]`**（64 层解码器第 50 层），
  不用 LM head；tokenizer Qwen2TokenizerFast、Qwen3VLProcessor。
- **组件清单（本期）**：text_encoder、tokenizer、processor、vae、
  scheduler(shift=12)、transformer（不含 audio_vae / audio_scheduler /
  transformer_ref）。
- **模型仓库** `MiniMaxAI/MiniMax-H3` 总量 ~210GB；组件式加载只拉命名子目录
  （transformer/ 61.7GB bf16 + vae/ + text_encoder/ 等，合计约 70GB）。
- PR 自评：无 CFG/negative_prompt；768p 单卡 80GB 可推理（auto CPU offload）；
  混合精度 checkpoint（proj_in/audio_proj_in/time_embedder/proj_out/
  audio_proj_out/rope 保持 fp32）。

### 2.3 UTrainer 编排器（原 ScheduledTrainer，已更名）

- Watcher(inbox) + Scheduler(cron) + Dispatcher(worker 池) + GPU Guard，
  SQLite 单一事实源；worker = `train.py --task-id N --db ...` 子进程，
  退出码 0/42/其他；`CUDA_VISIBLE_DEVICES` 绑卡，gpus>1 + `multi_gpu=ddp`
  走 `accelerate launch`。GPU Guard 准入：空机放行，否则需
  `free > total×3/4`（80GB 卡≈60GB free）。
- 约定（doc/REQUIREMENTS.md）：不建虚拟环境；代码隔离（外部代码复制进来再改，
  但 diffusers/transformers 等库依赖按 krea2 先例直接 import）；md/ 一功能一文档；
  progress/ 每阶段总结；agent/ 记录决定。

---

## 3. 核心设计决定

| # | 决定 | 说明与理由 |
|---|------|-----------|
| D1 | **依赖固定到 diffusers `pr-14355` 分支** | 训练与推理共用同一 diffusers 源码。`E:\diffusers` 保持该分支（`git fetch origin refs/pull/14355/head:pr-14355` 可随时更新，PR 可能 force-push）。requirements 注释注明"需要含 MiniMax-H3 的 diffusers PR #14355"。新增依赖：`av`（PyAV，视频解码，里程碑 2 必需，P0 一并安装）；不装 torchaudio。 |
| D2 | **Transformer 先直接 import diffusers，LoRA + 梯度检查点起步；block swap 作为 P4 增强** | 与 krea2（vendor 后加 BlockSwap subclass）不同：H3 前向是单 packed 序列，vendor 同步成本高。`MiniMaxH3Transformer3DModel` 已支持 PeftAdapterMixin + gradient checkpointing。若单卡训练 OOM，再按 krea2 模式 vendor + `BlockSwapMiniMaxH3Transformer3DModel`（复用 `utils/block_swap.ModelOffloader`，操作 `transformer_blocks` ModuleList 即可）。 |
| D3 | **数据层一次建成：统一媒体管线（图像/视频共用），音频本期不扩展** | 里程碑 1 即落地：① `ImageConfig.media: "image"\|"video"`（默认 "image"）、`DatasetConfig.video_frames/video_fps` 字段一次加入 schema；② **统一 5D 缓存格式**：所有 H3 媒体 latents 一律存 (C,T,H,W) npz，图像 = (C,1,H,W)（T 维统一，无 2D/3D 双轨）；③ cache_builder 单一媒体分发（image→PIL→(1,C,1,H,W)；video→PyAV 解码→(1,C,T,H,W)），共用 `adapter.encode_video`；④ `data/video_utils.py` 一次实现（图像/视频两个 loader + 17n+5 对齐函数），P1 端到端验证 image 分支，video 分支 P2 用真实数据激活——**里程碑 2 只加测试数据与验证输出，不改 schema/缓存格式/collate**。不做 `media="audio"`。 |
| D4 | **新增 `MiniMaxH3Adapter`，实现全部抽象方法 + 2 个新可选钩子** | 新可选钩子（带默认实现，不影响现有模型）：`encode_video`（里程碑 1 即用，单帧 T=1；内部可复用 encode_image 归一化逻辑）、`decode_validation_video`（里程碑 2）。`velocity_sign` 属性（默认 `"standard"`），H3 适配器声明 `"data_ward"`。`encode_audio/decode_audio/load_audio_vae` **延后到音频阶段再设计**。 |
| D5 | **速度方向约定：`unpack_prediction` 返回模型原值（data-ward），loss 按 `adapter.velocity_sign` 取反 target** | 这样 MiniMaxH3Scheduler.step 可直接驱动验证循环（它要求 data-ward 输入）。`LossContext` 已携带 `adapter`，`flow_matching.py` 一行改动：`target = learning_target - noise if adapter.velocity_sign == "data_ward" else noise - learning_target`。引擎侧 `x0_hat = noise - unpacked` 仅被 loss 模块消费，flow_matching 不使用 → H3 配置只启用 flow_matching 时无影响（新 loss 按需同步）。 |
| D6 | **时间约定：σ→t=1−σ；每前向最多 2 个去重 timestep；音频行省略** | 引擎插值 `(1-σ)x0 + σ·noise` 与 H3 前向 `x_t = t·x0 + (1−t)·noise`（t=1−σ）**逐位等价，引擎无需改**。`prepare_model_input` 内部构造：目标行 t=1−σ（引擎传入 σ）；关键帧条件行钉在 t=0.999（PR noise_aug=0.999 混噪 = 0.999·x0+0.001·noise，即引擎 σ_cond=0.001；已按 PR 源码核实：packing.py L82-84 / before_denoise.py L417 / scale_noise docstring）；**文本行继承目标行时间步**（已核实 `build_row_timesteps`）。音频行整体省略（`num_audio_latents=0`，代码级已核实可行）。`sample_timesteps` 用 logit-normal（沿用 Krea2 思路），配置可选。 |
| D7 | **LoRA 目标模块按 H3 命名** | H3 attention：`to_q/to_k/to_v/to_out`；FeedForward（diffusers swiglu）：`ff.net.*` 或 `ff.gate_proj/up_proj/down_proj`（P1 用 `named_modules()` 实测）；`proj_in/audio_proj_in/context_embedder/proj_out/audio_proj_out/time_embedder` 保持冻结（fp32 混合精度契约）。默认 rank 8–16（61.7GB 模型 + 长序列，显存优先）。 |
| D8 | **验证生成：`load_scheduler` 返回 MiniMaxH3Scheduler(shift=12)，CFG 关闭，解码按帧数分发** | 引擎验证循环迭代 `scheduler.timesteps`（=1−σ[:-1]，H3 约定）并调 `step(v, t, l)` —— MiniMaxH3Scheduler 语义完全匹配，**引擎循环无需改**。`val_guidance_scale` 必须 =1（H3 无 CFG）。解码：里程碑 1 `decode_latent` 对 T==1 返回 PIL（走引擎现有图像路径）；里程碑 2 新增 `decode_validation_video`（VAE 解码 → PyAV 写静音 mp4 或帧序列目录），引擎按能力分发。 |
| D9 | **编排器零代码改动，GPU 准入靠任务字段** | 任务即 `train.py --task-id`；单卡 80GB + NF4/int8 量化或 2×80GB DDP（`gpus: 2, multi_gpu: ddp`）均可。GPU Guard `free>total×3/4` 对 61.7GB 未量化模型过严 → 提供任务级 `min_free_mb` 可选字段（gpu_guard 小改）或直接用量化配置规避。 |
| D10 | **验收以数值 parity + 过拟合测试为准；图像对过拟合是里程碑 1 完成标准** | 借用 PR 的位级 parity 方法论：先速度符号校验（构造已知 x0/noise/σ，断言输出 ≈ x0 − noise），再**图像对单样本过拟合**（loss 收敛到 ~0，里程碑 1 达成），再视频过拟合（里程碑 2），最后真实 LoRA 训练验证保存/恢复与推理质量。 |

---

## 4. 分阶段实施

### P0 — 环境与前置（0.5 天，可并行）

- [ ] 固定 diffusers：确认 `E:\diffusers` 在 `pr-14355`（`git -C E:\diffusers status`），
      记录 commit；有更新先 `fetch` 再快进。
- [ ] `pip install av`（PyAV）；确认 `import av` 正常。
- [ ] 下载 MiniMax-H3 组件（组件式，**不含 audio_vae/audio_scheduler/transformer_ref**）：
      `hf download MiniMaxAI/MiniMax-H3 --include "transformer/*" "vae/*" "text_encoder/*" "tokenizer/*" "processor/*" "scheduler/*"`
      （估算 ~70GB；`vae/` 子目录名先与 modular_model_index.json 核对）。
- [ ] 核实脚本（写 `.tmp/`，跑完可删）：
      - ① 视频 VAE **单帧编码**：1 张图 → (1,3,1,H,W) → **`vae._encode_clip`**
        （PR 关键帧同款路径，**不要走 `_encode`**——视频分块路径会补帧到 17
        再掉 3 帧，产出 2 个 latent 帧）→ 断言 latent 为 (C,1,H,W)；记录
        latent 通道数（预期 24）、scale（预期 8）；同时验证 `_decode_clip`
        单帧解码（1 latent → 4 像素帧裁前 3 取首帧，或重复补 5 latent →
        20 像素帧裁 3 取首帧），供验证循环使用；
      - ② `build_packed_sequence(num_latent_frames=1, num_audio_latents=0,
        keyframe_anchors=("first",))` 形状断言：无音频行、条件行在前、目标行在后、
        token_tags/position_ids 布局正确（CPU，不需模型）。
- [ ] 冒烟：`MiniMaxH3Transformer3DModel` / `AutoencoderKLMiniMaxH3` /
      `MiniMaxH3Scheduler` / Qwen3VL 文本编码路径可 import。

**验收**：① 断言通过（1 帧 → 1 latent 帧）；② 布局断言通过；组件下载完成；
无 diffusers import 报错。若有任一失败，回改本文档对应小节（如实记录到
`progress/012`）。

### P1 — 图像对训练（里程碑 1，2–3 天，优先交付）

**改动面**：loss 一行 + 新适配器 + **统一媒体数据管线（一次建成，图像/视频共用）**
+ 1 个示例配置。**不动**引擎与编排器。

- `losses/flow_matching.py`：`compute()` 内按 `adapter.velocity_sign` 取 target
  方向（见 D5，一行 + 注释）。权重公式不变（σ 语义不变）。
- 数据层（**统一媒体管线，一次建成，P2 不返工**）：
  - `UnifiedTrainer/data/config_schema.py`：`ImageConfig.media`（"image"|"video"，
    默认 "image"）与 `DatasetConfig.video_frames=124/video_fps=24` 一次加入
    schema；validate() 规则（video 键只能被同名媒体 target 引用、帧数 17n+5
    对齐）一并落地；
  - 新增 `UnifiedTrainer/data/video_utils.py`（P1 实现全量，P2 只激活）：
    `load_image_frames(path, size) -> (1,C,1,H,W)`（PIL → 5D，图像媒体）；
    `load_video_frames(path, num_frames, fps=24) -> (1,C,T,H,W)`（PyAV 逐帧
    解码、24fps 均匀抽帧、等比缩放+中心裁剪、17n+5 对齐、5–15s 时长校验）；
    `snap_frames(n)`、`video_latent_num_frames(n)`（包装 PR packing 函数，
    供缓存与形状断言复用）；
  - `cache_builder.py`：单一媒体分发 `_construct_media`（按 `media` 字段：
    image → `load_image_frames`；video → `load_video_frames`）→
    `adapter.encode_video(vae, frames)` → **统一 (C,T,H,W) npz**
    （图像 = (C,1,H,W)，B 维折叠进 index 样本维度；每样本 JSON 记录
    media/num_frames）；
  - `dataset.py`/collate/bucket：5D 堆叠与 bucket 逻辑在 P1 就用图像样本
    （(C,1,H,W)）打通并硬化——P2 视频只是 T 变大，代码零改动；
  - 收益：图像与视频共用同一缓存格式/加载路径/适配器钩子；里程碑 2 无需
    重建任何数据代码，图像缓存与视频缓存同格式、可互相复用。
- 新增 `UnifiedTrainer/models/minimax_h3/__init__.py`（注册名 `minimax_h3`）：
  - 属性：`latent_channels=24`（P0 核实）、`vae_scale_factor=8`、`patch_size=2`
    （spatial；temporal patch=1 在 prepare_model_input 内处理）、
    `embedding_dim=5120`、`velocity_sign="data_ward"`、
    `supports_image_conditioning=True`（关键帧条件）、`resolution_config`
    （768p 档：短边 768、面积 ≤768×1344、16 整除）。
  - 加载：`load_transformer` → `MiniMaxH3Transformer3DModel.from_pretrained(
    path/transformer, torch_dtype=bf16)`（`_keep_in_fp32_modules` 由 diffusers
    自动处理）；`load_vae` → `AutoencoderKLMiniMaxH3.from_pretrained(path/vae)`；
    `load_scheduler` → `MiniMaxH3Scheduler(shift=12)`；`load_text_encoder` →
    `Qwen3VLForConditionalGeneration.from_pretrained(path/text_encoder)`（复用
    krea2 的 rope_scaling 补丁）；`load_tokenizer/load_processor` → 复用 krea2
    的 fallback 构造。
  - 编码：
    - `encode_image(vae, image)`（图像阶段主路径）：ImageNet 归一化 →
      (1,3,1,H,W) → **`vae._encode_clip`（PR 关键帧同款路径，1 帧 → 1 latent
      帧，已核实；禁止走 `_encode`）**（fp16 autocast over fp32 权重，PR
      同款）→ `{"latent": (C,1,H,W)}`；
    - `encode_video(vae, frames_5d)`：T≥1 通用（P1 先实现，P2 直接用）；
      视频 T>1 时走 **`vae._encode`**（17n+5 分块 + token_drop=3，与 PR 一致）；
    - `encode_text(text, image=None)`：**带关键帧图像时**构造 PR presentation
      （`"<Picture 1>: "` + vision block，无 chat template），Qwen3VL forward
      （`output_hidden_states=True`，不走 LM head，取 `hidden_states[50]`；
      注意 accelerate offload 的 `hook.pre_forward` 模式，照 PR 复制），返回
      `{"prompt_embed": (L,5120), "prompt_embeds_mask": (L,)}`；
    - `decode_latent(vae, latent)`：T==1（图像）时走 **`_decode_clip` 单帧解码**
      （已核实：`_decode` 分块路径对 1 latent 帧得 0 chunk，不可用）——重复补
      帧到 5 latent → `_decode_clip` → 20 像素帧 → 裁前 3 帧
      （frame_pre_padding）→ 取首帧 → PIL 列表（引擎验证路径原样工作）；
      T>1 时走 `_decode`（P2）。
  - 训练钩子：
    - `prepare_model_input(batch, noisy_latents(list), sigmas)`：
      1. 目标行：noisy_latents[0]（(B,24,1,H,W)）→ `patchify_video_latents`
         patch (1,2,2)（复用 PR 函数）；
      2. 条件行：batch["latents"] 源图（干净）→ 按 `keyframe_condition_noise`
         语义混噪（PR noise_aug=0.999 → 0.999·x0+0.001·noise，即引擎
         σ_cond=0.001；噪声每步重抽、generator 可复现）→ patchify；
      3. `build_packed_sequence(num_latent_frames=1, num_audio_latents=0,
         keyframe_anchors=("first",))`（复用 PR 函数）→
         position_ids/token_tags/video_indices/audio_indices/text_indices；
      4. 时间：`t_target = 1 - sigmas`（引擎 σ），`t_cond = 0.999`
         （before_denoise.py：`max(t, 0.999)`，已按 PR 源码核实）；
         `timestep = unique([t_cond, t_target])`（排序），
         `timestep_indices` 按行组填（文本行继承 t_target，已核实）；
      5. `audio_hidden_states` = 空张量 (B, 0, 5376)；
      6. 返回 forward kwargs dict。
    - `unpack_prediction(model_pred)`：`return_dict=False` 取 (video, audio空)
      → 返回 `[video_pred]`（与 noises 对齐）；
    - `compute_target(noise, x0)`：返回 `x0 - noise`（data-ward；引擎虽不调用，
      保持协议一致性）；
    - `sample_timesteps`：logit-normal（mu 可配）→ 返回 σ（引擎语义）。
- 引擎适配检查（P1 内核实，必要时小改）：collate/`_get_target_latents` 对
  (B,24,1,H,W) 的堆叠（数据层已按统一 5D 硬化，这里只做端到端确认）；
  `sigmas.view(-1,1,1,1)` 广播（预期 OK，见 2.1）。
- 配置 `configs/minimax_h3_image_smoke.json`（图像对）：
```json
{
  "model": "minimax_h3",
  "model_path": "E:/models/MiniMax-H3",
  "data": {
    "cache_dir": "cache/minimax_h3_image",
    "dataset_configs": [{
      "train_data_dir": "E:/data/h3_image_pairs",
      "resolution": 768,
      "image_configs": {
        "S": {"suffix": "_s", "media": "image"},
        "P": {"suffix": "_t", "media": "image"}
      },
      "target_configs": {"T": [{"image": "P"}]},
      "reference_configs": {"S": [{"image": "S", "sample_type": "from_same_name"}]},
      "caption_configs": {"C": {"ext": ".txt", "image": "S"}},
      "batch_configs": [{"target_config": "T", "caption_config": "C", "reference_config": "S"}]
    }]
  },
  "training": {"batch_size": 1, "learning_rate": 1e-4, "max_steps": 100,
               "lora_rank": 8, "mixed_precision": "bf16"},
  "losses": [{"type": "flow_matching", "weight": 1.0}],
  "output": {"dir": "output/h3_image_smoke", "save_name": "h3_image_smoke"}
}
```
  - 说明：S=源图（关键帧条件，suffix `_s`）、P=目标图（suffix `_t`）；cache_builder
    按 name-marker suffix 分 pool 配对，S/P 必须不同后缀；caption 与 S 关联——
    caption 阶段会把 S 的 PIL 作为 condition_image 传给 `adapter.encode_text`
    （通过 `encode_text_accepts_image` 钩子）生成 vision block。
- 过拟合测试：单样本（1 对图 + 1 条 caption），batch=1，~100 步，断言 loss
  下降 ≥80%；速度符号校验脚本先行（构造已知 x0/noise/σ，断言模型输出 ≈ x0−noise）。

**验收（里程碑 1 完成标准）**：
- G2：统一管线图像分支缓存构建通过（S/P 的 5D latent + 含 vision block 的
  embedding）；视频分支代码已就绪（待 G5 激活）；
- G3：dry-run 前向+反向无 NaN，速度符号校验通过；
- G4：单样本过拟合 loss 下降 ≥80%，验证循环（val split）产出 PIL 图像。

### P2 — 视频训练（里程碑 2，2–3 天）

**改动面**：激活 P1 已建成的视频分支 + 适配器验证解码。**不改** schema/缓存格式/
collate（统一管线已就位）；**音频不碰**。

- `data/video_utils.py`：已在 P1 实现；本阶段以真实视频端到端验证（解码帧数、
  24fps 抽帧、17n+5 对齐、5–15s 时长校验）。
- `config_schema.py`：已在 P1 落地（`media="video"`、`video_frames/video_fps`、
  validate 规则），本阶段无需改动。
- `cache_builder.py`：已在 P1 落地（单一媒体分发）；本阶段以真实 .mp4 构建
  缓存，验证 (C,T,H,W) 形状（124 帧 → 37 latent 帧）。
- `dataset.py`/collate：已在 P1 以 (C,1,H,W) 打通 5D 路径；本阶段视频
  (C,T,H,W) 走同一路径，仅帧数 T 变大。
- 适配器（增量）：`encode_video` 已通用（P1 实现）；`decode_validation_video(
  vae, latent, output_dir)`：VAE 解码 → (B,C,T,H,W) → 帧序列转 PIL → PyAV 写**静音 mp4**
  （无音频轨道；音频训练是本期之外的事，不 mux）；引擎验证循环按钩子分发。
- 配置 `configs/minimax_h3_video_smoke.json`（无音频 target）：
```json
{
  "model": "minimax_h3",
  "model_path": "E:/models/MiniMax-H3",
  "data": {
    "cache_dir": "cache/minimax_h3_video",
    "dataset_configs": [{
      "train_data_dir": "E:/data/h3_videos",
      "resolution": 768,
      "video_frames": 124,
      "image_configs": {"V": {"suffix": ".mp4", "media": "video"}},
      "target_configs": {"T": [{"image": "V"}]},
      "reference_configs": {"none": []},
      "caption_configs": {"C": {"ext": ".txt", "image": "V"}},
      "batch_configs": [{"target_config": "T", "caption_config": "C"}]
    }]
  },
  "training": {"batch_size": 1, "learning_rate": 1e-4, "max_steps": 100,
               "lora_rank": 8, "mixed_precision": "bf16"},
  "losses": [{"type": "flow_matching", "weight": 1.0}],
  "output": {"dir": "output/h3_video_smoke", "save_name": "h3_video_smoke"}
}
```
  - 说明：视频训练同样省略音频行（D6 已核实可行）；本期视频 = 无声视频建模，
    时间对齐由共享 rotary clock 保证（只有视频时无跨模态对齐问题）。
  - 说明：`reference_configs: {"none": []}` 合法——视频训练默认**无关键帧**
    （适配器支持 `keyframe_anchors=()`，纯 t2v 布局；schema 仅要求该键
    存在，空条目列表跳过参考校验）。可选 i2v 扩展需加 reference_configs。
  - （可选，后续迭代）视频首帧作关键帧条件（`keyframe_anchors=("first",)`，
    源=视频自身首帧），作为视频训练的 i2v 扩展，不在本期必做范围。

**验收（里程碑 2 完成标准）**：
- G5：视频缓存构建通过（(C,T,H,W) 形状断言：124 帧 → 37 latent 帧
  = 5×7+2，17×7+5=124 ✓）；
- G6：视频单样本过拟合 loss 下降 ≥80%；验证循环产出静音 mp4（可播放）。

### P3 — 编排器接入与冒烟（0.5–1 天）

- 新增示例配置 `configs/minimax_h3_train.json`（图像对或视频，含
  `training.multi_gpu` 说明）。
- CLI/API 建任务（单卡 NF4 或 `gpus: 2, multi_gpu: ddp`），跑 10 步冒烟；
  心跳/退出码/日志正常。
- 确认 `--resume`（LoRA 权重）与 `--resume-full` 路径对 H3 检查点可用
  （`engine/checkpoint.py` 的 save/load 与 diffusers LoRA 格式对齐——P1 验证
  `save_lora_weights` 兼容）。
- （可选）`gpu_guard.py`：任务字段 `min_free_mb` 支持（默认沿用 3/4 规则）。

**验收**：编排器建任务 → GPU Guard 准入 → 训练 → done 全链路走通。

### P4 — 文档更新 + 回归加固（1 天，可选增强）

- 文档更新见 §5。
- 与 PR 推理 parity：对同一 σ、同一 packed 布局（含关键帧行），训练前向输出
  vs `MiniMaxH3Blocks` 推理同步输出一致（速度符号约定内）；
- 量化（NF4/int8）下的训练冒烟；多卡 DDP 冒烟；
- （OOM 时才做）vendor transformer + BlockSwap subclass（D2）。

---

## 5. 文档更新清单（"update related documents"）

| 文件 | 更新内容 |
|------|---------|
| `md/00-architecture.md` | 索引表新增一行：`07-minimaxh3-training.md`（MiniMax-H3 训练接入：图像对 → 视频，音频延后；状态：设计完成/实施中→as-built） |
| `md/07-minimaxh3-training.md`（新建） | 本功能的 as-built 设计文档：统一媒体数据管线（图像/视频共用、一次建成）、MiniMaxH3Adapter 钩子、data-ward 速度约定、t=1−σ 时间约定、关键帧条件行（PR noise_aug=0.999 → 引擎 σ_cond=0.001、t_cond=0.999）、音频行省略依据、验证循环调度器语义、LoRA 目标模块清单、配置模板 |
| `md/minimaxh3_implementation.md`（本文档） | 实施完成后状态改为"已完成"，或归档为 plan 历史 |
| `requirements.txt`（UTrainer 根）+ `UnifiedTrainer/requirements.txt` | diffusers 注释追加：训练 MiniMax-H3 需 PR #14355 分支（`refs/pull/14355/head`）；新增 `av`（PyAV）依赖行 |
| `AGENTS.md`（仓库根 e:\UnifiedTrainer） | "Available Models" 表加 `minimax_h3`（架构 MiniMax-H3 33B joint video+audio，latent 通道 24，VAE scale 8；当前支持图像对与视频训练） |
| `UnifiedTrainer/data/cache_system.md` | 图像对 5D 缓存格式（(C,1,H,W)）+ 视频媒体缓存（(C,T,H,W)、media 字段、17n+5 帧对齐规则）；音频缓存标注"延后" |
| `UnifiedTrainer/models/base.py` | 文档化新可选钩子：`encode_video/decode_validation_video/velocity_sign`（默认实现保持现有行为） |
| `doc/Improvement_k3.md` | 追加 MiniMax-H3 训练接入章节（决策 D1–D10 摘要 + 图像对/视频数据改动点 + 音频延后原因） |
| `progress/012-minimaxh3.md`（新建） | 每阶段小结（对照本文档验收标准逐项打勾） |
| `agent/decisions.md` | 记录 D1–D10 决定与实现过程中的偏差 |

## 6. 验收标准（fitness gates，goal-md 风格）

按序通过即视为"训练可用"：

1. **G1 环境**：`import diffusers; diffusers.__version__ == 0.40.0.dev0` 且
   `MiniMaxH3Transformer3DModel` 可 import；`import av` 成功。
2. **G2 图像对数据**：统一媒体管线以图像为第一个验证媒体——缓存构建成功
   （S/P 5D latents + 含 vision block 的 embedding），`__getitem__` 形状断言
   通过（同一管线已含视频分支，G5 只做真实数据激活）。
3. **G3 训练步**：dry-run 前向+反向无 NaN，速度符号校验通过
   （构造已知 x0/noise/σ，断言模型输出 ≈ x0 − noise）。
4. **G4 图像对过拟合**：单样本 100 步 loss 下降 ≥80%；验证产出 PIL 图像。
   **→ 里程碑 1 完成。**
5. **G5 视频数据**：复用 G2 同一管线（仅解码分支不同），真实 .mp4 缓存构建
   成功，形状断言通过（17n+5 → 5n+2，如 124 → 37 latent 帧）。
6. **G6 视频过拟合**：单样本 100 步 loss 下降 ≥80%；验证产出静音 mp4（可播放）。
   **→ 里程碑 2 完成。**
7. **G7 全链路**：编排器任务 done，产物 LoRA 可被推理侧
   （`MiniMaxH3Blocks` + `pipe.load_lora_weights`）加载。

## 7. 行动目录（按影响排序）

| 行动 | 影响 | 方式 |
|------|------|------|
| P0 组件下载（~70GB，无 audio_vae）+ 单帧 VAE 核实 | 阻塞一切 | 尽早启动下载（后台），并行跑核实脚本 |
| P1 统一媒体数据管线（schema+video_utils+5D 缓存） | 里程碑 1+2 共用，一次建成 | 图像先行端到端验证，视频解码同批实现 |
| P1 速度符号校验 + flow_matching data_ward | 训练正确性核心 | 先写符号校验脚本再实现 |
| P1 适配器 + 关键帧条件行 + 图像对过拟合 | 里程碑 1 | 复用 PR packing/presentation，最小引擎改动 |
| P2 激活视频解码 + 真实数据缓存 + 验证 mp4 | 里程碑 2 | 仅激活与测试，不重建数据代码 |
| P3 编排器冒烟 | 日常使用 | 复用现有任务机制 |
| Block swap / 量化调优 | 显存（单卡可行与否） | OOM 后按 D2 决策 |

## 8. 约束

1. **不改 orchestrator 主体代码**（除非 gpu_guard `min_free_mb` 小改获准）——
   任务机制已支持 H3 所需（多卡 DDP、心跳、resume）。
2. **diffusers 版本锁定在 pr-14355**；不得在未通知的情况下把
   `E:\diffusers` 切回 main（会丢 MiniMaxH3 组件）。
3. **不建虚拟环境**（REQUIREMENTS.md 硬性规定），新依赖直接进系统 Python。
4. **本期不含音频训练**：不下载 audio_vae/audio_scheduler，不实现音频数据/
   损失/解码代码，实现中不得为音频预留半成品分支（音频行整体省略是 D6 的
   正式决定，不是临时 hack）。
5. **验证不造假**：G2–G6 必须真实跑出数字；OOM/形状错误记录到
   `progress/012` 而不是静默改测试。
6. **不下载全量 210GB 仓库**：只拉组件子目录（磁盘与流量约束）。
7. 每个里程碑后提交一次，commit 信息含 gate 编号（如 `[G4] minimax_h3 image-pair overfit`）。

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| PR #14355 未合并且 API 变动（force-push） | 组件签名漂移 | 锁定本地分支 + 记录 commit；升级前跑 G3 符号校验 |
| 61.7GB transformer 显存压力 | 视频长序列单卡 80GB 训练可能不可行 | 图像对阶段序列短（1 latent 帧），80GB 应可行；视频阶段 NF4/int8 量化（预计 16–25GB）、block swap、2×80GB DDP |
| "单帧 → 1 latent 帧"用错编码路径 | 图像 latent 变 2 帧、与关键帧条件不一致 | 已核实：图像必须走 `_encode_clip`（PR 关键帧同款）；P0 脚本 ① 断言兜底；解码同理走 `_decode_clip` |
| 关键帧条件行与 PR 语义不一致（噪声增强、锚定、vision block） | 图像对训练学到错误条件 | 已按 PR 源码核实（packing.py L82-84 / before_denoise.py L417 / scale_noise docstring）：混噪 = 0.999·x0+0.001·noise、钉步 t=0.999；直接复用 PR `keyframe_condition_noise`/`build_packed_sequence`/presentation 构造，不自己发明 |
| vision-block caption 编码需关联图像 | cache_builder caption 阶段改动面 | P1 核实现有调用点；加可选钩子，默认实现退化为现有行为 |
| 视频帧对齐/时长约束（17n+5、5–15s） | 缓存形状错误 | video_utils 统一对齐 + 形状断言；过拟合测试兜底 |
| 图像/视频管线分两次建导致返工 | 里程碑 2 工作量膨胀、缓存不兼容 | D3 统一媒体管线一次建成；P1 验收即含 5D 路径硬化；图像缓存与视频同格式可复用 |
| collate/`_get_target_latents` 5D 混合 | 里程碑 2 数据 bug | P1 先以 (C,1,H,W) 打通 5D 路径，P2 只是 T 变大 |
| LoRA 目标模块命名不匹配 | PEFT 报错 | P1 用 `named_modules()` 实测后固化清单 |
| 模型仓库下载中断/磁盘不足 | 阻塞 P0 | 组件式 include 下载 + 断点续传（hf 默认），预留 100GB 磁盘 |
