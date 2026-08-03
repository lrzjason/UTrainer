# MiniMax-H3 训练接入 ScheduledTrainer — 实施方案

> 状态：规划中（plan）｜目标产出：`UnifiedTrainer` 新增 `minimax_h3` 模型适配器，
> 支持视频+音频联合扩散训练，并接入 ScheduledTrainer 编排器与文档体系。
> 本文档按"设计决定 → 分阶段实施 → 文档更新 → 验收 → 风险"组织，实施前请通读。

---

## 1. 背景与目标

MiniMax-H3 是 MiniMax 发布的 33B 联合视频+音频生成模型（单一去噪过程同时产出
带同步立体声的视频）。目前 HuggingFace diffusers PR
[#14355 "Add MiniMax-H3"](https://github.com/huggingface/diffusers/pull/14355)
提供了完整推理侧组件（Modular Diffusers blocks，无标准 `DiffusionPipeline`），
本地 `E:\diffusers` 已切换到 `pr-14355` 分支（commit `abc5e9b`）。

**目标**：
1. 在 ScheduledTrainer 的 `UnifiedTrainer` 框架内实现 MiniMax-H3 **训练**能力
   （LoRA，冻结 base 模型），复用现有模型无关训练引擎；
2. 打通视频+音频数据管线（抽帧 24fps、17n+5 帧、32kHz 音频重采样）；
3. 接入 ScheduledTrainer 编排器（任务即 `train.py --task-id`，无需改编排器主体）；
4. 更新相关文档（md/ 索引、设计文档、requirements、AGENTS.md 等）。

**不做**（本期范围外）：ref2va（omni 参考）训练、CFG 训练、完整微调（全参）、
distill 蒸馏、多机并行。

---

## 2. 现状盘点（关键事实，全部经代码核实）

### 2.1 UnifiedTrainer 训练引擎（模型无关层）

- 入口 `UnifiedTrainer/train.py`：`--model` 选适配器、`--config` 或 `--task-id/--db`
  提供配置；支持 LoRA（PEFT `add_adapter`）/ LoKR（`networks/lokr_module.py`）、
  NF4/int8 量化（bnb）、torchao 量化、block swap（`utils/block_swap.py`）、
  梯度检查点、DDP（accelerate）。
- 适配器协议 `models/base.py`（`BaseModelAdapter`）：`load_transformer/vae/
  scheduler/text_encoder/tokenizer`、`latent_channels/vae_scale_factor/patch_size/
  embedding_dim/resolution_config`、`encode_image/encode_text/decode_latent`、
  `prepare_model_input/unpack_prediction/compute_target/sample_timesteps`。
- 训练步（`engine/trainer.py` `train_epoch`，L629–750）：
  1. `noisy_latents = (1 - σ)·x0 + σ·noise`（标准 FM 插值，逐 target）；
  2. `adapter.prepare_model_input(batch, noisy_latents, sigmas)` → 模型 kwargs；
  3. `model_pred = transformer(**model_input)`；
  4. `unpacked = adapter.unpack_prediction(model_pred)`（多 target 返回 list，
     与 `noises` 顺序一一对应）；
  5. 逐 target 构建 `LossContext`，loss 求和后按 target 数归一。
- **引擎硬编码了标准流匹配约定**：
  - `losses/flow_matching.py`：`target = noise - learning_target`（即标准速度
    v_s = noise − x0），未调用 `adapter.compute_target`；
  - 验证生成循环（L1383–1431）迭代 `self.noise_scheduler.timesteps`、取
    `self.noise_scheduler.sigmas[i]`、调 `self.noise_scheduler.step(v, t, l)`，
    由 `adapter.load_scheduler()` 返回的调度器驱动；
  - 验证/边距 loss 用 `x0_hat = noise - model_pred`。
- 数据管线（`data/`）：五层配置（image_configs / target_configs /
  reference_configs / caption_configs / batch_configs），图片对 + 缓存
  （npz latents + npz embeddings + 每样本 JSON + 显式 index 文件），
  bucket 按分辨率分桶。**全部面向 2D 图片**。

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
  目标视频(V) ]`；视频 VAE 像素为 ImageNet 归一化
  （mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)）。
- **调度器** `MiniMaxH3Scheduler`：与 `FlowMatchEulerDiscreteScheduler` 三个不兼容点：
  ① 速度方向相反（**data-ward**：x0 = x_t + σ·v，即 v_d = x0 − noise = −v_s）；
  ② timesteps 用 t = 1 − σ ∈ [0,1]，t=1 干净，不缩放；③ σ 网格从
  `linspace(1,0,N)` 起（终点 0 计入步数）。视频 shift=12、音频 shift=3
  （两个实例）。`step(v,t,l)` 实现 x0 估计 + r 混合 Euler，正是验证循环需要的语义。
- **VAE**：视频 `AutoencoderKLMiniMaxH3`（3D 因果卷积，时间压缩 4×，需从
  checkpoint config 核实精确 latent 帧数公式）；音频
  `AutoencoderKLMiniMaxH3Audio`（DAC：`[B,1,samples]→[B,32,samples/800]`，
  32kHz、40 latents/s、**单声道**；立体声 = 两个 channel-major 音频行块）。
- **文本编码器**：Qwen3VLForConditionalGeneration（4B 级，与 Krea2 同族），
  取 `text_encoder.model(...)` 输出 **`hidden_states[50]`**（64 层解码器第 50 层），
  不用 LM head；tokenizer Qwen2TokenizerFast、Qwen3VLProcessor。
- **组件清单**（t2va/fl2va 任务）：text_encoder、tokenizer、processor、vae、
  audio_vae、scheduler(shift=12)、audio_scheduler(shift=3)、transformer、
  video_processor（+ ref2va 用 transformer_ref）。
- **模型仓库** `MiniMaxAI/MiniMax-H3` 总量 ~210GB；组件式加载只拉命名子目录
  （transformer/ 61.7GB bf16、transformer_ref/、vae/、audio_vae/、text_encoder/…）。
- PR 自评：无 CFG/negative_prompt；768p 单卡 80GB 可推理（auto CPU offload）；
  混合精度 checkpoint（proj_in/audio_proj_in/time_embedder/proj_out/audio_proj_out/rope 保持 fp32）。

### 2.3 ScheduledTrainer 编排器

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
| D1 | **依赖固定到 diffusers `pr-14355` 分支** | 训练与推理共用同一 diffusers 源码。`E:\diffusers` 保持该分支（`git fetch origin refs/pull/14355/head:pr-14355` 可随时更新，PR 可能 force-push）。requirements 注释注明"需要含 MiniMax-H3 的 diffusers PR #14355"。新增依赖：`av`（PyAV，视频/音频解码与重采样，PR 推理侧同款 gating）；`torchaudio` 可选。 |
| D2 | **Transformer 先直接 import diffusers，LoRA + 梯度检查点起步；block swap 作为 P3 增强** | 与 krea2（vendor 后加 BlockSwap subclass）不同：H3 前向是单 packed 序列，vendor 同步成本高。`MiniMaxH3Transformer3DModel` 已支持 PeftAdapterMixin + gradient checkpointing（`_no_split_modules` 含 block）。若 80GB 单卡训练 OOM，再按 krea2 模式 vendor + `BlockSwapMiniMaxH3Transformer3DModel`（复用 `utils/block_swap.ModelOffloader`，操作 `transformer_blocks` ModuleList 即可）。 |
| D3 | **数据管线扩展"媒体类型"，复用五层配置框架** | `image_configs` 条目增加可选 `"media": "image" \| "video" \| "audio"`；target_configs 引用 video/audio 键 → 多 target 机制天然支持（引擎已支持 target 列表）。新增 `data/video_utils.py`（抽帧、17n+5 对齐、音频重采样）+ `data/cache_builder` 的媒体分支。视频以 4D/5D 张量 (B,C,T,H,W) 入缓存，音频 (B,1,samples) 或直接存 (B,C_a,T_a) 潜变量。 |
| D4 | **新增 `MiniMaxH3Adapter`，实现全部抽象方法 + 3 个新可选钩子** | 新可选钩子（带默认实现，不影响现有模型）：`encode_video` / `encode_audio` / `decode_audio`，以及 `velocity_sign` 属性（默认 `"standard"`）。H3 适配器声明 `velocity_sign = "data_ward"`。 |
| D5 | **速度方向约定：`unpack_prediction` 返回模型原值（data-ward），loss 按 `adapter.velocity_sign` 取反 target** | 这样 MiniMaxH3Scheduler.step 可直接驱动验证循环（它要求 data-ward 输入）。`LossContext` 已携带 `adapter`，`flow_matching.py` 一行改动：`target = learning_target - noise if adapter.velocity_sign == "data_ward" else noise - learning_target`。新 loss（lcs 等）按需同步。 |
| D6 | **时间约定：`prepare_model_input` 内把引擎 σ 转换为 t = 1 − σ** | 引擎侧插值公式 `(1-σ)x0 + σ·noise` 与 H3 前向 `x_t = t·x0 + (1−t)·noise`（t=1−σ）**逐位等价，引擎无需改**。只有喂给 transformer 的 `timestep` 与 `timestep_indices` 要按 H3 约定构造。训练时视频行与音频行共用同一 σ（单次去噪联合训练）；`sample_timesteps` 用 logit-normal（沿用 Krea2 思路）或均匀，配置可选。文本行的时间索引取值（t=1 或当前步）P2 时对照 PR 的 `build_row_timesteps`（before_denoise.py）核实。 |
| D7 | **LoRA 目标模块按 H3 命名** | H3 attention：`to_q/to_k/to_v/to_out`；FeedForward（diffusers swiglu）：`ff.net.*` 或 `ff.gate_proj/up_proj/down_proj`（P2 用 `named_modules()` 实测）；`proj_in/audio_proj_in/context_embedder/proj_out/audio_proj_out/time_embedder` 保持冻结（fp32 混合精度契约）。默认 rank 8–16（61.7GB 模型 + 长序列，显存优先）。 |
| D8 | **验证生成：`load_scheduler` 返回 MiniMaxH3Scheduler(shift=12)，CFG 关闭，解码走适配器钩子** | 引擎验证循环迭代 `scheduler.timesteps`（=1−σ[:-1]，H3 约定）并调 `step(v, t, l)` —— MiniMaxH3Scheduler 语义完全匹配，**引擎循环无需改**。`val_guidance_scale` 必须 =1（H3 无 CFG）。视频/音频验证解码：新增适配器可选方法 `decode_validation_video(latent)`（VAE 解码 + 可选 ffmpeg/PyAV mux 音频），引擎按能力分发，否则 fallback 取中间帧 PIL（现有路径）。 |
| D9 | **编排器零代码改动，GPU 准入靠任务字段** | 任务即 `train.py --task-id`；单卡 80GB + NF4/int8 量化（~16–25GB）或 2×80GB DDP（`gpus: 2, multi_gpu: ddp`）均可。GPU Guard `free>total×3/4` 对 61.7GB 未量化模型过严 → 提供任务级 `min_free_mb` 可选字段（gpu_guard 小改）或直接用量化配置规避。 |
| D10 | **验收以数值 parity + 过拟合测试为准，不追求一步到位** | 借用 PR 的位级 parity 方法论：训练前先跑推理侧 parity（噪声/速度一致性），再单样本过拟合（loss 收敛到 ~0），最后 1–2 步真实 LoRA 训练验证保存/恢复与推理质量。 |

---

## 4. 分阶段实施

### P0 — 环境与前置（半天，可并行）

- [ ] 固定 diffusers：确认 `E:\diffusers` 在 `pr-14355`（`git -C E:\diffusers status`），
      记录 commit；`E:\diffusers` 若有更新先 `fetch` 再快进。
- [ ] `pip install av`（PyAV）；确认 `import av` 正常。
- [ ] 下载 MiniMax-H3 组件（组件式，只拉训练需要的子目录）：
      `hf download MiniMaxAI/MiniMax-H3 --include "transformer/*" "transformer_ref/*" "vae/*" "audio_vae/*" "text_encoder/*" "tokenizer/*" "processor/*" "scheduler/*" "audio_scheduler/*"`
      （估算 >125GB，注意磁盘与网络；`vae/`、`audio_vae/` 先确认子目录名与
      modular_model_index.json 一致）。
- [ ] 用脚本核实 VAE 关键配置（写 `E:\diffusers` 外的临时脚本或 `.tmp/`）：
      `AutoencoderKLMiniMaxH3.from_pretrained(...).config` → latent 通道数、
      时间压缩、tiling；`AutoencoderKLMiniMaxH3Audio` → latent_dim、800 跳。
      记录到本阶段 progress 小结。
- [ ] 冒烟：`MiniMaxH3Blocks` / `MiniMaxH3Transformer3DModel` 可 import；
      `build_packed_sequence` 可调用（CPU 小样例）。

**验收**：上述脚本全部通过；组件下载完成；无 diffusers import 报错。

### P1 — 数据管线（1–2 天）

新增文件：
- `UnifiedTrainer/data/video_utils.py`
  - `load_video_frames(path, num_frames=124) -> (B,C,T,H,W) float [0,1]`：
    PyAV 逐帧解码（24fps），等比缩放 + 中心裁剪到 bucket 分辨率，帧数
    对齐 17n+5（不足尾部复制尾帧/超出均匀抽帧），ImageNet 归一化在
    encode_video 内做（VAE 前）。
  - `load_audio_waveform(path, sample_rate=32000, duration=None) -> (B,1,samples)`：
    PyAV 解码 + `aresample` 到 32kHz、立体声→双单声道（与 PR 的
    channel-major 音频行约定一致）。
  - `snap_frames(n) -> 17n+5`、`audio_latent_len(duration) -> duration×40`。
- `UnifiedTrainer/data/config_schema.py` 扩展：
  - `ImageConfig` 增加 `media: str = "image"`（"image"|"video"|"audio"）；
  - `DatasetConfig` 增加 `video_frames: int = 124`、`video_fps: int = 24`、
    `audio_sample_rate: int = 32000`（dataset 级默认，可被 batch_configs 覆盖）；
  - validate() 补充：video/audio 键只能被同名媒体类型的 target 引用。
- `UnifiedTrainer/data/cache_builder.py`：`_construct_image_pairs` 与编码阶段按
  `media` 分发：video → `adapter.encode_video(vae, frames)`；audio →
  `adapter.encode_audio(audio_vae, waveform)`；输出 latents npz 保存
  video: (C,T,H,W)、audio: (C_a,T_a)（B 维折叠进 index 样本维度，与图片一致）。
- `UnifiedTrainer/data/dataset.py`：`__getitem__` 已按 latents dict 读取，
  只需保证 `_get_target_latents`/collate 能处理 4D/5D 混合（当前按 torch.stack
  逐 role 堆叠——已满足；验证 `bucket` 键用 "HxW" 字符串不变）。

配置模板（`configs/minimax_h3_video_smoke.json`）：
```json
{
  "model": "minimax_h3",
  "model_path": "E:/models/MiniMax-H3",
  "data": {
    "cache_dir": "cache/minimax_h3",
    "dataset_configs": [{
      "train_data_dir": "E:/data/h3_videos",
      "resolution": 768,
      "video_frames": 124,
      "image_configs": {
        "V": {"suffix": ".mp4", "media": "video"},
        "A": {"suffix": ".wav", "media": "audio"}
      },
      "target_configs": {"T": [{"image": "V"}, {"image": "A"}]},
      "caption_configs": {"C": {"ext": ".txt", "image": "V"}},
      "batch_configs": [{"target_config": "T", "caption_config": "C"}]
    }]
  },
  "training": {"batch_size": 1, "learning_rate": 1e-4, "max_steps": 100,
               "lora_rank": 8, "mixed_precision": "bf16"},
  "losses": [{"type": "flow_matching", "weight": 1.0}],
  "output": {"dir": "output/h3_smoke", "save_name": "h3_smoke"}
}
```

**验收**：缓存构建跑通（1 个视频样本 + wav + txt）；`dataset.py` 单样本
`__getitem__` 返回 video/audio/embedding；bucket 无报错。

### P2 — 模型适配器（2–3 天）

新增 `UnifiedTrainer/models/minimax_h3/__init__.py`（注册名 `minimax_h3`）：

- 属性：`latent_channels`（视频 VAE 输出通道，P0 核实，预期 24）、
  `vae_scale_factor=8`、`patch_size=2`（spatial；temporal patch=1 在
  prepare_model_input 内处理）、`embedding_dim=5120`、`resolution_config`
  （768p 档：1344×768 等 16 整除分辨率）、`supports_image_conditioning=False`
  （本期不做 keyframe 训练）、`velocity_sign="data_ward"`。
- 加载：
  - `load_transformer` → `MiniMaxH3Transformer3DModel.from_pretrained(
    path/transformer, torch_dtype=bf16)`（注意 `_keep_in_fp32_modules` 由
    diffusers 自动处理）；
  - `load_vae` → `AutoencoderKLMiniMaxH3.from_pretrained(path/vae)`；
  - 新增 `load_audio_vae` → `AutoencoderKLMiniMaxH3Audio.from_pretrained(
    path/audio_vae)`；
  - `load_scheduler` → `MiniMaxH3Scheduler(shift=12)`（验证循环用）；
  - `load_text_encoder` → `Qwen3VLForConditionalGeneration.from_pretrained(
    path/text_encoder)`（复用 krea2 的 rope_scaling 补丁逻辑）；
  - `load_tokenizer` / `load_processor` → 复用 krea2 的 fallback 构造（同一套
    Qwen3VL 补丁，抽公共函数或复制）。
- 编码：
  - `encode_video(vae, frames)`：ImageNet 归一化 → VAE encode（5D）→
    latent（可 tiling；fp16 autocast over fp32 权重，PR 同款）→ 返回
    `{"latent": (C,T,H,W)}`；
  - `encode_audio(audio_vae, waveform)`：→ `{"latent": (C_a,T_a)}`；
  - `encode_text`：Qwen3VL forward（`output_hidden_states=True`，**不走 LM
    head**，直接 `text_encoder.model(...)` 取 `hidden_states[50]`，注意
    accelerate hook 问题——PR 用 `hook.pre_forward` 手工触发，复制该模式），
    返回 `{"prompt_embed": (L, 5120), "prompt_embeds_mask": (L,)}`；
  - `decode_latent(vae, latent)`：视频 VAE 解码 → (B,C,T,H,W)；
    新增 `decode_audio(audio_vae, latent)` → waveform。
- 训练钩子：
  - `prepare_model_input(batch, noise(list), sigmas)`：
    1. 从 batch 取视频目标行与音频目标行（noise[0]/noise[1] 即 noisy 潜变量）；
    2. `patchify` 视频（patch (1,2,2)，可用 PR 的 patchify 函数）；
    3. 调 `build_packed_sequence(...)`（无关键帧：keyframe_anchors=()）得到
       `position_ids / token_tags / video_indices / audio_indices / text_indices`；
    4. `t = 1 - sigmas`（sigmas 是引擎传入的噪声水平）；训练单步只有一个去重
       timestep → `timestep = t`，`timestep_indices` 视频/音频行=0，文本行取值
       P2 对照 PR `build_row_timesteps` 核实（先用 0 跑通，文档注明）；
    5. 返回前向 kwargs dict。
  - `unpack_prediction(model_pred)`：`return_dict=False` 时取 `(video, audio)`
    元组 → 按 target 顺序返回 `[video_pred, audio_pred]`（与 noises 对齐）；
  - `compute_target(noise, x0)`：返回 `x0 - noise`（data-ward；引擎虽不调用，
    保持协议一致性）；
  - `sample_timesteps`：logit-normal（mu 可配），返回 `(σ, σ)`——引擎用 σ 做
    插值，H3 时间转换在 prepare_model_input 内完成。
- `models/__init__.py` 无需改（注册靠 import，train.py 按 `--model` 动态导入；
  如按包发现机制需在 train.py 的导入表加一行——P2 核实 train.py 的 adapter
  导入方式后补）。

**验收**：`--model minimax_h3 --dry-run-steps 1` 能完成一次前向+反向；
loss 有限值；`--list-models` 出现 minimax_h3。

### P3 — 引擎与 loss 微调（1 天）

- `losses/flow_matching.py`：`compute()` 内
  `target = (context.learning_target - context.noise) if (context.adapter and
  getattr(context.adapter, "velocity_sign", "standard") == "data_ward") else
  (context.noise - context.learning_target)`；权重公式不变（σ 语义不变）。
  其他 loss（lcs/edge 等）若用于 H3 同理处理（本期只保证 flow_matching）。
- `engine/trainer.py` 验证解码分发：`decode_validation_video` 存在时走新路径
  （视频 + 音频 mux 成 mp4 存到验证目录），否则现有 PIL 路径（取中间帧）。
- （可选）`gpu_guard.py`：任务字段 `min_free_mb` 支持（默认沿用 3/4 规则）。
- （可选，OOM 时才做）vendor transformer + BlockSwap subclass（D2）。

**验收**：过拟合单视频样本（batch=1, ~100 步）loss 稳定下降至接近 0；
验证循环（val split）产出 mp4（视频+音频）。

### P4 — 编排器接入与冒烟（0.5–1 天）

- 新增示例配置 `configs/minimax_h3_train.json`（含 dataset_configs、
  `training.multi_gpu` 说明）。
- CLI/API 建任务（`gpus: 2` + `multi_gpu: ddp` 或单卡 NF4），跑 10 步冒烟；
  心跳/退出码/日志正常。
- 确认 `--resume`（LoRA 权重）与 `--resume-full` 路径对 H3 检查点可用
  （`engine/checkpoint.py` 的 save/load 与 diffusers LoRA 格式对齐——P2 验证
  `save_lora_weights` 兼容）。

**验收**：编排器建任务 → GPU Guard 准入 → 训练 → done 全链路走通。

### P5 — 文档更新（见 §5）

### P6 — 回归与加固（1 天，可选增强）

- 与 PR 推理 parity：对同一 σ、同一 packed 布局，训练前向输出 vs
  `MiniMaxH3Blocks` 推理同步输出一致（速度符号约定内）；
- 量化（NF4/int8）下的训练冒烟；
- 多卡 DDP 冒烟。

---

## 5. 文档更新清单（"update related documents"）

| 文件 | 更新内容 |
|------|---------|
| `md/00-architecture.md` | 索引表新增一行：`07-minimaxh3-training.md`（MiniMax-H3 训练接入，实施期 P7，状态：设计完成/实施中→as-built） |
| `md/07-minimaxh3-training.md`（新建） | 本功能的 as-built 设计文档：媒体类型数据管线、MiniMaxH3Adapter 钩子、data-ward 速度约定、t=1−σ 时间约定、验证循环调度器语义、LoRA 目标模块清单、配置模板 |
| `md/minimaxh3_implementation.md`（本文档） | 实施完成后状态改为"已完成"，或归档为 plan 历史 |
| `UnifiedTrainer/requirements.txt` + `requirements.txt`（根） | diffusers 注释追加：训练 MiniMax-H3 需 PR #14355 分支（`refs/pull/14355/head`）；新增 `av`（PyAV）依赖行 |
| `AGENTS.md`（仓库根） | "Available Models" 表加 `minimax_h3`（架构 MiniMax-H3 33B joint video+audio，latent 通道 24+32，VAE scale 8） |
| `UnifiedTrainer/data/cache_system.md` | 视频/音频媒体类型缓存格式（latents npz 形状、media 字段、帧对齐规则） |
| `UnifiedTrainer/models/base.py` | 文档化新可选钩子：`encode_video/encode_audio/decode_audio/decode_validation_video/velocity_sign`（默认实现保持现有行为） |
| `doc/Improvement_k3.md` | 追加 MiniMax-H3 训练接入章节（决策 D1–D10 摘要 + 数据/引擎改动点） |
| `progress/012-minimaxh3.md`（新建） | 每阶段小结（对照本文档验收标准逐项打勾） |
| `agent/decisions.md` | 记录 D1–D10 决定与实现过程中的偏差 |

## 6. 验收标准（fitness gates，goal-md 风格）

按序通过即视为"训练可用"：

1. **G1 环境**：`import diffusers; diffusers.__version__ == 0.40.0.dev0` 且
   `MiniMaxH3Transformer3DModel` 可 import；`import av` 成功。
2. **G2 数据**：视频+音频缓存构建成功，`__getitem__` 形状断言通过
   （video (C,T,H,W)、audio (C_a,T_a)、embedding (L,5120)）。
3. **G3 训练步**：dry-run 前向+反向无 NaN，速度符号校验脚本通过
   （构造已知 x0/noise/σ，断言模型输出 ≈ x0 − noise）。
4. **G4 过拟合**：单样本 100 步 loss 下降 ≥80%。
5. **G5 验证产出**：验证循环生成 mp4（含音频轨道），可播放。
6. **G6 全链路**：编排器任务 done，产物 LoRA 可被推理侧
   （`MiniMaxH3Blocks` + `pipe.load_lora_weights`）加载。

## 7. 行动目录（按影响排序）

| 行动 | 影响 | 方式 |
|------|------|------|
| P0 组件下载 + VAE 配置核实 | 阻塞一切 | 尽早启动下载（后台），并行核实 |
| P2 适配器 + 速度/时间约定 | 训练正确性核心 | 先写符号校验脚本再实现 |
| P3 flow_matching data_ward | 训练正确性 | 一行改动 + 符号测试 |
| P1 视频/音频数据管线 | 训练可开始的前提 | 复用现有缓存框架，最小 schema 扩展 |
| D8 验证解码钩子 | 质量反馈闭环 | 先 fallback 中间帧，再 mux 音频 |
| P4 编排器冒烟 | 日常使用 | 复用现有任务机制 |
| Block swap / 量化调优 | 显存（单卡可行与否） | OOM 后按 D2 决策 |

## 8. 约束

1. **不改 orchestrator 主体代码**（除非 gpu_guard `min_free_mb` 小改获准）——
   任务机制已支持 H3 所需（多卡 DDP、心跳、resume）。
2. **diffusers 版本锁定在 pr-14355**；不得在未通知的情况下把
   `E:\diffusers` 切回 main（会丢 MiniMaxH3 组件）。
3. **不建虚拟环境**（REQUIREMENTS.md 硬性规定），新依赖直接进系统 Python。
4. **验证不造假**：G3/G4 必须真实跑出数字；OOM/形状错误记录到
   `progress/012` 而不是静默改测试。
5. **不下载全量 210GB 仓库**：只拉组件子目录（磁盘与流量约束）。
6. 每个里程碑后提交一次，commit 信息含 gate 编号（如 `[G3] minimax_h3 adapter dry-run`）。

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| PR #14355 未合并且 API 变动（force-push） | 组件签名漂移 | 锁定本地分支 + 记录 commit；升级前跑 G3 符号校验 |
| 61.7GB transformer 显存压力 | 单卡 80GB 训练不可行 | NF4/int8 量化（预计 16–25GB）；block swap；2×80GB DDP |
| 视频 VAE 精确 latent 帧数公式未核实 | bucket/帧对齐错误 | P0 从 checkpoint config 核实并写入文档；过拟合测试兜底 |
| 文本行 timestep 约定不明确 | 文本条件弱化 | P2 对照 PR `build_row_timesteps`；先用 t=1 假设 + 消融 |
| 音频 VAE 单声道/立体声处理错误 | 音频质量/对齐错误 | 严格按 PR channel-major 约定；验证产出 mp4 试听 |
| LoRA 目标模块命名不匹配 | PEFT 报错 | P2 用 `named_modules()` 实测后固化清单 |
| 模型仓库下载中断/磁盘不足 | 阻塞 P0 | 组件式 include 下载 + 断点续传（hf 默认），预留 150GB 磁盘 |
