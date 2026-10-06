# 07 — MiniMax-H3 训练接入（as-built 设计文档）

> 状态：as-built（2026-08-05）。本文档描述 `UnifiedTrainer` 中
> `minimax_h3` 适配器与统一媒体数据管线的**最终实现形态**，供后续开发/
> 排障对照代码核对。
> **诚实声明**：P0–P3 代码与文档已交付，全部无权重验证通过；但
> **G2–G7 运行时验收（缓存构建、dry-run、过拟合、视频缓存、视频过拟合、
> 编排器全链路）需要模型权重，本机未下载权重，一律未运行**。
> 实现计划见 [minimaxh3_implementation.md](minimaxh3_implementation.md)，
> 阶段小结见 [progress/012-minimaxh3.md](../progress/012-minimaxh3.md)。

## 1. 功能范围

- **图像对训练（里程碑 1）**：源图（S）作关键帧条件行，目标图（P）作目标
  视频行（1 帧视频即图像），配文本 caption。
- **视频训练（里程碑 2）**：同一条统一媒体管线，仅 T>1 的视频分支被激活；
  本期视频 = 无声视频建模（音频行整体省略，见 §7）。
- **音频训练：延后**（D6 正式决定，不是临时 hack）。
- 不做：ref2va（omni 参考）、CFG 训练、完整微调、distill、多机并行。

## 2. 统一媒体数据管线（D3：图像/视频共用、一次建成）

里程碑 1 即落地全部数据层，里程碑 2 只激活视频解码，**不重建管线**。

| 层 | 实现 | 说明 |
|----|------|------|
| schema | `data/config_schema.py` | `ImageConfig.media: "image"\|"video"`（默认 image）+ `DatasetConfig.video_frames/video_fps` 一次加入；validate 规则（video 键只能被同名媒体 target 引用、17n+5 对齐）已落地 |
| 媒体加载 | `data/video_utils.py` | `load_image_frames(path, size) -> (1,C,1,H,W)`（PIL → 5D）；`load_video_frames(path, num_frames, fps=24) -> (1,C,T,H,W)`（PyAV 逐帧解码、24fps 均匀抽帧、等比缩放+中心裁剪、17n+5 对齐、5–15s 时长校验）；`snap_frames(n)` / `video_latent_num_frames(n)`（包装 PR packing 函数） |
| 缓存 | `cache_builder.py` 单一媒体分发 | 按 `media` 字段分发 → `adapter.encode_video(vae, frames)` → **统一 (C,T,H,W) npz**（图像 = (C,1,H,W)，B 维折叠进 index 样本维度）；每样本 JSON 记录 media/num_frames |
| dataset/collate/bucket | `dataset.py` | 5D 堆叠与 bucket 逻辑在 P1 以图像样本 (C,1,H,W) 打通并硬化；P2 视频只是 T 变大，代码零改动 |
| 音频 | — | `media="audio"` 不做；schema validate 直接拒绝（D3/D6） |

收益：图像与视频共用同一缓存格式/加载路径/适配器钩子；图像缓存与视频
缓存同格式、可互相复用。

## 3. MiniMaxH3Adapter 钩子清单（`models/minimax_h3/__init__.py`）

注册名 `minimax_h3`（`@ModelRegistry.register`）。协议继承
`BaseModelAdapter`。

| 钩子 | 实现要点 |
|------|---------|
| `latent_channels = 24` / `vae_scale_factor = 8` / `patch_size = 2`（spatial；temporal patch=1 在 prepare_model_input 内处理）/ `embedding_dim = 5120` / `supports_video = True` / `supports_image_conditioning = True` | 架构规格；resolution_config 768p 档（短边 768、面积 ≤768×1344、16 整除） |
| `load_transformer` | `MiniMaxH3Transformer3DModel.from_pretrained(path/transformer, torch_dtype=bf16)`（`_keep_in_fp32_modules` 由 diffusers 自动处理） |
| `load_vae` | `AutoencoderKLMiniMaxH3.from_pretrained(path/vae)` |
| `load_scheduler` | `MiniMaxH3Scheduler(shift=12)`（返回 data-ward 调度器） |
| `load_text_encoder` / `load_tokenizer` / `load_processor` | Qwen3VLForConditionalGeneration + krea2 的 rope_scaling 补丁 / Qwen2TokenizerFast fallback（tokenizer_config 的 `extra_special_tokens` list 补丁）/ Qwen3VLProcessor 手动构造（无 preprocessor_config.json） |
| `encode_image` | 单帧关键帧路径：统一 (1,3,1,H,W) [0,1] → ImageNet 归一化 → **`vae._encode_clip`**（PR 关键帧同款，1 帧 → 1 latent 帧；禁止走 `_encode`）→ fp16 舍入 → 归一化 → `(24,1,h,w)` |
| `encode_video` | T==1 委托 `encode_image`；T>1 走 `vae._encode`（17n+5 分块 + token_drop=3，与 PR 一致）→ `(24,T_lat,h,w)` |
| `encode_text` | 带关键帧图时构造 PR presentation（`"<Picture i>: "` + vision block，无 chat template）；Qwen3VL forward `output_hidden_states=True` 取 `hidden_states[50]`（不走 LM head；注意 accelerate offload `hook.pre_forward`）；返回 `prompt_embed (L,5120)` + `prompt_embeds_mask` + `text_token_tags` |
| `decode_latent` | T==1：`_decode_clip` 单帧解码（重复补 5 latent → 20 像素帧 → 裁 frame_pre_padding 前 3 帧取首帧 → [-1,1]）；T>1：`vae._decode` 分块解码（fp32，非 PR fp16 autocast，见 P2 注释） |
| `decode_validation_video` | **仅 MiniMaxH3Adapter 定义**（base 刻意不定义，见下）；`decode_latent`(T>1) → PyAV 写静音 h264 mp4（24fps、yuv420p、无音频轨道），返回路径 list；`prefix` 参数防跨样本/跨 epoch 覆盖 |
| `encode_text_accepts_image = True` | 缓存 caption 阶段把关键帧 PIL 作为 `condition_image` 传入 |
| `prepare_model_input` | 见 §4/§5/§6 |
| `unpack_prediction` | `return_dict=False` 取 `(video, audio)` → 按 `_packed_geometry` 切 condition/target 行 → `unpatchify_video_tokens` 回 5D → 返回 `[video_pred]`（与 noises 对齐） |
| `compute_target` | `learning_target - noise`（data-ward；引擎 flow_matching 不直接调它，保持协议一致性） |
| `sample_timesteps` | logit-normal（`timestep_mu` 可配，默认 0.0）；**mu 守卫**：σ = `sigmoid(u).clamp(1e-5, 1-1e-5)`；单 σ 广播到 batch（packed 布局 batch 统一） |

**base.py 可选钩子约定**（P1.4 文档化）：

- `encode_video` / `velocity_sign` / `compute_x0_hat` 在 `BaseModelAdapter`
  上有默认实现（图像适配器行为不变）；
- `decode_validation_video` **刻意不定义为 base 方法**：引擎验证循环用
  `hasattr(self.adapter, "decode_validation_video")` + `latent.ndim == 5 and
  latent.shape[2] > 1` 分发，base 定义会翻转所有图像适配器的 hasattr，
  破坏 `decode_latent` → PIL 的回退契约（详见 base.py 类 docstring）。

## 4. data-ward 速度约定（D5）

- MiniMax-H3 调度器是 **data-ward**：`v = x0 − x_t`（标准流匹配是
  `v_s = noise − x0`）。
- `velocity_sign` 属性：`BaseModelAdapter` 默认 `"standard"`，
  MiniMaxH3Adapter 声明 `"data_ward"`。
- `losses/flow_matching.py`：`target = learning_target - noise`（data_ward）
  或 `noise - learning_target`（standard）；未知值直接 `ValueError` 拒绝
  （防止符号写反静默训错方向）。
- `engine/noise_selector.py` `_eval_velocity_mse`：同样按 `velocity_sign`
  分发（P1-r 修复，standard 逐位一致 / data_ward 生效 / 未知拒绝）。
- `compute_x0_hat`（base）：`data_ward → noise + velocity`；
  `standard → noise - velocity`——trainer 不得写死公式（P1-r 符号盲区修复：
  M1 接入两处）。
- `compute_target(noise, x0)`：返回 `x0 - noise`。

## 5. 时间约定（D6：σ → t = 1−σ）

- 引擎插值 `(1-σ)·x0 + σ·noise` 与 H3 前向 `x_t = t·x0 + (1−t)·noise`
  （t=1−σ）**逐位等价**，引擎无需改。
- `prepare_model_input`：目标行 `t_target = 1 - sigma`；关键帧条件行钉在
  `t_cond = 0.999`；`build_row_timesteps` 生成 `timestep`（去重、排序）与
  `timestep_indices`（文本行继承目标行时间步——已按 PR 源码核实）。
- 每前向最多 2 个去重 timestep（目标 t 与关键帧 t）。

## 6. 关键帧条件行（图像对训练核心）

- 条件行**不是干净源图**：按 PR `scale_noise(clean, 0.999)` 混噪 =
  `0.999·x0 + 0.001·noise`（`MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999`，
  packing.py L82-84）——引擎 σ 术语即 **σ_cond = 0.001**。
  即计划 §5 表述：**PR `noise_aug=0.999` → 引擎 `σ_cond=0.001`、`t_cond=0.999`**。
- 钉在 `t_cond = 0.999`（`max(t, 0.999)`，before_denoise.py L417，每个去噪
  步不变）；行锚定 `"first"`。
- **混噪在 float32 下执行**（P1 审查 M2 修复）：0.001·noise 项幅度 ~0.001
  远低于 bf16 ulp（~0.0078@1.0），若按 bf16 混合会被舍入丢弃、条件行退化为
  确定性 0.999·x0；f32 混合让 seed/σ 派生的噪声增强语义真实生效（packed
  序列进 transformer 时仍转 bf16，与 PR 行为一致）。
- 噪声每步重抽、generator 可复现：`_condition_noise_generator(sigmas, ...)`
  从 `_keyframe_noise_seed + round(σ·1e6)` 派生 seed——同 σ 同条件行
  （CFG 一致），不同训练 σ 抽新噪声。
- 文本侧 presentation：逐字 prompt 前加 `"<Picture i>: "` 标签 + vision
  block（`<|vision_start|>` + 每 vision patch 一个 `<|image_pad|>` +
  `<|vision_end|>`），**无 chat template**，vision block 行 tag=0（视频）
  而非 1（文本）。

## 7. 音频行省略依据（D6）

- Packed 布局 `[ text(L) | 关键帧条件(C) | 目标音频(A) | 目标视频(V) ]` 中
  音频块可整体省略：`num_audio_latents=0` → 无 A 块；transformer 内
  `audio_proj_in(空)` + `index_copy` 空索引均为 no-op（已逐行核实）。
- 适配器传 `audio_hidden_states = torch.empty(B, 0, 32)`——`audio_proj_in`
  是 `nn.Linear(audio_in_channels=32, hidden=5376)`，空行仍校验特征维，
  所以空张量带 **32** 特征（不是 5376；P1 阶段曾写错，已修复并注释）。
- 不下载 audio_vae/audio_scheduler，不实现音频数据/损失/解码代码。

## 8. 验证循环调度器语义（D8）

- `load_scheduler` 返回 `MiniMaxH3Scheduler(shift=12)`——与
  `FlowMatchEulerDiscreteScheduler` 的差异：① 速度方向 data-ward
  （`x0 = x_t + σ·v`）；② timesteps 用 `t = 1−σ ∈ [0,1]`，t=1 干净，
  不缩放；③ σ 网格从 `linspace(1,0,N)` 起（终点 0 计入步数）。
- 引擎验证循环迭代 `scheduler.timesteps`、取 `scheduler.sigmas[i]`、调
  `scheduler.step(v, t, l)`——MiniMaxH3Scheduler 语义完全匹配，**引擎循环
  无需改**。
- `val_guidance_scale` 必须 = 1（H3 无 CFG）。
- 解码按能力分发：T==1 → `decode_latent` → PIL；T>1（且有
  `decode_validation_video`）→ 静音 mp4。
- 采样：`sample_timesteps` logit-normal（`timestep_mu` 可配，mu 守卫
  clamp 到 [1e-5, 1−1e-5]）；返回引擎 σ 语义（t = 1−σ 在
  prepare_model_input 内构造）。

## 9. LoRA 目标模块清单（D7/D2）

- H3 attention：`to_q / to_k / to_v / to_out`；
- FeedForward（diffusers **swiglu** FF 命名）：`ff.net.0.proj` / `ff.net.2`
  （**不是** `ff.gate/ff.up/ff.down`——P1 用 `named_modules()` 实测后固化）；
- 固化清单：`["to_q","to_k","to_v","to_out","ff.net.0.proj","ff.net.2"]`；
- `proj_in / audio_proj_in / context_embedder / proj_out / audio_proj_out /
  time_embedder` 保持冻结（fp32 混合精度契约；注：`adaln_proj / norm_out`
  是 bf16 的 AdaLN 调制头，也在冻结清单中但**不属于** fp32 契约）；
- 默认 rank 8–16（61.7GB 模型 + 长序列，显存优先）。

## 10. 配置模板

### 10.1 图像对 smoke — `configs/minimax_h3_image_smoke.json`

- `model: minimax_h3`；S/P 图像对：`S`（suffix `_s`，media image，关键帧
  条件）、`P`（suffix `_t`，media image，目标）；`target_configs: T→P`、
  `reference_configs: S`（from_same_name）、`caption_configs: C→S`；
- 训练：batch_size 1、lr 1e-4、max_steps 100、lora_rank 8、
  mixed_precision bf16；`losses: [flow_matching]`；
- 输出 `output/h3_image_smoke`。

### 10.2 生产图像对 — `configs/minimax_h3_train.json`

- 同 10.1 的 S/P/C wiring；`_comment` 记录 D2/D8/D9 决策；
- `training`: `max_steps: -1`、`num_epochs: 10`、`optimizer: adamw8bit`、
  `gradient_checkpointing: true`、`weight_dtype: bf16`、
  `mixed_precision: bf16`、**`quantize: nf4`**、`gpus: 1`、
  `multi_gpu: "reserve"`（编排器约定：reserve=单进程绑卡，ddp 才走
  accelerate launch）；
- `lora_target_modules`: §9 固化清单；
- `validation`: `generate_images: true`、`num_inference_steps: 50`、
  **`guidance_scale: 1.0`**（D8 无 CFG）。

### 10.3 视频 smoke — `configs/minimax_h3_video_smoke.json`

- `V`（suffix `.mp4`，media video）；`video_frames: 124`（17×7+5）→
  37 latent 帧；
- `reference_configs: {"none": []}` 合法——视频训练默认**无关键帧**
  （纯 t2v 布局 `keyframe_anchors=()`）；可选 i2v 扩展需加 reference_configs
  + `keyframe_anchors=("first",)`；
- 音频 target 一律不配（D6）。

## 11. 已知限制 / 延后项

- **G2–G7 未运行**（需模型权重）：G2 图像对缓存、G3 dry-run + 符号校验、
  G4 图像对过拟合、G5 视频缓存、G6 视频过拟合、G7 编排器全链路。
- P4 可选增强延后：与 PR 推理 parity、量化（NF4/int8）训练冒烟、多卡 DDP
  冒烟、（OOM 时才做）vendor transformer + BlockSwap subclass。
- 遗留（范围外）：trainer `_get_reference_latent` / `_decode_reference_images`
  对视频 reference 走 try/except → None（不崩溃，建议后续清理）。
- 同批混 T（图像+视频混合 batch）延后；每 forward 需 batch 统一 packed
  布局与统一 σ。
