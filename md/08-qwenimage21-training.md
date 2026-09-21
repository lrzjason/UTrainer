# 08 — Qwen-Image 2.1（transformer_qwenimage21）LoRA/LoKr 训练接入（as-built 设计文档）

> 状态：as-built（2026-09-20）。描述 `UnifiedTrainer` 中 `qwen_image21` 适配器与
> 独立 transformer 副本的最终实现形态，供后续开发/排障对照代码核对。
> **诚实声明**：代码已交付，且通过了 stub 化端到端契约验证（34 项，见 §7），
> 但 **G1–G5 运行时验收（缓存构建、dry-run、过拟合、验证采样）需要模型权重，
> 权重尚未发布，一律未运行**。所有训练相关数值约定均以
> `diffusers` 官方 `QwenImage21Pipeline` 推理实现为准（逐条对齐）。

## 1. 功能范围

- **T2I LoRA 训练**（milestone 1，完整）：PEFT `add_adapter` 路径，注册名
  `qwen_image21`，与全部既有引擎/缓存/数据集组件即插即用。
- **T2I LoKr 训练**（milestone 1，完整）：`networks/lokr_module.py` 新增
  `qwen21` 目标层预设。
- **图像条件（edit）路径**（milestone 2，已实现、带硬校验）：条件图 latent
  进入联合序列、`img_shapes` 条件在前目标在后、槽位严格校验
  `ref_tokens == 4 × VLM 槽位数`。
- **不做**：多目标序列（2.1 架构定义每序列恰好一个 noisy 目标块，
  `build_token_metadata` 只把最后一个块标为目标；收到多目标直接
  `NotImplementedError`）、视频、完整微调、CFG 蒸馏。

> 相关：对 guidance-distilled 检查点（"cfg distilled model"）做微调时，
> 使用 `md/09-corrected-flow-matching-guide.md` 的 `guide_flow_matching`
> loss（DC-Gen Eq. 10）替代标准 `flow_matching`——那是修正**训练目标**，
> 不是蒸馏，与本文档的适配器范围正交。

## 2. 交付物

| 文件 | 说明 |
|------|------|
| `UnifiedTrainer/models/qwen_image21/transformer_qwenimage21.py` | diffusers `transformer_qwenimage21.py` 的**独立训练副本**（1329 行，19/19 模型专属 def 与源 AST 级一致）。不 `import diffusers` 的任何 `QwenImage21*` 类；仅引用通用基础设施（ModelMixin/ConfigMixin/RMSNorm/TimestepEmbedding/dispatch_attention_fn 等，与 krea2/qwen_image 副本同约定） |
| `UnifiedTrainer/models/qwen_image21/__init__.py` | `@ModelRegistry.register("qwen_image21")` 适配器（944 行）。diffusers/transformers 全部方法级惰性导入（diffusers 顶层不可用的机器上仍可注册） |
| `UnifiedTrainer/networks/lokr_module.py` | 新增 `_QWEN21_PATTERNS` + `_MODEL_PATTERNS["qwen21"]` |
| `UnifiedTrainer/configs/qwen_image21_lora_example.json` | LoRA 示例（rank16/alpha16，targets `to_q/to_k/to_v/to_out`） |
| `UnifiedTrainer/configs/qwen_image21_lokr_example.json` | LoKr 示例（`lokr_model_type: "qwen21"`，rank64） |

## 3. 架构规格（全部来自 diffusers 源，勿凭记忆改动）

| 项 | 值 |
|----|-----|
| VAE | `diffusers.AutoencoderKLQwenImage21`（通用基础设施导入）：z_dim=64 通道、16× 空间压缩、`latents_mean/latents_std`（64 值，取自 vae.config）归一化、5D (B,C,T,H,W) |
| latent | pixel/16；**不 patchify**（`patch_size=1`），token=latent 像素格点，每 token 覆盖 16×16 像素 |
| 像素对齐 | 必须被 32 整除（pipeline `vae_scale_factor*2`；`bucket_divisibility=32` 覆写） |
| Transformer | 单流 32 块；inner_dim=4096（32 头×128）；context_in_dim=4096（Qwen3-VL）；mlp_ratio=3（SwiGLU）；axes_dims_rope=(16,56,56)；eps=1e-6；**全块共享一个 `modulation`**（SiLU+Linear→4×dim，无 per-block 调制参数） |
| 注意力 | block-causal：`(q_idx >= kv_idx) or same_image_block`；默认 `QwenImage21AttnProcessor`（精确分段 SDPA，无需 compile/flex）＝训练路径；`QwenImage21FlexAttnProcessor` 仅推理/编译用 |
| `causal_condition=True` | 额外 t=0 调制行：文本与条件图 token 全程用 t=0 调制（`_select_modulation_rows` + `target_token_mask`），目标块用训练 timestep |
| timestep | **[0,1]** 直接传入（pipeline `t/1000`；`QwenImage21TemporalTimesteps` 内部 ×1000） |
| 调度器 | FlowMatchEulerDiscreteScheduler：`use_dynamic_shifting=True`，seq 256→4096 对应 shift 0.5→1.15，num_train_timesteps=1000 |
| 文本编码器 | `Qwen3VLForConditionalGeneration` + `Qwen3VLProcessor`（本机 transformers 5.2.0 实测可用）；hidden states 取最后一层 decoder **final RMSNorm 之前**（forward hook 挂 `text_model.norm` 返回其输入；transformers 5.x 起 `hidden_states[-1]` 已被归一化） |
| 联合序列 | `[文本 token（VLM 图像槽 1→4 展开）, 目标槽（每槽=2×2 latent 组=32×32 像素）]`；`_IMG_TOKENS_PER_SLOT=4` |

## 4. 关键实现约定（与代码逐条对应）

### 4.1 编码缓存（`encode_text`，pipeline `_get_qwen_prompt_embeds` 逐条对齐）

- 原始模板字符串（**不用** `apply_chat_template` 拼 prompt）：t2i 模板
  `system\nComprehend and analyze the provided prompt.<|im_end|>...`；ti2i 模板在
  user 内容前插 `<image{i}><|vision_start|><|image_pad|><|vision_end|>` 占位。
- 空prompt → `" "`（Qwen 无 BOS）；left padding；多模态路径**不截断**，纯文本
  按 `training.text_max_length`（默认 1024）+ 模板 headroom 截断。
- `_drop_idx` = system 消息 token 数（`processor.apply_chat_template(sys_message,
  tokenize=True, return_dict=False)`，缓存一次）。
- 返回 dict：`prompt_embed (L,4096)`、`prompt_embeds_mask (L,) bool 全 True`
  （masked 抽取后无 padding，pipeline 同款）、`image_token_mask (L,) bool`、
  **`img_mask (L,) bool 同一张量双键**。
  双键原因：`cache_builder._encode_caption` 会 `pop("image_token_mask")`，npz 里
  只留 `img_mask`；多模态槽位掩码因此能存活到训练时。`prepare_model_input` 按
  `img_mask` → `image_token_mask` → 全 False（caption dropout / 纯文本）顺序读取。
- RGBA 参考图对视觉编码器平铺白底（pipeline 同款）；VAE 仍读四通道（本 trainer
  参考图经 bucket 裁剪，RGB）。

### 4.2 `prepare_model_input`（联合序列装配）

- 单目标：`noise` list 长度≠1 → `NotImplementedError`（见 §1）。
- 打包 = 纯空间展平 `view(B,C,H*W).transpose(1,2)`（无 patchify，pipeline
  `_pack_latents`）；H/W 必须为偶数（2×2 分组），否则 ValueError。
- 条件（edit）路径由**缓存的槽位掩码驱动**，不由 `batch["latents"]` 驱动——
  caption dropout 换空嵌入（无图像槽）时参考 latent 自动被排除，与
  CFG 无条件分支天然一致：
  - 批内槽位数必须一致（transformer 只读 img_mask 行 0），否则 ValueError；
  - `num_cond > 0` 时必须有参考 latent，且严格校验
    `sum(ref_tokens) == 4 × num_cond`，失败给出可操作提示（文本编码器看到的
    参考图与 VAE 看到的必须同尺寸：配置 `reference_list.resize == 图像分辨率`）；
  - `hidden_states = cat([ref_seqs…, target_seq])`、`img_shapes = [(1,rh,rw)…,
    (1,H,W)]`（条件在前、目标最后——`build_token_metadata` 只标最后一块）。
- **目标槽只按目标 latent 追加**：`img_mask_full = cat([img_mask,
  ones(B, H*W//4)])`（pipeline `append_target_slots` 只对目标 noise 取 `//4`；
  实现文档里明确记录了这一处对早期计划的修正——按
  `(ref_tokens+H*W)//4` 会把条件槽重复计入，`build_token_metadata` 在 edit 路径
  直接崩溃）。
- 返回 forward kwargs：`hidden_states / timestep(=sigmas, [0,1]) /
  encoder_hidden_states / encoder_hidden_states_mask / img_shapes(元组列表,
  每样本一份) / img_mask / return_dict=False` —— 与 transformer forward 签名
  AST 级核对，键完全一致。

### 4.3 `unpack_prediction`

- 取联合输出**最后 H*W 行**（pipeline `noise_pred[:, -latents.size(1):]`），
  `transpose(1,2).reshape(B,64,H,W)`（纯展平的精确逆）。返回 `[tensor]`
  （trainer zip 契约）。

### 4.4 `sample_timesteps`

- `timestep_shift_mode`（config，默认 `"dynamic_shift"`）：μ = 线性插值
  0.5→1.15 over 256→4096 **图像 token 数 = H*W**（latent 尺寸由
  noise_selector 传入；缺失时回退 μ=0 logit-normal，inline 实现——基类只认
  `"sigma"`，不能走 super()）。
- `"flow_shift"`（固定 s，krea2 配方）与 `"sigma"`（均匀，musubi 配方）可选；
- 返回 `(sigmas*1000, sigmas)`。

### 4.5 BlockSwap / 梯度检查点

- `BlockSwapQwenImage21Transformer2DModel`：T2ITrainer class-2 约定
  （`blocks_to_swap` int 真值门控、`offloader_double`、
  `move_to_device_except_swap_blocks`、`prepare_block_swap_before_forward`），
  梯度检查点分支按源顺序透传全部 block kwargs
  （`modulation_mask→target_token_mask`、`block_segments→segments`、
  `block_key_valid→key_valid`…）。基类保留 class-1 `_offloader` 钩子（None 门控）。
- train.py 的 block-swap 接线是 hasattr 门控 → 零改动生效。

### 4.6 LoRA / LoKr / 检查点

- LoRA：PEFT 默认路径。示例 targets `to_q/to_k/to_v/to_out`（子串匹配命中
  `to_out.0`）；`norm_q/norm_k`（RMSNorm）天然不在目标内。每块 4 个 Linear ×
  32 块 = 128 层。
- LoKr：`lokr_model_type: "qwen21"` → 块内 attn 4 投影 + `img_mlp
  .proj/.gate_layer/.out`（SwiGLU 命名与 qwen v1 的 `ff.net` / krea2 的
  `ff.gate` 都不同，预设不可复用）。`img_in/modulation.1/norm_out.linear/
  proj_out/txt_in.*` 可通过显式 `lokr_target_modules` 选择性加入。
- 检查点：LoRA 走 `get_peft_model_state_dict`、LoKr 走 `lycoris_net.save_weights`
  —— 全部通用，无需改动。ComfyUI 副本走通用 krea2 风格键名替换
  （`transformer_blocks→blocks`、`to_q→wq`…）；**注意**：ComfyUI 对 2.1 的键名
  约定无法在本地核实（本地 ComfyUI 快照尚无 2.1 支持），权重发布后如需对齐
  再补映射。

## 5. 数据集接入（零 schema 改动）

动态配置数据集照常使用：`image_configs/target_configs/caption_configs/
bucket_configs` 不变；adapter 以 `resolution_config`（512/768/1024/1536/2048
五档 bucket，全部 32 整除，脚本断言 0 违规）、`bucket_divisibility=32`、
`latent_channels=64`、`embedding_dim=4096` 接入。latent 走统一 5D 缓存
（图像=(C,1,H,W)）；embedding 走 int8+scale npz（bool 掩码原样保留）。

## 6. 时间/方向约定

- 引擎插值 `(1-σ)·x0 + σ·noise`，标准速度 `v = noise - x0`
  （`velocity_sign="standard"`，`compute_target = noise - x0`，`compute_x0_hat`
  继承基类）。flow_matching loss 零改动。
- 验证生成：trainer 用 `patch_size` 计算 mu —— `patch_size=1` 时
  `total_image_seq_len = H*W` 正确（无需 trainer 改动）；调度器动态 shifting 由
  trainer 既有 `mu` 分支处理。

## 7. 验证（本机已完成 / 待权重）

**已完成（无权重、无 diffusers 运行时）：**

1. `py_compile`：transformer、adapter、lokr_module 全过。
2. AST 级对照：transformer 19/19 def 与 diffusers 源一致（基类差异仅为预设的
   offloader 增行）；forward 签名与 `prepare_model_input` 返回键互相核对。
3. **stub 化端到端冒烟（`.tmp/qwen21_smoke_test.py`，34/34 PASS）**：用最小
   diffusers 基础设施桩（真实 SDPA/真 RMSNorm 语义）加载真实 transformer +
   真实 adapter，跑通：`build_token_metadata` 块标号/目标掩码 → T2I
   prepare→forward→unpack 全链 → pack/unpack 精确互逆 → 联合序列梯度回传
   （img_in/modulation/blocks/proj_out 均有梯度）→ edit 路径（4 槽↔16 参考行、
   槽位数不一致即抛、joint 行数 (L−槽)+ref+target=26 精确吻合）→ 多目标拒绝 →
   三种 timestep 模式 → BlockSwap 表面 + KV-cache 签名。
4. registry 实测：`--list-models` 列出 `qwen_image21`（且在 diffusers 顶层导入
   损坏的机器上仍能注册——惰性导入纪律的直接收益）。
5. EmbeddingCache 往返：save→pop→npz→load 后 `img_mask` 存活且槽位数正确。

**待权重（G1–G5）：** 缓存构建 → dry-run → 单图过拟合 → 验证采样出图 →
ComfyUI 键名对齐核实。

## 8. 已知边界

- 多目标 batch_config 不支持（明确报错，见 §1）。
- edit 路径的槽位对齐依赖文本编码器与 VAE 看到同尺寸参考图；不满足时报错
  （不是静默错训练）。
- flex 注意力路径保留但训练不用；KV-cache 参数保留仅作推理形态兼容。
- `minimax_h3` 适配器在本机因 diffusers 顶层导入损坏而注册失败——既有问题，
  与本次交付无关（qwen_image21 不受影响）。
