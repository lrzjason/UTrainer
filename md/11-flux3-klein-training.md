# FLUX.3 klein 训练支持（flux3_klein）

> 状态：as-built（2026-10-16）。描述 `UnifiedTrainer` 中 `flux3_klein` 适配器与
> vendored transformer 的设计、与 diffusers 参考实现的对齐关系，以及验收方法。
> 参考源：`E:\diffusers` @ `7ecb1e1b4`（"Flux3 image (#15046)"）。

## 0. 交付物

| 文件 | 说明 |
|------|------|
| `UnifiedTrainer/models/flux3_klein/transformer_flux3.py` | diffusers `transformer_flux3.py` 的**独立训练副本**（自实现，不 import diffusers 的 Flux3 模型类），外加 `BlockSwapFlux3Transformer2DModel` |
| `UnifiedTrainer/models/flux3_klein/__init__.py` | `@ModelRegistry.register("flux3_klein")` 适配器 |
| `UnifiedTrainer/configs/flux3_klein_lora_example.json` | LoRA 示例配置 |
| `UnifiedTrainer/networks/lokr_module.py` | `_FLUX3_KLEIN_PATTERNS` + `_MODEL_PATTERNS["flux3_klein"]` |
| `tests/test_flux3_klein_adapter.py` | 38 项无权重测试 |

> 与 ai-toolkit（Ostris AI Toolkit）flux3 实现的逐项对照、以及对其中可用部分的
> 复制落点，见 `md/12-aitoolkit-flux3-compare.md`（复制产物在
> `UnifiedTrainer/vendor/flux3_aitoolkit/`，含原作者与上游链接）。

## 1. 架构对齐（vs diffusers 参考实现）

### 1.1 三模态 MMDiT

`Flux3Transformer2DModel` 用**一条模型 + 三种 token 模态**：

| 模态 | forward 参数 | 时间步 | RoPE `l` |
|------|--------------|--------|----------|
| 目标图（带噪） | `hidden_states` | 真实 flow timestep | `0` |
| 参考图（干净） | `image_cond_hidden_states` + `image_cond_ids` | **0** | `-(1+i)` |
| 文本 | `encoder_hidden_states` + `txt_ids` | **0** | `0`（仅 `l` 轴递增） |

执行顺序：各模态先跑自己的 per-mode blocks（`txt_mode_blocks` /
`mode_blocks["image"]` / `mode_blocks["image_cond"]`，**只在模态内部 self-attention**），
随后 `[txt, image, image_cond]` 拼接跑 joint `single_transformer_blocks`，
最后只取 image token 过 `norm_out` / `proj_out`。

### 1.2 关键超参（klein checkpoint）

| 参数 | 值 |
|------|-----|
| `in_channels` / `cond_in_channels` | 128 / 128 |
| `num_layers` / `num_single_layers` | 5 / 28 |
| `attention_head_dim` × `num_attention_heads` | 128 × 24 → inner 3072 |
| `joint_attention_dim` | 20480（8 tap × 2560） |
| `axes_dims_rope` / `rope_theta` | `(32,32,32,32)` / 10000 |

### 1.3 与 FLUX.2 klein 的三个关键差异

1. **参考图不是"额外目标 token"**，而是独立模态（`image_cond_*`），时间步恒为 0，
   RoPE `l` 为负。FLUX.2 klein 则把参考图当序列里的普通 token（`l` 轴正数分隔）。
2. **没有 `scaling_factor` / `shift_factor`**：改用 VAE 的 **BatchNorm 统计量**
   `(x - bn.running_mean) / sqrt(bn.running_var + bn.batch_norm_eps)`，
   且归一化发生在 **2×2 patchify 之后**（packed 128 通道）。
3. **文本编码器是 Qwen3-VL**（非 Qwen3 causal LM），且 transformer
   **不接收 text attention mask** —— padding 会被真实 attend，
   因此文本窗口长度是模型配方的一部分。

## 2. 数据流

### 2.1 latent 空间（重要）

```
原图 (B,3,H,W)
  └─ vae.encode().latent_dist.mode()      → (B, 32, H/8,  W/8)     原始 VAE latent
       └─ _patchify_latents()             → (B, 128, H/16, W/16)   2×2 折进通道
            └─ BatchNorm 归一化            → (B, 128, H/16, W/16)   ← **缓存存这个**
```

`encode_image()` 完成全部三步，所以**缓存里存的是 transformer 空间 latent**，
`latent_channels = 32 * 2 * 2 = 128`。这样做使重活（patchify + BN）只在建缓存时
算一次，训练每步只做 tokenize；`decode_latent()` 是严格逆运算
（BN 反归一化 → unpatchify → `vae.decode`）。

> ⚠️ `bucket_divisibility = vae_scale_factor * patch_size = 8 * 2 = 16`，
> 所有 bucket 宽高必须是 16 的倍数。`resolution_config` 提供的桶已全部 16 对齐。

### 2.2 文本编码

对齐 `Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds`：

1. `apply_chat_template(..., add_generation_prompt=True)`（经 processor/tokenizer）；
2. 量出真实 token 数，**向上取整到 `pad_multiple`（默认 80）的倍数**，上限
   `max_sequence_length`（默认 8192）；
3. 右 padding 到该窗口；
4. 显式传 `position_ids`（padding 位置固定为 1 —— 不同 Qwen3-VL 版本对 padding
   位置 id 的处理不一致，而 FLUX.3 会 attend padding，必须显式固定）；
5. `hidden_states[k] for k in (4,8,12,16,20,24,28,32)` 沿通道维堆叠 → 20480。

### 2.3 RoPE 位置 id

4 轴 `(t, h, w, l)`，均为 `(seq_len, 4)`（未 batch，全 batch 共享）：

- 文本：`(0, 0, 0, i)`
- 图像：`t=0`，`h/w` 乘以 `image_rope_scale=0.5`（模型训练时的缩放因子），
  `l = 0`（目标）/ `l = -(1+i)`（第 i 张参考图）

## 3. flow_map（蒸馏 checkpoint）

`flux-3-image-klein`（16 步蒸馏）的 transformer 是 flow map，forward **强制要求**
`target_timestep`；`flux-3-image-klein-base`（CFG，guidance 3.5）则要求**不得传**。

适配器在 `load_transformer()` 里从 checkpoint config 自动读取（`flow_map` 字段），
无需手工配置；显式配置 `"flow_map": true/false` 可覆盖自动识别。

训练时 `target_timestep = 0`：采样到的 sigma 即当前步，目标是干净样本，
即"一步跳到 x0"——这正是 flow-map 的训练目标。

## 4. 配置

```json
{
    "model": "flux3_klein",
    "model_path": "E:/hf_models/flux-3-image-klein",
    "vae_path": "E:/hf_models/flux-3-image-klein/vae",
    "text_encoder_path": "E:/hf_models/flux-3-image-klein/text_encoder",
    "tokenizer_path": "E:/hf_models/flux-3-image-klein/tokenizer",
    "processor_path": "E:/hf_models/flux-3-image-klein/processor",

    "max_sequence_length": 8192,
    "pad_multiple": 80,
    "image_rope_scale": 0.5
}
```

| key | 默认 | 说明 |
|-----|------|------|
| `flow_map` | 自动（读 checkpoint） | 是否 flow-map 蒸馏模型；显式设置可覆盖 |
| `max_sequence_length` | 8192 | 文本窗口上限 |
| `pad_multiple` | 80 | 文本窗口向上取整粒度；**会改变模型输入，勿随意改** |
| `image_rope_scale` | 0.5 | 图像 H/W RoPE 缩放；**训练配方的一部分** |
| `text_encoder_select_layers` | `(4,8,12,16,20,24,28,32)` | 层 tap；长度 × 2560 必须等于 `joint_attention_dim` |
| `timestep_shift_mode` | `"flow_shift"` | 训练 sigma 采样模式：`"flow_shift"`（默认）/ `"dynamic_shift"` / `"sigma"`（均匀） |
| `flow_shift` | 16.0 | `flow_shift` 模式的 s，`mu = ln s`。**16 = 官方工作流推理 alpha**（mean σ≈0.92，72% 抽样 ≥0.9）；2.5 = krea2、6.93 = FLUX.3 参考、1.0 = 裸 logit-normal |
| `sigmoid_scale` | 1.0 | logit-normal 的标准差 |
| `scheduler_shift` | 16.0 | 推理/验证 timeshift 的 alpha（官方工作流默认 16；6.93 = 模型参考），见 `scheduler.py` |

参考图训练（image-conditioned）沿用既有 `batch_configs[].reference_config` 约定，
无需新增字段。

## 5. LoRA / LoKr 目标模块

- **LoRA**：`train.py` 用子串匹配，默认 `["to_q","to_k","to_v","to_out","..."]`
  已覆盖三个模态的所有 attention 线性层。
- **LoKr**：预设 `_MODEL_PATTERNS["flux3_klein"]`（27 条 pattern），覆盖
  `txt_mode_blocks.*` / `mode_blocks.image.*` / `mode_blocks.image_cond.*` /
  `single_transformer_blocks.*` 的 `attn.to_{q,k,v,out.0}` 与 `ff.linear_{in,out}`，
  外加 `x_embedders.*` / `context_embedder` / `proj_out`。
  用 `training.lokr_model_type = "flux3_klein"` 选取；
  `lokr_target_modules = null` 仍表示"所有 nn.Linear"。

## 6. Block swap

`BlockSwapFlux3Transformer2DModel` 提供 T2ITrainer 兼容的三件套
（`enable_block_swap` / `move_to_device_except_swap_blocks` /
`prepare_block_swap_before_forward`），即使 `blocks_to_swap = 0` 也应使用本类。

offload 顺序（`_flat_swap_blocks()`，与 forward 循环一致）：

```
[txt_mode_blocks | mode_blocks.image | mode_blocks.image_cond | single_transformer_blocks]
```

`ModelOffloader` 交换的是该列表**末尾** `blocks_to_swap` 项，
即 28 层 joint single blocks（参数主体）优先被 offload。

## 7. 验收

```bash
cd <repo>
python -m pytest tests/test_flux3_klein_adapter.py -q     # 38 passed
```

覆盖：注册/规格、patchify↔unpatchify 往返、RoPE id（含参考图负 `l` 与 0.5 缩放）、
`prepare_model_input`（T2I / 参考图 / flow_map）、`unpack_prediction`、
encoding 往返（fake VAE）、真实维度 forward/backward（三模态梯度）、
block-swap 顺序、config 持久化、`flow_map` 自动识别与覆盖。

另有端到端形状校验（production 超参 + 极小层数）：输出的
`unpacked` 形状与输入 latent 一致，三个 embedder 均获得非零梯度。

## 8. 环境备注

本机 `E:\diffusers` 为 editable install。2026-10-16 发现
`huggingface_hub==1.22.0` 会导致 `import diffusers` 失败
（`cannot import name 'resolve_revision'`）；该 checkout 需要
`huggingface_hub>=1.3.0`。已装 `1.33.0` 解决。
若再次出现 diffusers 导入失败，先检查 `huggingface_hub` 版本。

> `minimax_h3` 适配器仍依赖 diffusers PR #14355 分支的
> `diffusers.modular_pipelines.minimax_h3.packing`，当前 checkout 无该模块，
> 故 `--list-models` 扫描时会跳过它。此为既有状态，与本次交付无关。
