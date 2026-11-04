# FLUX.3 klein：ai-toolkit 对照与本仓库落地（2026-10-16）

> 状态：comparison（2026-10-16）。对照对象：**AI Toolkit**（Ostris, LLC）
> `E:\UnifiedTrainer\ref\ai-toolkit` @ `12395a103fd0ca499c631dc6f460a8778e11490d`
> （2026-10-15，"Load comfy files if they exist in flux.3 image"），MIT。
> 本文档回答两个问题：(1) ai-toolkit 的 flux3 实现与本仓库
> `models/flux3_klein/` 有何异同；(2) 其中值得借鉴的部分以**复制 + 抬头署名**的方式
> 落进 UTrainer（**不 import** ai-toolkit）。
>
> 配套：`md/11-flux3-klein-training.md`（本仓库 as-built）。
> 复制产物：`UTrainer/UnifiedTrainer/vendor/flux3_aitoolkit/`（含 `ATTRIBUTION.md`）。
> 验证：`tests/test_flux3_aitoolkit_keymap.py`、`tests/test_flux3_aitoolkit_vendor.py`。

---

## 0. 结论速览

| # | 事项 | 结论 | 动作 |
|---|------|------|------|
| 1 | 三模态 MMDiT 结构、超参、RoPE 4 轴约定、`l=-1-i` 参考图、`image_rope_scale=0.5` | **完全一致**（逐值核对 + 单测钉死） | 不复制 |
| 2 | latent 空间：2×2 patchify 后接 VAE BatchNorm、`mode()` 编码、解码逆序、bucket 16 对齐 | **一致**（数值等价，实现位置不同） | 不复制 |
| 3 | flow target（`noise - clean`）、t=1→0、推理 timeshift | **官方工作流 α=16**（`mu = ln(16) = 2.77259`，见 §4 补记），模型参考 α=6.93；本仓库验证调度已改为 α=16 | 已改 `scheduler.py`，`scheduler_shift` 可回 6.93 |
| 4 | 训练 sigma 分布 | **有意不同**：上游默认 `timestep_type='sigmoid'` 走 logit-normal 且**忽略** scheduler 的 shift（≈ mu 0，mean σ 0.5）；本仓库默认 `flow_shift` **16**（mu = ln16 = 2.7726，mean σ 0.92，对齐官方推理 alpha）。要复刻上游型分布用 `flow_shift: 1.0` | 默认已改为 16（2026-10） |
| 5 | 输出 adaLN 的 `[shift, scale]` vs diffusers `[scale, shift]` | **硬差异**（三份独立实现互证）→ 转换时必须交换半区 | 已写入 `keymap.py` + 单测 |
| 6 | 原生（BFL/ComfyUI）单文件权重键名 | 本仓库**无法加载**（只吃 diffusers 目录） | 已复制/新写键映射、fused-qkv 切分、布局嗅探 |
| 7 | 参考图（image_cond）尺寸策略 | **配方分歧**：上游按目标面积匹配 + 16 对齐；本仓库按 dataset bucket 各自缩放 | 已复制 `recipe.fit_reference`，适配器尚未接线（见 §6） |
| 8 | 文本窗口 padding | **配方分歧**：上游按 batch 最长者统一取整到 80 的倍数；本仓库逐条取整、批内零填充 | 已复制 `padded_token_length`，适配器尚未接线（见 §6） |
| 9 | step-distilled（flow-map）checkpoint | 上游**明确拒绝**（见 `ref_time_in` 即报错）；本仓库支持（`target_timestep=0`） | 本仓库更强；仅补键嗅探 |
| 10 | int8_convrot / nvfp4 反量化内核 | 上游有（`convrot_quant.py` 2264 行 + `nvfp4_quant.py`） | **不复制**（本仓库用 torchao 量化路线） |

---

## 1. 对照范围与方法

**上游（A）关键文件**

| 文件 | 作用 |
|------|------|
| `extensions_built_in/diffusion_models/flux3/flux3.py` | `Flux3Model(BaseModel)`：加载、VAE 编解码、训练 hook、采样入口 |
| `.../flux3/src/transformer.py` | 自研 `Flux3Transformer`（**模块树 == 权重文件键名**） |
| `.../flux3/src/text_encoder.py` | Qwen3-VL 隐层堆叠 + Comfy 单文件键转换 |
| `.../flux3/src/image_vae.py` | 独立 `Flux3ImageVAE`（FLUX.2 打包 AE，含 BN 归一化） |
| `.../flux3/src/pipeline.py` | 采样配方 + latent/条件打包器 |
| `.../flux3/src/{positional,checkpoints}.py` | RoPE id 打包、权重来源常量 |
| `toolkit/models/v2/text_encoders/qwen3_vl.py` | Qwen3-VL 基类（Comfy 键布局、去视觉塔、Conv3d→GEMM） |
| `toolkit/util/comfy_quant_import.py` | `split_fused_quantized_keys`（int8 repack 的 qkv 融合拆分） |

**本仓库（B）**：`UnifiedTrainer/models/flux3_klein/{__init__.py,transformer_flux3.py}`、
`engine/{checkpoint,trainer}.py`、`data/cache_builder.py`、`utils/quantize.py`。

**第三方裁判（用于打破分歧）**：
- ComfyUI `E:\ComfyUI\comfy\ldm\flux3\model.py` + 共用层 `comfy\ldm\flux\layers.py`
  （直接加载 BFL 的 bf16 repack，是原生键名的消费方）；
- diffusers `E:\diffusers\src\diffusers\loaders\single_file_utils.py:4017-4077`
  （官方 `convert_flux3_transformer_checkpoint_to_diffusers`）。

**方法**：逐项读源码比对数值/键名/顺序；对无法凭读确定的项（半区顺序、键映射）
用权重无关单测在真实模块树上验证，并与 diffusers 官方转换器逐键逐值交叉验证。

---

## 2. 已经一致、不要重复抄的项（含证据）

| 项 | 上游 | 本仓库 | 结论 |
|----|------|--------|------|
| 三模态 + 每模态 5 层 / joint 28 层 / inner 3072 / mlp_ratio 3.0 / axes (32,32,32,32) / θ=10000 | `src/transformer.py:51-73` | `transformer_flux3.py:404-419` | 结构等价（键名不同，见 §3.2） |
| per-mode 块与 joint 块内部：`x + gate*(attn+ff)`、`(1+scale)*norm(x)+shift`、SwiGLU `linear_in(dim→inner*2)` + `linear_out(inner→dim)`、QK-RMSNorm(eps 1e-6) | `transformer.py:243-385` | `transformer_flux3.py:206-304,48-88,160-203` | 等价 |
| 块内 modulation 三分量顺序 `(shift, scale, gate)` | `transformer.py:212-217` | `transformer_flux3.py:238,283` | 等价（ComfyUI `flux\layers.py:131-139` 同） |
| 图像 id `t=0`、`h/w` 网格 × 0.5、文本 id `(0,0,0,i)` | `positional.py:16-47`、`pipeline.py:70-76` | `__init__.py:560-588` | **逐值相同**（单测 `test_*_ids_match_vendored_reference`） |
| 参考图 `t=0, l=-1-i`，可多张 | `pipeline.py:79-110` | `__init__.py:668-686` | 相同 |
| latent：`vae.encode().mode()` → 2×2 patchify（`(c pi pj)` 通道序）→ BatchNorm(affine=False, 128 ch) | `image_vae.py:329-350` | `__init__.py:309-395` | **逐位等价**（单测 `test_patchify_matches_vendored_vae_rearrange`、`test_bn_normalization_matches_vendored_vae`） |
| 解码：反归一化 → unpatchify → decoder | `image_vae.py:352-361` | `__init__.py:397-420` | 相同顺序/数值 |
| bucket 16 对齐（8× 下采样 × 2×2 patchify） | `flux3.py:137-138` | `__init__.py:136` | 相同 |
| 目标 `noise - clean`，t=1 噪声 → t=0 干净 | `flux3.py:565-568`、`pipeline.py:27-32` | `__init__.py:786-790` | 相同 |
| 文本/参考图调制时间步恒为 0（上游显式传、本仓库由 transformer 内部硬编码） | `pipeline.py:61,109` | `transformer_flux3.py:570-583` | 等价 |
| 采样 timeshift：官方工作流 α=16（mu=ln16），模型参考 α=6.93；guidance 3.5 | `pipeline.py:23-24` | `__init__.py:125-144` + `scheduler.py` | 本仓库验证调度用官方 α=16（步骤网格与模型一致） |
| BN eps 取值 1e-4 | `image_vae.py:310`（硬编码） | `__init__.py:342,367`（读 `config.batch_norm_eps`，diffusers 默认 1e-4） | 实际一致；回退值 1e-5 是隐患，见 §6(d) |
| 参考图 `[0,1]→[-1,1]` 后再编码 | `flux3.py:528-531` | 图像管线同 | 相同 |

> ⚠️ **别照抄的一处**：上游 `DEFAULT_TIMESTEP_SHIFT = DEFAULT_ALPHA = 6.93` 挂在
> `get_train_scheduler()`（`flux3.py:84-91,131-135`）上，看着像"训练用 6.93"，但上游
> 默认 `timestep_type='sigmoid'`（`toolkit/config_modules.py:557`）走的分支**不使用
> scheduler 的 shift**（`toolkit/samplers/custom_flowmatch_sampler.py:117-134`），
> 实际与本仓库默认的 logit-normal 相同。6.93 是**推理/验证**常量。本仓库现状正确。

### 4.1 补记（2026-10-17）：官方推理 alpha 是 16，不是 6.93

官方 16 步蒸馏工作流（ComfyUI 模板 `flux3_klein_int8_t2i.json`）把 timeshift 显式
放在一条链上：`alpha (PrimitiveFloat) = 16` → `mu = ln(alpha)`（`ComfyMathExpression`
`log(a)` → **2.77259**）→ `ModelSamplingFlux(max_shift=mu, base_shift=mu)`（节点标题
**"Timestep shift (from alpha)"**）。两个 shift 端点相同 ⇒ `mu` 与分辨率无关，
`alpha = exp(mu) = 16` 在所有分辨率下恒定（不是只在 1024²）。BFL 官方参考仓库
`black-forest-labs/flux-3-image-klein` 同样以 `Pipeline.generate(..., alpha=16.0)` 为默认
（`src/flux3/pipeline.py`；`timesteps(steps, alpha)` 即 `alpha*t/(1+(alpha-1)t)`）。
工作流注释里 "the model's reference is 6.93" 指的是 50 步 base 模板 / ComfyUI 内建
`sampling_settings`（`mu = ln(6.93) = 1.93586`），二者是两个不同的调度。

本仓库原先用 `FlowMatchEulerDiscreteScheduler(shift=6.93)`，有两点偏差：
(1) alpha 取的是参考值 6.93 而不是官方的 16；
(2) diffusers 的 `shift` 会被**用两次**（`__init__` 里对 `sigma_min` 一次、
`set_timesteps` 里对插值网格再一次），且网格终点落在被 shift 过的 `sigma_min` 而不是
`t=0`。所以 20 步时最后一个采样 sigma 是 0.046，而官方参考（α=6.93）应为 0.267、
官方工作流（α=16）应为 0.457。现改为 `scheduler.py` 直接生成配方网格
（`sigma(t) = alpha*t/(1+(alpha-1)*t)`，`t=1..0`，末端严格 0），默认
`alpha = 16`，可用 config `"scheduler_shift": 6.93` 回到参考调度。
差异量化（20 步）：平均 sigma 0.778 → 0.891，σ≥0.8 的步数 13/20 → 17/20，
最后一步的 σ 0.046 → 0.457。

**训练侧同步（2026-10）**：adapter 默认 `flow_shift` 由 2.5 改为 **16**（`timestep_shift_mode`
默认仍是 `"flow_shift"`），即训练 sigma 中心 mu = 2.7726（mean σ 0.92，72% 抽样 ≥0.9）。
原来的 `dynamic_shift 1.5 → 2.5` 在 1024² 是 mu ≈ 1.984（≈ alpha 7.27），比官方低 0.79
（logit 空间的 0.79 个标准差）。三个已启动的 test config 仍显式写 `dynamic_shift`，
因此不受默认值改动影响；要试官方档位可把 `flow_shift: 16` 单独跑一条对照。

---

## 3. 值得借鉴的增量（按价值排序）

### 3.1 参考图（image_cond）尺寸策略 —— 价值最高，属训练配方

- 上游 `_fit_reference`（`flux3.py:440-457`，开关 `match_target_res` 默认 `True`，
  `flux3.py:126-128`）：**参考图按目标像素面积等比缩放**（保自身长宽比），并 snap 到
  16 的倍数；参考图在自己的位置网格上、`l` 轴独立，因此 token 数≈目标 token 数。
- 本仓库：`data/cache_builder.py:712-787` 按 `BucketSystem.find_bucket_for_image()`
  用同一个 `resolution_config` 给参考图**单独选桶**（分辨率可经
  `image_configs.<key>.resolution` 覆盖），参考图与目标的 token 数可以差很多。
- 影响：参考图 token 数与尺度直接改变联合序列构成与条件强度，是训练配方的一部分。
- 已落地：`vendor/flux3_aitoolkit/recipe.py::fit_reference`（含
  `target_pixels_from_latent`）。**适配器/data 管线尚未接线**，见 §6(b)。

### 3.2 原生（BFL/Comfy）单文件权重的键映射 —— 打开一整个加载路径

- 上游 `src/transformer.py` 的模块树**就是**权重文件键名：
  `emb_in.* / txt_in.* / time_in.* / ref_time_in.* / early_stream_modulations.*.lin.* /
  content_mode_blocks.* / txt_mode_blocks.* / single_blocks.* / final_layer.image.*`；
  diffusers（本仓库 vendored）叫 `x_embedders / context_embedder / timestep_embedder /
  target_timestep_embedder / *_stream_modulations.*.linear / mode_blocks /
  single_transformer_blocks / proj_out / norm_out`。
- 本仓库现状：`flux3_klein.load_transformer()` 只做
  `BlockSwapFlux3Transformer2DModel.from_pretrained(path)`（`__init__.py:185-201`），
  即**只吃 diffusers 目录**；BFL 单文件 `model.safetensors`、ComfyUI repack
  (`flux3_image_klein_{bf16,int8_convrot,nvfp4}.safetensors`) 都进不来。
- 两个必须处理的细节：
  1. **输出 adaLN 半区顺序**（见 §4.1）：`[shift, scale]` → `[scale, shift]`；
  2. **int8 repack 把每个块的 q/k/v 融合成一个 `.qkv`**（行序 q,k,v），bf16/nvfp4 不融合。
- 已落地：`vendor/flux3_aitoolkit/keymap.py`
  （`convert_native_to_diffusers` / `convert_diffusers_to_native` / `split_fused_qkv` /
  `detect_layout` / `is_flow_map_state_dict`）。它是**新写的**（上游没有显式映射表），
  但以 `src/transformer.py` 的模块树为准，并与 diffusers 官方转换器**逐键逐值交叉验证**
  （`test_matches_diffusers_official_converter`，12/12 通过）。
- 提示：若不想要本仓库这份，diffusers 自带
  `convert_flux3_transformer_checkpoint_to_diffusers`
  （`single_file_utils.py:4017`）已能覆盖 bf16 与 `_sd_` 单文件；本模块的额外价值在
  **逆向映射**（把 LoRA 导出成 ComfyUI 键名）、fused-qkv、布局/flow-map 嗅探与回归测试。

### 3.3 flow-map（step-distilled）checkpoint 的**键嗅探**

- 上游用 `ref_time_in.*` 直接判定并**拒绝**训练（`transformer.py:397-401`；
  `flux3.py:266-270` 拒绝 `dit.flow_map=true` 的包；`_sd_` repack 文件不算候选）。
- 本仓库支持蒸馏模型（`__init__.py:727-735` 传 `target_timestep=0`），但**只信加载后的
  `config.flow_map`**（`__init__.py:195-200`）；forward 侧是硬校验
  （`transformer_flux3.py:558-560`），配置缺失/单文件场景下会静默用错训练目标。
- 已落地：`keymap.is_flow_map_state_dict()`——同时认 `ref_time_in.*` 与
  `target_timestep_embedder.*`，可在转换前后任一时刻判断。

### 3.4 文本窗口按 batch 统一 padding

- 上游 `padded_token_length`（`text_encoder.py:85-99`）：一批取**最长** prompt，向上取整到
  `pad_multiple=80` 的倍数，全批用同一窗口；并且强调 **DiT 会 attend 到 padding**，
  所以窗口大小是模型输入的一部分。
- 本仓库 `encode_text`（`__init__.py:424-546`）**逐条**按自己的长度取整，
  训练时 `_extract_encoder_hidden_states`（`794-835`）再把短行**零填充**到批内最长
  ——短行多出来的位置是"零状态"而非模型训练时见过的 pad-token 状态。
  batch_size=1 时两者等价，batch>1 时不等价。
- 已落地：`vendor/flux3_aitoolkit/text_encoder.py::padded_token_length`，
  以及整段 `encode_flux3_prompt`（含"不要剥掉 padding"的说明）。

### 3.5 Qwen3-VL 的加载工程细节（VRAM/速度）

- `drop_vision_tower()`（`toolkit/models/v2/text_encoders/qwen3_vl.py:64-69`，
  `flux3.py:318`）：FLUX.3 条件化是纯文本，视觉塔是纯显存占用，直接置空。
- `patch_qwen_vl_patch_embed()`（同文件 `12-32`）：Qwen-VL 的 vision `patch_embed`
  是 kernel==stride 的 Conv3d，bf16 下无 fast cuDNN 路径，等价换成一个 `F.linear`
  GEMM。
- Comfy 单文件键转换（`text_encoder.py:56-71`）：`model.layers.*`/`model.norm.*`/
  `model.embed_tokens.*` → `model.language_model.*`，`visual.*` → `model.*`，
  并补 tied `lm_head.weight`（单文件里没有该键，strict load 会报错）。
- 本仓库 `load_text_encoder/load_tokenizer/load_processor`（`__init__.py:236-272`）走
  `from_pretrained(dir)`：读不了 Comfy-Org 的单文件 Qwen3-VL repack，也没丢视觉塔。
- 已落地：`vendor/flux3_aitoolkit/text_encoder.py`（上述三个函数都收进来了）。

### 3.6 量化排除表与 LoRA 键前缀

- `get_quantization_exclude_modules()`（`flux3.py:633-645`）：
  `emb_in* / txt_in / time_in* / *modulations* / final_layer*` 保持全精度。
  本仓库 `utils/quantize.py::quantize_model` 可复用同一份清单。
- `transformer.* ↔ diffusion_model.*`（`flux3.py:647-657`）：与本仓库
  `engine/checkpoint.py:131-145` 的 comfy→peft 约定方向一致（本仓库已有，无需重复）。
- 已落地：`recipe.QUANTIZATION_EXCLUDE_MODULES`、`convert_lora_weights_before_{save,load}`。

### 3.7 明确**不建议**复制的部分

| 项 | 原因 |
|----|------|
| `convrot_quant.py`（2264 行）+ `nvfp4_quant.py` + `comfy_quant_import.py` 的反量化内核 | 只为吃 int8/nvfp4 repack；本仓库量化走 torchao（`utils/quantize.py`），bf16 单文件已足够 |
| 离线下载/`MODELS_PATH` 解析/暂存改名（`flux3.py:143-216`） | 本仓库无下载器；仅当将来引入"按名取权重"时才需要（`vae/ae.safetensors` 与 FLUX.1 VAE 撞名那条经验值得记） |
| `Flux3ImageVAE` 替换现有 diffusers VAE | 本仓库 diffusers 版支持 `decoder_block_out_channels`，本机 `E:\hf_models\vae\flux2\config.json` 声明 `[96,192,384,384]`，上游版本反而更脆。已复制仅作**可选**单文件路径参考 |
| 训练 sigma 改成 6.93 | 见 §2 警告：上游训练实际也是 logit-normal |

---

## 4. 两处硬差异（本轮最重要的发现）

### 4.1 输出 adaLN 的半区顺序：原生 `[shift, scale]` vs diffusers `[scale, shift]`

同一份权重，两种实现的消费顺序相反，因此**转换时必须交换半区**，只改键名会静默算错：

| 实现 | 代码 | 语义 |
|------|------|------|
| ai-toolkit（原生） | `src/transformer.py:229-240`：`shift_w, scale_w = weight.chunk(2)`；`x*(linear(s,scale_w)+1) + linear(s,shift_w)` | 前半 = shift |
| ComfyUI（原生，加载同一 bf16 repack） | `comfy/ldm/flux/layers.py:376-383`：`shift, scale = adaLN_modulation(vec).chunk(2, dim=-1)`；`modulated_norm(x, norm, scale, shift)`（`156-159`） | 前半 = shift |
| diffusers（本仓库 vendored） | `transformer_flux3.py:347-350`：`scale, shift = chunk(emb, 2, dim=-1)` | 前半 = scale |

diffusers 官方转换器对此有**显式注释与交换**：
`single_file_utils.py:4046-4060`（"original adaLN weights produce [shift, scale];
Flux3AdaLayerNorm expects [scale, shift]" → `swap_scale_shift(value, dim=0)`，
`swap_scale_shift` 定义在 `1990-1993`）。

**对本仓库的含义**
- 现在走 `from_pretrained(diffusers 目录)`（HF 上 `flux-3-image-klein-base` 等）：权重已被
  官方转换器交换过，**当前实现正确**，无需改动。
- 一旦有人绕过转换器直接加载 BFL 单文件 / ComfyUI repack：必须交换，否则速度场
  在输出端被错误调制（图像仍能出，但质量明显劣化——属于最难察觉的一类 bug）。
- 已用测试钉死：`test_output_adaln_swap_is_required`（断言"不交换就不等于期望值"）、
  `test_matches_diffusers_official_converter`（与 diffusers 转换器逐键逐值相同）。

### 4.2 flow-map 支持度：本仓库更强

| | 上游 | 本仓库 |
|---|------|--------|
| 蒸馏 checkpoint（`flux-3-image-klein`，16 步） | load 阶段**直接报错**（`transformer.py:397-401`、`flux3.py:266-270`），comfy `_sd_` 文件被排除在候选外 | 支持：`config.flow_map` 自动识别 + `target_timestep=0`（`__init__.py:191-200,727-735`）；`transformer_flux3.py:435-439` 建 `target_timestep_embedder` |
| 风险 | 无（不支持） | 只信 `config.flow_map`。单文件/手搭目录若缺该字段，会按 base 目标训练蒸馏模型 |

> 结论：不抄上游的"拒绝"，而是抄它的**键嗅探**（§3.3）来补本仓库的风险点。

---

## 5. 已复制进本仓库的文件

目录：`UTrainer/UnifiedTrainer/vendor/flux3_aitoolkit/`
（每个文件**抬头**都写了原作者、上游仓库/文件、commit、许可与变更类型；
详见同目录 `ATTRIBUTION.md`）。

| 文件 | 类型 | 上游来源 | 变更 |
|------|------|----------|------|
| `positional.py` | **逐字复制** | `flux3/src/positional.py` | 仅加抬头 |
| `checkpoints.py` | **逐字复制** | `flux3/src/checkpoints.py` | 仅加抬头 |
| `pipeline.py` | **逐字复制** | `flux3/src/pipeline.py` | 仅加抬头 |
| `image_vae.py` | **逐字复制** | `flux3/src/image_vae.py` | 仅加抬头 |
| `text_encoder.py` | 改编 | `flux3/src/text_encoder.py` + `toolkit/models/v2/text_encoders/qwen3_vl.py` | 去掉依赖 `toolkit.*` 的类，改为自由函数；数值不变 |
| `recipe.py` | 摘录 | `flux3/flux3.py` + `src/pipeline.py` | 抽常量与 `_fit_reference`/LoRA 前缀/量化排除 |
| `keymap.py` | 新写（依上游模块树） | `flux3/src/transformer.py`、`toolkit/util/comfy_quant_import.py` | 双向键映射、adaLN 半区交换、fused-qkv、嗅探 |
| `__init__.py` / `ATTRIBUTION.md` | 新增 | — | 包说明与署名/许可 |

**验证**

```bash
cd E:\UnifiedTrainer
python -m pytest tests/test_flux3_klein_adapter.py \
                 tests/test_flux3_aitoolkit_keymap.py \
                 tests/test_flux3_aitoolkit_vendor.py -q      # 64 passed
```

- `test_flux3_aitoolkit_keymap.py`（12 项）：在真实（微型）`Flux3Transformer2DModel` 上
  做 native↔diffusers 往返，断言张量逐位相等；adaLN 半区交换为"必须项"；
  fused-qkv 拆分/融合；布局判定；量化与未知键必须报错；**与 diffusers 官方转换器交叉验证**。
- `test_flux3_aitoolkit_vendor.py`（10 项）：id 约定（图像/参考图/文本）、α/divisibility
  常量、patchify 与 BN 归一化同上游逐位一致、`fit_reference`、vendored VAE 往返、
  batch-common padding、`drop_vision_tower`/`patch_embed` 助手。

---

## 6. 本轮**未做**的事（可选的后续接线）

本轮只做"对照 + 复制 + 验证"，**没有改动任何现有适配器/数据管线代码**。以下按收益排序：

| 字母 | 待接线项 | 改动点 | 代价/风险 |
|------|----------|--------|-----------|
| (a) | 单文件 checkpoint 加载 | `models/flux3_klein/load_transformer()` 增加分支：`safetensors.load_file` → `keymap.detect_layout` → `convert_native_to_diffusers` → `load_state_dict(strict=True)`；同时用 `is_flow_map_state_dict` 覆盖 `flow_map` | 低；建议同时保留 diffusers 目录路径 |
| (b) | 参考图面积匹配 | `data/cache_builder.py::_encode_reference_entries` / `_construct_media` 增加 `match_target_res`（或 per-reference `resize` 取值 `"target"`），用 `recipe.fit_reference` | 中；**必须重建缓存**（token 数变化），需先确认对现有实验的影响 |
| (c) | 文本窗口批级统一 | `encode_text` 改为先用 `padded_token_length` 求批窗口再编码；缓存侧需能按批重建 | 中；**必须重建 embedding 缓存**；batch>1 才有效果 |
| (d) | BN eps 回退值 | `__init__.py:342,367` 的 `1e-5` → `1e-4`（与 checkpoint 一致；仅当 config 缺键时生效） | 极低，一行 |
| (e) | 量化排除表 | `utils/quantize.py::quantize_model` 调用处传入 `recipe.QUANTIZATION_EXCLUDE_MODULES` | 低；仅量化训练时相关 |
| (f) | LoRA 导出为 ComfyUI 键名 | `engine/checkpoint.py` 增加 flux3 的 `diffusers_key_to_native` 分支 | 低；提高与 ComfyUI 生态的互通性 |

需要哪一项告诉我即可，改动会按 (a)→(d)→(e)→(f)→(b)/(c) 的顺序（低风险优先，
(b)(c) 涉及缓存重建，建议单独一次实验）。

---

## 附：证据索引

| 主题 | 上游 | 本仓库 | 第三份旁证 |
|------|------|--------|-----------|
| 三模态 forward / 超参 | `flux3/src/transformer.py:51-73,391-743` | `transformer_flux3.py:353-674` | diffusers `transformers/transformer_flux3.py` |
| adaLN 输出半区 | `flux3/src/transformer.py:229-240` | `transformer_flux3.py:338-350` | ComfyUI `comfy/ldm/flux/layers.py:376-383`；diffusers `loaders/single_file_utils.py:4046-4060` |
| 原生键名/融合 qkv | `flux3/src/transformer.py:391-420` | `keymap.py` | diffusers `single_file_utils.py:4017-4077` |
| 参考图尺寸 | `flux3/flux3.py:126-128,440-457` | `data/cache_builder.py:712-787` | — |
| 文本窗口 | `flux3/src/text_encoder.py:85-150` | `models/flux3_klein/__init__.py:424-546,794-835` | — |
| latent/BN/解码 | `flux3/src/image_vae.py:286-361` | `models/flux3_klein/__init__.py:309-420` | diffusers `autoencoders/autoencoder_kl_flux2.py:99,133-139` |
| RoPE id | `flux3/src/positional.py:16-47`、`src/pipeline.py:70-110` | `models/flux3_klein/__init__.py:550-588` | ComfyUI `comfy/ldm/flux3/model.py:84,109-110` |
| flow-map | `flux3/src/transformer.py:397-401`（拒绝） | `__init__.py:191-200,727-735` | diffusers `single_file_utils.py:4043-4044`（`ref_time_in`→`target_timestep_embedder`）；ComfyUI `flux3/model.py:94,168` |
| 训练 sigma | `toolkit/config_modules.py:557`、`toolkit/samplers/custom_flowmatch_sampler.py:117-134` | `models/base.py:455-457` | — |
