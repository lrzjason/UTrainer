# MOSS-Image → UnifiedTrainer 适配计划（含参考图 VL 条件）

> 状态：计划（2026-10-19）。目标：`python -m UnifiedTrainer.train --model moss ...` 能跑通一次
> **完整的测试训练**（1 图过拟合 → 小数据集短跑），并支持**参考图训练**（参考图缩放后送
> Qwen3.5 的 vision tower，行为对齐 krea2）。
> 参考源：`<repo>/ref\KohakuMossTrainer`（作者 **Kohaku-Blueleaf**，只读、运行时不 import）。
> 搬过来的每个文件首行标注原作者；`UnifiedTrainer/vendor/kohakumoss/ATTRIBUTION.md` 记逐文件出处。

---

## 0. 已核对的事实（不是推测）

| 项 | 事实 |
|---|---|
| 权重（modelscope `openmoss/MOSS-Image-Base`） | **25.36 GiB / 45 文件**：transformer 9.47、text_encoder 15.63、vae 0.23 |
| 磁盘 | **E: 只剩 27.5 GB**（装 25.4 GiB 会几乎填满）；C: 剩 1421 GB |
| transformer | `MOSSImageTransformer`（diffusers 里没有这个类）：dim 3072、ffn 8192、24 头、**42 层**、`latent_channels=128`、`text_dim=4096`、`freq_dim=256`、rope_base 1e4 |
| vae | `diffusers.AutoencoderKLFlux2`（本地 diffusers 0.41 **有**），32ch、2×2 patch、`force_upcast` |
| text_encoder | 仓库里是 `Qwen3_5Model`，**带 vision_config**（depth 27 / hidden 1152 / patch 16）→ 本身就是多模态；text_config 4096 hidden / 32 层 / 16 头 |
| 本地 | `E:\hf_models\Qwen3.5-9B`（`Qwen3_5ForConditionalGeneration`，带 `preprocessor_config.json`）——参考图 processor 可以从它取 |
| transformers 5.2.0 | 有 `Qwen3_5Config`，没有 `Qwen3_5VL*` 类名；仓库指定的类就是 `Qwen3_5Model` |
| 参考图链路（UTrainer 已有） | `cache_builder` 会把 `reference_image=[PIL…]`（**已按 resize 面积缩放**）+ `processor` 传给 `adapter.encode_text`；krea2 用 `Picture {}: <|vision_start|><|image_pad|><|vision_end|>` 占位符走 Qwen3-VL |
| MOSS backbone | `MossBackbone(img[N,128], text[N_txt,4096], sigma[B], SeqInfo) -> v[N,128]`，**没有**参考 latent token 路径（参考图只能走文本/vision 流） |

### 参考图通路：与 krea2 的接口对齐 + idea.txt 规格

**接口照 krea2**（拼接在 adapter，模型只收一条 token 序列）：

| krea2 | MOSS（本实现） |
|---|---|
| `prepare_model_input` 里 `torch.cat(packed_parts, 1)` → `hidden_states (B, N, C)` | `pack.prepare_model_input` 里 `torch.cat([target, refs…], 1)` → `hidden_states (B, N, C)` |
| 传 `position_ids`（refs 帧号 1,2,…）+ `reflen` | 传 `target_grids=[(h,w)]` + `reference_grids=[[(rh,rw),…]]`，帧号由模型按 1,2,… 生成 |
| `split = text_len + img_noisylen` → 前段 sigma、后段 t=0 | `split = h*w + S` → 目标+文本用 sigma，参考尾段用 `conditioning(0)` |
| refs 追加在 target 之后 | refs 追加在 `[target | text]` 之后（官方序列是 `[image, text]`，末尾追加不动输出切片） |

| idea.txt 规格 | 实现 |
|---|---|
| frame 0 = 预测 latent（加噪、画布网格） | ✓ `r=0`、`f=text_length`、h/w 画布网格 |
| 参考图帧 `r = 1, 2, 3, …`（与 krea2 的 `i+1` 一致） | ✓ 按顺序连续编号；`annotation_reference_config` 声明的标注图组**只影响排序**（排在最前），因此自然占 `r=1`，全尺寸参考 `r=2`、物体级 `r=3+` |
| 参考 token t=0 调制、目标才加噪 | ✓ |
| RoPE h/w 整张参考图按画布比例映射并取整 | ✓（**不做逐 item bbox**） |
| 同一图同时进 VL | ✓ `Picture N:` + `image_token_mask` |
| 损失只算预测 token / 只训 LoRA | ✓ |
| `z_index` 走文本 JSON | ✓ 无需改代码 |
| §6.3 bbox 派生的注意力掩码 | ✗ 未实现（依赖 bbox，按要求不做） |
| 参考图原生分辨率编码 | ⚠ 目前按 `data.resolution` 分桶（与画布无关）；要原生就把 `data.resolution` 设为参考图尺寸 |

实测日志（真实权重 + 真实参考图）：

```
MOSS reference spans: 1 reference image(s) per sample at RoPE frames [1], tokens=[1024]
(target r=0, references r=1..N like Krea-2; t=0 modulation, whole image mapped over the canvas)
```


---

## 1. 架构决定（2026-10-19 修订：改用官方自包含实现）

下载时发现 ModelScope 仓库根目录带着 **`pipeline_moss_image.py`（440 行，自包含，
"Copy this file anywhere and import it normally"）**，里面就是官方的
`MOSSImageTransformer` / `MOSSImageScheduler` / `MOSSImagePipeline`。它的 forward 是
标准 diffusers 形态，**没有 packed 轴、没有 SeqInfo**：

```python
forward(hidden_states [B,128,H,W], timestep = sigma*1000,
        encoder_hidden_states [B,S,4096], encoder_attention_mask [B,S] bool)
```

因此把「搬 ref 的 2.5k 行模型 + ops shim」整个换掉，改成搬这 440 行。收益：代码量 1/6、
语义由官方保证、与 UTrainer 现有 adapter 形态一致（和 krea2/flux2 一样是 `(B,C,H,W)`）。

```
UnifiedTrainer/models/moss/
    __init__.py   MossAdapter(BaseModelAdapter)              我
    pack.py       batch → forward kwargs + MOSS 时间步 pmf    我（时间步算法搬自 ref）
    encoders.py   Qwen3.5 文本/多模态 + Flux2 VAE             子代理 B
UnifiedTrainer/vendor/openmoss_moss_image/
    pipeline_moss_image.py   官方实现，verbatim + banner      子代理 A
    ATTRIBUTION.md           OpenMOSS 为原作者；时间步算法另注 Kohaku-Blueleaf
```

三个关键取舍（修订后）：

1. **模型用官方实现，不 vendor ref 的框架。** ref 的价值在训练配方（时间步 pmf、
   参考图/编码器实践、kernel 选型），不在它的模型框架；那套 packed/ops/registry 对
   UTrainer 是负资产。
2. **batch 直接 (B,C,H,W)。** 官方 forward 自己按样本循环并丢掉 padding 文本 token，
   所以多样本天然正确；`batch_size: 1` 仍然是最省显存的设置，但不是结构限制。
3. **参考图双通路（2026-10-20 修订）。** 同一张图既经 Qwen3.5 的 vision tower 进文本流，
   又由 VAE 编码成 latent 追加进 DiT 序列（`models/moss/forward.py`），与 krea2 一致；
   帧编号按 `E:\MOSS_data_pipeline\idea.txt` §3.1：`r=0` 目标、`r=1` 标注图、
   `r=2` 全尺寸参考、`r=3+` 物体级参考，参考段用 t=0 调制。

---

## 2. 阶段、产出与验收门（Gate）

### P0.5 干跑（已完成，2026-10-19 23:30）

权重还在下载时，先用「官方 transformer 按 config 随机初始化 + 本地 Qwen3.5-9B 编码器 + 真实
VAE」把整条链路跑通（`configs/moss_dryrun.json`，`init_from_config: true`）：

```
cache build (VAE + TE nf4) → transformer from config → LoRk 269 modules / 9.93M params
→ 4 steps: loss 2.3337 / 2.3455 / 2.3395 / 2.3416, fwd 1.7s bwd 3.2s (4.2 s/step)
→ peak VRAM 12212 MB → checkpoint saved → exit 0
```

干跑抓到并修掉的 4 个真问题（都会在真实权重下复现）：

| # | 现象 | 修法 |
|---|------|------|
| 1 | `MOSSImageTransformer does not support gradient checkpointing`（diffusers 拒绝） | `models/moss/train_forward.py::install_training_hooks`：给每个 joint block 的 `forward` 打一层 `torch.utils.checkpoint` 包装（实例属性，不动类、不改 state_dict 键） |
| 2 | 缓存 embedding 是 **numpy 数组**，`.to()` 报错 | `pack._as_tensor` 统一转 tensor |
| 3 | `unpack_prediction` 返回 `Transformer2DModelOutput`，trainer 按样本迭代时拿到字符串键 | `pack.unpack_prediction` 取 `.sample` |
| 4 | `quantize: nf4` → `silu_cuda not implemented for 'Byte'`（官方 forward 按 `weight.dtype` 转型，bnb 打包权重是 uint8） | **MOSS 不支持 bnb nf4/int8 与 torchao fp8**；训练用 `quantize: none`（bf16）。同时在 `train_forward` 加了首个 forward 前的显式报错，避免静默训坏 |

### P0 权重下载（子代理 D，后台）
- `modelscope download --model openmoss/MOSS-Image-Base --local_dir C:\hf_models\MOSS-Image-Base`
  （CLI 不支持就退回 `snapshot_download(local_dir=…)`，可断点续传）。
- 完成后建 junction：`E:\hf_models\MOSS-Image-Base` → `C:\hf_models\MOSS-Image-Base`
  （用户要求的路径可用，字节落在 C:；E: 塞不下 25.4 GiB）。
- **Gate**：45 个文件齐全、总大小 25.36 GiB ±1%；每个 `.safetensors` 能用 `safetensors`
  读出 header；`transformer/…index.json` 的 shard 全部存在；junction 生效（从 E: 路径能打开文件）。

### P1 模型本体（子代理 A）
- 搬 `model/**` + `types/**` + `logging/log.py` + `common/{io,units}` + `kernels_eager/**`
  （ref `backends/eager/*` 6 文件 ≈ 450 行），写 `shim.py` 提供 ref 的
  `registry`/`ops`/`dtypes` 最小 API（只 eager，不 triton/tilelang）。
- 写 `io/load_hf.py`：按 `io/keymap.py` 把 HF `transformer/*.safetensors` 映射进 backbone，
  支持 3 个 shard + fp16-fix key。
- **Gate**：
  1. `build_backbone("…/transformer/config.json")` 成功，参数量 ≈ 4.7B；
  2. 加载真实权重 → **missing=0, unexpected=0**（打印清单）；
  3. 随机 `(img[N,128], text[L,4096], sigma[1], SeqInfo)` 前向在 5060 Ti 上得到有限值；
  4. 同一个输入两次前向逐位一致（无隐式状态）。

### P2 编码器（子代理 B）
- `encoders.py`：
  - `load_text_encoder(path, dtype)`：transformers 5.2 的 `Qwen3_5Model`（或等价 AutoModel），
    优先 `attn_implementation="sdpa"`；返回可前向的模块 + 取 hidden states 的方式
    （**先确认 ref 取哪一层**：`encoders/text/forward.py`）。
  - `encode_prompt(te, tokenizer, processor, prompt, reference_images, device, dtype)`
    → `[L, 4096]`：文本模板与 ref/Comfy 对齐；有参考图时按 krea2 方式拼
    `Picture N: <|vision_start|><|image_pad|><|vision_end|>` 并把 PIL 交给 processor。
  - `load_vae(path, dtype)`（diffusers `AutoencoderKLFlux2`）+ `encode_image/decode_latent`：
    MOSS 的 latent 归一化（查 ref `encoders/vae/__init__.py` 的 scale/shift），输出 32ch，
    backbone 侧再做 2×2 patchify → 128ch（以 ref 为准，别猜）。
- **Gate**：
  1. 真实图片 → latent `(32, H/8, W/8)`；同图两次编码差异 < 1e-5；
  2. caption → `(L,4096)`；加 1–2 张参考图后 L 变长且 `image_token_mask` 非空；
  3. VAE 解码回图（`decode_latent`）能出正常图（不是噪声）。

### P3 适配器 + 配置（我写，子代理 C 补 LoRA 目标）
- `MossAdapter`：`latent_channels=128`、`bucket_divisibility=16`、`load_*`、`encode_*`、
  `prepare_model_input`（batch → packed）、`unpack_prediction`（`[N,128]` → `(B,128,h,w)`）、
  `compute_target`（standard velocity）、`sample_timesteps`（MOSS shift 从 ref `diffusion/timestep/moss.py`）。
- `networks/lokr_module.py` 增 `_MODEL_PATTERNS["moss"]`（按 keymap 的真实模块名）。
- `UnifiedTrainer/configs/moss_lora_example.json`。
- **Gate**：`--list-models` 有 `moss`；用真实权重加载 + 一次 forward/backward（无数据，合成 latent）
  显存 < 15 GB，梯度非空。

### P4 参考图（子代理 B，在 P2 之后）
- 打开 `reference_conditioning` / `multimodal_reference` 两个 config 开关（语义对齐 krea2）。
- `encode_text(..., reference_image=[PIL…], processor=…)` 走 VL；缓存 key 要包含参考图，
  否则同一 caption 不同参考图会复用错误 embedding（**krea2 的缓存键处理要照抄**）。
- **Gate**：同一 caption、换参考图 → embedding 不同；训练一个 step 不报错；关掉开关时
  行为与纯文本完全一致（逐位）。

### P5 验收训练（我）

**已通过（2026-10-19 23:50–23:56）**：

| 验收项 | 结果 |
|--------|------|
| 真实权重加载 | 463/463 键，missing 0 / unexpected 0；**5.082 B** 参数（计划里估的 4.7 B 是错的），9.47 GiB bf16 |
| 单图过拟合（`moss_overfit_1img.json`，60 step，bf16） | exit 0；loss 2.33（第 1 步）→ 均值 **0.618**；峰值显存 **12.3 GB**；4.2 s/step；checkpoint 可存 |
| 参考图训练（`moss_ref_1img.json`，3 图 × 8 repeats，36 step） | exit 0；缓存里 caption embedding **L=179（144 个 image token + 35 文本 token）**，证明参考图确实过了 VL；峰值显存 **10.8 GB**；1.0–2.0 s/step |
| LoRA 是否真的在学 | 高学习率探针（lr=1e-2，10 step）：LoKR 因子张量从 0 变成 absmax 0.32–0.57（269/269 非零）→ **梯度确实进了 adapter** |
| 单元/集成测试 | `tests/test_moss_pack.py`（10 项）+ `tests/test_moss_adapter_weights.py`（4 项，真实权重）全绿；全套 **288 passed**（4 个失败是既有的、与 MOSS 无关的 dummy/cache-dir 用例） |
| **出图验收（最强证据）** | 用 vendored 官方 transformer + `MOSSImageScheduler` + Flux2 VAE，以缓存里的 caption embedding（1218 token）跑 20 步 CFG=3 采样 → 512×512 图，内容与 caption 一致（`.tmp/moss_sample.png`）。证明权重加载、latent 归一化/patchify、时间步约定（sigma×1000 在 forward 内部）、VAE 解码全部对齐 |

训练收敛观察（诚实记录）：单图过拟合 150 步（lr 5e-4）loss 0.588 → 0.552（前 30 vs 后 30 均值，
min 0.445）；`lokr_alpha=16` 的 40 步探针没有更快下降。**这说明管线是通的、梯度确实进了
adapter，但"训出效果"需要真实的几百到几千步**（5B 模型 + rank16 LoKR），不要用 150 步的
loss 判断质量。

两个已知限制（**已于 2026-10-20 00:25 全部解决**）：

1. ~~不支持 bnb nf4/int8 / torchao fp8~~ → **已支持**。根因是官方 forward 用
   `weight.dtype` 转型；`models/moss/forward.py` 的 `_lin()` 改成按模块的
   `compute_dtype`（bnb）或浮点权重 dtype 转型，fp8 权重则保留 bf16 激活。
   实测 nf4 训练：**峰值显存 12.3 GB → 5.7 GB**，4.5 s/step，loss 正常
   （`configs/moss_nf4_probe.json`，6 step，exit 0）。
2. ~~参考图只走 VL，不进 DiT 序列~~ → **已按 krea2 + `E:\MOSS_data_pipeline\idea.txt`
   的冻结设计实现参考 latent 拼接**：
   - 参考 latent（VAE 编码的同一张图）用**同一个 `video_embedder`** 编码，追加在
     `[target, text, refs…]` 之后，与目标共享 128ch latent 空间；
   - **RoPE 用第四轴 `r` 作参考索引**（官方该轴恒为 0，正是"预留给参考"）：
     **目标/预测 = `r = 0`，第 k 张参考图 = `r = 1 + k`**（第 1 张 r=1、第 2 张 r=2…，
     从 1 开始连续编号；标注图、全尺寸参考、物体级参考在 RoPE 里一律平等，类型交给
     caption/JSON 表达，不进 RoPE）；`f = text_length` 与图像 token 对齐，
     `h/w` 把参考图自身网格**整张按比例铺满画布网格**并取整（不做逐 item bbox 映射），
     多张参考图 token 落在同一画布格也靠 `r` 区分；
   - 参考图**不加噪**，用 **t = 0 的调制**；损失仍只作用于预测 token；
   - 同一张图**同时**走 Qwen3.5 的 vision tower（`encoders.encode_prompt` 的
     `image_token_mask`），与 krea2 完全一致（同一图两个通路）。
   实测（真实权重 + 真实参考图）：日志
   `MOSS reference conditioning: 1 reference latent(s) per sample, tokens=[1024]
   (appended to the DiT sequence on RoPE r=1.., t=0 modulation)`，
   12 step 训练 exit 0，峰值 11.5 GB。

模型通路已按 `idea.txt` 对齐（见上面的对齐表）：`r=0` 目标 / `r=1` 标注图 / `r=2` 全尺寸
参考 / `r=3+` 物体级参考，参考段 t=0 调制，整张参考图按画布比例映射。**刻意不做**的只有两件：
逐 item bbox→RoPE（按你的要求，RoPE 只按整张参考图）与由 bbox 派生的注意力掩码；
`z_index` 走文本 JSON，无需改代码。剩下的是数据侧：标注图生成、参考图原分辨率编码策略、
caption 模板与 JSON 校验（`idea.txt` §8.2）。


---

## 3. 风险与对策

| 风险 | 对策 |
|---|---|
| `Qwen3_5Model` 在 transformers 5.2.0 里前向接口/类名不同 | P2 第一步就写最小脚本验证 `from_pretrained` + forward；不行就用 `AutoModel` + `trust_remote_code=False` 的组合，或退到本地 `Qwen3.5-9B` 的 `Qwen3_5ForConditionalGeneration` |
| vendored ops shim 覆盖不全（attention/rope/ffn 变体） | 以 ref `backends/eager/*` 为唯一清单，逐个 op 名对齐；缺哪个会在 build 时报错，不会静默错 |
| MOSS 是否真被 VL token 训练过（参考图效果） | 先按用户要求实现并训 LoRA；若 loss 不降/出图无参考特征，在文档里给出「需要更多步/只训文本流」的结论，不假装有效 |
| 16 GB 卡 + 42 层 4.7B | LoRA + bf16 + gradient checkpointing；VAE/文本编码器训练前预缓存（离线 cache），训练中只留 DiT |
| 权重下载 25 GiB | C: 落盘 + junction；断点续传；先校验 index 再建 junction |

## 4. 依赖顺序

```
P0 下载 ──┐
P1 模型 ──┼─→ P3 适配器 ─→ P5 验收
P2 编码器 ┘        ↑
P4 参考图 ─────────┘
```


## 三个训练验收（2026-10-20，全部 exit 0）

| # | 训练 | 数据 | 步数 | avg_loss（e0 → e9） | 峰值显存 | 配置 |
|---|------|------|-----:|---------------------|---------:|------|
| 1 | 文生图 | `platform_search/r1ona_a1ra`（74 图 + txt） | 740 | 0.593 → 0.577 | 11.5 GB | `moss_t2i_r1ona.json`（bf16） |
| 2 | Stage 1 纯文本编辑 | `MOSS_edit_filter/ad-creative`（38 组，JSON 指令为 caption） | 380 | 0.616 → 0.638（平稳） | 13.1 GB | `moss_stage1_textedit.json`（bf16） |
| 3 | Stage 2 参考编辑 | 同上 + BBOX(r=1)/R(r=2) latent + VL 双图 | 380 | **0.715 → 0.448（-37%）** | 9.1 GB | `moss_stage2_refedit.json`（nf4） |

结论：
- Stage 2 单调下降 37%，说明参考 latent 通路确实在学（文本指令 alone 的 loss 地板高——
  同类指令对应的目标图差异大，条件方差决定了下限）；
- Stage 1 平稳符合预期：文本通路里基模型已很强，380 步 LoRA 对它的改动有限；
  它的作用是让 DiT 习惯"JSON 指令"这种文本分布；
- bf16（12-13 GB）与 nf4（9.1 GB，带两段参考）都可在 16 GB 卡上完成 10 epochs；
- bf16 下 Stage 2 曾贴到 15.8 GB（大参考图样本），nf4 后无压力——**Stage 2 建议 nf4**。

已知杂音：`minimax_h3` 适配器导入失败（缺 `diffusers.modular_pipelines.minimax_h3.packing`）
与 LoKR 的 ComfyUI 转换警告（`'NoneType' object has no attribute 'replace'`）均为既有问题，
不影响 MOSS 训练。
