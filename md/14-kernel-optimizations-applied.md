# 从 KohakuMoss 学到的 kernel 优化，落到 UnifiedTrainer 的模型上

> 状态：as-built（2026-10-19）。
> 参考源：`E:\UnifiedTrainer\ref\KohakuMossTrainer`（作者 **Kohaku-Blueleaf**，`kohakumoss`
> 0.1.0，KohakuMoss License 1.0 / Apache-2.0 + 命名条款）。只读阅读，**运行时不 import**。
> 技术萃取（逐项 + `path:line`）：`md/13-kohakumoss-kernels.md`。
> 代码落点：`UnifiedTrainer/kernels/`，逐文件出处见 `kernels/ATTRIBUTION.md`。
> 本文只记录**实测**结论与判定过程；未测的不写。

目标：不移植 MOSS，而是把那套「**eager 是基准、按 (op, arch, dtype, shape) 实测选型**」的做法
搬到 UnifiedTrainer 现有的 7 个模型适配器上，并给出每项改动的实测收益。

---

## 0. TL;DR

| # | 改动 | 作用面 | 实测（单 block fwd+bwd，bf16，base 冻结） |
|---|------|--------|------------------------------------------|
| 1 | GQA 不再传 `enable_gqa=True`，改为展开 KV 头 | krea2（任何 GQA 适配器）；落到 `kernels/eager.py::sdpa_gqa` + hook | L=2048 **165.0 → 120.3 ms（1.37x）**；L=1024 64.3 → 52.6 ms（1.22x） |
| 2 | Triton flash attention（GQA、可选 key mask、fwd+bwd） | 全部适配器（同一个 hook） | vs SDPA：**1.5–1.75x**；krea2 相对原始写法 **4.68x（L=1024）/ 5.91x（L=2048）** |
| 3 | 融合 block op（rms_norm+modulate / rope / silu_mul / 残差门控），build 时换 `forward` | 全部适配器（krea2/flux2/flux3 已接） | 单 op **13.9x / 33.7x / 1.68x / 1.29x**；一个 block ≈ −12 ms |
| 4 | 1+2+3 合起来（`kernels: auto`，不开 compile） | krea2 | L=2048 **165.0 → 109.5 ms（1.51x）**；L=1024 64.3 → 48.1 ms（1.34x） |
| 5 | `torch.compile` 作为一等候选（`training.compile` + 静态形状） | 全部 7 个适配器 | L=2048 102.4 ms（1.61x）；与 kernels 不叠加 |
| 6 | 按形状实测选型（`kernels/select.py`） | attention 后端 | 每个形状首调测一次并缓存，防「Triton 在某个形状更慢」 |

三个最重要的判定（都与直觉相反，值得记住）：

1. **Windows + sm120 上「优化 attention」的第一件事不是写 kernel，而是别让 SDPA 掉进
   math 后端。** `enable_gqa=True` 在 flash / mem-efficient 内核不可用的构建里会静默退到
   fp32 math 后端，同一份数学慢 2.8–3.4x；改三行（展开 KV 头）即可。写 kernel 是第二件
   事：Triton 再拿 1.3–1.75x。
2. **`torch.compile` 常常赢过手写 kernel**（这正是 KohakuMoss 自己的结论），本机 4 个
   block op 里它赢 3 个；所以本包的每个 op 都有三列候选 eager / Triton / compile，
   **只按实测排序**，并且 compile 开着时包装函数主动让路给 inductor。
3. **eager elementwise 比「带宽下限」贵一个数量级**（多趟 + fp32 upcast + 分配），
   在 D=6144 的 block 上仍占 ~14%（17 ms）；融合后剩 2.4 ms。它排在 attention 与
   compile 之后，但不是「趋近于 0」。

---

## 1. 方法：照搬 KohakuMoss 的判定纪律

参考其 `src/kohakumoss_kernels/backends/eager/*`（「eager 是唯一基准」）、
`error_stats.py`（fp64 作标尺，cosine 与 abs/rel 成对读）、`tuning/tuner.py`（按
card/op/dtype/shape 实测落表）。本包落到 4 条规则：

1. **每个 op 先有 eager 定义**（`kernels/eager.py`），Triton 实现必须等于它；
2. **正确性用 fp64 标尺**：cosine、mean/max abs、mean/max rel 一起报，不看单点；
3. **选型只来自本机实测**：`python -m UnifiedTrainer.kernels.bench`，eager 永远是第一列；
4. **跑不动的后端自动降级**，绝不静默改语义（`is_supported_*` 返回 False 就走 eager）。

测试与实测分离：`tests/test_kernels_*.py` 用 fp64 标尺判正确性；
`kernels/bench.py` 出性能表；本文的所有数字都能用 §8 的命令复现。

---

## 2. 实测发现一：`enable_gqa=True` 是性能陷阱

`Krea-2` 是 GQA（48 个 query 头 / 12 个 KV 头）。原实现走 diffusers
`dispatch_attention_fn(..., enable_gqa=True)`，数学上正确，但在这台机器上：

```
flash              FAIL  "Torch was not compiled with flash attention"
mem-efficient      FAIL  "both fused kernels require query, key and value to have the same num_heads"
cuDNN attention    FAIL  runtime disabled
   => math backend, 概率/矩阵乘全部 fp32
```

单次 attention（B=1, L=2048, H=48, KV=12, D=128, bf16, fwd+bwd）：

| 路径 | 时间 |
|------|------|
| `enable_gqa=True`（原来的写法）→ fp32 math | **67.7 ms** |
| KV 展开成 48 头 + SDPA（mem-efficient cutlass 内核） | **24.2 ms**（2.8x） |
| `torch.compile(flex_attention)` | 16.3 ms（需要 compile，且不支持任意 mask） |

fp32 math 后端的特征在内核名里一目了然：`cutlass_80_simt_sgemm_*`（fp32 GEMM，单次
block 里 41.7 ms）+ `softmax_warp_forward` + `BinaryFunctor<float,float,float>` 的
elementwise。**注意这是在一次 `Krea2TransformerBlock` 里，占了 204 ms 的 30%。**

修法（两处，都不改数学）：

- `models/krea2/transformer_krea2.py::Krea2AttnProcessor`：先
  `key/value.repeat_interleave(48//12, dim=2)`，再普通 SDPA，删掉 `enable_gqa`；
- `kernels/eager.py::sdpa_gqa`：同一逻辑的通用实现，所有适配器经 hook 共用；
- 若 Triton 内核可用，hook 直接走 `kernels.attention_triton`（§5.2），连展开都省了。

> 判定：**必须在模型里修，不能只靠 hook。** hook 默认开启，但 `training.kernels: off`
> 之后模型仍应是快的；因此 krea2 的处理器自己展开，hook 只负责可选加速。

---

## 3. 实测发现二：`torch.compile` 是合法赢家

KohakuMoss 的结论（其 `docs/internals/block-ops.md` 表）是：单样本 norm/residual/loss 上
`torch.compile` 赢手写 kernel 8–40%，FFN/matmul 层几乎总是 compile 赢。本机复现一致：

| 变体 | L=1024 | L=2048 |
|------|--------|--------|
| 原始（`enable_gqa` math） | 64.3 ms | 165.0 ms |
| KV 展开（= 本仓库修复后的 eager） | 52.6 ms | 120.3 ms |
| `torch.compile(block)`（KV 展开后） | 45.2 ms | 102.4 ms |
| 相对原始 | **1.42x** | **1.61x** |

所以 `training.compile: true` 不是「可选的锦上添花」，而是当前 GPU 上**最大的单项之一**，
且它的收益与 §2/§4 的 kernel 改造**不叠加**（二者走的是同一条融合路径，见 §5.4）。
本次顺带补了 `training.compile_dynamic`（默认 `null` = inductor 自己决定；bucketed
数据集建议 `false`，每个 bucket 形状 trace 一次）。

注意编译与 kernel hook 的**组合规则**：`torch.compile` 期间 `kernels.attention()` 直接
返回 SDPA（`kernels/__init__.py::_is_compiling`），避免自定义 autograd Function 造成
graph break；因此 `compile` + `kernels: auto` 的组合是「inductor 融合 + mem-efficient
attention」，而 `compile: false` + `kernels: auto` 是「Triton attention + eager 其余」。

---

## 4. 实测发现三：eager elementwise 的开销远超「带宽」估算

把 §5.2 / §5.3 的孤立测量按 block 结构加起来（L=2048，**修 GQA 之后**，即 §5.1 的
"修 GQA" 列 120.3 ms）：

| 类别 | 时间 | 占比 |
|------|------|------|
| bf16 GEMM（8 个大矩阵乘，fwd+bwd） | ~80 ms | 66% |
| attention（mem-efficient cutlass） | 23.4 ms | 19% |
| elementwise / copy（rope、modulate、残差、norm、fp32 upcast、KV 展开） | ~17 ms | 14% |

单看 op 本身，Krea-2 的两种 block op 在 eager 下比「带宽下限」慢一个数量级——
因为它们是**多趟**的（fp32 upcast → 统计 → 再乘 → cast → modulate → 再加）。
实测（`bench ops`，bf16，单次调用，取中位数）：

| op（shape） | eager | Triton | compile | 谁赢 |
|-------------|------:|-------:|--------:|------|
| `rms_norm_modulate` (1,2048,6144) | 2.278 ms | 0.164 ms (13.9x) | **0.113 ms (20.2x)** | compile |
| `rope_apply` (1,2048,48,128) | 4.412 ms | **0.131 ms (33.7x)** | 0.176 ms (25.0x) | **Triton** |
| `silu_mul` (1,2048,16384) | 0.848 ms | 0.505 ms (1.68x) | 0.499 ms (1.70x) | 平 |
| `add_gated` (1,2048,6144) | 0.401 ms | 0.312 ms (1.29x) | 0.299 ms (1.34x) | compile（微弱） |

一个 block 有 2 次 norm、2 次 rope、1 次 silu_mul、2 次 gated residual：eager ≈ 14.7 ms，
Triton ≈ 2.4 ms，**省 ~12 ms/block（约 block 的 10%）**。结论修正：

- **大 D 模型上 elementwise 融合依然值 10% 左右**，但天花板由 GEMM 与 attention 决定——
  所以顺序是「先修 §2（attention），再开 §3（compile），最后才是这些 kernel」；
- 小 D 的模型（文本融合 block、refiner block）里 elementwise 占比更高，收益更大；
- **`torch.compile` 在 4 个 op 里有 3 个赢**，这正是 KohakuMoss 的结论；本包的策略是
  「compile 开着时让 inductor 做，compile 关着时用 Triton」，见 §5.4。

接入方式见 §5.4：`kernels/patch.py` 在 build 时把这几个 `forward` 换掉（可逆），
不改模型源码。

---

## 5. 交付物与实测

新增 `UnifiedTrainer/kernels/`（**3196 行**，全部带原作者 banner，见
`kernels/ATTRIBUTION.md`）：

| 文件 | 作用 | 行数 |
|------|------|-----:|
| `eager.py` | **规范参考**：`sdpa_gqa`、`rms_norm(_modulate)`、`silu_mul`、`rope_apply`、`add_gated`，纯 torch | 154 |
| `attention_triton.py` | flash attention（GQA + 可选 key mask，fwd+bwd，两平面行统计） | 876 |
| `blockops_triton.py` | 4 个融合 block op（含反向） | 1252 |
| `__init__.py` | 三档模式、op 选型、diffusers attention hook | 325 |
| `select.py` | 按形状实测一次的选型表（`UTR_KERNELS_TUNE`） | 158 |
| `patch.py` | build 时替换 4 个 `forward`（可逆） | 145 |
| `bench.py` | 三列候选的基准 CLI | 286 |

测试：`tests/test_kernels_{patch,attention,blockops}.py`，共 **141 项**（18 + 27 + 96）。

### 5.1 Krea-2 单 block（fwd+bwd，bf16，base 冻结 = LoRA 训练真实算力形态）

| tokens | 原始（GQA math） | 修 GQA | + kernels | compile | compile + kernels |
|-------:|-----:|--------:|--------:|--------:|------------------:|
| 1024 | 64.3 ms | 52.6 ms (1.22x) | **48.1 ms (1.34x)** | 45.2 ms (1.42x) | 44.5 ms (1.45x) |
| 2048 | 165.0 ms | 120.3 ms (1.37x) | **109.5 ms (1.51x)** | 102.4 ms (1.61x) | 103.0 ms (1.60x) |

（free 7.5 GB / util 1–3%，`bench block --tokens 1024 2048`。「+ kernels」= `patch.install()`
+ attention hook + Triton attention。）

读法：

- **只在模型里修 GQA 就拿到 1.22–1.37x**（三行代码，零风险）；
- **kernels 层（block op 补丁 + Triton attention）把总量推到 1.34–1.51x，且不需要
  `torch.compile`**——这对 `offload_base_weights` / `block_swap` / gradient checkpointing
  这些会跟 compile 打架的配置很重要；
- **compile 再快一点（1.42–1.61x），但与 kernels 不叠加**：compile 开着时，包装函数与
  自定义 autograd Function 都主动退回原实现，让 inductor 自己融合（§5.4 的组合规则，
  `compile+kernels` ≈ `compile` 就是这个原因）。
- 两项都开没有额外损失，所以默认配置用 `kernels: auto`、`compile` 按需开。

### 5.2 attention op（B=1，bf16，fwd+bwd，跨模型真实头配置）

`python -m UnifiedTrainer.kernels.bench attention --seq-lens 1024 2048`
（free 3.0 GB / util 1% 时测得；`legacy` = 各适配器修 GQA 之前的写法，`sdpa` = 展开 KV 后的
SDPA，`triton` = `kernels.attention_triton`；括号内是相对 legacy 的倍数）：

| 模型（heads） | L | legacy | sdpa（展开 KV） | Triton | Triton 相对 legacy |
|---|---:|---:|---:|---:|---:|
| krea2（48/12） | 1024 | 17.63 ms | 5.90 ms (2.99x) | **3.77 ms** | **4.68x** |
| krea2（48/12） | 2048 | 78.45 ms | 23.35 ms (3.36x) | **13.28 ms** | **5.91x** |
| flux2_klein（24/24） | 1024 | 3.32 | 3.49 | **1.96** | 1.69x |
| flux2_klein（24/24） | 2048 | 11.37 | 11.17 | **6.60** | 1.72x |
| flux3_klein（24/24） | 1024 | 2.93 | 2.93 | **1.95** | 1.50x |
| flux3_klein（24/24） | 2048 | 11.30 | 11.46 | **7.11** | 1.59x |
| qwen_image（24/24） | 2048 | 11.15 | 11.46 | **6.42** | 1.74x |
| qwen_image21（32/32） | 2048 | 15.04 | 15.09 | **8.88** | 1.69x |
| minimax_h3（56/56） | 2048 | 26.79 | 26.59 | **15.31** | 1.75x |

两个要点：

1. **对非 GQA 模型，legacy 与 sdpa 打平**（本来就是同一条 SDPA 调用），
   Triton 稳定快 1.5–1.75x（因为省掉了 `(B,L,H,D)→(B,H,L,D)` 的 4 次 permute/copy，
   并且没有 KV 展开的两次 `repeat_interleave`）；
2. **krea2 是唯一的「被 math 后端坑到」的模型**：修 GQA 得到 3.0–3.4x，
   再上 Triton 到 4.7–5.9x。

**过程中修掉的一个 kernel bug**（`tests/test_kernels_attention.py` 抓到的）：
additive mask 若用「有限极小值」而不是 `-inf` 表示屏蔽（HuggingFace 风格
`torch.finfo(torch.float32).min`），`s + mask` 仍是有限数，行最大值与行和相差 ~1e38，
反向落到 `inf - inf` → `dQ` 全 NaN。修法：`_apply_mask` 把 `<= -1e30` 的权重当作硬屏蔽
（等价于 bool mask / `-inf`），真实的 additive bias（-5、-1e9）不受影响。修后
bool / `-inf` / `finfo.min` 三种写法在整行屏蔽时都得到 `O=0, dQ=0`、无 NaN
（`tests/test_kernels_attention.py::test_fully_masked_attention`）。

> 该表在共享卡上测得（另一进程长期占用 ~11 GB，数字会整体漂移 3–5x）；
> 判据是**同一行内的相对倍数**，在两轮独立测量中一致（±10%）。第一轮曾测得
> krea2 L=1024 的 Triton 慢于 SDPA（9.6 vs 5.9），第二轮为 3.77 vs 5.90——第一轮
> 是冷缓存 + 显存压力下的异常值，这也是 §5.5 保留「按形状实测」的原因。

### 5.3 融合 block op（vs eager vs compile）

`python -m UnifiedTrainer.kernels.bench ops`，bf16，单次调用：

| op | shape | eager | Triton | compile |
|----|-------|------:|-------:|--------:|
| `rms_norm_modulate` | (1,2048,6144) | 2.278 ms | 0.164 ms | **0.113 ms** |
| `rope_apply` | (1,2048,48,128) | 4.412 ms | **0.131 ms** | 0.176 ms |
| `silu_mul` | (1,2048,16384) | 0.848 ms | 0.505 ms | **0.499 ms** |
| `add_gated` | (1,2048,6144) | 0.401 ms | 0.312 ms | **0.299 ms** |

正确性：`tests/test_kernels_blockops.py` 用 fp64 标尺（cosine + abs/rel）逐 op 钉住，
bf16 下 rel ≈ 1e-3，rope 的 rel ≈ 2e-8（纯乘法，没有可丢精度的归约）。

### 5.4 绑定方式：build 时换 `forward`，可逆

`kernels/patch.py::install()` 在模型加载后把 4 个 `forward` 换成融合实现，`remove()` 原样
还原（`tests/test_kernels_patch.py` 断言还原后**逐位相同**）：

| 被替换者 | 换成 | 触发条件 |
|----------|------|----------|
| `Krea2RMSNorm.forward` | `kernels.rms_norm_modulate(offset=1.0)` | `training.kernels != off` |
| `Krea2SwiGLU.forward` | `kernels.silu_mul` | 同上 |
| `Flux2SwiGLU.forward` | `kernels.silu_mul` | 同上 |
| `apply_rotary_emb`（各 adapter 模块内的同名符号） | `kernels.rope_apply` | 仅 `sequence_dim=1` 且 bf16/fp16 且 4-D 输入；其它一律转发原函数 |

**与 compile 的组合规则**：包装函数里先判 `torch.compiler.is_compiling()`，是则调用原实现
——自定义 autograd Function 在编译区里会 graph break，而 inductor 本来就能把这几个点式
链融合掉（§5.3 的 compile 列）。所以：

- `compile: false` + `kernels: auto` → Triton：norm 13.9x、rope 33.7x、silu 1.68x、gate 1.29x；
- `compile: true` + `kernels: auto` → 这些 op 交给 inductor（norm 20.2x、rope 25x），
  attention 仍由 hook 决定（compile 下走 SDPA）。

### 5.5 按形状实测选型（`kernels/select.py`）

§5.2 里 krea2 的 L=1024 反例说明：**没有一个后端在所有形状上都最快**。这正是
KohakuMoss 的 `tuning/` 要解决的问题。本包取了它的最小形式：

1. 第一次见到某个 `(sm, q.dtype, 形状, mask)` 时，在**同形状的合成张量**上把
   `sdpa_gqa` 与 `attention_triton` 各跑一次 fwd+bwd（warmup 2 + 中位数 5 次）；
2. 把结论存进进程内表（可选落盘：`UTR_KERNELS_CACHE=path.json`）；
3. 之后每次调用只查表——**每步路径上没有 `if` 之外的代价**；
4. 判定失败（显存不足 4 倍探针占用、Triton 报错、`torch.compile` 追踪中）→ 退回 SDPA。

实测（本机，krea2 形状）：

| 形状 | 首调（含探针 + Triton 编译） | 之后 | 判据 |
|------|----------------------------:|-----:|------|
| 48/12, L=1024 | 329 ms | 0.91 ms | Triton 赢（0.91 vs SDPA 1.63，仅前向） |
| 48/12, L=2048 | 337 ms | 3.82 ms | Triton 赢 |
| 24/24, L=1024 | 55 ms | 0.49 ms | Triton 赢 |
| 24/24, L=2048 | 151 ms | 1.74 ms | Triton 赢 |

> 注意：探针测的是 **fwd+bwd**，表中「之后」列是**仅前向**，两者不可直接比较。
> 在本机上探针对四个形状都判 Triton 胜（与 §5.2 的最终表一致）。保留这个机制，是因为
> 第一轮测量曾出现 krea2 L=1024 的 Triton 慢于 SDPA（9.6 vs 5.9）的反例——冷缓存、
> 共享卡显存压力都会造成这种翻转。护栏是第 4 条：探针判输就走 SDPA，判赢才用 Triton，
> 每个新形状（每进程）只测一次。

开关：`UTR_KERNELS_TUNE=auto|off|eager`（`off` = 不测、直接用 Triton；`eager` = 不测、
永远 SDPA）。

---

## 6. 对每个适配器意味着什么

| 模型 | attention 头 | GQA | attention hook | block-op patch | 备注 |
|------|--------------|-----|----------------|----------------|------|
| `krea2` | 48 / 12，D=128 | 是 | ✅ 修正 math 回落（1.37x）+ Triton | ✅ norm / SwiGLU / rope | 收益最大；本就是最常用的训练目标 |
| `flux2_klein` | 24 / 24，D=128 | 否 | ✅ Triton（若赢过 mem-efficient） | ✅ SwiGLU / rope | 其 block 已把 QKV 与 MLP 融合进一个 GEMM（`to_qkv_mlp_proj`），没有可再融合的线性层 |
| `flux3_klein` | 24 / 24，D=128 | 否 | ✅ | ✅ rope（sequence_dim=1） | 三模态 + joint block，序列更长，attention 占比更高 |
| `qwen_image` | 24 / 24，D=128 | 否 | ✅ | ⛔ 自带 `apply_rotary_emb_qwen`（复数形式），未接 | rope 需单独写 kernel |
| `qwen_image_edit` | 继承 qwen_image | 否 | ✅ | ⛔ 同上 | |
| `qwen_image21` | 32 / 32，D=128 | 否 | ✅（BlockMask 路径原样转发） | ⛔ 自带 rope | 其 block-causal 走 flex，见 §9 |
| `minimax_h3` | 56 / 56，D=128 | 否 | ✅ | ⛔ 自带 rope | 视频序列最长，attention 占比最高 |
| `sd3` / `sd35` / `z_image` | diffusers 原生处理器 | 否 | ✅（hook 同样生效） | ⛔ | |

一句话：**hook 装一次，7 个模型都用上**；真正需要改模型代码的只有 krea2 的 GQA 那三行。

---

## 7. 没有采用的（以及为什么）

按 `md/13-kohakumoss-kernels.md` §14 的移植清单逐项裁决：

| 技术 | 裁决 | 理由 |
|------|------|------|
| packed varlen（`SeqInfo` + `cu_seqlens`） | 不采用 | UnifiedTrainer 按 bucket 出定长 batch（且 krea2 已 pad 到 256 的倍数），没有变长轴；引入 packed 布局要改数据层与所有 scale/shift 语义，收益只在「batch 内多分辨率」时才出现 |
| fp8 compute（sm89+） | 不采用 | 参考实现自己的实测：LoRA 场景下 base 冻结但激活仍需逐次量化，收益被吃掉；本仓库已用 `torchao` fp8 量化 base |
| int8 weight-only + ConvRot（Hadamard 旋转） | 暂不采用 | 需要「已旋转」的权重与离线预处理；当前 `quantize: torchao_float8` 已覆盖显存诉求。若要进一步压显存，这是下一个候选 |
| TileLang 后端 | 不采用 | 本机未安装，且只对 sm70/sm75 有意义；sm120 由 Triton 覆盖 |
| 240 条 profile 目录 / 逐卡 tuning 表 | 暂不采用 | 只有少数几个 op、一种卡型时，表带来的复杂度大于收益；`bench.py` 是将来落表的入口 |
| fused AdamW / 8-bit optimizer | 已有 | `optimizer: adamw8bit` 已在用；可训练参数只有 adapter，优化器不是瓶颈 |
| 文本编码器 kernel / delta-rule / GLA | 不采用 | 这些是 LLM 侧的 kernel，UTrainer 的文本编码器只做前向 |

---

## 8. 复现

```bash
cd E:\UnifiedTrainer\UTrainer
set UTR_KERNELS=auto            # off | auto | triton

# attention：跨模型真实头配置，三列候选
python -m UnifiedTrainer.kernels.bench attention

# 单 block：原始 / KV 展开 / compile / compile+kernels
python -m UnifiedTrainer.kernels.bench block --tokens 1024 2048
# 真实训练形态（gradient checkpointing 打开，会慢 ~1.6x）
python -m UnifiedTrainer.kernels.bench block --tokens 2048 --checkpoint

# 融合 block op：eager / Triton / compile
python -m UnifiedTrainer.kernels.bench ops
```

正确性：

```bash
python -m pytest tests\test_kernels_patch.py tests\test_kernels_attention.py tests\test_kernels_blockops.py -q
```

---

## 9. 后续（按收益排序）

1. **把 Triton attention 包成 `torch.library.custom_op` + `register_autograd`**，这样
   `compile` 与自定义 kernel 可以叠加（现在是二选一），预期在 §5.1 的 102.4 ms 上再省
   attention 的差值（Triton 15.3 ms vs 编译后 attention）。
2. **`BlockMask` 路径的 Triton 化**：qwen_image21 的 block-causal attention 目前走
   flex（未 compile 时极慢，其自身有 warning）。把 mask 结构化成「段表」传给 kernel，
   可以同时替代 flex 的 compile 依赖。
3. **逐卡 tuning 表**（KohakuMoss 的 `tuning/` 模式）：当 op 数或卡型变多时，把
   `bench.py` 的结果落成 JSON，按 (op, arch, dtype, shape) 选型。
4. **fp16 残差流（residual-stream fold）**：无 bf16 的卡（sm75）上训练 fp16 需要它；
   本机 sm120 有 bf16，暂不需要。
