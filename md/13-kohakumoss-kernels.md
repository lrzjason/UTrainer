# KohakuMoss kernel 库：可移植技术萃取（`kohakumoss_kernels`）

> 状态：技术调研（2026-10-19）。**不是移植计划**：本文把
> `ref/KohakuMossTrainer/src/kohakumoss_kernels/`（113 文件 / ~14k 行）里可复用的
> kernel 与工程手法拆成 9 项技术，每项给「省什么 / 源码 / 外来 trainer 必须提供的契约 /
> 在 UTrainer 的可移植性裁决」。
>
> 裁决基准：**RTX 5060 Ti 16 GB，sm120，torch 2.10 + triton（已装），tilelang 未装**，
> Windows；UTrainer 侧是纯 PyTorch MMDiT/DiT（flux2_klein / flux3_klein / krea2 /
> qwen_image / qwen_image21 / minimax_h3），LoRA/LoKr。
>
> `path:line` 约定：kernel 包内文件相对 `src/kohakumoss_kernels/`（如 `registry.py:414`、
> `backends/triton/attention/varlen.py:97-102`），`docs/…`、`tests/…`、`pyproject.toml`、
> `CLAUDE.md` 相对仓库根。
> 作者的设计笔记在 `docs/internals/{kernel-dev,tuning,quant-kernels,fp16}.md` 与
> `docs/performance/{attention,block-ops,quant-rotate}.md`，本文只引用其中的**实测数据**。

## 0. TL;DR：9 条结论

| # | 技术 | 在 16 GB sm120 上的真实收益 | 成本 | 是否做 |
|---|------|------------------------------|------|--------|
| 1 | varlen（packed）attention | batch=1 单段**打平**；**Windows 上 2.07–3.62x**（SDPA 退化 mem-efficient）；变长打包 1.6–4.4x | 中（~800 行） | **建议**（Windows 这一条就够） |
| 2 | fused block ops（norm+adaLN / residual / RoPE / GLU） | 单 op 2–30x（vs eager），省 HBM pass；峰值显存 −几十~几百 MB | 低（每 op 80–200 行） | **强烈建议**，性价比最高 |
| 3 | fused linear（GEMM + epilogue） | vs eager 2.5x；sm120 上比 compile 快 **3–36%**；省一趟 `[N,2F]` 读写（**512 MiB/趟** @16384 token） | 中（~600 行） | **建议**，先做 `linear_gated_residual` |
| 4 | 不物化的 loss | 省 **384–576 MiB** 瞬时显存（16384×3072 fp32 临时量） | 极低（~120 行） | **建议** |
| 5 | `torch.compile` 作为一等候选 | 单样本 norm/residual/loss 上 compile **赢 8–40%** | 极低（~200 行） | **强烈建议**（先做它再写 kernel） |
| 6 | fp8 / int8 weight-only + ConvRot | int8 把 base 权重压到 1/2；fp8 在 LoRA 下被「x 必须 16-bit」抵消 | 高（fp8 1.5k 行；int8 600 行） | **只做 int8 weight-only**；fp8/mxfp8 暂缓 |
| 7 | fused optimizer | 8-bit state：状态 8 B → 2.06 B/参数；multi 把 N 次 launch 变 1 次 | 低（~400–570 行） | **建议**（LoRA 参数多而小） |
| 8 | tuning harness（表 + 预热） | 不是加速，是**让前几项稳定生效**且不污染 step | 中（609 行可裁到 ~250） | **建议**（裁剪版） |
| 9 | precision bar（fp64 标尺） | 不是加速，是**证明移植正确** | 极低（68 + 64 行） | **强烈建议**，第一天就装 |

反向结论：**不要移植整个框架**。价值在这 9 个独立手法；1/2/3/4/5/7/9 只依赖
`torch`（+`triton`），与 MOSS 模型结构、`SeqInfo`、TileLang、KohakuEngine 无关。

## 1. 文件地图与依赖边界

### 1.1 子包 → 职责 → 关键文件

| 子包/文件 | 职责 | 关键位置 |
|-----------|------|----------|
| `ops.py` | op 名字的唯一真相：每组 op 的 tuple + 全部 eager 签名（docstring 即契约） | `ops.py:52-77`（`NORM/ROPE/ATTENTION/FFN/RESIDUAL/LOSS/FUSED/OPTIM/QUANT/ROTATION/TEXT/ALL`），签名 `:5-46` |
| `registry.py` | 「选择而非分派」：`register`/`resolve`/`Bound`/回退链/guarded autograd | 454 行；`DEFAULT_PRIORITY` `:32-43`；`Bound` `:101-224`；`_Guarded` `:280-314`；`resolve` `:414-429` |
| `constraints.py` | 声明式 `supports(query)`：device kind / sm 区间 / dtype / bf16 能力 | `constraints.py:10-31` |
| `errors.py` | 「算不能跑」的错误分类、粘性 CUDA 错误、`InputError` | `errors.py:12-20`、`:111-136`、`:76-85` |
| `error_stats.py` | fp64 标尺：cosine + abs/rel 的 mean/median/min/max + dtype 舍入地板 | `:11`（`KEYS`）、`:15-33`、`:36-49`、`:62-67` |
| `status.py` | 每 (op,backend,arch,dtype,static) 的失败记录；探测；按环境指纹落盘 | `:71-83`（`failure_key`）、`:90-108`（`probe`）、`:111-133`（`record`） |
| `cache_tag.py` | 把「注册过 custom op 的源码 hash」塞进 compile cache key | `:15-16`（`MARKERS`）、`:20-25`、`:58-62` |
| `saved_ops.py` | selective checkpoint 的白名单（这些 forward 的输出值得留） | `:8-19`（`KEPT`/`keep`/`keep_named`） |
| `recording.py` | 录制一次跑批里的 `Bound` 调用（张量快照；小张量存值，大整型存 min-max） | `:13`（`KEEP_NUMEL=1<<16`）、`:17-45`（`Snap`）、`:92-104` |
| `dist.py` / `dtypes.py` | 进程组 (rank,world) / dtype↔字符串 | `dist.py:6`、`:14-23`；`dtypes.py:7-22` |
| `hadamard.py` | ConvRot 的数学定义（H4 的 Kronecker 幂）+ fp32 蝴蝶 eager 实现 | `:14-15`（`H4`/`GROUPS`）、`:88-98` |
| `run_state.py` / `compile_setup.py` | run 的进程态复位 / compile 设置（cache tag + 不 quiesce 的编译池） | `run_state.py:10-17`；`compile_setup.py:11-34` |
| `backends/eager/` | **规范参考**：7 个模块，`__all__` 里的名字都注册成该 op 的 eager 实现 | `eager/__init__.py:17-19`；`norm/rope/attention/ffn/loss/textenc/hadamard.py` |
| `backends/compile/` | `torch.compile(eager)` 作为一等候选 + kernel-vs-compile 决策 | `compile/ops.py:37-51`、`compile/choice.py:123-280` |
| `backends/triton/` | sm80+ 主路径：attention/blocks/textenc/fp8/optim/quant | 见 §3–§9 |
| `backends/tilelang/` | sm70/sm75 与 sm80+ 的备选 | **目标机不需要**（未装） |
| `backends/_fused_linear.py` | 三个 fused linear 的共享宿主代码（op 定义、autograd、fused/split 选择） | 502 行；`FusedBackend` `:52-77`、`FusedLinear` `:80-302` |
| `backends/_master.py` / `_memo.py` | fp32 master 参与 16-bit matmul / 按「张量+version」缓存派生量 | `_master.py:10-38`；`_memo.py:16-35` |
| `tuning/` | 每卡配置表：解析排序 → 计时短名单 → 落盘 → 预热回放 | `tuner.py`(609)、`buckets.py`(64)、`window.py`(72)、`plan.py`(38)、`timing.py`(86)、`prewarm.py`(106)、`replay.py`(293)、`manifest.py`(64)、`decisions.py`(101) |
| `fp8/` / `quant/` / `optim/` | fp8 参考 / weight-only 量化 / 优化器 step 的参考与选择 | `formats.py`(136)、`op.py`(177)、`train.py`(125)；`quant/__init__.py`(142)、`rotated.py`(144)；`reference.py`(131)、`choice.py`(112) |
| `hardware/` | arch 识别、设备实测、当前设备守卫、显存窗口、工具链 | `arch.py`(53)、`device.py`(188)、`current.py`(72)、`memory.py`(227)、`toolchain.py`(333) |

### 1.2 依赖边界（真正要一起搬的东西）

kernel 包只 import 一个兄弟包，且只用到 4 个模块：`kohakumoss_common.calls`（253 B，
`nothing`）、`.devices`（351 B，`STREAMED`）、`.io`（2.3 K，`read_json`/`write_json_atomic`）、
`.log`（18 K，`get_logger/hl/log_table/*_once`）—— 用法见 `registry.py:15`、`tuner.py:7`、
`status.py:17`、`hardware/device.py:14-15`。

| 依赖 | 谁需要 | 目标机 |
|------|--------|--------|
| `torch` | 全部 | 有 |
| `triton` | `backends/triton/*` + inductor 的 GPU codegen | 有（`errors.py:7-10` 对 `TritonError` 做了可选 import） |
| `tilelang>=0.1.15` | `backends/tilelang/*` | **无** → `load_backends` 标 unavailable（`registry.py:446-450`），不影响 triton |
| `anyschedule` / `kohakuvault` | `pyproject.toml:22-23` 的**核心**依赖，但只被 `kohakumoss.training`/`scripts` 用；kernel 包内零引用 | 无需 |
| `kohakuengine` | 只被 `scripts/config`、`*_schema.py` 用 | 无需 |

结论：kernel 包本体 = **torch + triton + 10 行日志/IO 垫片**。

## 2. 布局契约：packed flat token axis（一切的前提）

- **省什么**：变长打包把「padding 到 `B × L_max`」换成「`Σ L_i`」；同时让 attention 之外
  的每个 op 都退化成**没有 batch 维的 last-dim op**，一次 launch 覆盖整个 batch。
- **契约**（`ops.py:1-3`、`docs/concepts/seqinfo.md:1-11`）：激活统一是 `[N, ...]` 扁平
  token 轴，sample-major，`N = Σ L_i`；`seg[N] int32` 是每个 token 的样本号且**非降序**
  （`ops.py:3` 明写 segment reductions 依赖它）；attention 用
  `cu_seqlens[B+1] int32` + `max_seqlen: int` + `scale: float`（`ops.py:10-11`）；逐样本
  调制向量是 `[B, D]`，按 `seg` gather（`ops.py:6-7`、`:13`）。
- **两种编码可互推**：仓库用二分从 `seg` 在 **device 上**造 `cu`，全程不回读 host
  （`backends/triton/blocks/_blockops.py:23-37`、`:139-144`）。
- **裁决**：✅ 纯契约。UTrainer 的 bucket dataloader 有每样本 token 数，拼 `cu_seqlens`
  只需一次 `cumsum`（样例 `tuning/prewarm.py:86-87`）。

## 3. 技术 1：varlen（packed）attention，无 mask

### 3.1 契约

```python
# ops.py:10-11；参考实现 backends/eager/attention.py:48-55
attention_varlen(q[N,H,Dh], k[N,H,Dh], v[N,H,Dh], cu_seqlens[B+1] int32,
                 max_seqlen: int, scale: float) -> o[N,H,Dh]
```
`[N,H,Dh]` 允许任意 token/head stride，只有最后一维必须连续（`varlen.py:402` 才按需
contiguous）。Triton 内部入口多返回两个 fp32 平面 `m [H,N]`、`l [H,N]` 供 backward
复用，并用 `register_autograd` 挂反向（`triton/attention/varlen.py:322-338`、`:387`）；
用户可见的 op 只返回 `o`（`:395-423`）。

### 3.2 实现要点

| 手法 | 做法 | 源码 |
|------|------|------|
| 网格 | `grid=(cdiv(max_seqlen,BM), H, B)`；**第一维按最长段算，超出自己段长的 program 直接 `return`**，host 不必按 batch 重算 grid | `_kernels.py:46-51`；`varlen.py:191-195` |
| int64 token 偏移 | `start` 转 int64 后再算 `row*stride`（strided qkv view 里「段起点 × row stride」在 ~233k token 处越过 2^31 元素） | `_kernels.py:48`、`:57`；`kernel-dev.md:508-515` |
| 在线 softmax（**两平面**） | 存**原始 score max `m`** 与 **`log2(l)`** 两个 fp32 平面，而不是一个 LSE：`c*m+log2(l)` 会在 logit 大时吃掉 `log2(l)` | `_kernels.py:84-85`；`kernel-dev.md:517-526` |
| shift-before-scale | 概率=`exp2((s-m)*c - log2 l)`；`s-m` 在行最大附近精确，主概率恰为 1，`delta=rowsum(dO*O)` 才能抵消 | `_kernels.py:33-36`、`:304-308` |
| scale 留在累加器 | dS 以**未缩放**的 16-bit 舍入，softmax scale 最后乘到 fp32 的 dQ/dK 累加器 | `_kernels.py:199`、`:281`、`:362` |
| padded key / 空段 | tail block 走 `MASKED` 分支（掩码列 `-inf`、显式 `tl.where(cols,p,0)` 避免 `inf*0`）；空段直接 `return`，`n==0` 或 `max_seqlen==0` 返回空张量 | `_kernels.py:22-37`、`:210-213`、`:50`；`varlen.py:331-332` |
| 反向两策略 | `fused`：每 key block 算 dK/dV，dQ 用 **fp32 atomic** 累到 `[N,H,D]`，`delta_kernel` 顺手清零，最后 `dq_cast_kernel` 一次乘 scale+转 dtype；`split`：dK/dV + dQ 两个 kernel，无 atomic | `_kernels.py:205-284`、`:287-293`；`varlen.py:230-247` |
| 精度 | 每次 MMA 16-bit 操作数 + fp32 累加；P、dS 各只舍入一次；delta 取自 16-bit 的 O | `_kernels.py:15-18`；`kernel-dev.md:612-617` |

sm120 已实测的 seed（`varlen.py:97-102`）：

```python
ARCH_SEEDS[120] = {
    "fwd":   [_cfg(128, 64, 8, 3), _cfg(128, 64, 8, 2), _cfg(128, 32, 4, 2)],
    "dkdv":  [_cfg(32, 32, 4, 3), _cfg(64, 64, 8, 3), _cfg(32, 32, 4, 2)],
    "dq":    [_cfg(128, 32, 8, 3), _cfg(128, 64, 8, 2), _cfg(128, 32, 8, 2)],
    "fused": [_cfg(16, 128, 8, 4), _cfg(16, 128, 8, 3), _cfg(64, 64, 8, 3)],
}   # 另：ARCH_SEEDS[89] = ARCH_SEEDS[86]（varlen.py:106）
```
fused backward 的 `BM=16` 是关键：128 个 key 的 dK/dV 在 fp32 寄存器里，query tile
一大就输给寄存器压力（`kernel-dev.md:563-567`）。候选的合法性用两个**解析式**判
（`varlen.py:47-63` 的 `_smem` / `_acc_regs`，阈值 192 寄存器见 `:135-138`），不用试编译。

### 3.3 何时打得过 `F.scaled_dot_product_attention` + padding

| 场景 | 实测（`docs/performance/attention.md:38-59`） | 结论 |
|------|-----------------------------------------------|------|
| 单段长序列（2 MP×1） | 1.01–1.02x（3090/5090），1.42x（V100） | **打平**：FA2 已跑在卡的 MMA 上限（`attention.md:77-95`） |
| batch>1 等长 1 MP×2 | 1.08–1.11x（3090/5090），1.63x（V100） | 收益 = 1 次 launch 代替 B 次 + 省每段 transpose |
| batch>1 等长 512 px×8（短段） | 2.01–2.21x | 每段只有 ~1k token，SDPA 是 launch-bound |
| 变长打包 / text refiner | 1.16–4.39x | padding→`Σ L_i` 的收益 |
| **Windows（无 flash）** | 4090 Windows：1 MP×2 **13.0 vs 26.9 ms**（2.07x）、packed mixed 2.27x、text refiner 3.62x | **2.07–3.62x，移植的第一理由** |

Windows 这一栏的机制在 `attention.md:15`：flash 只存在于 Linux sm80+ 的 torch build，
Windows 的 SDPA 落 memory-efficient；仓库的 `flash_varlen_usable()` 也据此不给
`torch_varlen` 候选（`backends/varlen_choice.py:30-43`，要求 `torch.nn.attention.varlen`）。

### 3.4 裁决

- **省什么**：Windows 上的 2–3.6x；变长打包的 padding 浪费；一个 launch 覆盖整个 batch。
- **契约**：`cu_seqlens int32 [B+1]`（device）、`max_seqlen`（Python int，决定 grid 第一维）、
  `scale`（float）、`q/k/v` 最后一维连续。
- **成本**：`_kernels.py` 367 + `varlen.py` 431，去掉 tuning/choice 后约 **500 行**可跑；
  外面套一个 `torch.autograd.Function` 即可，不必搬 `triton_op`（除非也要 fullgraph compile）。
- **裁决**：✅ 可移植，且是「不做会明显吃亏」的一项（前提：跑 Windows）。
  batch 全由同分辨率 bucket 组成时，先做 §4/§6 再回来。

## 4. 技术 2：fused block ops（norm+adaLN / residual / RoPE / GLU）

### 4.1 norm + adaLN modulation

```python
# ops.py:5-7；语义 backends/eager/norm.py:21-39
rms_norm(x[N,D], weight[D]|None, eps) -> y[N,D]
rms_norm_mod(x[N,D], weight[D]|None, shift[B,D]|None, scale[B,D]|None,
             seg[N] int32|None, eps) -> y[N,D]      # y = rms_norm(x)*(1+scale[seg])+shift[seg]
```
一个 program = `ROWS` 行 × `next_pow2(D)` 列，全程 fp32，`rstd` 存 fp32 给反向
（`backends/triton/blocks/norm.py:39-82`）。**`shift/scale` 用 stride 读、绝不拷贝**：
kernel 收 `sh_row/sh_col/sc_row/sc_col`，按 `seg*row_stride + col*col_stride` 寻址
（`norm.py:67-80`、`:215-237`）——这是为 MOSS 的 `mod[:,k]`（`[B,6,D]` 切片，row stride=6D）
设计的；单样本时每行都读 row 0，只有 `B≥2` 才暴露布局错误（`kernel-dev.md:86-98`）。

反向不物化 `[N,D]` 中间量：`dx` 直接写，同时把 `sum(g·x̂)`、`sum(g)` 的 fp32 列部分和写进
`part [tiles,2,D]`；第二个 kernel 按样本把 tile 部分和按**固定顺序**相加，得
`d_shift=S_g`、`d_scale=w·S_gx`、`d_weight=Σ_b(1+scale_b)·S_gx`（`norm.py:144-207`）——
固定顺序 ⇒ **梯度逐位可复现**（`docs/internals/fp16.md:87-99`）。

⚠️ **移植注意**：MOSS 用 RMSNorm；flux2/flux3 的 adaLN 是
`LayerNorm(elementwise_affine=False) * (1+scale) + shift`。骨架（一行 tile、stride 读调制、
per-tile 部分和）完全复用，但**要 mean-center**：把 `ms = sum(x*x)/D`（`norm.py:69-71`）
换成 mean 与 var，反向 `dx = rstd*(gm - x̂·c)` 不变。**不要直接复用数值。**

### 4.2 residual（+ scale / gate）

```python
# ops.py:13-14；eager backends/eager/norm.py:42-51
gated_residual(x[N,D], o[N,D], gate[B,D]|None, seg[N]|None, r: float) -> x*r + o*gate[seg]
```
1 读 `o` + 1 读 `x` + 1 写 `y`；`SCALED = r != 1.0` 是编译期常量，`r==1` 时连乘法都不生成
（`blocks/residual.py:32-63`）。反向 `d_o=g*gate[seg]`、`d_x=g*r`，`d_gate` 用 per-tile 部分和
（`:66-109`、`:183`）。

### 4.3 RoPE（split-half，多轴）

```python
# ops.py:8-9；eager backends/eager/rope.py:28-41
qk_norm_rope(q[N,H,Dh], k[N,H,Dh], q_weight[Dh], k_weight[Dh],
             pos[N,A] f32, bands[Dh//(2A)] f32, eps) -> (q, k)
```
- **配对是 split-half**：`a,b = x.chunk(2,-1)`，旋转 `(j, j+Dh/2)`（`rope.py:23-25`），
  **不是 interleaved**。换 interleaved 要把 `_norm_rotate_fwd` 的 `offs`/`offs+HALF`
  改成 `2j`/`2j+1`（`rope.py:50-59`）。
- **频率是 axis-major**：第 `j` 个半维用 `pos[:, j//NB]` 与 `bands[j%NB]`
  （`rope.py:33-42`；eager 同构 `:14-16`）。`bands` 是**单轴** `inv_freq`（长度
  `NB=Dh/(2A)`），多轴只是 concat。
- `sin/cos` 在 kernel 里从 `pos*band` 现算（`libdevice.sin/cos`），省掉两张 `[N,Dh]` 表
  （4096×128 fp32 = 2 MiB/层/次）。
- **q 与 k 在同一个 program**：rows 是 `(token, head)` 对，角度只算一次（`rope.py:83-87`）
  ——这是把 q/k 两个 norm+rope op 合成一次 launch 的关键。
- 反向：先把梯度旋回去（`dan=ga*c+gb*s`）再走 RMSNorm 反向，per-program 的 4 个 fp32
  列部分和（qa/qb/ka/kb）事后求和（`rope.py:90-173`）。

⚠️ **移植核对点**：若 UTrainer 的 vendored transformer 沿用 diffusers 的 `rotate_half`
（`chunk(2,-1)`）+ `cos/sin.repeat(1,2)`，则与这里**逐位一致**，`bands` 就是
`get_1d_rotary_pos_emb` 里的 `inv_freq`（`Dh=128, A=4 → NB=16`）。移植前用 §11 的方法核一次。

### 4.4 FFN gating（SiTU-GLU = SwiGLU 的同构替换）

```python
# ops.py:12；eager backends/eager/ffn.py:11-15
situ_glu(x[N,2F]) -> a[N,F]      # x = [up | gate]
# a = (4*tanh(g/4)*sigmoid(g)) * (25*tanh(u/25))
```
kernel 一个 program 一次读入 `[N,2F]` 的前后半（`ffn.py:47-56`），反向 1 读 1 写（`:60-72`）。
`situ()`/`situ_grad()` 被**复用为 GEMM 的 epilogue**（`ffn.py:26-43` 被 `_gemm_kernels.py:11`
import）——§5 的伏笔。换 SwiGLU 只改这两支：`silu(u)*g` 及其导数。

### 4.5 省了多少次 HBM pass

| op | eager 组合的 kernel 数（每个逐元素 op 一趟，未复用 allocator） | fused | sm120 fp16 实测 Triton / compile / **eager**（`block-ops.md:100-125`） |
|----|------|-------|------|
| `rms_norm_mod`（N=16400,B=4,dx） | ≥10 | 1 读+1 写 | 0.368 / 0.363 / **10.78 ms** |
| `rms_norm`（N=16400） | ≥6 | 1 读+1 写 | 0.462 / 0.530 / **8.63** |
| `gated_residual`（gate frozen） | ≥4 | 2 读+1 写 | 0.343 / 0.524 / **4.56** |
| `qk_norm_rope`（N=17381） | ≥25（q/k 各一遍）+2 张表 | 2 读+2 写（q、k 各一） | 0.662 / 2.520 / **26.06** |
| `situ_glu`（N=16400） | ≥8 | 1 读+1 写 | 1.261 / 1.752 / **41.25** |

**省什么**：① HBM 带宽（10–30x）；② 峰值显存（每省一趟 `[N,D]` 就是 `N·D·2` 字节，
16384×3072 fp16 = **96 MiB**）；③ 反向归约不用 atomic 且逐位可复现。

**契约**：`x [N,D]` 最后一维连续；`shift/scale/gate` 可为任意 stride 的 `[B,D]`；`seg`
非降序 int32；`r` 是 Python float（编译期常量）。

**裁决**：✅ 纯 Triton+torch，每 op 80–200 行。**先做 `rms_norm_mod`（LayerNorm 版）与
`gated_residual`**，它们同时决定 §5 能不能接上。

## 5. 技术 3：fused linear（GEMM + epilogue），以及「QKV 一个 GEMM」

### 5.1 契约

```python
# ops.py:21-23；实现 backends/_fused_linear.py:237-302
linear_qk_rope(y, wq, wk, q_weight, k_weight, pos, bands, eps, heads) -> (q, k)
linear_situ(y, w_up_gate, bias) -> situ_glu(y W^T + b)
linear_gated_residual(a, w, bias, x, gate, seg, r) -> gated_residual(x, a W^T + b, ...)
```

### 5.2 结构：一套宿主代码 + 每后端只写 launcher

`FusedBackend`（`_fused_linear.py:52-77`）只描述后端差异：`define`（怎么注册 custom op）、
`fused[op]`（GEMM+epilogue launcher）、`cfgs[op]`（tile 配置）、`situ/residual/rope`
（epilogue 的 fwd/bwd kernel）、`pick`（tuner 入口）、`fusable/dead/fallback`（能力与失败）。
`FusedLinear`（`:80-302`）提供 6 个 custom op body（`:312-398`）、3 组 autograd setup
（`:404-429`，**只在 autograd 记录时存反向要的东西**，`_kept()` 见 `:43-45`）、以及
**每问题在 `fused` 与 `split` 之间二选一**（`:105-108`）。`split` = cuBLAS + 独立 epilogue
kernel，`fused` = 一个 kernel。

### 5.3 值得偷的 kernel 手法

| 手法 | 说明 | 源码 |
|------|------|------|
| progid swizzle | 1-D grid + `GROUP_M=8`，让同组 program 复用 L2 里的 A tile | `_gemm_kernels.py:14-22` |
| SiTU 一个 program 算两半 | `acc_u = A·W_up^T`、`acc_g = A·W_gate^T` 都留寄存器，epilogue 直接组合，只写 `out`（需要时同时写 `ug`） | `:42-85` |
| residual 折进 epilogue | fp32 `acc` → 乘 `gate[seg]` → 加 `x*r` → 一次舍入写出；`SAVE` 时投影另写一份 | `:88-138` |
| qk 一 program = BM token × 1 head | grid=`cdiv(M,BM)*2H`，`pn<H` 算 q 否则算 k；GEMM→per-head RMSNorm→split-half RoPE 全在寄存器 | `:141-214` |
| dq/dk 拼一个 buffer | 反向把 dq、dk 写进 `[N,2D]` 两半，`dy = [dq|dk] @ [Wq;Wk]` **一个 GEMM**：一次 fp32 累加一次舍入 | `_fused_linear.py:381-386`、`_master.py:26-32` |
| 权重可以是 fp32 master | matmul 前 cast 到激活 dtype，梯度按存储 dtype 回写 | `_master.py:19-38` |

**关于「QKV 一个 GEMM」**：仓库**没有** q/k/v 合成一个 GEMM 的 op；`linear_qk_rope` 只把
**q 和 k** 放进一个 kernel（v 不需要 RoPE，走普通 linear）。真正的「单 GEMM」在两处可借：
① `stacked(wq,wk)` 把两个权重拼成一个 buffer 做反向（`_master.py:26-32`）；
② `linear_situ` 的 `[W_up; W_gate]` 本来就是**一个** `[2F,K]` 权重、一个 GEMM 出 `[N,2F]`，
epilogue 内部切两半（`_gemm_kernels.py:59-69`）。推广到 QKV = 在 dim 0 拼 `[3D,K]`，
epilogue 按 head 切；op schema 的参数个数照 `_fused_linear.py:172-182` 加即可。

### 5.4 内存账（16 GB 卡上最实际的部分）

`linear_situ` 的 `save` 判据是「grad 开着且 `y`、`w`、`bias` 任一 requires_grad」
（`_fused_linear.py:240-242`）：

| 训练形态 | `save` | 结果 |
|----------|--------|------|
| LoRA 在 `ffn_up_gate`（`y` 需要 grad、base `w` 冻结） | **True** | `ug` 仍要存（反向要 `situ_grad(ga,ug)`），但**省掉「读 ug 再算」那一趟**：N=16384、2F=16384、fp16 → **512 MiB 读** |
| 冻结层 + 输入也不需要 grad | **False** | `[N,2F]` **根本不出现**，省 512 MiB 峰值 |

同理 `linear_gated_residual` 的 `save = grad and gate is not None and gate.requires_grad`
（`:262`）。**关键洞察**：若 timestep embedding 冻结、输入是整数 timestep，则 adaLN 的
`gate` 不 requires_grad，`[M,N]` 的投影**不落盘**。这条判据比 kernel 本身更值钱。

### 5.5 裁决

- **省什么**：vs eager 2.5x（sm75 实测 74.9 vs 185.6，`block-ops.md:44`）；sm120 上比
  compile 快 3–36%（`linear_situ` 15.84 vs 17.07、`linear_qk_rope` 6.51 vs 10.23，
  `block-ops.md:122-125`）；外加 §5.4 的那一趟 `[N,2F]`。
- **契约**：`activation(y)`（fp32 输入在 autocast 下先 cast，`_master.py:10-16`）；权重任意
  存储 dtype；`gate/seg` 可 stride；`r` 编译期常量；qk 还要 `pos/bands/heads`，且 q/k norm
  权重以 **fp32** 进 op（`_fused_linear.py:293-296`）。
- **成本**：`_fused_linear.py` 502（可砍 TileLang 分支与一半 fake）+ Triton 侧 149
  + `_gemm_kernels.py` 214 ≈ **600 行**。
- **裁决**：✅ 建议顺序：`linear_gated_residual` → `linear_situ`（换 SwiGLU epilogue）→
  `linear_qk_rope`。**不做也能拿 80%**：`split` 路径（cuBLAS + standalone epilogue kernel）
  只多一次 `[M,N]` 往返，一天可落地。

## 6. 技术 4：不物化的 loss

```python
# ops.py:15-16；eager backends/eager/loss.py:10-19
token_mse(pred[N,C], target[N,C], seg[N] int32, weight[B] f32, mask[N] f32|None)
    -> fp32 scalar = sum_tokens weight[seg]*mask*mean_c((pred-target)^2)
```
- **eager 会物化什么**（`loss.py:14-19`，`dt=acc_dtype → fp32`）：`pred.to(fp32)`、
  `target.to(fp32)`、`(pred-target)`、`.square()`。`N=16384, C=3072` 时每个 `[N,C]` fp32
  张量 **192 MiB**，瞬时峰值 **~384–576 MiB**。16 GB 卡上这决定「能不能再塞一个 1024² batch」。
- **kernel**（`blocks/loss.py:27-55`）：一 program 处理 `ROWS=16` 行，行内 fp32 累计
  `sum((p-t)^2)/C`，乘 `weight[seg]*mask`，**每 program 只写一个 fp32 标量**到 `part[progs]`；
  最后 `total(part)` 固定顺序求和（**确定性**，`_blockops.py:205-208`）。没有任何 `[N,C]` 中间量。
- **反向**（`loss.py:58-91`）：只读一个标量 `gs`，`coef = gs*2/C*w`，`dp = d*coef`；
  target 的梯度是 `-dp`（`NEED_T` 控制）。forward 不保存任何张量。
- ⚠️ 语义是 **sum over tokens**、只在 channel 维取 mean（`ops.py:16`）。移植时必须核对
  你的 loss 是 mean 还是 sum，否则梯度尺度差 `N` 倍。
- **裁决**：✅ 成本最低、收益最确定（~120 行）。flow-matching MSE 正好是这个形状。

## 7. 技术 5：`torch.compile` 作为一等候选

### 7.1 它做了什么

- 每个 block op 一个 `compiled_op(name)` = `torch.compile(eager, dynamic=True)`
  （`compile/ops.py:219-224`），进程内每个 op 一个。
- 按 **call pattern** 缓存：`device / grad 模式 / intra-op 线程数 / autocast dtype /
  每参数的 dtype+requires_grad+rank / None 与 Python 标量`（`ops.py:83-105`）——
  **shape 不在里面**（`dynamic=True` 让 inductor 只编一次，靠 guard 复用）。
- 每个 pattern 用 `_fresh()` **复制一份函数对象与 code object**（`ops.py:108-116`），
  Dynamo 缓存互不干扰——这是绕开「同函数多 pattern 互相重编译」的关键技巧。
- `GRAPH_LIMIT=8`（`ops.py:54`）：一个 pattern 超过 8 张图就**永久退化为 eager**，且把
  Dynamo 自己的 `recompile_limit=64` / `accumulated_recompile_limit=4096` 抬高
  （`ops.py:63-69`），杜绝 Dynamo 静默 fallback。
- 编译池：`compile_setup.apply()` 设 `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0`
  （`compile_setup.py:11-34`）：否则 Linux 上 sidecar 的 32 个 worker 空闲 60 s 被 quiesce、
  下次编译重新 fork，**可能死锁**（`kernel-dev.md:353-369`）。

### 7.2 决策层：`KernelOrCompile`

- 候选只有 `{"impl":"kernel"}` 与 `{"impl":"compile"}`（`compile/choice.py:39`）。
- **problem key** = 各参数 shape（**首维按 `TOKEN_FLOOR=256` 向上取 2 的幂**，`choice.py:53-55`、
  `:336-351`）+ dtype + 标量 + 「哪些张量 requires_grad」的位串（`:143-148`）。
- **call key**（绑定键）= device + grad 模式 + 线程数 + autocast dtype + 每参数
  (bucketed shape, dtype, requires_grad)（`:58-75`）。problem 用于查表，call 用于免重入。
- **调用时绝不计时**：`_lookup` 只做 pinned→表→默认 kernel，查不到就进 `pending` 队列
  （`:169-192`）；`retune_pending()` 在预热结束/每个 retune step 后统一测（`:309-333`）。
- `_step()`（`:98-120`）保证测的是 **fwd+bwd**，与真实训练工作量一致。

### 7.3 实测决策规则（反直觉结论）

`block-ops.md:127-145`：
- **单样本、N 几千 token** 的 norm/residual/loss：**compile 赢 8–40%**——Triton 路径有
  custom-op dispatch + 2 个 Python autograd node + 2 次 launch 的 host 开销
  （5090 上 `rms_norm_mod` fwd+bwd：host 430 µs vs device 170 µs，`block-ops.md:132-133`）。
- **qk_norm_rope / situ_glu / 大 N 的 norm**：kernel 赢 2–4x。
- 结论：「kernel 还是 compile」必须按 (op, arch, dtype, shape) **实测出表**；sm86 与 sm120
  的表不一样（3090 上单样本 `rms_norm_mod` 是 compile、16400 token 是 kernel，`block-ops.md:81-98`）。

### 7.4 裁决

✅ **第一天就做**（~200 行：`compile/ops.py` 224 + `choice.py` 351，砍掉 pending/retune）。
有了它，§4 的每个 kernel 都可以「先上 compile，测完再决定要不要手写」。

## 8. 技术 6：fp8 / MXFP8 / int8 weight-only + Hadamard 旋转

### 8.1 两套东西，别混

| 套件 | 对象 | 存储 | 硬件 | 入口 |
|------|------|------|------|------|
| **fp8 compute**（`fp8/`） | 冻结 base 的 **W 与激活都量化** | e4m3，1 B/权重 | `sm89+`（`quant/__init__.py:130-137`、`triton/fp8/linear.py:319`） | `fp8_linear`（`fp8/op.py:148`）、`fp8_linear_train`（`fp8/train.py:115`） |
| **weight-only**（`quant/`） | **只有 W 量化**，激活 16-bit | int8 或 e4m3 + per-row fp32 scale | int8 `sm80+`；fp8 存储 `sm89+`（`triton/quant.py:271-276`） | `quant_linear`（`quant/__init__.py:117`）、`quant_linear_rotated`（`quant/rotated.py:109`） |

sm120 两者都能跑；**LoRA 下只有 weight-only 有意义**（见 §8.5）。

### 8.2 fp8 的四种 recipe（精确数值）

```python
# fp8/formats.py:9-13（docstring 即规范）；RECIPES=("tensor","row","block","mx")  :25
tensor  一个 fp32 scale [1]
row     每行一个 fp32 scale [rows]
block   激活：每行 × 128 列一个 fp32 [rows, K/128]；权重：128×128 块一个 [ceil(rows/128), K/128]
mx      OCP MX：每 32 列一个 e8m0（2 的幂）scale [rows, K/32]，uint8（biased +127）
```
- `s = max(amax,1e-12)/448`，`q = clamp(x/s).to(e4m3)`（`formats.py:44-52`）。
- **fp32 scale 必须「张量除张量」**：PyTorch 标量除走倒数、某些 amax 差 1 ulp；仓库让
  eager 除 tensor、Triton 用 `tl.div_rn`，于是**kernel 与参考字节相等**
  （`docs/internals/quant-kernels.md:319-324`）。
- **e4m3 转换先在 fp32 上做 RNE**（magic-number add）再交给硬件：Triton 在 sm89 的
  fp32→e4m3 会经 fp16（两次舍入），字节不同（`triton/fp8/quantize.py:36-45`；
  `quant-kernels.md:324-327`）。
- **mx 指数向上取整**：最小 `e` 使 `2^e*448 >= amax`，从 amax 的位里算
  （`formats.py:55-61`：`biased=((bits>>23)&0xFF)-8+up`，`up = mantissa>0x600000` 即 `m>1.75`）。
  理由：RNE 会**裁掉 block 最大值**，是 NVIDIA 与 KohakUwULLM 都测到的发散来源
  （`quant-kernels.md:328-332`）。
- `swizzle_mx`（`formats.py:128-136`）把 scale 排成厂商 `SWIZZLE_32_4_4` 的 128×4 tile 布局。

### 8.3 训练期 fp8 怎么跑

- `fp8_linear(x, q, scale, aux, q_t, scale_t, aux_t, bias, recipe)`（`fp8/op.py:148-155`）：
  冻结 base，`x` 每次调用现量化（straight-through，反向不过 quantizer）。
- 输入梯度两条路：有 `q_t/scale_t`（W^T 的 fp8 副本）走 fp8 dgrad，否则 16-bit
  `dy @ dequant(W)`（`fp8/op.py:34-40`、`train.py:60-64`）。`fp8_linear_train` 多一个 `w`，
  权重梯度是 16-bit 的 `dy^T x`（`train.py:75-86`），fp8 副本由调用方每步重建（`train.py:1-5`）。
- kernel 侧三种累加（`triton/fp8/gemm.py:1-13` 的 docstring 是权威表）：`acc="promote"`
  （每 BK 深的 MMA 结果乘激活 scale 再加进 fp32，`gemm.py:91-98`）、`"mma"`（MMA 累加器
  跨 K 链下去，scale 在 epilogue 乘，`:99-105`）、`"upcast"`（e4m3 精确转 fp16 再走 fp16 MMA，
  `:52-58`，硬件无 fp8 单元时的兜底）。per-K-block scale（block/mx）走 `promote_mm`
  （`:108-134`），mx 的 uint8 scale 用 `_pow2` 位运算还原 `2^(b-127)`（`:61-63`）；
  mx 原生路径是 `tl.dot_scaled`（`:137-171`），**只在 `sm>=100` 启用**（`linear.py:108-113`）。
- **一 launch 版**（`triton/fp8/fused.py`）：把激活量化塞进 tile loop——`fused_row_mm`
  先扫一遍 K 求行 amax 再在主循环量化（`:35-63`）、`fused_block_mm` 每 128 深量化后乘
  scale 累加（`:66-83`）、`fused_mx_mm` 每 32 列量化 + `tl.dot_scaled`（`:86-107`）。
- 选路：`candidates(recipe, sm, m, fp8, k)`（`linear.py:88-118`）组装
  `vendor/triton/promote/dot_scaled/fused/w16`，**每个 config 先 `_probe`**（`:226-246`），
  编译失败就 `_DEAD` 掉、问题重选（`:266-286`）——「部分 tile 编不过」的通用解法。

### 8.4 ConvRot：旋转量化是什么、为什么

- **定义**：`H_g` 是 `H4=[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]` 的**归一化
  Kronecker 幂**，`g=4^j`，`GROUPS=(4,16,64,256,1024)`（`hadamard.py:14-15`、`:47-55`）。
  对称、自逆、自伴随，元素 `±2^-j`（fp16/bf16 精确）。是 **regular Hadamard**，不是
  Sylvester（`quant-kernels.md:136-140`）。
- **op**：`group_hadamard(x[...,K], group) -> x R`，`R=blockdiag(H_g)` 在最后一维
  （`ops.py:26-28`）；自逆自伴随 ⇒ **backward 就是它自己**（`triton/hadamard.py:122-131`）。
- **为什么**：weight-only int8 的误差由**权重行的 outlier** 决定；右乘一个正交（元素只有
  4 个符号）的 H 把能量摊平，per-row absmax scale 变小 ⇒ 同 int8 精度更高。代价是一次
  `[N,K]` 读写（3090 上 16384×3072 仅 0.243 ms，~830 GB/s，`triton/hadamard.py:84-90`、
  `quant-rotate.md:43-45`）。
- **数学**（存的是**已旋转**权重，`W R ≈ q*scale`）：`forward: y = quant_linear(x R, q, scale, b)`；
  `dgrad: dx = (dy (q*scale)) R`——注意是 **dx 再旋转**，不是 dy
  （`quant/rotated.py:1-11`、`quant-kernels.md:170-176`）。
- **两条实现**（`triton/quant_rotated.py`）：`separate` = 先跑一次旋转 kernel 再跑普通 int8
  linear（`:170`，`compose` 于 `quant/rotated.py:42-52`）；`fused` = forward 在 prologue 旋转
  `[BM,BK]` tile（`:72-77`）、dgrad 在 epilogue 旋转 fp32 累加器（`:118`）。
  **实测 separate 在每个 DiT 形状上都赢**（fused forward ~2x 慢、dgrad 1.2–1.3x 慢），
  因为 fused 要把每个 x tile 按输出 tile 数重复旋转 `N/BN` 次
  （`quant-rotate.md:28-48`、`quant-kernels.md:188-201`）。**不要为「少一趟」去做 fused ConvRot。**
- 旋转在 Triton 里是 `j` 级 4 点蝴蝶：`y_k = sum(x) - 2·x_(3-k)`，末尾乘 `2^-j`
  （`triton/hadamard.py:28-48`），一 program 持 `[BR,C]` fp32 tile。

### 8.5 LoRA 训练的适用性

| 方案 | LoRA（base 冻结） | 说明 |
|------|-------------------|------|
| `quant_linear`（int8/fp8 weight-only） | ✅ **推荐** | base 不收梯度（`quant/__init__.py:1-9`），`dx` 走 dgrad kernel；LoRA 分支仍读 **16-bit 的 x**，所以别在 base 之前量化 x |
| `quant_linear_rotated`（ConvRot） | ✅ 但 base 必须是**已旋转**存储 | 需一次性预处理（`W → W R` 再量化），否则数学不成立；导出时用反量化版（`quant-kernels.md:222-236`） |
| `fp8_linear`（W 与 x 都 fp8） | ⚠️ 不建议 | 每次调用都量化 x（一趟带宽），而 LoRA 分支本来就要 16-bit x；仓库自己承认 q/k/v 会**重复量化同一个归一化输入三次**（`quant-kernels.md:355-360`） |
| `fp8_linear_train` | ❌ 与 LoRA 无关 | 每步重建 fp8 副本（`fp8/train.py:1-5`） |

### 8.6 裁决

- **省什么**：int8 把 base 从 2 B/参数压到 1 B + 每行 4 B，直接决定 16 GB 卡「能不能训
  1024² 的 flux3」。
- **契约**：预量化的 `q [N,K]`、`scale [N]`、可选 `bias`、`fmt ∈ {"int8","fp8"}`；
  ConvRot 再加 `group`（`4^j`，须整除 K）。
- **成本**：`quant/__init__.py` 142 + `rotated.py` 144 + `triton/quant.py` 276 + `hadamard.py` 99
  + `triton/hadamard.py` 152 + `triton/quant_rotated.py` 274 ≈ **1080 行**（去 tuner 后 ~700）。
- **裁决**：⚠️ 分两步。①只做 int8 weight-only（`triton/quant.py` + `quant/__init__.py`，
  ~420 行，无旋转）；②按需加 ConvRot（**只有真从 ComfyUI 拿 `convrot` 权重时才需要**）。
  **fp8/mxfp8 暂缓**：LoRA 下收益被抵消，而 `torch._scaled_mm` 的 mx 路径要
  `float8_e8m0fnu` + `SWIZZLE_32_4_4`，需按 torch 版本核对 API（`fp8/linear.py:136-143`）。

## 9. 技术 7：fused optimizer kernels

```python
# ops.py:59；语义 optim/reference.py:48-131（全部 in-place）
adamw_step(p, g, m, r, comp, lr, b1, b2, eps, wd, step, mode="nearest", seed=0)
adamw8bit_step(p, g, mq, ms, rq, rs, comp, lr, ...)
adamw_step_multi(params, grads, ms, rs, comps, lr, ...)
```

四个可偷的设计：

1. **存 `m` 与 `r = sqrt(v)`，不存 `v`**（`reference.py:48-76`）：fp16 的 `v` 在 6e-8 以下
   下溢，它的平方根不会（`fp16.md:103-104`）。kernel 里就是
   `r = sqrt_rn((1-b2)*r*r + b2*g*g)`（`triton/optim/adamw.py:58-64`）。
2. **参数回写三模式**（`reference.py:34-45`、`triton/optim/adamw.py:38-55`）：`nearest` 直接
   cast；`stochastic`（bf16）对 `(index, seed)` 做**计数器哈希**得到被丢弃的 16 bit 加到 fp32
   位模式上再截断（期望值等于 fp32 更新，且无需 RNG 状态）；`kahan`（fp16/bf16）用同 dtype
   补偿缓冲，参数真值是 `p + comp`。
3. **8-bit 状态**：`BLOCK=256`（`reference.py:10`），`m` 存 int8、`r` 存 uint8，编码是
   `sqrt(|x|/block_absmax)` 的**平方根压扩**（`:79-102`）——把码值范围花在小量级上。
   kernel 一次读写完成 dequant→step→quant（`triton/optim/adamw.py:86-117`）。
4. **multi-tensor**：一张 device 表存每张量的 5 个基地址（p/g/m/r/comp），一张 block 表映射
   program→(tensor, offset)，`SEED_STRIDE=1000003` 保证张量 t 的随机种子与单独调用一致
   （`triton/optim/multi.py:30-63`、`optim/__init__.py:28`、`:31-44`）；表只在地址/大小变化时
   重建（`multi.py:74-97`）。

**抉择**：与 block op 同一套（`{"impl":"kernel"}` vs `{"impl":"torch"}`），按
`problem_key`（dtype + 选项 + **桶化 numel**，`NUMEL_FLOOR=4096`，`optim/choice.py:14`、
`:46-57`）计时一次；**只在真要计时时才做 scratch 拷贝**（`:75-81`）。

**裁决**：✅ 可移植（`reference.py` 131 + `adamw.py` 180 + `multi.py` 143 + `choice.py` 112
≈ 570 行，去掉 8-bit/writeback 后 ~250）。优先级 `adamw_step_multi` > `adamw8bit_step` >
`adamw_step`。契约：全 in-place；`step` 从 1 起；`seed` 每步前进且 DDP 各 rank 一致
（`fp16.md:115`）；`mode` 是 kernel 的 `tl.constexpr` 字符串。

## 10. 技术 8：tuning harness（每卡表 + 预热回放）

### 10.1 三层结构

| 层 | 文件 | 作用 |
|----|------|------|
| 解析排序 | `plan.py:10-38` | `Work(programs, mma_flops, exp_ops, dram_bytes, resident)` + roofline `estimate_ms`（`mma_eff=0.55`，wave 量化 `fill`） |
| 计时短名单 | `tuner.py:207-278` `select()` | 查表→命中即返回；否则取 `ranked[:shortlist]` 计时，**工作内存上界** = `WORKING_FACTOR=8` × 最省者（`:49-50`、`:348-355`） |
| 落盘 | `tuner.py:94-142` | `<cache>/tuning/<card>/<TABLE_TAG>/<kernel>.json`（`:180-188`）；`TABLE_TAG` 是 triton/tilelang/torch 版本哈希（`:58-60`），换版本自动失效 |
| 键 | `buckets.py` | `TOKEN_FLOOR=256`、`LENGTH_FLOOR=64`、`SEGMENT_CAP=8`、`KEY_VERSION="k2"`（`:6-10`）；`pow2(n,floor)` 单调向上取桶（`:18-23`） |
| 窗口 | `window.py:9-72` | `freeze / tuning_window / timing_here / seal / reopen` |
| 录制+回放 | `recording.py`、`replay.py` | 录一次跑批的所有 `Bound` 调用，再在计划中的每个 token 数上重放 fwd+bwd |
| 已决清单 | `manifest.py:23-63` | `prewarm_problems` 表：上次跑过哪些 `(kernel,key)`，全有存储值时**跳过** |
| 决策日志 | `decisions.py:38-53`、`:88-95` | 每 `(kernel,key)` 只记一次，按批 flush 成一张 INFO 表 |

### 10.2 TUNE 只有两个值

```python
MODES = ("table", "off")   # tuning/tuner.py:47
```
| 值 | 计时 | 表 | 缺 key |
|----|------|----|--------|
| `table`（训练默认） | 是（仅窗口内） | 读 + 写 | 窗口内计时并存储；窗口外 `_bind_untimed` |
| `off`（推理默认） | 否 | 只读 | 取第一候选，**不存储**（`:255-259`） |

### 10.3 一个 key 缺失时的 4 种命运

| 条件 | 结果 | 源码 |
|------|------|------|
| 在窗口内（预热 / `tuning_window` / `timing_here`） | 计时 shortlist，存表，绑赢家 | `tuner.py:262-278` → `_measure` |
| `TUNE="off"` 或只有一个候选 | 第一候选，记录 `default`，**不存表** | `:255-259` |
| 已 `freeze` 或 DDP go-live 后 `defer` | `_bind_untimed`：试第一个「能跑一次」的候选，**不计时不存表**，进 `DEFERRED`，WARNING（`met outside a tuning window`） | `:311-338`；`window.py:12-28` |
| 有进程调优预算（`set_cap`）且该 key 超预算 | `_over_budget`：第一候选，untimed/unstored，INFO | `:297-308` |

全部候选失败 → `ConfigsFailed`（`errors.py:60-73`）→ registry 回退下一 backend
（`registry.py:184-205`）；同问题再查**直接重抛，不再计时**（`tuner.py:243-245`）。

### 10.4 预热的两种形态

| 形态 | 适用 | 做法 | 源码 |
|------|------|------|------|
| **合成输入** | 参数少、可控（attention） | 对 `max_len` 的每个 2 的幂桶 × 每个 segment-count 桶造 `q/k/v` 跑 fwd+bwd | `prewarm.py:46-80`、`:20-43` |
| **录制回放** | 整模型、形状组合多 | `recording()` 录一次真实 batch → `Target(rows={n: rows_map})` 把每个 token 数映射到计划值 → 逐个重放 | `recording.py:92-104`、`replay.py:74-101`、`:164-185`、`:245-288` |

回放两个细节值得抄：**同 (op, backend, arg 签名) 的多次调用只留一个**（`replay.py:109-119`，
28 层 block 只测一份）；**边界张量会按目标长度重算**（`:136-148`：有显式 `segments` 就用它，
否则按比例缩放）。

### 10.5 裁决

- **契约**：每个 kernel 给 ① `candidates() -> list[dict]`；② `launch(cfg) -> Callable[[], None]`；
  ③ `problem_key(*args) -> str`；④ 一张 `(op, device, key)` 存储表。
- **成本**：**最小 250 行**——`buckets.py`(64) + `timing.py`(86) + `window.py`(72) +
  `plan.py`(38) + `select/_measure` 核心 ~150（砍掉多进程 `agree/broadcast`、`set_cap`、
  `decisions` 日志）。
- **裁决**：✅ 建议；**别搬 609 行的 `tuner.py`**，只搬状态机 + `select`。`timing.py:56-60`
  的 `l2_flush` 与 `:63-86` 的 `graph_ms`（CUDA graph 回放）是测量可信度的关键，照抄。

## 11. 技术 9：precision bar（fp64 标尺）

### 11.1 方法

对每个输出/输入梯度，把**候选 kernel** 与 **eager（同 dtype）** 分别和 **fp64 参考**比：

```python
# error_stats.py:15-33  stats() -> finite/cos/cos_deficit + abs_{mean,median,min,max} + rel_{...}
# :36-49  floors()：dtype eps × median|ref|（abs）/ eps（rel、cos）
# :52-59  compare() -> {"kernel":..., "eager":..., "floor":...}      :62-67  worse()
```
判定（`tests/_util/error_stats.py:30-42`，`MAX_ERROR_RATIO=10.0` 于 `:21`）只有三种失败：
① kernel 非有限而 eager 有限；② `cos_deficit > 10 × max(eager, 地板)`；③ `abs_*` 与 `rel_*`
（同一 mean 或同一 median）**都** > `10 × max(eager, 地板)`。`min`/`max` 永不参与失败判定
（只做爆炸检测）。

### 11.2 为什么不是固定阈值

- `abs` 与 `rel` **必须成对读**：参考值接近 0 时 `rel` 会被放大到 1e18（attention 的
  huge-logit 用例里 fp64 的 dQ/dK 元素在 1e-190 量级，`kernel-dev.md:635-647`），单看 `rel` 会误杀；
- 单看 `abs` 会放过「大值错得离谱但相对还行」；
- `floors()` 解决「eager 本身已在地板附近」：eager 的统计量低于 dtype 舍入尺度时按地板算
  （`tests/_util/error_stats.py:8-9`）。
- 于是规则是**「不比 PyTorch 差」是阈值**，不是「绝对精度」；满足阈值的候选里**选最快的**
  （`fp16.md:42-46`）。

### 11.3 落进 UTrainer 的最小做法

```python
ref64 = eager_op(*(a.double() for a in args))   # fp64 标尺
table = compare(kernel_op(*args), eager_op(*args), ref64)
assert not regressions(table), (name, table)
```
抄 `error_stats.py`（68 行，纯 torch）+ `tests/_util/error_stats.py` 的 `regressions`
（64 行）；每条写 jsonl 便于跨卡对比。**必须做的变体**：fp16 与 bf16 各一遍、fwd 与 bwd
各一遍、外加一个**故意写坏**的 kernel 要让检查失败（`kernel-dev.md:80-82`，
`tests/_util/negative.py`）——否则不知道检查有没有接上。

### 11.4 配套的三条数值纪律（`CLAUDE.md` "Numerics rules"）

1. **归约在 fp32**；token 计数用 int64；逐样本数学（time embedding、modulation、head modulation）用 fp32。
2. **生产路径里没有 fp64**：fp64 只作标尺（测试 / `error_stats` / bench / eager 跑 fp64 输入），绝不是 fp64 张量或寄存器。
3. **fp32 只做向量运算**：所有 matmul（linear、attention 的 QK^T/PV、adapter）都是 16-bit 输入 + fp32 累加；不做 fp32/tf32 matmul，也不做 hi/lo 拆分多趟 MMA。fp32 存储（master、优化器状态）与逐元素/归约数学不受限。

`fp16.md:48-53` 是这三条在 kernel 层的展开：「load 到 fp32、算在 fp32、只在 store 时舍入一次」。
**裁决**：✅ 第一天就装，零成本。

## 12. 硬件层（只列值得抄的）

| 位置 | 值得抄的东西 |
|------|--------------|
| `hardware/arch.py:7`、`:26-36` | `SM_TABLE=(70,75,80,86,89,90,100,120)`，**小版本继承最近的更低档**（sm103→100）；`has_bf16=sm>=80`、`has_fp8=sm>=89` |
| `hardware/device.py:54-88`、`:142-157`、`:173-188` | 每卡**实测** `mma_tflops/dram_gbs/l2_gbs`（4096³ mm、大 copy、L2 内 copy），按 `card-sm-cu-torch` 缓存到磁盘 |
| `hardware/device.py:160-170` | `tracing()`：compile / fake mode / graph capture 下**绝不测量** |
| `hardware/current.py:37-42` | `on_device(device)`：每次 launch 前切当前设备（否则 `cuda:1` 的张量在 device 0 是 current 时读错卡） |
| `hardware/current.py:52-66` | `rng_kept(devices)`：调优造的随机数**不能扰动调用方的随机流** |
| `hardware/memory.py:47-136`、`:167-199` | **peak window**（嵌套不丢峰值）+ `working(device, bound)` 用 `set_per_process_memory_fraction` 把工作内存上界变成真 OOM |
| `cache_tag.py:15-62` | 把「注册过 custom op 的源文件 hash」塞进 compile cache key |

对 sm120 最有用的是 `device.measure_rates`：它让选 tile 有真实 `mma_tflops/dram_gbs`
（`plan.py` 的 roofline 就靠这两个数），而不是硬编码。

## 13. 落地代价评估

### 13.1 依赖矩阵

| 技术 | 纯 torch | 需要 triton | 需要 tilelang | 需要 MOSS 内部 |
|------|----------|-------------|---------------|----------------|
| precision bar（§11） | ✅ | — | — | 否 |
| compile 决策层（§7） | ✅ | ✅（inductor GPU codegen） | — | 否 |
| loss（§6） | ✅（参考） | ✅ | — | 否（只要 `seg`） |
| block ops（§4） | ✅（参考） | ✅ | — | 否（只要 `seg` + 可 stride 的 `[B,D]`） |
| fused linear（§5） | ⚠️ split 只要 torch + 一个小 epilogue kernel | ✅（fused） | — | 否 |
| varlen attention（§3） | ⚠️ 参考可用（SDPA 循环） | ✅（要赢必须） | — | 否（只要 `cu_seqlens`） |
| int8 weight-only（§8） | ✅（dequant+matmul 参考） | ✅ | — | 需**预量化** base |
| ConvRot（§8.4） | ✅（`group_hadamard_eager`） | ✅ | — | 需**已旋转**的 base |
| fp8 compute / MXFP8 | ⚠️（要 `torch._scaled_mm`） | ✅ | — | 需预量化 base + 布局 |
| optimizer（§9） | ✅（`reference.py`） | ✅ | — | 否 |
| tuning harness（§10） | ✅ | — | — | 否 |

**完全不参与**：`backends/tilelang/**`（~20 文件）、`textenc/**`（Qwen3.5 DeltaNet）、
`bench/**`、`hardware/xpu_triton.py`、`dist.py`、`run_state.py`、`kohakuvault` /
`kohakuengine` / `anyschedule`。

### 13.2 调用方必须提供的 5 条

1. **序列信息**：`cu_seqlens int32 [B+1]`（attention）与 `seg int32 [N]` 非降序（block op 反向归约）；两者可由每样本 token 数推出。
2. **dtype**：`fp16/bf16/fp32` 三选一（`DTYPES` 常量，如 `blocks/norm.py:35`）；bf16 需 `sm>=80`（`constraints.py:27-28`）。
3. **arch 分派**：至少实现 `arch_of(device)`（`hardware/arch.py:43-52`），因为 `supports()` 的 `min_sm` 与 sm120 的 tile seed 都靠它。
4. **per-sample 向量的 stride**：`shift/scale/gate` 允许任意 stride（`_blockops.strides2`，`:211-213`），不要假设 contiguous。
5. **autograd 语义**：kernel 要支持 `torch.compile(fullgraph=True)`（custom op + `register_fake` + `register_autograd`），否则 §7 的 compile 候选接不上。

### 13.3 最小可用子集（varlen attention + block ops + fused linear）

**能跑但不快（纯 torch）**：直接抄 `backends/eager/*` 的 7 个文件（其中 UTrainer 需要的 5 个：
`attention.py` 65 行、`norm.py` 51、`rope.py` 41、`ffn.py` 15、`loss.py` 19），它们本身就是
可用的 PyTorch 实现（`attention_varlen` 逐段 SDPA）。

**要拿到加速（triton，≈3.4k 行）**，按依赖顺序：

| # | 文件 | 行数 | 为什么需要 |
|---|------|------|------------|
| 1 | `triton/blocks/_blockops.py` | 250 | `segment_bounds`/`tile_range`/`segment_sum`/`SegPlan`/`strides2`/`row_candidates`：**所有 block op 共用** |
| 2 | `triton/blocks/norm.py` | 478 | `rms_norm(_mod)`；同时是「per-tile 部分和 → 逐样本归约」的范式 |
| 3 | `triton/blocks/residual.py` | 250 | `gated_residual`；`linear_gated_residual` 的 epilogue 依赖它 |
| 4 | `triton/blocks/rope.py` | 359 | `qk_norm_rope`；`linear_qk_rope` 依赖它 |
| 5 | `triton/blocks/ffn.py` | 177 | `situ_glu`（换 SwiGLU 只改 2 行）+ 共享的 `situ/situ_grad` |
| 6 | `triton/attention/_kernels.py` + `varlen.py` | 367+431 | varlen attention（tuning/choice 可先硬编码 seed） |
| 7 | `triton/blocks/_gemm_kernels.py` + `fused_linear.py` + `_fused_linear.py` | 214+149+502 | fused linear（可先只留 `gres`） |
| 8 | `triton/blocks/loss.py` | 181 | 不物化的 loss |
| 9 | 垫片 | ~150 | `Launcher`（`_common.py:11-26`）+ `TensorMemo`（`_memo.py`）+ `error_stats` |

只依赖 `torch + triton`；`kohakumoss_common` 的 4 个模块换成 10 行等价物即可。

**`_blockops.py` 是真正的枢纽**：二分从 `seg` 造 `cu`（`:23-37`，不回读 host）、前缀和把
program 映射到 segment tile（`:40-62`）、`TensorMemo` 保证同一 step 内所有 op 共享同一份
bounds（`:139-144`）。不抄它，每个 block op 都要重写一遍。

## 14. 移植清单

收益按「16 GB sm120、1024² 训练、LoRA、Windows」估；成本按「一个熟悉 triton 的工程师在
UTrainer 里落地」估。**所有收益都是对比 eager**，对比 compile 会小很多（§7.3）。

| 技术 | 关键源码 | 预期收益（16 GB sm120） | 成本 | 依赖 | 优先级 |
|------|----------|------------------------|------|------|--------|
| **precision bar** | `error_stats.py`、`tests/_util/error_stats.py` | 无速度收益；**决定后面每项敢不敢上** | 0.5 天（~130 行） | torch | **P0** |
| **compile 作为候选** | `compile/ops.py`、`choice.py` | 单样本 norm/residual/loss **+8–40%**；其余 op 免手写 | 1 天（~200 行） | torch+triton | **P0** |
| **不物化 loss** | `triton/blocks/loss.py` | 省 **384–576 MiB 瞬时显存**；速度 launch-bound（±） | 0.5 天（~120 行） | triton | **P0** |
| **fused block ops** | `triton/blocks/{norm,residual,rope,ffn}.py` + `_blockops.py` | 单 op **2–30x**；每省一趟 `[N,D]` = 96 MiB；反向逐位可复现 | 3 天（~1.4k 行） | triton | **P0**（先 norm+residual） |
| **fused linear** | `_fused_linear.py` + `_gemm_kernels.py` | vs eager 2.5x；vs compile +3–36%；省一趟 `[N,2F]`（**512 MiB**）；`save=False` 时省整块 | 1 天（split）/ 3 天（fused） | torch / triton | **P1** |
| **varlen attention** | `triton/attention/{_kernels,varlen}.py` | **Windows 2.07–3.62x**；变长打包 1.6–4.4x；batch=1 单段**打平** | 3 天（~800 行） | triton | **P1**（跑 Windows 才做） |
| **fused optimizer** | `triton/optim/*`、`optim/reference.py` | 8-bit：状态 **8 B → 2.06 B/参数**；multi：N 次 launch → 1 次 | 1.5 天（~400 行） | triton | **P1** |
| **tuning harness（裁剪）** | `tuning/{buckets,timing,window,plan}.py` + `select` | 让上面每项**稳定生效**；不污染 step | 2 天（~250 行） | torch | **P1** |
| **int8 weight-only** | `quant/__init__.py` + `triton/quant.py` | base **2 B → 1 B/参数**；16 GB 上等于「能多训 1.5–2 倍大的 base」 | 2 天（~420 行） | triton | **P2** |
| **ConvRot** | `hadamard.py` + `triton/hadamard.py` + `triton/quant_rotated.py` | int8 精度（outlier 摊平）；旋转 ≈830 GB/s 一趟 | 2 天（~450 行）+ 权重预处理 | triton | **P2**（有 convrot 权重才做） |
| **fp8 compute / MXFP8** | `fp8/*` + `triton/fp8/*` | 权重 1 B、激活带宽减半；**LoRA 下被「x 必须 16-bit」抵消** | 5 天（~1.5k 行） | triton + sm89/100 | **P3**（暂缓） |
| **textenc（DeltaNet）** | `backends/*/textenc/*` | UTrainer 不用 | — | — | **不做** |
| **tilelang 后端** | `backends/tilelang/**` | 目标机未装；sm120 上 triton 已够 | — | tilelang | **不做** |

### 三条落地顺序建议

1. **P0 一起做**：precision bar → compile 候选 → loss → block ops。做完这四项，「小 op 的
   host 开销」与「大临时量」两类问题同时消失，且有了证明正确性的工具。
2. **P1 按测量取舍**：先用 §11 的 harness + `torch.profiler` 量一次 1024² 的 step。
   attention 占比 <20% 就**不做 varlen**（Windows 的 2–3.6x 只作用在那 20% 上）；
   base 权重占显存 >50% 就先做 int8 weight-only 而不是 fused optimizer。
3. **P2 之前不要碰 fp8**：LoRA 下收益最不确定，代码量最大。
