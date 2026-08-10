# 012 — MiniMax-H3 训练接入（图像对 → 视频，音频延后）

> 对应 [md/minimaxh3_implementation.md](../md/minimaxh3_implementation.md)（P0–P4）。
> 编排模式：主 agent 调度/实现 + CodeReview subagent 逐 step 审查 + Debug subagent 定位失败。
> 每阶段按验收标准（G1–G7）逐项打勾；commit 信息含 gate 编号。

## P0 — 环境与前置

- [x] diffusers 固定 pr-14355（commit `abc5e9b`）— 2026-08-03
- [x] `av` 已安装（16.1.0，系统 Python）
- [x] **不下载模型组件**（用户决定：仅代码开发；运行时验证延后到有权重环境）
- [x] packing 空音频布局断言（CPU）通过 — 2026-08-03，21 项断言全绿
      （图像对 8192 行=128 文本+4032 条件+4032 目标；空音频；关键帧锚定；
      空间网格解析精确值；"first"/"last" 锚点消歧（3 帧）；视频 37 帧 149312 行）
- [x] G1 环境验收 CPU 部分 — 2026-08-03（diffusers 0.40.0.dev0 / pr-14355@abc5e9b；
      H3 transformer/vae/scheduler/packing/text-encoder 全部可导入；av 16.1.0；
      torch 2.10.0+cu130，RTX 5060 Ti 可用）
- [ ] VAE 单帧核实（`_encode_clip` → 1 latent 帧）— **延后**（需权重）

> 运行时依赖权重的验收（G2/G3/G4/G5/G6/G7、dry-run、过拟合、编排器冒烟）
> 全部延后；当前只交付代码与文档。

## P1 — 图像对训练（里程碑 1）

- [x] flow_matching data_ward 一行改动（`velocity_sign` 钩子 + 符号分发）— 2026-08-03
- [x] schema media 字段 + `data/video_utils.py` — 2026-08-03（general-agent 实现，
      27/27 验证通过；含 snap_frames/video_latent_num_frames 包装、
      load_image_frames/load_video_frames、时长 [5,15]s 校验、17n+5 对齐）
- [x] P1.2 审查修复闭环 — 2026-08-03（首审 APPROVE_WITH_NOTES：HIGH caption 规则冲突、
      MEDIUM 解码上限绕过时长校验、MEDIUM 抽帧不足破坏 17n+5 对齐；general-agent 修复后
      32/32 断言 + krea2 44 配置回归；聚焦复审 APPROVE_WITH_NOTES）
- [x] P1.2 残余 LOW 修复 — 2026-08-04（计划 §P1/P2 模板补 `reference_configs`、
      README exit-code 行回退；general-agent 修复 + 验证通过）
- [x] cache_builder 统一媒体分发（5D 缓存）— 2026-08-04（general-agent 实现 +
      修复闭环；28/28 断言全绿：图像链路 (C,1,H,W)、视频链路 (C,T,H,W)=(16,124,8,8)、
      dataset 5D 堆叠 (B,C,T,H,W)/legacy 压 T、encode_video 负例、krea2 44 配置回归、
      pair 扫描回归；CodeReview APPROVE）
      ⚠ 引擎偏差（D6 授权）：`sigmas.view(-1,1,1,1)` → ndim 通用展开
      `view(-1, *(1,)*(ndim-1))`（trainer.py L614/L1008 + noise_selector.py L186）——
      旧 4D view 对 5D latent 广播右对齐成 (1,B,1,1,1)，B>1 必炸；4D 行为逐位一致
      ⚠ 前瞻：collate 同批混 T（图像+视频）延后 P2；验证解码 4D 假设由 P1.4 按 D8 分发
- [x] `models/minimax_h3/__init__.py` 适配器 — 2026-08-04（general-agent 实现 + 审查修复闭环；
      53/53 断言全绿：注册/属性、协议完整性、prepare_model_input 行数/tag/timestep 布局、
      条件行混噪公式 0.999·x0+0.001·noise 精确匹配、可复现/重抽、负例、
      unpack_prediction 往返、encode/decode 桩 VAE 路径、decode T>1 拒绝）
- [x] 图像对 smoke 配置 + dry-run 符号校验 + 过拟合（G2/G3/G4）— 2026-08-04
      （general-agent 实现；计划模板 reference_configs 硬性要求 + S/P name-marker suffix 修正；
      17/17 无权重验证：smoke 配置解析/validate、_eval_velocity_mse 符号修复回归
      （standard 逐位一致 / data_ward 生效 / 未知值拒绝）、--list-models 回归；
      符号校验/过拟合脚本代码就绪、缺权重分支返回码 2，运行延后）
- [x] P1 审查：CodeReview APPROVE_WITH_NOTES + 修复闭环 — 2026-08-04
      （无 HIGH；修复 M1 x0_hat 符号盲区→BaseModelAdapter.compute_x0_hat 两处接入、
      M2 条件行混噪 f32、L1 视频缓存重复 load、L2 时长下界 5.17s、L3 17n+5 仅视频配置；
      26/26 断言 + 全量回归（p1_4 53/53、p1_5 17/17、p1_3 28/28、krea2 44 配置、
      --list-models）全绿）

**里程碑 1（图像对）代码交付完成** — 运行验收（G2/G3/G4）需权重，延后。

## P2 — 视频训练（里程碑 2）

- [x] 视频分支代码（P2 实现，运行延后）— 2026-08-04（general-agent 实现；
      40/40 无权重验证全绿：prepare_model_input 视频布局（T=37 latent，
      无 reference → keyframe_anchors=() / 有 reference → ("first",) 条件行在前；
      T==1 无 reference 仍报 e12）、unpack_prediction 视频往返、
      decode_latent T>1 桩 VAE 路径（_decode 复用 + 反归一化逐位一致）、
      decode_validation_video 产出静音 mp4（PyAV 可打开、无音频轨道）、
      引擎验证解码分发（真实 Trainer + EchoTransformer + MiniMaxH3Scheduler +
      桩 VAE：视频 T>1 → mp4 / 图像 T==1 → PNG / 无能力属性 → PNG）、
      视频 smoke 配置 validate（17n+5=124 ✓、reference_configs {"none": []}）、
      回归（p1_4 52/53 仅 i2 按计划翻转：T>1 decode 已实现；krea2 44 配置；
      --list-models））
      ⚠ 遗留（本次范围外，已记录）：trainer `_get_reference_latent`/
      `_decode_reference_images` 仍用旧 `k != tc_key` fallback（视频 reference
      走 try/except → None，不崩溃，建议后续清理）
- [ ] 视频分支激活 + 真实视频缓存（G5）— 代码就绪（.tmp/p2_5_video_cache.py，
      缺权重返回码 2），运行需权重，延后
- [ ] 视频过拟合 + 静音 mp4 验证（G6）— 代码就绪（.tmp/p2_6_video_overfit.py，
      缺权重返回码 2），运行需权重，延后
- [x] P2 审查：CodeReview APPROVE_WITH_NOTES + 修复闭环 — 2026-08-04
      （无 HIGH/MEDIUM；修复 LOW mp4 文件名跨样本/跨 epoch 静默覆盖 →
      decode_validation_video 增 prefix 参数 + trainer 按 PNG 身份传
      {save_name}_val_epoch{epoch}_{sample_idx}[_{ti}]，新增断言 5.9/5.10
      （路径含身份 + 两样本两个 mp4 不覆盖）；NOTE 注释修正 ×4：视频 ref
      跳过理由（mp4 已覆盖 target，重复解码冗余）、latents 按 target-config
      名键控（image-key 并集为防御）、fps 几何注释（37 latent → 124 像素帧
      ×4 → 24fps = 5.17s）、fp32 解码为有意偏离 PR fp16 autocast（已注释）；
      42/42 全绿 + 全量回归（p1_4 52/53 计划内 i2 翻转、p1_5 17/17、
      p1_3 28/28、p1_r 26/26）全绿；唯一偏差：.tmp/p1_r_fix_verify.py 6b.1
      陈旧断言（要求 p1_4 53/53）与计划 i2 翻转矛盾，按 52/53 对齐，仅 harness）

**里程碑 2（视频）代码交付完成** — 运行验收（G5/G6）需权重，延后。

## P3 — 编排器接入

- [x] 生产示例配置 `configs/minimax_h3_train.json` — 2026-08-04（图像对生产配置：
      S/P name-marker、reference S、caption C→S；`lora_target_modules` = H3 实测
      固化清单 ["to_q","to_k","to_v","to_out","ff.net.0.proj","ff.net.2"]
      （diffusers swiglu FF 命名 ff.net.0.proj/ff.net.2，非 ff.gate/ff.up/ff.down）；
      multi_gpu="reserve"/gpus=1 按编排器约定；nf4+bf16+adamw8bit（D9：
      61.7GB transformer 需量化过 GPU Guard）；guidance_scale=1.0（D8 无 CFG）；
      _comment 记录 D2/D8/D9 决策）
- [x] LoRA 目标模块固化 + 无权重验证 — 2026-08-04（general-agent 实现；
      28/28 全绿：config 全路径 validate + validate_gpu_request、H3 transformer
      **微型实例化**（见下方 RAM 事故）、LoRA 模式解析（默认 flux 模式死 4 个 →
      H3 固化清单全覆盖 6 族）、frozen 头排除、负例、checkpoint save_lora/
      load_lora 往返（PEFT 桩模型 + safetensors 权重逐位相等）、
      save/load_training_state 恢复 step/epoch/optimizer/scheduler、
      get_training_state_path stem 逻辑（含 _comfyui）、train.py CLI
      --resume/--resume-full/--task-id/--db 接线、--list-models、p2 42/42 回归）
      ⚠ RAM 事故与修复：原验证脚本以全尺寸 H3 配置（50+2 层、hidden 5376、
      ffn 14336）from_config 随机初始化 → ~14B 参数 ≈ 56-60GB fp32，仅为了
      枚举模块名；修复为同结构微型配置（2+2 层、hidden 256、ffn 512，
      heads×head_dim=hidden）→ 5.35M 参数 ≈ 20MB，实测峰值 873MB（torch 导入
      开销）；断言按 num_layers+num_refiner_layers 派生计数，结构断言不变
- [ ] 编排器任务冒烟 + resume/checkpoint 验证（G7）— 代码就绪
      （CLI/DB 模式接线已验；checkpoint 往返已验），运行需权重，延后
- [x] P3 审查：CodeReview APPROVE_WITH_NOTES — 2026-08-04（28/28 复跑全绿；
      全部 NOTE 无阻塞：① validation.resolution 惰性字段（trainer L117 读了
      不用，值同为 768 无功能影响，不修）；② _comment "2×80GB DDP" 为计划
      D9 继承表述、非虚假陈述（multi_gpu 约定 / GPU Guard 3/4 规则 /
      guidance_scale==1.0 均已对照 orchestrator/validation.py、gpu_guard.py、
      trainer L1408 核实准确）；③ harness 2.6 注释把 adaln_proj/norm_out 标为
      "fp32-contract" 不准确（PR 仅 proj_in/audio_proj_in/time_embedder/
      proj_out/audio_proj_out/rope 为 fp32 契约，AdaLN 为 bf16）——断言本身
      有效（frozen 头排除），仅注释措辞，随 P4 顺带修正；
      另核实：pin 在 quantize="nf4" 下仍解析成功（bnb Linear4bit 继承
      nn.Linear，MRO 确认）；D1 全字段对照真实代码路径（adamw8bit→trainer
      L216 bnb AdamW8bit、nf4→train.py L646、weight_dtype→L303、val 字段→
      trainer L109-118/L1417/L1399/L1408）无运行时拒绝）

**里程碑 3（编排器接入）代码交付完成** — G7 运行验收需权重，延后。

## P4 — 文档更新 + 回归

- [x] §5 文档清单（9 项全部交付，见下方 P4 总结）
- [ ] 量化/DDP/parity 回归 — **延后**（P4 可选增强 per plan L386-392；需权重环境，未运行）


## P4 总结（2026-08-05，实施完成 / 运行验收延后）

- [x] **§5 文档清单全部交付**（9 项）：
      - `md/00-architecture.md` — 索引表新增 `07-minimaxh3-training.md` 行（as-built）
      - `md/07-minimaxh3-training.md` — **新建 as-built 设计文档**（统一媒体数据管线、
        MiniMaxH3Adapter 钩子清单、data-ward 速度约定、t=1−σ 时间约定、关键帧条件行
        （PR noise_aug=0.999 → 引擎 σ_cond=0.001、t_cond=0.999、f32 混噪）、音频行省略
        依据（D6，空 audio (B,0,32)）、验证循环调度器语义（shift=12、mu 守卫）、
        LoRA 目标模块清单（swiglu FF 命名）、三份配置模板要点、已知限制/延后项）
      - `md/minimaxh3_implementation.md` — 头部状态改为"实施完成（代码）/运行验收延后"，
        内容章节未动，链接本文档
      - `requirements.txt`（UTrainer 根）+ `UnifiedTrainer/requirements.txt` — diffusers
        注释追加 PR #14355 分支（refs/pull/14355/head）+ 新增 `av`（PyAV ≥16.1.0）
      - `AGENTS.md`（仓库根）— "Available Models" 表加 `minimax_h3` 行（33B joint
        video+audio、latent 通道 24、VAE scale 8；图像对与视频训练，音频延后）
      - `UnifiedTrainer/data/cache_system.md` — 图像对 5D 缓存 (C,1,H,W) + 视频媒体缓存
        (C,T,H,W)、media 字段、17n+5 对齐（124 → 37 latent）；音频缓存标注"延后"
      - `UnifiedTrainer/models/base.py` — 类 docstring 文档化可选钩子
        `encode_video`/`decode_validation_video`/`velocity_sign`/`compute_x0_hat`
        （文档 only，零行为变更；decode_validation_video 刻意不定义为 base 方法，
        保持 hasattr 分发契约）
      - `doc/Improvement_k3.md` — 追加 §11 MiniMax-H3 训练接入章节（H3-D1–H3-D10
        摘要 + 数据改动点 + 音频延后原因）
      - `agent/decisions.md` — 补齐 H3-D1–H3-D10（命名空间化，编号与计划 §3 一致）
        + 实施偏差与事故记录（P1-r x0_hat 符号盲区修复、M2.3 bf16 非确定性、
        P1.3 引擎 sigmas.view ndim 展开、P2 _reserved_latent_keys/t2v guard、
        P2 mp4 覆盖修复、P3 RAM 事故与微型实例化、空音频 (B,0,32) 等）
- [x] **P3 审查 NOTE ③ 顺带修正**：`.tmp/p3_verify_noweight.py` 2.6 注释
      "frozen fp32-contract heads" → "frozen heads / AdaLN-modulation heads excluded"
      （断言不变，28/28 复跑全绿）
- [x] **最终全量回归**（e:\UnifiedTrainer，全部无权重脚本）：
      | 脚本 | 计数 | 退出 |
      |------|------|------|
      | `.tmp/p1_3_verify_media_pipeline.py` | 28/28 | 0 |
      | `.tmp/p1_4_verify_minimax_h3_adapter.py` | 52/53（仅计划内 i2 翻转） | 1（预期） |
      | `.tmp/p1_5_verify_noweight.py` | 17/17 | 0 |
      | `.tmp/p1_r_fix_verify.py` | 26/26 | 0 |
      | `.tmp/p2_verify_noweight.py` | 42/42 | 0 |
      | `.tmp/p3_verify_noweight.py` | 28/28 | 0 |
      | `train.py --list-models` | 含 `minimax_h3` | 0 |
      krea2 44 配置回归：内嵌于 p1_3/p1_r/p2 各 44 parsed，全绿。
- [ ] **G2–G7 运行验收一律延后（权重环境）**：G2 图像对缓存、G3 dry-run+符号校验、
      G4 图像对过拟合、G5 视频缓存、G6 视频过拟合、G7 编排器全链路 ——
      **未运行，本机未下载模型权重**，不得视为已完成。
- [ ] 量化/DDP/parity 回归（P4 可选增强，per plan L386-392）— 延后：PR 推理 parity、
      NF4/int8 量化训练冒烟、多卡 DDP 冒烟、（OOM 时才做）vendor transformer +
      BlockSwap subclass。

**P4 完成标准**：代码 + 文档交付完成（P0–P3 无权重验证全绿、P4 全量回归全绿）；
运行验收（G2–G7 与 P4 可选增强）待权重环境。

- [x] **最终整体审查：CodeReview APPROVE** — 2026-08-04（无 HIGH/MEDIUM；逐条核对
      as-built 文档 md/07 与代码一致：钩子表/空音频 (B,0,32)/17n+5→5n+2 几何
      （37→124 像素帧 = 7×17+5 独立重推）、t_cond=0.999/σ_cond=0.001、
      data-ward 三消费者一致（未知值 ValueError）、mu 守卫、guidance_scale=1.0
      无 CFG、LoRA pin 与 harness SUMMARY 一致；跨里程碑端到端走查（图像对/
      视频前向、resume 路径）无接缝问题；四个已知遗留（旧 fallback 优雅降级、
      validation.resolution 惰性、p1_4 52/53 计划翻转、G2–G7 延后标注）全部
      已记录且可接受；全量回归复跑全绿）

**MiniMax-H3 接入（P0–P4）代码与文档交付完成** — 本机无权重，运行验收
（G2–G7、parity/量化/DDP）在权重环境执行。
