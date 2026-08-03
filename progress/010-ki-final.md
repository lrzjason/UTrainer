# 010 — KI 收尾清零（med KI-12/13 + low KI-17/19/20）

> 日期：2026-07-19。范围：`UnifiedTrainer/engine/hook_manager.py`、`hot_keys.py`、
> `orchestrator/db.py`、`server.py`、`main.py`。依据 `progress/007-code-review.md`
> §7 编号与描述。修复后 **007 known-issues 全部清零**（KI-18 已在 009 修复）。
> 决策记录：`agent/decisions.md` D25。验证：`tmp/test_ki_low.py`（28 项新增断言）
> + 全套回归（smoke / P2 / P3 / ki_fixes / ki_med）。

---

## 1. 修复清单（KI → 修法 → 测试）

### KI-12（med）sample_from_weights 备份键集合

- **修法**：`hook_manager._backup_adapter_params()` 替代 `"lora_" in k` 前缀备份——
  优先 `peft.get_peft_model_state_dict(transformer)` 查询 adapter 实际键
  （与 `CheckpointManager.load_lora` 的 PEFT 加载路径键空间一致），交集
  state_dict 后备份；PEFT 不可用/查询失败时退化为宽松名称匹配
  （键含 `lora` 或 `adapter`，大小写不敏感）并 logger.warning，
  备份范围只多不少，保证换回完整。顺手删除该函数内未使用的 `import torch`。
- **测试**：`test_ki_low.py` KI-12——静态断言不再按 `lora_` 前缀；
  假 transformer 验证 fallback 匹配 `lora_*` 与非 lora 命名的
  `custom.adapter` 键、不备份基底权重。

### KI-13（med）set_dotted 静默跳过

- **修法**：`hot_keys.set_dotted` 返回 bool（losses[i] 越界或该位置非 dict →
  False，且不再 setdefault 创建空 losses 列表）；`apply_live` 签名改为返回
  `(applied, skipped)`：config 写入失败、trainer.losses 下标越界均进 skipped。
  `hook_manager._do_patch_config` 把 skipped 明细写进 hook.result
  （`{"value", "applied_to", "skipped"}`），kv 落库语义不变。
  顺手删 `hot_keys.py:60 _REJECT_PREFIXES` 死代码（validate_key 早已内联
  `data.`/`losses[` 判断）。
- **测试**：`test_ki_low.py` KI-13——set_dotted 三种返回值；apply_live
  skipped/applied 断言；patch_config hook 端到端 result 含 skipped 且
  kv 仍落库；死代码静态断言。

### KI-17（low）db.py 四项

- **修法**：
  1. 连接注册表（`_conns` + lock），`close()` 关当前线程连接并注销，
     新增 `close_all()` 统一关闭全部跨线程连接；
  2. 新增 `prune_heartbeats(keep_days=7)`（按 ts 删旧行）与
     `prune_gpu_snapshots(keep_rows=10000)`（每 gpu_index 保留最新 N 行，
     ROW_NUMBER 窗口）；`main.py` 主循环每小时调用一次，失败仅告警；
  3. 删 `connect(timeout=30)`，`PRAGMA busy_timeout=30000` 为锁等待
     唯一来源（原 30s 参数在 PRAGMA 设置后即被覆盖，是死代码兼误导）；
  4. `enqueue_hook(task_id, type_, payload)` 参数改名 `type_`
     （调用方 `hooks.py` 为位置传参，无需改动）。
- **测试**：`test_ki_low.py` KI-17——PRAGMA 值、签名断言、注册表跨线程
  计数与 close_all 清空、30 天前心跳被裁剪而新行保留、gpu 快照按
  gpu_index 各保留 N 行。

### KI-19（low）hook_manager 四项

- **修法**：
  1. `_sample_prompts` JSONL 行 `json.loads` 后 `isinstance(obj, dict)`
     校验，非 dict（数组/字符串/数字）按原文本行处理，不再 AttributeError；
  2. 新增 `db.requeue_stale_acked_hooks(task_id)`（acked→queued 重置），
     `HookManager.__init__` 启动时调用——worker 在 ack 后崩溃不再让 hook
     永远卡 acked；
  3. `_sample_prompts` 的 DB tag 通道改传 `self.project_id`（__init__ 时
     从 task 记录），不再跨项目取提示词；db 层 `prompts_by_tag(None)` 语义
     保留给管理类查询；
  4. dry-run `save` 的 result 与真实路径对齐：`with_optimizer=True` 时同样
     返回 `optimizer_state` 键（此前只写文件不进 result）。
- **测试**：`test_ki_low.py` KI-19——JSONL 混合行解析、跨项目提示词隔离、
  acked 卡死→新 HookManager 启动回收、dry-run save result 键断言。

### KI-20（low）server.py 三项 + 顺手 bug

- **修法**：
  1. `_static` 先 `unquote(path)` 再拼接与穿越校验（abspath 前缀校验不变），
     `%20` 等文件名可访问；
  2. `_json_body` 加 1MB 上限，超限 ValueError → 400；
  3. `_sse` 循环兜底 `except Exception`：记录后静默关闭，不再把异常抛回
     `_dispatch` 的 500 分支往已流式 socket 再写响应头；
  4. **顺手修复（本轮新发现）**：`_route` 处理器一律 `return self._send(...)`
     （None），`_dispatch` 把"已处理"误判为"未命中"，每个成功响应后都追加
     一个 404——keep-alive 连接被污染（实测日志 200 后紧跟 404）。
     `_send`/`_send_file`/`_sse` 现返回 True 作为已处理标记。
- **测试**：`test_ki_low.py` KI-20——`/a%20b.txt` 200 且内容正确；>1MB
  body → 400；同一 keep-alive 连接连发两请求均 200 且 body 正确（证明不再
  双发）；SSE 兜底静态断言。

---

## 2. 行为变化（已同步文档）

| 变化 | 文档 |
|------|------|
| `enqueue_hook` 参数 `type`→`type_`；新增 `close_all` / `requeue_stale_acked_hooks` / `prune_heartbeats` / `prune_gpu_snapshots`；busy_timeout 5000→30000 单源 | `md/01-database.md` |
| sample_from_weights 备份策略、apply_live (applied, skipped)、acked 回收、提示词项目隔离、dry-run save result 对齐 | `md/03-hooks.md` |
| _static URL 解码、_json_body 1MB、SSE 流内异常静默、响应双发修复 | `md/05-api-frontend.md` |
| main.py 每小时 retention 裁剪 | 代码注释 + 本文档 |

---

## 3. 回归结果

```
tmp/smoke_test.py     … Archived project_smoke.json -> done   PASS
tmp/test_ki_fixes.py  === KI fixes test PASSED ===
tmp/test_ki_med.py    === 57/57 checks passed ===  PASSED
tmp/test_p2_hooks.py  === P2 E2E PASSED ===
tmp/test_p3.py        === 66/66 checks passed ===  PASSED
tmp/test_ki_low.py    === 28/28 checks passed ===  （新增）
```

`progress/007-code-review.md` §7 的 KI-12/13/17/19/20 已标 ✅ 并更新
尾部小结：**known-issues 全部清零**。

## 4. 遗留与说明

- `prune_gpu_snapshots` 用 ROW_NUMBER 窗口函数，要求 SQLite ≥ 3.25
  （Python 3.12 内置 SQLite 3.45+，满足；test_ki_low 已实测通过）。
- `_backup_adapter_params` 的 PEFT 精确路径在本环境（无 torch/peft）无法
  端到端验证，靠 fallback 路径单测覆盖；真实训练机上 PEFT 查询命中时
  备份键与 load_lora 加载键同源，语义严格。
- server 响应双发 bug 为 007 清单外的新发现，按 KI-18 先例顺手修复并记入
  KI-20 行与 D25。
