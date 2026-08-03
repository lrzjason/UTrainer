"""多卡请求与 launch 模式校验（P5c）。CLI / watcher / server 三入口复用。

与 scheduler.validate_schedule 同模式：入库前统一校验，避免非法值落库。
"""

from __future__ import annotations

from typing import Optional

TORCHAO_MODES = {"torchao_float8", "torchao_int8", "torchao_int4"}
LAUNCH_MODES = {"reserve", "ddp"}


def parse_gpu_ids(raw: Optional[str]) -> Optional[list]:
    """"0,1" → [0,1]；空串/None → None；非法值抛 ValueError。"""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(f"invalid gpu_ids entry: {part!r} (expect indices)")
        out.append(int(part))
    return out


def validate_gpu_request(config: dict, gpus=None, gpu_ids=None):
    """校验并规范化多卡请求，返回 (gpus:int, gpu_ids:Optional[list])。

    规则：
    - gpus 缺省 1；必须 >= 1 的整数；
    - gpu_ids 存在时长度必须等于 gpus（CSV 或列表）；
    - launch 模式：config training.multi_gpu ∈ {reserve, ddp}（缺省 reserve）；
    - ddp + gpus>1 与 block_swap>0 / torchao 量化组合直接拒绝
      （CPU 卸载钩子与 DDP 不兼容，见 md/06 §3.5）；
    - ddp + gpus>1 与 mixed_precision='no' 组合拒绝（DDP 无梯度同步精度
      会异常，保持保守）。
    """
    train_cfg = config.get("training", {}) if isinstance(config, dict) else {}
    if gpus is None:
        gpus = train_cfg.get("gpus", 1)
    try:
        gpus = int(gpus)
    except (TypeError, ValueError):
        raise ValueError(f"invalid gpus value: {gpus!r} (expect integer >= 1)")
    if gpus < 1:
        raise ValueError(f"invalid gpus value: {gpus} (expect >= 1)")

    pinned = gpu_ids
    if isinstance(pinned, str):
        pinned = parse_gpu_ids(pinned)
    if pinned is not None:
        if len(pinned) != gpus:
            raise ValueError(
                f"gpu_ids length {len(pinned)} != gpus {gpus}: {pinned}")
        if len(set(pinned)) != len(pinned):
            raise ValueError(f"duplicate gpu_ids: {pinned}")

    mode = train_cfg.get("multi_gpu", "reserve")
    if mode not in LAUNCH_MODES:
        raise ValueError(
            f"training.multi_gpu must be one of {sorted(LAUNCH_MODES)}, "
            f"got {mode!r}")
    if mode == "ddp" and gpus > 1:
        block_swap = train_cfg.get("block_swap", 0)
        if block_swap:
            raise ValueError(
                f"ddp multi-GPU incompatible with block_swap={block_swap} "
                "(CPU offload hooks are not DDP-safe); use multi_gpu=reserve "
                "or block_swap=0")
        q = train_cfg.get("quantize", "none")
        if q in TORCHAO_MODES:
            raise ValueError(
                f"ddp multi-GPU incompatible with quantize={q} "
                "(torchao device_placement=[False] is not DDP-safe); "
                "use multi_gpu=reserve or quantize=none")
        if train_cfg.get("mixed_precision", "bf16") in ("no", "none"):
            raise ValueError(
                "ddp multi-GPU requires training.mixed_precision "
                "(bf16/fp16), got 'no'")
    return gpus, pinned
