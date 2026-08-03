"""采样能力抽离（P2 hook 机制）。

与训练循环解耦：输入 = 模型句柄 + 提示词 + 参数 + 输出目录，
输出 = 生成文件路径列表。训练循环只在 step 边界通过 HookManager 调用。

- 采样期间 model.eval() + torch.no_grad() + requires_grad_(False)，
  结束恢复 train() 与原有 requires_grad 状态；
- torch 惰性 import：dry-run（或无 generate_fn）时生成占位文件，
  全流程无需 torch 即可端到端验证 hook 协议。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# 1x1 像素 PNG（dry-run 占位图，保证图片查看器可打开）
_PLACEHOLDER_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d49444154789c626001000000ffff030000060005"
    "57bfabd40000000049454e44ae426082"
)


def run_sample(
    output_dir: str,
    step: int = 0,
    tag: str = "sample",
    prompts: Optional[List[str]] = None,
    params: Optional[dict] = None,
    n: int = 1,
    model_handle: Any = None,
    generate_fn: Optional[Callable[[List[str], dict, str], List[str]]] = None,
    dry_run: bool = False,
) -> List[str]:
    """执行一次采样，返回生成文件的绝对路径列表。

    Args:
        output_dir: 输出目录（workspace/samples/<project>/<task>）。
        step/tag:   用于文件命名 <step>_<tag>[_i].png。
        prompts:    提示词列表；空则用占位提示词。
        params:     采样参数（steps/seed/guidance 等，透传给 generate_fn）。
        n:          生成张数（dry-run 时每张一个占位文件）。
        model_handle: trainer 或 transformer（取其 .transformer 优先）。
        generate_fn: 真实采样回调 (prompts, params, out_dir) -> [path]；
                     None 或 dry_run=True 时走占位路径。
        dry_run:    无 torch 环境下验证 hook 流程用。
    """
    os.makedirs(output_dir, exist_ok=True)
    prompts = prompts or ["(placeholder prompt)"]
    params = params or {}

    if dry_run or generate_fn is None:
        files = []
        for i in range(max(1, n)):
            name = f"{step}_{tag}.png" if i == 0 else f"{step}_{tag}_{i}.png"
            path = os.path.abspath(os.path.join(output_dir, name))
            with open(path, "wb") as f:
                f.write(_PLACEHOLDER_PNG)
            files.append(path)
        logger.info(
            f"[sample{'(dry-run)' if dry_run else '(no generate_fn)'}] "
            f"{len(files)} placeholder image(s) -> {output_dir}")
        return files

    # ── 真实采样：惰性 torch，eval + no_grad + stop gradient ──────
    import torch  # noqa: 惰性导入，dry-run 路径不触达

    model = getattr(model_handle, "transformer", None) or model_handle
    grad_states = {}
    if model is not None and hasattr(model, "parameters"):
        for p in model.parameters():
            grad_states[id(p)] = p.requires_grad
            p.requires_grad_(False)
        was_training = model.training
        model.eval()
    else:
        was_training = False

    try:
        with torch.no_grad():
            files = generate_fn(prompts, params, output_dir)
    finally:
        if model is not None and hasattr(model, "parameters"):
            for p in model.parameters():
                if id(p) in grad_states:
                    p.requires_grad_(grad_states[id(p)])
            if was_training:
                model.train()
    files = [os.path.abspath(f) for f in (files or [])]
    logger.info(f"[sample] {len(files)} image(s) -> {output_dir}")
    return files
