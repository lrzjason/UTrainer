"""
统一媒体管线（Unified Media Pipeline）工具模块。

所有媒体 latents 一律 (C,T,H,W) 5D，图像=(C,1,H,W)，视频 P2 激活。

本模块在 P1（图像对训练/里程碑 1）一次建成，图像/视频共用，P2（视频训练）
不重建管线、只激活视频解码：
  - 图像 loader：load_image_frames  → (1, 3, 1, H, W)（T 维=1，统一 5D 约定）
  - 视频 loader：load_video_frames  → (1, 3, T, H, W)（PyAV 解码，P2 激活）
  - 帧数对齐：  snap_frames / video_latent_num_frames（包装 diffusers PR #14355
    packing 函数，供缓存与形状断言复用；17n+5 像素帧 → 5n+2 latent 帧）

里程碑 2 以真实视频数据端到端激活视频分支时，本模块不改动。
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from UnifiedTrainer.data.bucket import BucketSystem
from UnifiedTrainer.data.transforms import to_tensor

logger = logging.getLogger(__name__)

# 视频时长合法区间（秒）：PyAV 逐帧解码 + 均匀抽帧的前提约束。
# 下界 5.17 = ceil(124/24)，对应默认 video_frames=124 @24fps 的抽帧下限——
# 低于该时长必先触发本区间的"时长太短"错误，而不是在抽帧处报
# num_frames > n_total（自相矛盾：5.0s 过旧下界 5.0 却抽不出 124 帧）。
MIN_VIDEO_DURATION = 5.17
MAX_VIDEO_DURATION = 15.0
# 单视频最大解码帧数（防御性上限；默认 24fps 下 5–15s 视频 = 120–360 帧）。
# 解码一旦触及上限即视为视频超长——以截断后的帧数无法推断真实时长（例如 30fps
# 16s 视频会被截到 400 帧、误算成 13.33s），必须直接抛错而非静默截断。
MAX_DECODE_FRAMES = 400


# ── 帧数对齐（包装 diffusers PR #14355）───────────────────────────────

def snap_frames(n: int) -> int:
    """把像素帧数向上取整到 17n+5（视频 VAE 可分块编码的最小帧数）。

    包装 ``diffusers.modular_pipelines.minimax_h3.packing.align_num_frames``。
    """
    from diffusers.modular_pipelines.minimax_h3.packing import align_num_frames

    return align_num_frames(n)


def video_latent_num_frames(n: int) -> int:
    """17n+5 像素帧 → 5n+2 latent 帧（视频 VAE 时间压缩 4x 的实际输出）。

    包装 ``diffusers.modular_pipelines.minimax_h3.packing.video_latent_num_frames``。
    """
    from diffusers.modular_pipelines.minimax_h3.packing import video_latent_num_frames

    return video_latent_num_frames(n)


# ── 媒体 loaders（统一 5D 输出）───────────────────────────────────────

def load_image_frames(
    path: str,
    resolution: int,
    divisibility: int,
    resolution_config: dict = None,
) -> torch.Tensor:
    """加载一张图像为统一 5D 帧张量 ``(1, 3, 1, H, W)``，float32 [0, 1]。

    处理链：PIL 打开 RGB → ``BucketSystem.find_bucket_for_image`` 定桶 →
    ``crop_to_bucket`` 等比缩放 + 中心裁剪 → ``transforms.to_tensor``
    （float32 [0,1] CHW）→ 展开 T 维（=1）与 B 维，得到 (1, C, 1, H, W)。
    与视频 loader 共用同一 bucket 变换，图像/视频缓存格式一致。

    Args:
        path: 图像文件路径。
        resolution: bucket 基准分辨率（短边）。
        divisibility: 尺寸整除约束（vae_scale * patch_size）。
        resolution_config: BucketSystem 自定义桶配置（可选）。

    Returns:
        (1, 3, 1, H, W) float32 张量，取值 [0, 1]。
    """
    from PIL import Image

    pil = Image.open(path).convert("RGB")
    bucket_system = BucketSystem(
        divisibility=divisibility,
        resolution_config=resolution_config,
    )
    bucket = bucket_system.find_bucket_for_image(resolution, pil)
    pil = bucket_system.crop_to_bucket(pil, bucket)

    # (C, H, W) float32 [0,1] → (C, 1, H, W) → (1, C, 1, H, W)
    tensor = to_tensor(pil)
    return tensor.unsqueeze(1).unsqueeze(0)


def load_video_frames(
    path: str,
    num_frames: int,
    fps: int = 24,
    resolution: int = 1024,
    divisibility: int = 16,
    resolution_config: dict = None,
) -> torch.Tensor:
    """PyAV 解码视频并按 ``num_frames`` 均匀抽帧，返回 ``(1, 3, T, H, W)`` float32 [0, 1]。

    处理链：
    a) 解码：``container.decode(video=0)`` 逐帧解码，触及 ``MAX_DECODE_FRAMES``
       解码上限即 ``raise ValueError``（真实时长不可信，拒绝静默截断）；
    b) 时长校验：解码帧数/实际 fps ∈ [5, 15] 秒，否则 ``raise ValueError``；
    c) ``num_frames`` 超过解码帧数时 ``raise ValueError``；抽帧索引取
       ``linspace(0, n_total-1, num_frames).round()`` 并去重，抽样后断言
       ``(T-5) % 17 == 0``（缓存形状契约兜底）；
    d) 每帧走与 ``load_image_frames`` 相同的 bucket 变换：**首帧定桶**，
       所有帧同尺寸裁剪；
    e) 堆叠为 ``(1, C, T, H, W)``（T 维 = 实际抽帧数，去重后 ≤ num_frames）。

    Args:
        path: 视频文件路径。
        num_frames: 目标抽帧数（调用方保证满足 17n+5 对齐，如 124）。
        fps: 入参帧率兜底（容器元数据缺失时使用）。
        resolution: bucket 基准分辨率（短边）。
        divisibility: 尺寸整除约束（vae_scale * patch_size）。
        resolution_config: BucketSystem 自定义桶配置（可选）。

    Returns:
        (1, 3, T, H, W) float32 张量，取值 [0, 1]。

    Raises:
        ValueError: 解码触及上限、视频时长不在 [5, 15] 秒、无可解码帧、
            抽帧数非正、num_frames 超过可解码帧数、或抽样后不满足 17n+5 对齐。
    """
    import av

    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        actual_fps = float(stream.average_rate) if stream.average_rate else float(fps)
        if actual_fps <= 0:
            actual_fps = float(fps)

        # 逐帧解码（cap ~400 帧，病态/超长输入兜底）
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame)
            if len(frames) >= MAX_DECODE_FRAMES:
                break
    finally:
        container.close()

    n_total = len(frames)
    # a) 解码触及上限 → 以截断帧数推算时长会误报通过，直接拒绝（不静默截断）
    if n_total >= MAX_DECODE_FRAMES:
        raise ValueError(
            f"Video {path!r} exceeds decode cap "
            f"({MAX_DECODE_FRAMES} frames decoded) — true length likely longer "
            f"than the [{MIN_VIDEO_DURATION:.0f}, {MAX_VIDEO_DURATION:.0f}]s "
            f"window. Provide a shorter video (or raise MAX_DECODE_FRAMES)."
        )
    if n_total == 0:
        raise ValueError(f"No decodable video frames in {path!r}")

    # b) 时长校验：解码帧数/实际 fps ∈ [5, 15] 秒
    duration = n_total / actual_fps
    if not (MIN_VIDEO_DURATION <= duration <= MAX_VIDEO_DURATION):
        raise ValueError(
            f"Video duration must be in "
            f"[{MIN_VIDEO_DURATION:.2f}, {MAX_VIDEO_DURATION:.2f}] seconds, got "
            f"{duration:.2f}s ({n_total} frames at {actual_fps:.2f} fps): {path}"
        )

    # c) 抽帧：num_frames 超过解码帧数 → 报错（避免返回未对齐 T 破坏缓存形状契约）
    if num_frames > n_total:
        raise ValueError(
            f"video too short: need >= {num_frames / actual_fps:.2f}s "
            f"for {num_frames} frames @ {actual_fps:.2f}fps, got {duration:.2f}s "
            f"({n_total} decoded frames of {path!r}). Use a longer video or "
            f"reduce video_frames."
        )
    if num_frames == n_total:
        indices = list(range(n_total))
    else:
        indices = np.linspace(0, n_total - 1, num_frames).round().astype(int).tolist()
        indices = sorted(set(indices))

    # 抽样后 17n+5 对齐断言（缓存形状契约兜底；T 必须满足 (T-5) % 17 == 0）
    if (len(indices) - 5) % 17 != 0:
        raise ValueError(
            f"Sampled {len(indices)} frames from {path!r} does not satisfy "
            f"17n+5 alignment ((T-5) % 17 == 0) — cache shape contract "
            f"violated. Use a longer video or adjust num_frames/video_frames."
        )
    if len(indices) < num_frames:
        logger.warning(
            f"Uniform sampling deduped {num_frames} -> {len(indices)} frames "
            f"(source has {n_total} frames); T < num_frames."
        )

    # d) 首帧定桶，所有帧同尺寸裁剪（与 load_image_frames 同一 bucket 变换）
    bucket_system = BucketSystem(
        divisibility=divisibility,
        resolution_config=resolution_config,
    )
    first_pil = frames[indices[0]].to_image()
    bucket = bucket_system.find_bucket_for_image(resolution, first_pil)

    tensors = []
    for i in indices:
        pil = bucket_system.crop_to_bucket(frames[i].to_image(), bucket)
        tensors.append(to_tensor(pil))  # (C, H, W) float32 [0,1]

    # e) 堆叠：(C, T, H, W) → (1, C, T, H, W)
    video = torch.stack(tensors, dim=1)
    return video.unsqueeze(0)
