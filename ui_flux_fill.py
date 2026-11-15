"""ui_flux_fill.py — 云端镜像默认启动入口（保留 T2ITrainer 的同名调用约定）。

镜像每次开机都会把启动钩子脚本还原成固定内容：

    WORK_DIR="/root/T2ITrainer"
    cd $WORK_DIR
    setsid python ui_flux_fill.py >> /root/T2ITrainer/logs/t2itrainer.log 2>&1 &

所以这里保留同名入口，但实际启动的是重构后的 UTrainer 控制台
（orchestrator watcher + scheduler + dispatcher + HTTP API + 零构建 SPA
web/dist/index.html，单端口同时服务 API 与前端）。/root/T2ITrainer 是指向
本仓库的符号链接，因此 t2itrainer 的路径、日志目录都照旧可用。

命令行接口与旧 T2ITrainer 的 gradio 入口保持一致：

    python ui_flux_fill.py                        # 0.0.0.0:7860
    python ui_flux_fill.py --port 7860 --host 0.0.0.0
"""
import argparse
import os
import sys

# 解析符号链接：/root/T2ITrainer -> /root/UTrainer，统一用真实仓库根
_REPO_ROOT = os.path.dirname(os.path.realpath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _nvrtc_guard() -> None:
    """cuDNN fused attention 运行时用 NVRTC JIT，nvrtc 大版本需与 torch 的
    cudart 对齐（与 start.sh 同款）。缺包时静默跳过，不影响启动。"""
    try:
        import nvidia.cuda_nvrtc
        lib = os.path.join(os.path.dirname(nvidia.cuda_nvrtc.__file__), "lib")
        if os.path.isdir(lib):
            old = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{lib}:{old}" if old else lib
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="UTrainer console (T2ITrainer-compatible entry)")
    parser.add_argument("--port", type=int, default=7860,
                        help="服务端口（默认 7860，与镜像映射的 Web UI 端口一致）")
    parser.add_argument("--host", default="0.0.0.0",
                        help="绑定地址（默认 0.0.0.0，云端必须，编排器自身默认 127.0.0.1）")
    parser.add_argument("--share", action="store_true",
                        help="gradio 兼容参数；编排器没有公网分享功能，忽略")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="并行训练 worker 上限（默认 2）")
    args = parser.parse_args()

    if args.share:
        print("[ui_flux_fill] --share 是 gradio 兼容参数，UTrainer 控制台忽略。")

    os.chdir(_REPO_ROOT)
    _nvrtc_guard()

    workspace = os.path.join(_REPO_ROOT, "workspace")
    sys.argv = [
        "orchestrator.main",
        "--workspace", workspace,
        "--api",
        "--host", args.host,
        "--port", str(args.port),
        "--max-parallel", str(args.max_parallel),
    ]
    print(f"[ui_flux_fill] UTrainer console on http://{args.host}:{args.port}/", flush=True)

    from orchestrator import main as orch_main
    orch_main.main()


if __name__ == "__main__":
    main()
