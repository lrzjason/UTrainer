"""Diagnose the cuDNN Frontend 'cudart vs nvrtc major version mismatch'.

Run this ON THE CLOUD (where the GPU + error occur), in the SAME shell where you
will start training:
    python diag_cudnn_nvrtc.py

Verdict priority:
1. The TRAINING-LIKE masked test (seq 4096, heads 24, head_dim 128, bf16) is the
   real reproduction: for those shapes cuDNN has no precompiled engine, so JIT
   is mandatory. If it passes, the environment is FIXED for training.
2. nvrtc major alignment is checked TWICE: in-process (subject to dlopen handle
   caching — an old nvrtc may already be mapped by torch/cuDNN) and in a FRESH
   subprocess (authoritative for what a newly started training process loads).
"""
import os
import glob
import ctypes
import subprocess
import sys

FIX_DIR = "/opt/nvrtc13"
LINK_NAMES = ("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.13")

print("=== torch / cudart ===")
import torch

print("torch.__version__       :", torch.__version__)
print("torch.version.cuda      :", torch.version.cuda, "  <-- cudart major source")
print("cudnn.version()         :", torch.backends.cudnn.version())
print("cudnn.enabled           :", torch.backends.cudnn.enabled)
print("cuda.is_available()     :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device                  :", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("compute capability      :", cap, f"(sm_{cap[0]}{cap[1]}0)")

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
print("\n=== torch bundled libs ===")
print("torch lib dir           :", torch_lib)
for pat in ("libnvrtc*", "libcudnn*"):
    for f in sorted(glob.glob(os.path.join(torch_lib, pat))):
        print("   ", os.path.basename(f))

required = (torch.version.cuda or "").split(".")[0]


def nvrtc_version(libname):
    try:
        lib = ctypes.CDLL(libname)
        major = ctypes.c_int()
        minor = ctypes.c_int()
        lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
        return f"{major.value}.{minor.value}"
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def nvrtc_version_fresh(libname):
    """Resolve the name in a FRESH process — immune to dlopen handle caching."""
    code = (
        "import ctypes;"
        f"lib = ctypes.CDLL('{libname}');"
        "m = ctypes.c_int(); n = ctypes.c_int();"
        "lib.nvrtcVersion(ctypes.byref(m), ctypes.byref(n));"
        "print(f'{m.value}.{n.value}')"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        return out.stdout.strip() or f"<err: {out.stderr.strip()[:100]}>"
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


print("\n=== env ===")
print("CUDA_HOME               :", os.environ.get("CUDA_HOME"))
ld = os.environ.get("LD_LIBRARY_PATH", "")
print("LD_LIBRARY_PATH         :", ld or "<unset>")
fix_in_ld = FIX_DIR in [p for p in ld.split(":") if p]

print("\n=== fix state ===")
for name in LINK_NAMES:
    p = os.path.join(FIX_DIR, name)
    if os.path.lexists(p):
        print(f"    {name:15s} -> {os.path.realpath(p)}")
    else:
        print(f"    {name:15s} -> <missing in {FIX_DIR}>")

print("\n=== filesystem ground truth ===")


def ls_dir(d):
    try:
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.islink(p):
                kind = "-> " + os.readlink(p)
            elif os.path.isdir(p):
                kind = "<dir>"
            else:
                kind = f"{os.path.getsize(p)}B"
            print(f"        {f:42s} {kind}")
    except OSError as e:
        print(f"        <error: {e}>")


print("    /opt/nvrtc13/")
ls_dir(FIX_DIR)
try:
    import nvidia.cuda_nvrtc as _m

    pipdir = os.path.join(os.path.dirname(_m.__file__), "lib")
    print(f"    {pipdir}/")
    ls_dir(pipdir)
except Exception:  # noqa: BLE001
    pass

print("\n    /opt/nvrtc13 symlink integrity (exists= follows links; readable= dlopen-able):")
for name in LINK_NAMES:
    p = os.path.join(FIX_DIR, name)
    lex = os.path.lexists(p)
    ex = os.path.exists(p)  # follows symlinks -> False when dangling
    acc = os.access(p, os.R_OK) if lex else False
    tgt = os.readlink(p) if lex and os.path.islink(p) else "(not a link)"
    print(f"        {name:15s} lexists={lex} exists={ex} readable={acc} target={tgt}")

print("\n=== nvrtc versions: in-process dlopen (may be handle-cached) ===")
resolved_inproc = {}
for name in LINK_NAMES:
    v = nvrtc_version(name)
    resolved_inproc[name] = v
    print(f"    {name:15s} -> {v}")

print("\n=== nvrtc versions: FRESH subprocess (authoritative) ===")
resolved_fresh = {}
for name in LINK_NAMES:
    v = nvrtc_version_fresh(name)
    resolved_fresh[name] = v
    print(f"    {name:15s} -> {v}")


def dlopen_trace(libname):
    """Fresh-subprocess LD_DEBUG trace of which file dlopen actually opens."""
    env = dict(os.environ)
    env["LD_DEBUG"] = "libs"
    try:
        out = subprocess.run(
            [sys.executable, "-c", f"import ctypes; ctypes.CDLL('{libname}')"],
            capture_output=True, text=True, timeout=60, env=env,
        )
        lines = [
            l for l in (out.stdout + out.stderr).splitlines()
            if "nvrtc" in l or "calling init" in l
        ]
        return "\n".join(lines[-35:]) or "<no nvrtc lines in LD_DEBUG output>"
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


print("\n=== LD_DEBUG dlopen trace for libnvrtc.so (fresh subprocess) ===")
print(dlopen_trace("libnvrtc.so"))


def loaded_lib_paths(pat):
    """Real on-disk paths of libs already mapped into this process."""
    out = set()
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                p = line.split()[-1]
                if pat in p and p.startswith("/"):
                    out.add(p)
    except OSError:
        pass
    return sorted(out)


def dlopen_maps_path(name, pat):
    try:
        ctypes.CDLL(name)
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"
    paths = loaded_lib_paths(pat)
    mains = [p for p in paths if "builtins" not in p]
    return mains[0] if mains else (paths[0] if paths else f"<{name} loaded, path unknown>")


print("\n=== what the process ACTUALLY loaded (real paths) ===")
print("    libcudnn.so.9        ->", dlopen_maps_path("libcudnn.so.9", "libcudnn"))
print("    libcudart.so.13      ->", dlopen_maps_path("libcudart.so.13", "libcudart"))
print("    libcublas.so.13      ->", dlopen_maps_path("libcublas.so.13", "libcublas"))
all_nvrtc = sorted(set(loaded_lib_paths("libnvrtc")) | set(loaded_lib_paths("nvrtc")))
if all_nvrtc:
    for p in all_nvrtc:
        kind = "MAIN" if "builtins" not in p else "builtins (bitcode dep)"
        print(f"    [nvrtc map] {kind:22s} {p}")
else:
    print("    [nvrtc map] (none mapped yet)")

# The REAL file of the required major that the versioned name maps to — the
# correct symlink target (skip any 12.6 main lib mapped via other names).
real_nvrtc = None
for cand in [p for p in all_nvrtc if "builtins" not in p and "nvrtc" in p]:
    v = nvrtc_version(os.path.realpath(cand))
    if v.startswith(required + ".") or v == required:
        real_nvrtc = os.path.realpath(cand)
        break
if real_nvrtc:
    print(f"    REAL nvrtc (major {required} file): {real_nvrtc}")


def find_pip_nvrtc():
    try:
        import nvidia.cuda_nvrtc as mod  # type: ignore

        return os.path.join(os.path.dirname(mod.__file__), "lib")
    except Exception:  # noqa: BLE001
        try:
            out = subprocess.run(
                ["find", "/root/miniconda3", "-name", f"libnvrtc.so.{required}*"],
                capture_output=True, text=True, timeout=120,
            ).stdout.split()
            return os.path.dirname(out[0]) if out else None
        except Exception:  # noqa: BLE001
            return None


def run_sdpa_test(name, q, k, v, mask):
    from torch.nn.attention import sdpa_kernel, SDPBackend

    try:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.cuda.synchronize()
        print(f"    {name:45s} OK   -> out {tuple(o.shape)}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    {name:45s} FAIL -> {type(e).__name__}: {str(e)[:120]}")
        return False


print("\n=== cuDNN SDPA tests ===")
q = torch.randn(1, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)
mask = torch.zeros(128, 128, dtype=torch.bfloat16, device="cuda")
run_sdpa_test("MASKED small (1,8,128,64)  <- comparison only", q, k, v, mask)

print("\n=== TRAINING-LIKE masked test (the real reproduction) ===")
ok = True
for seq in (4096,):
    q = torch.randn(1, 24, seq, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = torch.zeros(seq, seq, dtype=torch.bfloat16, device="cuda")
    ok &= run_sdpa_test(f"MASKED train-shape seq={seq} h=24 hd=128", q, k, v, mask)

print("\n=== VERDICT ===")
fresh_ok = all(
    v.startswith(required + ".") or v == required for v in resolved_fresh.values()
)
if ok and fresh_ok:
    print(f"FIXED: fresh processes resolve every nvrtc name to major {required} and")
    print("       the training-shape test passes. Start training from this shell/env.")
elif ok:
    print(f"FIXED (effective): training-shape test passes; cuDNN gets nvrtc major {required}.")
    print("       Fresh subprocess resolution still shows an old major — check the")
    print("       symlinks and that /opt/nvrtc13 precedes system dirs. In-process 12.6")
    print("       readings are dlopen handle-cache artifacts (old handle pre-loaded).")
elif not fix_in_ld:
    print(f"PARTIAL: symlinks exist but LD_LIBRARY_PATH does not include {FIX_DIR}")
    print("         in THIS shell. The export is shell-local: run the export and this")
    print("         script in the SAME shell, and add the export to the launch script")
    print("         (start.sh / systemd unit) that starts the orchestrator.")
else:
    print("NOT FIXED: training-shape test still fails. Re-check symlink targets below.")

if not fresh_ok or not ok:
    nvrtc_dir = find_pip_nvrtc()
    # Prefer the REAL mapped file over the (possibly dangling) pip symlink chain.
    target = real_nvrtc or (os.path.join(nvrtc_dir, f"libnvrtc.so.{required}") if nvrtc_dir else None)
    if target:
        print("\n>>> APPLY / RE-APPLY (as root, in THIS shell), then re-run:")
        print(f"""    mkdir -p {FIX_DIR}
    ln -sf {target} {FIX_DIR}/libnvrtc.so
    ln -sf {target} {FIX_DIR}/libnvrtc.so.12
    ln -sf {target} {FIX_DIR}/libnvrtc.so.{required}
    export LD_LIBRARY_PATH={FIX_DIR}:$LD_LIBRARY_PATH
    python diag_cudnn_nvrtc.py""")
    else:
        print(f"\n>>> Install the matching nvrtc first: pip install nvidia-cuda-nvrtc-cu{required}")
else:
    print("\nOptional cleanup: point the /opt/nvrtc13 symlinks at the REAL file so EVERY")
    print(f"name resolves to major {required} (not just the versioned one):")
    if real_nvrtc:
        print(f"    ln -sf {real_nvrtc} {FIX_DIR}/libnvrtc.so")
        print(f"    ln -sf {real_nvrtc} {FIX_DIR}/libnvrtc.so.12")

cudnn_path = dlopen_maps_path("libcudnn.so.9", "libcudnn")
if cudnn_path and "site-packages/nvidia/cudnn" not in cudnn_path and "/usr/local/" in cudnn_path:
    print("\nNOTE: the loaded libcudnn comes from the container (/usr/local/...), not pip.")
    print("      Prepending the pip cu13 libcudnn dir makes the whole stack coherent:")
    print("      export LD_LIBRARY_PATH=$(python -c 'import nvidia.cudnn,os;print(os.path.join(os.path.dirname(nvidia.cudnn.__file__),\"lib\"))'):$LD_LIBRARY_PATH")
