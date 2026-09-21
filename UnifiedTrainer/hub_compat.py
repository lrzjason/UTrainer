"""Restore the `huggingface_hub` API surface that the editable diffusers
checkout (`E:\\diffusers`, 0.39.0.dev0) imports.

Why this exists
---------------
`E:\\diffusers` is installed as an editable package pinned to a
huggingface_hub generation that exposed a handful of symbols at the top level.
The interpreter has huggingface_hub 1.22.0, which dropped or renamed exactly
five of them, so *any* `import diffusers` fails:

    ImportError: cannot import name 'resolve_revision' from 'huggingface_hub'

Without diffusers, `UnifiedTrainer` cannot import its model adapters at all
(`models/qwen_image21/transformer_qwenimage21.py` pulls ModelMixin, RMSNorm,
dispatch_attention_fn, ... from diffusers), so no training run can start.

What it does
------------
Adds back the five missing symbols, best-effort, and nothing else:

    huggingface_hub.resolve_revision                   -> HfApi().repo_info().sha
    huggingface_hub.get_cached_repo_tree               -> local hub-cache walk
    huggingface_hub.errors.RevisionResolutionError     -> alias of
        huggingface_hub.errors.RevisionNotFoundError
    huggingface_hub.errors.CachedRepoTreeNotFoundError -> alias of
        huggingface_hub.errors.EntryNotFoundError
    huggingface_hub.utils.httpx                        -> the httpx package

Only missing attributes are patched, so this module is inert on a machine with
a matching huggingface_hub.  `resolve_revision` is never reached for a local
directory path — diffusers' `_resolve_revision` short-circuits on
`os.path.isdir` — which is the only loading mode this project uses (the local
snapshot is always passed by path).  Training therefore never depends on the
shimmed network paths.

Usage
-----
Deterministic and explicit: the entry point calls `install()` before importing
anything that pulls in diffusers::

    from UnifiedTrainer import hub_compat
    hub_compat.install()

`sitecustomize.py` in this directory does the same thing for interactive use
when the working directory is on `sys.path`, but Python does NOT import
`sitecustomize` for plain script runs (the script directory is added to
`sys.path` after site initialisation), so the explicit call is what actually
guarantees the patch.
"""

from __future__ import annotations

import os
import sys

__all__ = ['install']

_PATCHED: list[str] = []


def _patch_errors() -> None:
    from huggingface_hub import errors

    if not hasattr(errors, 'RevisionResolutionError'):
        errors.RevisionResolutionError = errors.RevisionNotFoundError
        _PATCHED.append('huggingface_hub.errors.RevisionResolutionError')

    if not hasattr(errors, 'CachedRepoTreeNotFoundError'):
        errors.CachedRepoTreeNotFoundError = errors.EntryNotFoundError
        _PATCHED.append('huggingface_hub.errors.CachedRepoTreeNotFoundError')


def _patch_utils() -> None:
    import huggingface_hub.utils as hu

    if not hasattr(hu, 'httpx'):
        import httpx

        hu.httpx = httpx
        _PATCHED.append('huggingface_hub.utils.httpx')


def _patch_top_level() -> None:
    import huggingface_hub as h

    if not hasattr(h, 'resolve_revision'):
        def resolve_revision(
            repo_id,
            *,
            revision=None,
            cache_dir=None,
            local_files_only=False,
            token=None,
            **_kwargs,
        ):
            """Return the commit sha for `revision` (unchanged when unresolvable).

            Diffusers only needs a pinned commit hash to pass down to
            `hf_hub_download`; when the Hub cannot answer it falls back to the
            un-resolved revision, matching upstream behaviour.
            """
            if local_files_only or not isinstance(repo_id, str):
                return revision
            try:
                info = h.HfApi().repo_info(repo_id, revision=revision, token=token)
            except Exception:  # best-effort, mirrors the upstream fallback
                return revision
            return getattr(info, 'sha', None) or revision

        h.resolve_revision = resolve_revision
        _PATCHED.append('huggingface_hub.resolve_revision')

    if not hasattr(h, 'get_cached_repo_tree'):
        import huggingface_hub.constants as hc

        def get_cached_repo_tree(
            repo_id=None,
            *,
            repo_type=None,
            revision=None,
            cache_dir=None,
            **_kwargs,
        ):
            """Minimal stand-in: list a cached repo revision's files.

            Returns `{'revision': ..., 'files': [...]}` or None when the
            revision is not present locally.  Only reached on cache
            introspection paths, never during local-snapshot training.
            """
            if not repo_id:
                return None
            base = cache_dir or hc.HF_HUB_CACHE
            folder = repo_id.replace('/', '--')
            prefix = {'model': 'models', 'dataset': 'datasets',
                      'space': 'spaces'}.get(repo_type or 'model', 'models')
            root_base = os.path.join(base, f'{prefix}--{folder}')
            snap = os.path.join(root_base, 'snapshots')
            if not os.path.isdir(snap):
                return None
            rev = revision or 'main'
            root = os.path.join(snap, rev)
            if not os.path.isdir(root):
                # 'main' is a ref, not a snapshot directory name.
                ref = os.path.join(root_base, 'refs', rev)
                if os.path.isfile(ref):
                    try:
                        with open(ref, encoding='utf-8') as fh:
                            rev = fh.read().strip()
                    except OSError:
                        return None
                    root = os.path.join(snap, rev)
                if not os.path.isdir(root):
                    return None
            files = []
            for dirpath, _dirnames, filenames in os.walk(root):
                for fn in filenames:
                    p = os.path.join(dirpath, fn)
                    files.append(os.path.relpath(p, root).replace(os.sep, '/'))
            return {'revision': rev, 'files': sorted(files)}

        h.get_cached_repo_tree = get_cached_repo_tree
        _PATCHED.append('huggingface_hub.get_cached_repo_tree')


def install(quiet: bool = False) -> list[str]:
    """Apply the compatibility patches. Returns the list of patched symbols.

    Idempotent and safe to call more than once.  Never raises: a failure here
    must not break interpreter startup or the training entry point.
    """
    if _PATCHED:
        return _PATCHED

    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        return _PATCHED

    try:
        _patch_errors()
        _patch_utils()
        _patch_top_level()
    except Exception as exc:  # noqa: BLE001
        print(f'[hub_compat] shim failed: {exc}', file=sys.stderr)
        return _PATCHED

    if _PATCHED and not quiet:
        print(
            '[hub_compat] huggingface_hub compatibility shim applied: '
            + ', '.join(_PATCHED),
            file=sys.stderr,
        )
    return _PATCHED
