"""Local sitecustomize: apply the huggingface_hub compatibility shim on startup.

Handles the interactive case — `python` started from this directory, where the
working directory is on `sys.path` and `site` can find this file.

Plain script runs (`python -m UnifiedTrainer.train ...`) do NOT reach this file:
`site.execsitecustomize()` runs before the script directory is prepended to
`sys.path`.  That path is covered by the explicit
`UnifiedTrainer.hub_compat.install()` call at the top of `train.py`, which is
what actually guarantees the patch.  See `UnifiedTrainer/hub_compat.py` for the
full rationale.

This file is intentionally a thin wrapper so there is exactly one copy of the
patching logic.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from UnifiedTrainer.hub_compat import install

    install()
except Exception as exc:  # never break interpreter startup
    print(f'[sitecustomize] hub_compat unavailable: {exc}', file=sys.stderr)
