"""
Registry — decorator-based registries for model adapters and loss modules.

Usage:
    @ModelRegistry.register("flux2_klein")
    class Flux2KleinAdapter(BaseModelAdapter):
        ...

    adapter_cls = ModelRegistry.get("flux2_klein")

    @LossRegistry.register("flow_matching")
    class FlowMatchingLoss(BaseLoss):
        ...

    loss_cls = LossRegistry.get("flow_matching")
"""
from __future__ import annotations

from typing import Dict


class _BaseRegistry:
    """Shared registry logic. Subclasses get their own _registry dict."""

    _registry: Dict[str, type] = {}
    _label: str = "Registry"

    @classmethod
    def register(cls, name: str):
        """Decorator that registers a class under the given name."""

        def decorator(target_cls: type) -> type:
            cls._registry[name] = target_cls
            return target_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Retrieve a registered class by name. Raises KeyError if not found."""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"{cls._label} '{name}' not registered. "
                f"Available: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list(cls) -> Dict[str, type]:
        """Return all registered name -> class mappings."""
        return dict(cls._registry)

    @classmethod
    def names(cls) -> list:
        """Return sorted list of registered names."""
        return sorted(cls._registry.keys())

    @classmethod
    def contains(cls, name: str) -> bool:
        """Check whether a name is registered."""
        return name in cls._registry


class ModelRegistry(_BaseRegistry):
    """Registry for model adapters. Use @ModelRegistry.register(name)."""

    _registry: Dict[str, type] = {}
    _label: str = "ModelAdapter"


class LossRegistry(_BaseRegistry):
    """Registry for loss modules. Use @LossRegistry.register(name)."""

    _registry: Dict[str, type] = {}
    _label: str = "LossModule"
