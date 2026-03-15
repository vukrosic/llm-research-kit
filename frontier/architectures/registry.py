"""
Architecture Registry
=====================
Central registry for all frontier architectures. Each architecture module
registers its classes here so the runner can instantiate any architecture
by name.
"""

from typing import Dict, Type, Any, Optional
from frontier.architectures.base import FrontierModel, FrontierConfig

# Global registry: arch_class_name -> (model_class, default_config_factory)
REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_arch(name: str, family: str, description: str):
    """
    Decorator to register an architecture class.

    Usage:
        @register_arch("MambaLM", "state_space", "Mamba-style selective SSM language model")
        class MambaLM(FrontierModel):
            ...
    """
    def decorator(cls):
        REGISTRY[name] = {
            "class": cls,
            "family": family,
            "description": description,
        }
        return cls
    return decorator


def build_model(arch_class: str, config: FrontierConfig) -> FrontierModel:
    """
    Instantiate a registered architecture by class name.

    Args:
        arch_class: registered name (e.g., "MambaLM")
        config: FrontierConfig with arch_config populated

    Returns:
        Instantiated FrontierModel
    """
    if arch_class not in REGISTRY:
        available = list(REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture '{arch_class}'. Available: {available}"
        )
    model_cls = REGISTRY[arch_class]["class"]
    return model_cls(config)


def list_architectures() -> Dict[str, Dict[str, str]]:
    """List all registered architectures with their family and description."""
    return {
        name: {"family": info["family"], "description": info["description"]}
        for name, info in REGISTRY.items()
    }


def get_family_architectures(family: str) -> Dict[str, Dict[str, str]]:
    """List all registered architectures in a given family."""
    return {
        name: {"family": info["family"], "description": info["description"]}
        for name, info in REGISTRY.items()
        if info["family"] == family
    }
