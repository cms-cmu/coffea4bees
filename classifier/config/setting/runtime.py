from __future__ import annotations

from src.classifier.task import GlobalSetting


class Torch(GlobalSetting):
    """Runtime PyTorch execution controls for reproducible CPU runs."""

    n_threads: int = ...
    deterministic_algorithm: bool = ...

    @classmethod
    def set__n_threads(cls, value: int):
        """Set PyTorch intra-op and inter-op CPU threads lazily without top-level torch import."""
        if value is ... or value is None:
            return None

        import torch

        n = int(value)
        torch.set_num_threads(n)
        if torch.get_num_interop_threads() != n:
            try:
                torch.set_num_interop_threads(n)
            except RuntimeError:
                pass
        return n

    @classmethod
    def set__deterministic_algorithm(cls, value: bool):
        """Toggle PyTorch deterministic algorithms and mkldnn lazily without top-level torch import."""
        if value is ...:
            return False

        import torch

        enabled = bool(value)
        torch.backends.mkldnn.enabled = not enabled
        torch.use_deterministic_algorithms(enabled)
        return enabled
