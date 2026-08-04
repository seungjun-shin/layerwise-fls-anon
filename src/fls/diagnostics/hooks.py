from __future__ import annotations

from torch import nn


class ActivationCollector:
    """Collect selected module outputs via forward hooks for small diagnostic batches."""

    def __init__(self) -> None:
        self.activations = {}
        self.handles = []

    def add(self, name: str, module: nn.Module) -> None:
        def hook(_module, _inputs, output):
            self.activations[name] = output.detach().cpu()

        self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

