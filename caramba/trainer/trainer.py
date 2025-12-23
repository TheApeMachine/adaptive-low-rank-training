"""
trainer provides the training loop.
"""

from __future__ import annotations

from caramba.config.manifest import Manifest


class Trainer:
    """
    Trainer composes the training loop.
    """

    def __init__(
        self,
        manifest: Manifest,
    ) -> None:
        self.manifest: Manifest = manifest

    def run(self) -> None:
        """
        Run the training loop.
        """
        pass


