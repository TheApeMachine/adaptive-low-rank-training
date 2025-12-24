"""
main provides the entrypoint to the production system.

This is a research platform developed for the paper
"Decoupled Bottleneck Attention: Scaling Efficient Transformers via Low-Rank Semantic Routing"
by Daniel Owen van Dommelen.

The paper is available at https://arxiv.org/abs/2025.XXXXX.

The paper is reproduced by running the following command:

```bash
python main.py --mode train --size medium --exp paper_decoupled --data fineweb_100m.npy
```

It is now being further developed under the name "caramba" to explore additional
architectures and techniques with the goal of optimizing machine learning models
both for efficiency and accuracy.
The idea is to make it easier to combine different components to quickly experiment
with new ideas.
"""
from __future__ import annotations

from caramba.cli import CLI
from caramba.trainer import Trainer


def main(argv: list[str] | None = None) -> None:
    """
    Entrypoint to the caramba system.
    """
    intent = CLI().parse(argv)

    if not intent.groups:
        raise ValueError("Manifest has no groups to run.")
    Trainer(manifest=intent).run()
