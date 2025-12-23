"""
validate provides compiler-time validation of lowered configs.
"""
from __future__ import annotations

from dataclasses import dataclass

from caramba.config.layer import (
    AttentionLayerConfig,
    DropoutLayerConfig,
    LayerConfig,
    LayerNormLayerConfig,
    LinearLayerConfig,
    MultiheadLayerConfig,
    RMSNormLayerConfig,
    SwiGLULayerConfig,
    _LayerConfigBase,
)
from caramba.config.topology import (
    BranchingTopologyConfig,
    CyclicTopologyConfig,
    NestedTopologyConfig,
    ParallelTopologyConfig,
    RecurrentTopologyConfig,
    ResidualTopologyConfig,
    SequentialTopologyConfig,
    StackedTopologyConfig,
    TopologyConfig,
    _TopologyConfigBase,
)


@dataclass(frozen=True, slots=True)
class _IO:
    d_in: int | None
    d_out: int | None


def validate_topology(config: TopologyConfig) -> None:
    """
    validate_topology checks basic cross-layer shape invariants.
    """
    _ = _infer_topology_io(config)


def _infer_layer_io(config: LayerConfig) -> _IO:
    match config:
        case LinearLayerConfig() as c:
            return _IO(d_in=int(c.weight.d_in), d_out=int(c.weight.d_out))
        case LayerNormLayerConfig() as c:
            d = int(c.weight.d_model)
            return _IO(d_in=d, d_out=d)
        case RMSNormLayerConfig() as c:
            d = int(c.weight.d_model)
            return _IO(d_in=d, d_out=d)
        case MultiheadLayerConfig() as c:
            d = int(c.weight.d_model)
            return _IO(d_in=d, d_out=d)
        case AttentionLayerConfig() as c:
            d = int(c.weight.d_model)
            return _IO(d_in=d, d_out=d)
        case SwiGLULayerConfig() as c:
            d = int(c.weight.d_model)
            return _IO(d_in=d, d_out=d)
        case DropoutLayerConfig():
            return _IO(d_in=None, d_out=None)
        case _:
            raise ValueError(f"Unsupported layer config: {type(config)!r}")


def _infer_seq_io(layers: list[LayerConfig | TopologyConfig]) -> _IO:
    cur: int | None = None
    for layer in layers:
        if isinstance(layer, _LayerConfigBase):
            io = _infer_layer_io(layer)
        elif isinstance(layer, _TopologyConfigBase):
            io = _infer_topology_io(layer)
        else:
            raise ValueError(f"Unsupported node type: {type(layer)!r}")

        if io.d_in is not None:
            if cur is not None and int(cur) != int(io.d_in):
                raise ValueError(f"Shape mismatch: expected d_in={cur}, got {io.d_in}")
            cur = int(io.d_in)

        if io.d_out is not None:
            cur = int(io.d_out)

    return _IO(d_in=None, d_out=cur)


def _infer_topology_io(config: TopologyConfig) -> _IO:
    match config:
        case StackedTopologyConfig() as c:
            return _infer_seq_io(list(c.layers))
        case SequentialTopologyConfig() as c:
            return _infer_seq_io(list(c.layers))
        case NestedTopologyConfig() as c:
            return _infer_seq_io(list(c.layers))
        case CyclicTopologyConfig() as c:
            return _infer_seq_io(list(c.layers))
        case RecurrentTopologyConfig() as c:
            return _infer_seq_io(list(c.layers))
        case ResidualTopologyConfig() as c:
            io = _infer_seq_io(list(c.layers))
            if io.d_out is None:
                return io
            # Residual requires shape-preserving layers end-to-end.
            # After lowering, the residual topology applies + at each layer, so
            # the running dimension must remain constant.
            for layer in c.layers:
                layer_io = _infer_layer_io(layer)
                if layer_io.d_in is not None and layer_io.d_out is not None:
                    if int(layer_io.d_in) != int(layer_io.d_out):
                        raise ValueError(
                            "Residual topology requires shape-preserving layers, "
                            f"got d_in={layer_io.d_in}, d_out={layer_io.d_out}"
                        )
            return io
        case ParallelTopologyConfig() as c:
            outs: set[int] = set()
            for layer in c.layers:
                if isinstance(layer, _LayerConfigBase):
                    layer_io = _infer_layer_io(layer)
                elif isinstance(layer, _TopologyConfigBase):
                    layer_io = _infer_topology_io(layer)
                else:
                    raise ValueError(f"Unsupported node type: {type(layer)!r}")
                if layer_io.d_out is not None:
                    outs.add(int(layer_io.d_out))
            if len(outs) > 1:
                raise ValueError(
                    f"Parallel topology requires consistent d_out, got {sorted(outs)}"
                )
            return _IO(d_in=None, d_out=next(iter(outs)) if outs else None)
        case BranchingTopologyConfig() as c:
            outs_branch: set[int] = set()
            for layer in c.layers:
                if isinstance(layer, _LayerConfigBase):
                    layer_io = _infer_layer_io(layer)
                elif isinstance(layer, _TopologyConfigBase):
                    layer_io = _infer_topology_io(layer)
                else:
                    raise ValueError(f"Unsupported node type: {type(layer)!r}")
                if layer_io.d_out is not None:
                    outs_branch.add(int(layer_io.d_out))
            if len(outs_branch) > 1:
                raise ValueError(
                    "Branching topology requires consistent d_out, "
                    f"got {sorted(outs_branch)}"
                )
            return _IO(
                d_in=None,
                d_out=next(iter(outs_branch)) if outs_branch else None,
            )
        case _:
            raise ValueError(f"Unsupported topology config: {type(config)!r}")


