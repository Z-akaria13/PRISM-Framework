"""
AUDIT CONFIG
Everything needed to run one extraction job, as plain data.

The GUI builds one from its tabs, the cluster dispatcher writes it to JSON,
and headless.py reads it back, so local and remote runs share one definition.
"""
from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from math import prod

POOLING_STRATEGIES = ("Mean Pool", "Max Pool", "Grid Pool", "Seq Pool", "CLS Token", "Flatten")
GRID_CELLS_PER_AXIS = 4  # Grid Pool: 4 x 4 cells for 2D maps, 4 x 4 x 4 for 3D volumes
DEFAULT_POOLING = "Mean Pool"
DEVICE_CHOICES = ("auto", "cpu", "cuda")  # "cuda:N" is accepted as well


def stored_shape(shape, strategy):
    """What one row of a layer is stored as, given the shape of one row's output and its pooling
    (the engine's pooling, core/execution._prepare, gives the same)."""
    shape = tuple(int(n) for n in shape)
    if len(shape) <= 1:
        return shape  # already one vector per row: stored as it is
    if strategy == "Flatten":
        return (prod(shape),)
    if strategy in ("Seq Pool", "CLS Token"):
        return shape[1:]  # averages, or takes the first of, axis 1 (the tokens)
    if strategy == "Grid Pool":
        return (shape[0] * GRID_CELLS_PER_AXIS ** (len(shape) - 1),)
    return (shape[0],)  # Mean Pool and Max Pool keep axis 1 (the channels)


def shape_text(shape):
    """(64, 28, 28) -> '64×28×28'."""
    return "×".join(str(int(n)) for n in shape) or "1"


@dataclass(frozen=True)
class AuditConfig:
    project_root: str
    script_path: str
    framework: str
    csv_path: str
    id_col: str
    target_col: str
    output_dir: str
    group_col: str | None = None
    layers: dict = field(default_factory=dict)  # {layer name: pooling strategy}, in selection order
    batch_size: int = 16
    device: str = "auto"  # "auto" = CUDA if the machine running the extraction has it
    overwrite: bool = False  # replace results already in output_dir (otherwise the run refuses)
    max_failed_percent: float = 5.0  # stop once more than this share of all rows has failed (100: never stop)

    def __post_init__(self):
        object.__setattr__(self, "layers", dict(self.layers))
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size!r}")
        unknown = sorted({s for s in self.layers.values() if s not in POOLING_STRATEGIES})
        if unknown:
            raise ValueError(f"Unknown pooling strategy {unknown}; expected one of {POOLING_STRATEGIES}")
        if not isinstance(self.overwrite, bool):
            raise ValueError(f"overwrite must be true or false, got {self.overwrite!r}")
        limit = self.max_failed_percent
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not 0 <= limit <= 100:
            raise ValueError(f"max_failed_percent must be a number from 0 to 100, got {limit!r}")
        if self.device not in DEVICE_CHOICES and not self.device.startswith("cuda:"):
            raise ValueError(f"Unknown device {self.device!r}; expected one of {DEVICE_CHOICES} or 'cuda:N'")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_dict(cls, data: dict) -> "AuditConfig":
        known = {f.name for f in fields(cls)}
        required = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"Job config is missing {missing}")
        values = {k: v for k, v in data.items() if k in known}
        if isinstance(values.get("layers"), list):
            values["layers"] = dict(parse_layer_label(label) for label in values["layers"])
        return cls(**values)

    @classmethod
    def from_json(cls, text: str) -> "AuditConfig":
        return cls.from_dict(json.loads(text))


def parse_layer_label(label: str) -> tuple[str, str]:
    """Read a pre-AuditConfig layer label, e.g. "layer4.1.conv2 [Max Pool]" -> ("layer4.1.conv2", "Max Pool").

    Job files written before AuditConfig stored layers as these Wiretap display strings.
    """
    name = label.split(" [")[0].strip()
    strategy = label.split("[")[1].replace("]", "") if "[" in label else DEFAULT_POOLING
    return name, strategy
