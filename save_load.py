"""Serialize and deserialize a trained Hierarchy to/from disk.

Uses pickle for simplicity — the entire Hierarchy (cells + LMs + priors) is
pure Python with no C extensions, so pickle is portable and version-stable
within a Python release series.

The saved file also records metadata (training source, n-gram order, countries)
so eval scripts can load a hierarchy without re-reading the training config.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from hierarchy import Hierarchy


@dataclass
class HierarchyBundle:
    hierarchy: Hierarchy
    metadata: dict = field(default_factory=dict)

    def countries(self) -> list[str]:
        return sorted({
            cell.region for cell in self.hierarchy.cells if cell.region is not None
        })

    def n_cells(self) -> int:
        return len(self.hierarchy.cells)


def save(hierarchy: Hierarchy, path: str | Path, **metadata) -> None:
    """Save a trained hierarchy to a .pkl file.

    Args:
        hierarchy: The trained Hierarchy object.
        path: Output file path (conventionally *.pkl).
        **metadata: Optional key-value info stored alongside (source, order, ...).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = HierarchyBundle(
        hierarchy=hierarchy,
        metadata={"saved_at": datetime.utcnow().isoformat(), **metadata},
    )
    with path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str | Path) -> HierarchyBundle:
    """Load a hierarchy bundle from a .pkl file.

    Returns a HierarchyBundle with .hierarchy and .metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained hierarchy found at {path}. Run train_geo.py first.")
    with path.open("rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, HierarchyBundle):
        raise ValueError(f"{path} does not contain a HierarchyBundle.")
    return bundle
