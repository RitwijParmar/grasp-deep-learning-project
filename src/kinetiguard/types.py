"""Small, explicit data containers used across the research pipeline."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class MolecularFrames:
    """Fixed-composition molecular configurations and first-principles labels.

    Positions have shape ``[frames, atoms, 3]`` and energies have shape ``[frames]``.
    A fixed composition is deliberate: it keeps the force field and curvature probes
    honest about which chemical system they are modelling.
    """

    atomic_numbers: torch.Tensor
    positions: torch.Tensor
    energies: torch.Tensor
    forces: torch.Tensor
    name: str = "unnamed"
    metadata: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if self.atomic_numbers.ndim != 1:
            raise ValueError("atomic_numbers must have shape [atoms]")
        if self.positions.ndim != 3 or self.positions.shape[-1] != 3:
            raise ValueError("positions must have shape [frames, atoms, 3]")
        if self.forces.shape != self.positions.shape:
            raise ValueError("forces must match positions shape")
        if self.energies.ndim != 1 or self.energies.shape[0] != self.positions.shape[0]:
            raise ValueError("energies must have shape [frames]")
        if self.positions.shape[1] != self.atomic_numbers.shape[0]:
            raise ValueError("atom count does not match atomic_numbers")

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def subset(self, indices: torch.Tensor, name: Optional[str] = None) -> "MolecularFrames":
        indices = torch.as_tensor(indices, dtype=torch.long)
        return MolecularFrames(
            atomic_numbers=self.atomic_numbers.clone(),
            positions=self.positions[indices].clone(),
            energies=self.energies[indices].clone(),
            forces=self.forces[indices].clone(),
            name=name or self.name,
            metadata=dict(self.metadata or {}),
        )

    def to(self, device: torch.device) -> "MolecularFrames":
        return MolecularFrames(
            atomic_numbers=self.atomic_numbers.to(device),
            positions=self.positions.to(device),
            energies=self.energies.to(device),
            forces=self.forces.to(device),
            name=self.name,
            metadata=dict(self.metadata or {}),
        )
