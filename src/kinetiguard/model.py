"""A compact conservative E(3)-equivariant molecular potential.

The model uses scalar channels and vector channels.  All scalar updates depend on
distances or vector dot-products, while vector updates are linear combinations of
relative directions and neighbour vectors.  The predicted energy is therefore
rotation/translation invariant; forces are its negative coordinate gradient.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    hidden_dim: int = 96
    radial_basis: int = 16
    interactions: int = 4
    cutoff: float = 5.0
    max_atomic_number: int = 100

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class GaussianRadialBasis(nn.Module):
    def __init__(self, count: int, cutoff: float) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, count)
        spacing = cutoff / max(count - 1, 1)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max(spacing * spacing, 1e-6)
        self.cutoff = float(cutoff)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        expanded = distances.unsqueeze(-1) - self.centers
        values = torch.exp(-self.gamma * expanded.square())
        ratio = torch.clamp(distances / self.cutoff, min=0.0, max=1.0)
        envelope = 0.5 * (torch.cos(torch.pi * ratio) + 1.0)
        envelope = envelope * (distances < self.cutoff).to(distances.dtype)
        return values * envelope.unsqueeze(-1)


class ScalarVectorInteraction(nn.Module):
    """One scalar/vector message-passing block with no external geometry library."""

    def __init__(self, hidden_dim: int, radial_basis: int) -> None:
        super().__init__()
        message_dim = 2 * hidden_dim + radial_basis + hidden_dim
        self.scalar_message = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vector_gate = nn.Sequential(
            nn.Linear(message_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.SiLU(), nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        relative: torch.Tensor,
        radial: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # relative[i, j] is r_j - r_i.  Shapes: B,A,A,3 and B,A,A,R.
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True).clamp_min(1e-8)
        direction = relative / distance
        source_scalar = scalar[:, None, :, :].expand(-1, scalar.shape[1], -1, -1)
        target_scalar = scalar[:, :, None, :].expand(-1, -1, scalar.shape[1], -1)
        source_vector = vector[:, None, :, :, :].expand(-1, scalar.shape[1], -1, -1, -1)
        directional_projection = (source_vector * direction[..., None]).sum(dim=-2)
        message_input = torch.cat([target_scalar, source_scalar, radial, directional_projection], dim=-1)

        mask = edge_mask[..., None].to(scalar.dtype)
        scalar_message = self.scalar_message(message_input) * mask
        gates = self.vector_gate(message_input)
        direction_gate, neighbour_gate = gates.chunk(2, dim=-1)
        vector_message = (
            direction_gate[..., None, :] * direction[..., None]
            + neighbour_gate[..., None, :] * source_vector
        ) * mask[..., None]

        scalar_aggregate = scalar_message.sum(dim=2)
        vector_aggregate = vector_message.sum(dim=2)
        vector_norm = torch.sqrt(vector_aggregate.square().sum(dim=-2) + 1e-10)
        updated_scalar = scalar + self.scalar_update(
            torch.cat([scalar, scalar_aggregate, vector_norm], dim=-1)
        )
        return self.norm(updated_scalar), vector + vector_aggregate


class EquivariantEnergyModel(nn.Module):
    """Conservative local molecular force field with scalar/vector equivariance."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.max_atomic_number + 1, config.hidden_dim, padding_idx=0)
        self.radial = GaussianRadialBasis(config.radial_basis, config.cutoff)
        self.blocks = nn.ModuleList(
            [ScalarVectorInteraction(config.hidden_dim, config.radial_basis) for _ in range(config.interactions)]
        )
        self.energy_head = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, atomic_numbers: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Return total energies for batches of configurations."""
        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0).expand(positions.shape[0], -1)
        if atomic_numbers.shape[:2] != positions.shape[:2]:
            raise ValueError("atomic_numbers and positions disagree on batch or atom dimensions")
        atom_mask = atomic_numbers.ne(0)
        scalar = self.embedding(atomic_numbers)
        vector = torch.zeros(
            (*scalar.shape[:2], 3, scalar.shape[-1]), dtype=scalar.dtype, device=scalar.device
        )
        relative = positions[:, None, :, :] - positions[:, :, None, :]
        distances = torch.linalg.vector_norm(relative, dim=-1)
        pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
        no_self = ~torch.eye(positions.shape[1], dtype=torch.bool, device=positions.device)[None]
        edge_mask = pair_mask & no_self & distances.lt(self.config.cutoff)
        radial = self.radial(distances)
        for block in self.blocks:
            scalar, vector = block(scalar, vector, relative, radial, edge_mask)
        vector_norm = torch.sqrt(vector.square().sum(dim=-2) + 1e-10)
        atom_energy = self.energy_head(torch.cat([scalar, vector_norm], dim=-1)).squeeze(-1)
        return (atom_energy * atom_mask.to(atom_energy.dtype)).sum(dim=-1)

    def energy_and_forces(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        create_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = positions.requires_grad_(True)
        energy = self(atomic_numbers, positions)
        forces = -torch.autograd.grad(energy.sum(), positions, create_graph=create_graph)[0]
        return energy, forces
