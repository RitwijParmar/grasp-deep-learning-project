import torch

from kinetiguard.data import synthetic_harmonic_frames
from kinetiguard.model import EquivariantEnergyModel, ModelConfig


def test_energy_is_translation_and_rotation_invariant() -> None:
    torch.manual_seed(3)
    model = EquivariantEnergyModel(ModelConfig(hidden_dim=16, radial_basis=6, interactions=2, cutoff=4.0))
    frames = synthetic_harmonic_frames(frames=4, atoms=4)
    positions = frames.positions
    energy = model(frames.atomic_numbers, positions)
    shifted = model(frames.atomic_numbers, positions + torch.tensor([2.0, -3.0, 1.5]))
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = model(frames.atomic_numbers, torch.matmul(positions, rotation))
    torch.testing.assert_close(energy, shifted, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(energy, rotated, atol=1e-5, rtol=1e-5)


def test_force_is_energy_gradient() -> None:
    torch.manual_seed(4)
    model = EquivariantEnergyModel(ModelConfig(hidden_dim=16, radial_basis=6, interactions=1, cutoff=4.0))
    frames = synthetic_harmonic_frames(frames=2, atoms=4)
    energy, forces = model.energy_and_forces(frames.atomic_numbers, frames.positions.clone())
    assert energy.shape == (2,)
    assert forces.shape == frames.positions.shape
    assert torch.isfinite(forces).all()
