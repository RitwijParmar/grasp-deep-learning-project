"""Dataset adapters with explicit units and a synthetic scientific smoke test."""

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .types import MolecularFrames

THREE_BPA_URL = "https://github.com/davkovacs/BOTNet-datasets/raw/refs/heads/main/dataset_3BPA.tar.gz"


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise ValueError("archive contains a path outside the requested destination")
    archive.extractall(destination)


def download_3bpa(destination: Path, url: str = THREE_BPA_URL) -> Path:
    """Download the public 3BPA benchmark without silently overwriting data."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "dataset_3BPA.tar.gz"
    extracted = destination / "dataset_3BPA"
    if extracted.exists():
        return extracted
    if not archive_path.exists():
        urllib.request.urlretrieve(url, archive_path)  # nosec B310: public benchmark URL is explicit
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract(archive, destination)
    if not extracted.exists():
        candidates = [path for path in destination.rglob("dataset_3BPA") if path.is_dir()]
        if not candidates:
            raise FileNotFoundError("3BPA archive did not contain dataset_3BPA")
        return candidates[0]
    return extracted


def _first_present(mapping: Dict[str, object], choices: Iterable[str]) -> object:
    for choice in choices:
        if choice in mapping:
            return mapping[choice]
    raise KeyError("None of the expected keys found: {}".format(", ".join(choices)))


def load_extxyz(path: Path, name: Optional[str] = None) -> MolecularFrames:
    """Read a single-composition extended XYZ trajectory through ASE.

    3BPA and Landscape17 use variants of the standard ``energy``/``forces`` keys;
    accepting the documented aliases avoids accidental reliance on a particular
    preprocessing release.
    """
    try:
        from ase.io import read
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Install chemistry dependencies: pip install -e '.[chemistry]'") from exc
    atoms_frames = read(str(path), index=":", format="extxyz")
    if not atoms_frames:
        raise ValueError("no frames in {}".format(path))
    numbers = torch.as_tensor(atoms_frames[0].numbers, dtype=torch.long)
    positions: List[np.ndarray] = []
    energies: List[float] = []
    forces: List[np.ndarray] = []
    for index, atoms in enumerate(atoms_frames):
        if not np.array_equal(atoms.numbers, numbers.numpy()):
            raise ValueError("mixed composition at frame {} in {}".format(index, path))
        try:
            energy = _first_present(atoms.info, ("energy", "REF_energy", "E"))
        except KeyError:
            energy = atoms.get_potential_energy()
        try:
            force = _first_present(atoms.arrays, ("forces", "REF_forces", "force"))
        except KeyError:
            force = atoms.get_forces()
        positions.append(np.asarray(atoms.positions, dtype=np.float32))
        energies.append(float(energy))
        forces.append(np.asarray(force, dtype=np.float32))
    return MolecularFrames(
        atomic_numbers=numbers,
        positions=torch.from_numpy(np.stack(positions)),
        energies=torch.tensor(energies, dtype=torch.float32),
        forces=torch.from_numpy(np.stack(forces)),
        name=name or path.stem,
        metadata={"source": str(path), "energy_unit": "eV", "distance_unit": "angstrom"},
    )


def save_frames(frames: MolecularFrames, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "atomic_numbers": frames.atomic_numbers.cpu(),
            "positions": frames.positions.cpu(),
            "energies": frames.energies.cpu(),
            "forces": frames.forces.cpu(),
            "name": frames.name,
            "metadata": frames.metadata or {},
        },
        path,
    )


def load_frames(path: Path) -> MolecularFrames:
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    return MolecularFrames(
        atomic_numbers=data["atomic_numbers"],
        positions=data["positions"].float(),
        energies=data["energies"].float(),
        forces=data["forces"].float(),
        name=data.get("name", Path(path).stem),
        metadata=data.get("metadata", {}),
    )


def materialize_3bpa_splits(source: Path, destination: Path) -> Dict[str, Path]:
    """Convert official 3BPA extended-XYZ splits to compact torch artifacts."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    aliases = {
        "train_300K": ("train_300K.xyz", "train_300K.extxyz"),
        "test_300K": ("test_300K.xyz", "test_300K.extxyz"),
        "test_600K": ("test_600K.xyz", "test_600K.extxyz"),
        "test_1200K": ("test_1200K.xyz", "test_1200K.extxyz"),
    }
    outputs: Dict[str, Path] = {}
    for split, names in aliases.items():
        candidates = [source / filename for filename in names]
        match = next((candidate for candidate in candidates if candidate.exists()), None)
        if match is None:
            raise FileNotFoundError("missing {} under {}".format(" or ".join(names), source))
        output = destination / (split + ".pt")
        save_frames(load_extxyz(match, name="3BPA-" + split), output)
        outputs[split] = output
    manifest = {
        "dataset": "3BPA",
        "source": str(source),
        "splits": {name: str(path) for name, path in outputs.items()},
        "units": {"energy": "eV", "forces": "eV/angstrom", "positions": "angstrom"},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return outputs


def synthetic_harmonic_frames(
    frames: int = 96, atoms: int = 5, seed: int = 7, spring: float = 0.8
) -> MolecularFrames:
    """Translation/rotation-invariant pairwise harmonic system for CI and GPU smoke tests."""
    if atoms < 2:
        raise ValueError("atoms must be at least two")
    generator = torch.Generator().manual_seed(seed)
    reference = torch.randn((atoms, 3), generator=generator) * 0.7
    positions = reference.unsqueeze(0) + 0.15 * torch.randn((frames, atoms, 3), generator=generator)
    positions = positions - positions.mean(dim=1, keepdim=True)
    pair = positions[:, :, None, :] - positions[:, None, :, :]
    reference_pair = reference[None, :, :] - reference[:, None, :]
    distances = torch.linalg.vector_norm(pair, dim=-1)
    target = torch.linalg.vector_norm(reference_pair, dim=-1)
    upper = torch.triu_indices(atoms, atoms, offset=1)
    energy = 0.5 * spring * (distances[:, upper[0], upper[1]] - target[upper[0], upper[1]]).square().sum(dim=-1)
    pair_direction = pair / distances[..., None].clamp_min(1e-8)
    scalar = spring * (distances - target) / 2.0
    scalar[:, torch.arange(atoms), torch.arange(atoms)] = 0.0
    forces = -(scalar[..., None] * pair_direction).sum(dim=2)
    return MolecularFrames(
        atomic_numbers=torch.tensor([6] * atoms, dtype=torch.long),
        positions=positions.float(),
        energies=energy.float(),
        forces=forces.float(),
        name="synthetic_pair_harmonic",
        metadata={"oracle": "pairwise harmonic", "distance_unit": "arbitrary"},
    )


def dataset_fingerprint(frames: MolecularFrames) -> str:
    """Stable small fingerprint to prevent accidental mixed-data comparisons."""
    digest = hashlib.sha256()
    for value in (frames.atomic_numbers, frames.positions[:2], frames.energies[:2], frames.forces[:2]):
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]
