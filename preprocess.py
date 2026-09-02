"""
Preprocessing: normalization + patch sampling
"""

import numpy as np


def normalize_modality(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize using only brain voxels (intensity > 0)."""
    brain_mask = volume > 0
    brain_voxels = volume[brain_mask]
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    std = std if std > 1e-8 else 1e-8

    normalized = (volume - mean) / std
    return normalized.astype(np.float32)


def normalize_case(volumes: dict) -> dict:
    """Normalize the 4 modalities. Leaves SEG untouched."""
    normalized = {}
    for name, vol in volumes.items():
        if name == "SEG":
            normalized[name] = vol
        else:
            normalized[name] = normalize_modality(vol)
    return normalized


def get_tumor_voxels(seg: np.ndarray) -> np.ndarray:
    """Compute tumor voxel coords once per case."""
    return np.argwhere(seg > 0)


def sample_patch_center(shape, tumor_voxels: np.ndarray, patch_size: int, tumor_bias_prob: float = 0.6):
    """Pick a patch center, biased toward tumor voxels with probability tumor_bias_prob."""
    half = patch_size // 2

    if np.random.rand() < tumor_bias_prob:
        if len(tumor_voxels) > 0:
            idx = np.random.randint(len(tumor_voxels))
            cx, cy, cz = tumor_voxels[idx]
            jitter = patch_size // 4
            cx += np.random.randint(-jitter, jitter + 1)
            cy += np.random.randint(-jitter, jitter + 1)
            cz += np.random.randint(-jitter, jitter + 1)
        else:
            cx, cy, cz = _random_center(shape)
    else:
        cx, cy, cz = _random_center(shape)

    cx = np.clip(cx, half, shape[0] - half)
    cy = np.clip(cy, half, shape[1] - half)
    cz = np.clip(cz, half, shape[2] - half)

    return int(cx), int(cy), int(cz)


def _random_center(shape):
    return (
        np.random.randint(0, shape[0]),
        np.random.randint(0, shape[1]),
        np.random.randint(0, shape[2]),
    )


def extract_patch(volume: np.ndarray, center, patch_size: int) -> np.ndarray:
    cx, cy, cz = center
    half = patch_size // 2
    return volume[
        cx - half : cx + half,
        cy - half : cy + half,
        cz - half : cz + half,
    ]


def extract_case_patch(volumes: dict, tumor_voxels: np.ndarray, patch_size: int = 96, tumor_bias_prob: float = 0.6) -> dict:
    """Sample one shared patch location, crop all 5 volumes at it."""
    center = sample_patch_center(volumes["SEG"].shape, tumor_voxels, patch_size, tumor_bias_prob)

    patches = {}
    for name, vol in volumes.items():
        patches[name] = extract_patch(vol, center, patch_size)

    return patches