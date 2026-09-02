"""
Dataset class, train/val/test split, modality dropout
"""

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from pathlib import Path

from preprocess import normalize_case, extract_case_patch, get_tumor_voxels


MODALITY_SUFFIXES = {
    "T1": "_t1.nii.gz",
    "T1ce": "_t1ce.nii.gz",
    "T2": "_t2.nii.gz",
    "FLAIR": "_flair.nii.gz",
}
SEG_SUFFIX = "_seg.nii.gz"
MODALITY_ORDER = ["T1", "T1ce", "T2", "FLAIR"]


def discover_cases(root_dir: Path) -> list:
    root_dir = Path(root_dir)
    return sorted([p for p in root_dir.iterdir() if p.is_dir()])


def create_or_load_split(root_dir: Path, split_dir: Path, train_frac=0.7, val_frac=0.15, seed=42):
    """Creates a train/val/test split and saves it, or loads it if it already exists."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_file = split_dir / "train_ids.txt"
    val_file = split_dir / "val_ids.txt"
    test_file = split_dir / "test_ids.txt"

    if train_file.exists() and val_file.exists() and test_file.exists():
        train_ids = train_file.read_text().splitlines()
        val_ids = val_file.read_text().splitlines()
        test_ids = test_file.read_text().splitlines()
        return train_ids, val_ids, test_ids

    cases = discover_cases(root_dir)
    case_names = [c.name for c in cases]

    rng = np.random.RandomState(seed)
    rng.shuffle(case_names)

    n = len(case_names)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = case_names[:n_train]
    val_ids = case_names[n_train : n_train + n_val]
    test_ids = case_names[n_train + n_val :]

    train_file.write_text("\n".join(train_ids))
    val_file.write_text("\n".join(val_ids))
    test_file.write_text("\n".join(test_ids))

    return train_ids, val_ids, test_ids


def load_case(patient_dir: Path) -> dict:
    patient_dir = Path(patient_dir)
    case_id = patient_dir.name

    volumes = {}
    for name, suffix in MODALITY_SUFFIXES.items():
        path = patient_dir / f"{case_id}{suffix}"
        volumes[name] = nib.load(path).get_fdata().astype(np.float32)

    seg_path = patient_dir / f"{case_id}{SEG_SUFFIX}"
    volumes["SEG"] = nib.load(seg_path).get_fdata().astype(np.float32)

    return volumes


def seg_to_regions(seg: np.ndarray) -> np.ndarray:
    """WT = 1+2+4, TC = 1+4, ET = 4. Overlapping regions, stacked as channels."""
    wt = (seg > 0).astype(np.float32)
    tc = ((seg == 1) | (seg == 4)).astype(np.float32)
    et = (seg == 4).astype(np.float32)
    return np.stack([wt, tc, et], axis=0)


def random_modality_dropout(image: np.ndarray, min_present=1):
    """Zeros out a random subset of modality channels, keeping at least min_present."""
    n_modalities = image.shape[0]

    while True:
        presence_mask = (np.random.rand(n_modalities) > 0.5).astype(np.float32)
        if presence_mask.sum() >= min_present:
            break

    dropped_image = image.copy()
    for i, present in enumerate(presence_mask):
        if present == 0:
            dropped_image[i] = 0.0

    return dropped_image, presence_mask


class BraTSDataset(Dataset):
    def __init__(self, root_dir, case_ids, patch_size=96, tumor_bias_prob=0.6, apply_dropout=True):
        self.root_dir = Path(root_dir)
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.tumor_bias_prob = tumor_bias_prob
        self.apply_dropout = apply_dropout

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        patient_dir = self.root_dir / case_id

        volumes = load_case(patient_dir)
        volumes = normalize_case(volumes)
        tumor_voxels = get_tumor_voxels(volumes["SEG"])

        patch = extract_case_patch(
            volumes, tumor_voxels, patch_size=self.patch_size, tumor_bias_prob=self.tumor_bias_prob
        )

        image = np.stack([patch[name] for name in MODALITY_ORDER], axis=0)
        label = seg_to_regions(patch["SEG"])

        if self.apply_dropout:
            image, presence_mask = random_modality_dropout(image)
        else:
            presence_mask = np.ones(len(MODALITY_ORDER), dtype=np.float32)

        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(label).float(),
            "presence_mask": torch.from_numpy(presence_mask).float(),
            "case_id": case_id,
        }