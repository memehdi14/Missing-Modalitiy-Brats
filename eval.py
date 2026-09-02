"""
Evaluation: Dice score across a representative set of modality combinations.
Run from a terminal: python eval.py
"""

import itertools
import torch
import numpy as np
from pathlib import Path

from dataset import create_or_load_split, load_case, seg_to_regions, MODALITY_ORDER
from preprocess import normalize_case
from models.unet_baseline import UNet3D


ROOT_DIR = Path(r"D:\MMD\IPA Assignment\BraTS2021\BraTS2021_Training_Data")
SPLIT_DIR = Path(r"D:\MMD\IPA Assignment\BraTS2021\splits")
CHECKPOINT_PATH = Path("checkpoints/best.pt")

PATCH_SIZE = 96
STRIDE = 64  # overlap between windows, smaller stride = more overlap = slower but smoother

# representative subset for now, expand to all 15 later if time allows
MODALITY_COMBOS = [
    (0, 1, 2, 3),  # all present
    (1, 3),        # T1ce + FLAIR, clinically common pair
    (3,),          # FLAIR only
    (1,),          # T1ce only
    (0,),          # T1 only
]


def dice_score(pred, target, eps=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return ((2 * intersection + eps) / (union + eps)).item()


def sliding_window_inference(model, image, patch_size, stride, device):
    """
    image: (4, H, W, D) tensor
    Runs the model over overlapping patches across the full volume,
    averaging predictions where windows overlap.
    """
    _, H, W, D = image.shape
    out_channels = 3

    pred_sum = torch.zeros((out_channels, H, W, D))
    count = torch.zeros((out_channels, H, W, D))

    x_starts = list(range(0, max(H - patch_size, 0) + 1, stride)) or [0]
    y_starts = list(range(0, max(W - patch_size, 0) + 1, stride)) or [0]
    z_starts = list(range(0, max(D - patch_size, 0) + 1, stride)) or [0]

    if x_starts[-1] + patch_size < H:
        x_starts.append(H - patch_size)
    if y_starts[-1] + patch_size < W:
        y_starts.append(W - patch_size)
    if z_starts[-1] + patch_size < D:
        z_starts.append(D - patch_size)

    model.eval()
    with torch.no_grad():
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    patch = image[:, x:x+patch_size, y:y+patch_size, z:z+patch_size]
                    patch = patch.unsqueeze(0).to(device)
                    logits = model(patch)
                    probs = torch.sigmoid(logits).squeeze(0).cpu()

                    pred_sum[:, x:x+patch_size, y:y+patch_size, z:z+patch_size] += probs
                    count[:, x:x+patch_size, y:y+patch_size, z:z+patch_size] += 1

    return pred_sum / count.clamp(min=1)


def evaluate_combo(model, case_ids, present_indices, device):
    dice_scores = {"WT": [], "TC": [], "ET": []}

    for i, case_id in enumerate(case_ids):
        volumes = load_case(ROOT_DIR / case_id)
        volumes = normalize_case(volumes)

        image = np.stack([volumes[name] for name in MODALITY_ORDER], axis=0)
        label = seg_to_regions(volumes["SEG"])

        # zero out modalities not in this combo
        mask = np.zeros(4, dtype=np.float32)
        mask[list(present_indices)] = 1.0
        for c in range(4):
            if mask[c] == 0:
                image[c] = 0.0

        image_t = torch.from_numpy(image).float()
        label_t = torch.from_numpy(label).float()

        pred = sliding_window_inference(model, image_t, PATCH_SIZE, STRIDE, device)
        pred_binary = (pred > 0.5).float()

        for c, region in enumerate(["WT", "TC", "ET"]):
            dice_scores[region].append(dice_score(pred_binary[c], label_t[c]))

        print(f"  case {i+1}/{len(case_ids)}  {case_id}  "
              f"WT {dice_scores['WT'][-1]:.4f}  TC {dice_scores['TC'][-1]:.4f}  ET {dice_scores['ET'][-1]:.4f}",
              flush=True)

    return {region: float(np.mean(scores)) for region, scores in dice_scores.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_ids = create_or_load_split(ROOT_DIR, SPLIT_DIR)

    model = UNet3D(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    print(f"evaluating on {len(test_ids)} test cases\n")

    for combo in MODALITY_COMBOS:
        present_names = [MODALITY_ORDER[i] for i in combo]
        print(f"--- {'+'.join(present_names)} ---")
        result = evaluate_combo(model, test_ids, combo, device)
        print(f"{'+'.join(present_names):25s}  WT {result['WT']:.4f}  TC {result['TC']:.4f}  ET {result['ET']:.4f}\n")


if __name__ == "__main__":
    main()