"""Load one sample, print shapes/label stats, and save a visualization."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd

MODALITIES = ["FLAIR", "T1w", "T1gd", "T2w"]
LABELS = {0: "background", 1: "edema", 2: "non-enhancing tumor", 3: "enhancing tumor"}

def main():
    os.makedirs("outputs", exist_ok=True)
    tfm = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])
    ds = DecathlonDataset(root_dir="data", task="Task01_BrainTumour",
                          section="training", download=False, transform=tfm, cache_rate=0.0)
    sample = ds[0]
    img, lbl = sample["image"].numpy(), sample["label"].numpy()

    print("Image shape (C,H,W,D):", img.shape)
    print("Label shape (C,H,W,D):", lbl.shape)
    print("Label classes present:", {int(k): LABELS[int(k)] for k in np.unique(lbl)})

    z = int(np.argmax(lbl[0].sum(axis=(0, 1))))
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i, name in enumerate(MODALITIES):
        axes[i].imshow(img[i, :, :, z].T, cmap="gray", origin="lower")
        axes[i].set_title(name); axes[i].axis("off")
    axes[4].imshow(img[0, :, :, z].T, cmap="gray", origin="lower")
    axes[4].imshow(np.ma.masked_equal(lbl[0, :, :, z].T, 0), cmap="jet", alpha=0.5, origin="lower")
    axes[4].set_title("Tumor mask overlay"); axes[4].axis("off")
    plt.tight_layout()
    plt.savefig("outputs/sample_slice.png", dpi=120)
    print("Saved outputs/sample_slice.png")

if __name__ == "__main__":
    main()