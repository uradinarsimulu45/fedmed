import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

from src.data.transforms import get_file_lists, make_dataset, ROI
from src.models.unet import build_model

os.makedirs("results/masks", exist_ok=True)
_, val_files = get_file_lists()
ds = make_dataset(val_files[:3], False)

model = build_model()
model.load_state_dict(torch.load("outputs/baseline_best.pt", map_location="cpu"))
model.eval()

for n in range(3):
    s = ds[n]
    img = s["image"].unsqueeze(0)
    with torch.no_grad():
        pred = sliding_window_inference(img, ROI, 2, model)
    pred = (torch.sigmoid(pred)[0] > 0.5).numpy()
    true = s["label"].numpy()
    image = s["image"].numpy()
    z = int(np.argmax(true[1].sum(axis=(0, 1))))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    base = image[0, :, :, z].T
    ax[0].imshow(base, cmap="gray", origin="lower"); ax[0].set_title("MRI (FLAIR)")
    ax[1].imshow(base, cmap="gray", origin="lower")
    ax[1].imshow(np.ma.masked_equal(true[1, :, :, z].T, 0), cmap="autumn", alpha=0.6, origin="lower")
    ax[1].set_title("True tumor")
    ax[2].imshow(base, cmap="gray", origin="lower")
    ax[2].imshow(np.ma.masked_equal(pred[1, :, :, z].T, 0), cmap="cool", alpha=0.6, origin="lower")
    ax[2].set_title("Predicted (baseline model)")
    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.savefig(f"results/masks/case{n}.png", dpi=110)
    plt.close()
    print("saved", f"results/masks/case{n}.png")
