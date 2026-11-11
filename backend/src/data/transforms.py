"""Preprocessing, augmentation, and dataset helpers for MSD Task01_BrainTumour."""
import numpy as np
import torch
from monai.apps import DecathlonDataset
from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, CropForegroundd,
    NormalizeIntensityd, RandSpatialCropd, RandFlipd, RandScaleIntensityd,
    RandShiftIntensityd, SpatialPadd, MapTransform, EnsureTyped,
)

ROI = (96, 96, 96)


class ConvertMSDLabelsd(MapTransform):
    """MSD labels (1 edema, 2 non-enh, 3 enh) -> 3 channels: TC, WT, ET."""
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lbl = d[key]
            lbl = lbl[0] if lbl.shape[0] == 1 else lbl
            tc = (lbl == 2) | (lbl == 3)
            wt = lbl > 0
            et = lbl == 3
            d[key] = torch.stack([tc, wt, et], dim=0).float()
        return d


def get_transforms(train: bool):
    keys = ["image", "label"]
    base = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", labels=None),
        ConvertMSDLabelsd(keys="label"),
        CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if train:
        base += [
            SpatialPadd(keys=keys, spatial_size=ROI),
            RandSpatialCropd(keys=keys, roi_size=ROI, random_size=False),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]
    base.append(EnsureTyped(keys=keys))
    return Compose(base)


def get_file_lists(root_dir="data"):
    """Return (train_files, val_files): lists of {'image': path, 'label': path}.
    Uses MONAI's built-in 80/20 split. Federated partitioning (Week 2) reuses this."""
    out = {}
    for section in ("training", "validation"):
        ds = DecathlonDataset(root_dir=root_dir, task="Task01_BrainTumour",
                              section=section, download=False, transform=None,
                              cache_rate=0.0)
        out[section] = [{"image": d["image"], "label": d["label"]} for d in ds.data]
    return out["training"], out["validation"]


def make_dataset(files, train: bool):
    return Dataset(data=files, transform=get_transforms(train))