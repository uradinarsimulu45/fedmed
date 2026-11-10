"""Download the MSD Task01_BrainTumour dataset (BraTS-derived) via MONAI."""
from monai.apps import DecathlonDataset

def main(root_dir: str = "data"):
    ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        download=True,
        transform=None,
        cache_rate=0.0,
    )
    print(f"Downloaded. Training split size: {len(ds)}")

if __name__ == "__main__":
    main()