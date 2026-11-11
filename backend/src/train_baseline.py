"""Centralized baseline: train a 3D U-Net on pooled data. Federated results are compared to this."""
import argparse, json, os, time
import torch
from torch.utils.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose

from src.data.transforms import get_file_lists, make_dataset, ROI
from src.models.unet import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-train", type=int, default=20)
    p.add_argument("--max-val", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="outputs")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_files, val_files = get_file_lists()
    train_files, val_files = train_files[:args.max_train], val_files[:args.max_val]
    print(f"Train cases: {len(train_files)}, Val cases: {len(val_files)}")

    train_loader = DataLoader(make_dataset(train_files, True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(make_dataset(val_files, False), batch_size=1,
                            shuffle=False, num_workers=args.workers)

    model = build_model().to(device)
    loss_fn = DiceLoss(sigmoid=True, squared_pred=True, smooth_nr=1e-5, smooth_dr=1e-5)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["image"].to(device), batch["label"].to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    pred = sliding_window_inference(x, ROI, sw_batch_size=2, predictor=model)
                dice_metric(y_pred=[post(i) for i in pred], y=list(y))
            tc, wt, et = dice_metric.aggregate().tolist()
            dice_metric.reset()

        mean_dice = (tc + wt + et) / 3
        rec = dict(epoch=epoch, train_loss=train_loss, dice_tc=tc, dice_wt=wt,
                   dice_et=et, dice_mean=mean_dice, seconds=time.time() - t0)
        history.append(rec)
        print(rec)
        torch.save(model.state_dict(), f"{args.out}/baseline_last.pt")
        with open(f"{args.out}/baseline_metrics.json", "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()