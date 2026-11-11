"""Centralized baseline with checkpoint/resume. FedAvg results are compared to this."""
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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-train", type=int, default=60)
    p.add_argument("--max-val", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", default="outputs")
    p.add_argument("--results", default="results")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_files, val_files = get_file_lists()
    train_files, val_files = train_files[:args.max_train], val_files[:args.max_val]
    print(f"Train cases: {len(train_files)}, Val cases: {len(val_files)}")

    train_loader = DataLoader(make_dataset(train_files, True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(make_dataset(val_files, False), batch_size=1,
                            shuffle=False, num_workers=args.workers)

    model = build_model().to(device)
    loss_fn = DiceLoss(sigmoid=True, squared_pred=True, smooth_nr=1e-5, smooth_dr=1e-5)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    ckpt_path = f"{args.out}/baseline_ckpt.pt"
    history, best, start_epoch = [], 0.0, 1
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        history, best, start_epoch = ck["history"], ck["best"], ck["epoch"] + 1
        print(f"Resumed from epoch {ck['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["image"].to(device), batch["label"].to(device)
                pred = sliding_window_inference(x, ROI, sw_batch_size=2, predictor=model)
                dice_metric(y_pred=[post(i) for i in pred], y=list(y))
            tc, wt, et = dice_metric.aggregate().tolist()
            dice_metric.reset()

        mean_dice = (tc + wt + et) / 3
        rec = dict(epoch=epoch, train_loss=round(train_loss, 4), dice_tc=round(tc, 4),
                   dice_wt=round(wt, 4), dice_et=round(et, 4),
                   dice_mean=round(mean_dice, 4), seconds=round(time.time() - t0))
        history.append(rec)
        print(rec, flush=True)

        if mean_dice > best:
            best = mean_dice
            torch.save(model.state_dict(), f"{args.out}/baseline_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "history": history, "best": best}, ckpt_path)
        with open(f"{args.results}/baseline_metrics.json", "w") as f:
            json.dump({"best_dice_mean": best, "history": history}, f, indent=2)

    print(f"Done. Best mean Dice: {best:.4f}")


if __name__ == "__main__":
    main()