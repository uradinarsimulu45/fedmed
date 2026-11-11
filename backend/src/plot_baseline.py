import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/baseline_metrics.json"))
h = d["history"]
ep = [r["epoch"] for r in h]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, [r["train_loss"] for r in h]); ax[0].set_title("Train loss"); ax[0].set_xlabel("Epoch")
for k in ("dice_tc", "dice_wt", "dice_et", "dice_mean"):
    ax[1].plot(ep, [r[k] for r in h], label=k)
ax[1].set_title("Validation Dice"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig("results/baseline_curves.png", dpi=120)
print("Best mean Dice:", d["best_dice_mean"])