import json, glob, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = []
for f in glob.glob("results/dp_clip1.0_noise*.json"):
    noise = float(re.search(r"noise([\d.]+)\.json", f).group(1))
    runs.append((noise, json.load(open(f))["history"]))
runs.sort()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for noise, h in runs:
    ax[0].plot([r["round"] for r in h], [r["dice_mean"] for r in h], label=f"noise {noise}")
ax[0].set_title("Global Dice per round"); ax[0].set_xlabel("Round"); ax[0].legend()
ax[1].plot(range(len(runs)), [h[-1]["dice_mean"] for _, h in runs], marker="o")
ax[1].set_xticks(range(len(runs))); ax[1].set_xticklabels([str(n) for n, _ in runs])
ax[1].set_title("Final Dice vs noise multiplier"); ax[1].set_xlabel("noise")
plt.tight_layout(); plt.savefig("results/dp_tradeoff.png", dpi=120)
for n, h in runs:
    print(f"noise {n}: final Dice {h[-1]['dice_mean']:.4f}")
