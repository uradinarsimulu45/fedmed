import json
b = json.load(open("results/baseline_audit_15epochs.json"))["history"]
f = json.load(open("results/federated_audit_15rounds.json"))["history"]
print(f"{'Step':>4} | {'Baseline Dice':>13} | {'Federated Dice':>14}")
for i in range(min(len(b), len(f))):
    print(f"{i+1:>4} | {b[i]['dice_mean']:>13.4f} | {f[i]['dice_mean']:>14.4f}")
bb = max(r["dice_mean"] for r in b); ff = max(r["dice_mean"] for r in f)
print(f"\nBest baseline: {bb:.4f} | Best federated: {ff:.4f} | Ratio: {ff/bb:.0%}")
