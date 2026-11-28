import json, glob, re, math

def eps(z, delta=1e-5):
    return math.sqrt(2 * math.log(1.25 / delta)) / z if z > 0 else float("inf")

runs = []
for f in glob.glob("results/dp_clip1.0_noise*.json"):
    z = float(re.search(r"noise([\d.]+)\.json", f).group(1))
    runs.append((z, json.load(open(f))["history"]))
runs.sort()

print("\n## Differential Privacy (Week 4)\n")
print("Each hospital clips its update to L2 norm 1.0 and adds Gaussian noise (std = noise x clip) before sending.")
print("5 rounds, 30 cases, fixed seed, one run per setting.\n")
print("| Noise | Final mean Dice | Approx. epsilon per round | After 5 rounds |")
print("|---|---|---|---|")
for z, h in runs:
    e = eps(z)
    es = "no noise" if z == 0 else f"{e:.1f}"
    e5 = "-" if z == 0 else f"{e*5:.1f}"
    print(f"| {z} | {h[-1]['dice_mean']:.4f} | {es} | {e5} |")
print("\n**Caveats**")
print("- Epsilon uses the classic Gaussian bound with simple composition and delta = 1e-5. It is a loose upper bound, and the classic bound is only valid for epsilon per round of 1 or less.")
print("- One run per setting on a barely trained model, so small Dice gaps are within run-to-run noise.")
print("- Clipping alone costs accuracy: updates have norm about 4.4 and are clipped to 1.0.")
print("- Encryption and DP were evaluated separately, not combined.")
