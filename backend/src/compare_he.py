import json, re
p = json.load(open("results/federated_audit_15rounds.json"))["history"]
e = json.load(open("results/federated_he_audit_15rounds.json"))["history"]
print(f"{'Round':>5} | {'Plain Dice':>10} | {'Encrypted Dice':>14}")
for i in range(min(len(p), len(e))):
    print(f"{i+1:>5} | {p[i]['dice_mean']:>10.4f} | {e[i]['dice_mean']:>14.4f}")
bp = max(r["dice_mean"] for r in p); be = max(r["dice_mean"] for r in e)
print(f"\nBest plain: {bp:.4f} | Best encrypted: {be:.4f}")
print(f"Upload per hospital per round: plain 4.8 MB | encrypted {e[0]['upload_mb_per_hospital']} MB")
try:
    t = re.search(r"Run finished (\d+) round\(s\) in ([\d.]+)s", open("logs/server_he.log").read())
    print(f"Encrypted: {float(t.group(2))/int(t.group(1)):.0f} s per round")
except Exception:
    pass
