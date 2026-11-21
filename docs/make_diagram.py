import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")

def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#333"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

for i, y in enumerate([3.6, 2.0, 0.4]):
    box(0.2, y, 3.0, 1.0, f"Hospital {i}\n20 private cases\nFlower client + 3D U-Net", "#dbeafe")
    ax.annotate("", xy=(5.3, 2.5), xytext=(3.3, y + 0.5),
                arrowprops=dict(arrowstyle="<|-|>", color="#333"))

box(5.3, 1.8, 2.6, 1.4, "Central server\nFedAvg + evaluation", "#dcfce7")
ax.text(4.3, 2.5, "weights only\ngRPC + TLS", ha="center", va="center", fontsize=9,
        bbox=dict(fc="white", ec="none"))
box(9.0, 2.0, 1.8, 1.0, "React\ndashboard", "#fef3c7")
ax.annotate("", xy=(8.95, 2.5), xytext=(8.0, 2.5), arrowprops=dict(arrowstyle="-|>", color="#333"))
ax.text(8.5, 2.75, "metrics JSON", ha="center", fontsize=8)
ax.set_title("FedMed architecture", fontsize=12)
plt.savefig("docs/architecture.png", dpi=130, bbox_inches="tight")
print("saved docs/architecture.png")
