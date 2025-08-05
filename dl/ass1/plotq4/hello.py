#!/usr/bin/env python3
"""
Plot MNIST optimizer comparison from JSON results.
"""

import json, matplotlib.pyplot as plt

with open("results.json") as f:
    data = json.load(f)

epochs = list(range(1, 11))

# 1. Accuracy vs Epoch
plt.figure(figsize=(6,4))
for name, vals in data.items():
    plt.plot(epochs, vals["accuracy"], marker='o', label=name)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("MNIST 10-epoch comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_curve.png")
plt.show()

# 2. Final accuracy + wall-time bar chart
labels = list(data.keys())
final_acc = [vals["accuracy"][-1] for vals in data.values()]
times     = [vals["time"] for vals in data.values()]

fig, ax = plt.subplots(figsize=(5,3))
ax.bar(labels, final_acc, color=['tab:blue', 'tab:orange', 'tab:green'])
ax.set_ylabel("Final Accuracy (%)")
ax.set_title("Final accuracy & training time")
for bar, t in zip(ax.patches, times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height+0.3,
            f"{t:.1f}s", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("final_accuracy.png")
plt.show()
