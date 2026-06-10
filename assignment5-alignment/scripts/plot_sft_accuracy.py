#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 150

runs_dir = "runs/sft_experiment"
sizes = ["128", "256", "512", "1024", "full"]
run_names = {
    "all": {
        "samples128_all": "128",
        "samples256_all": "256",
        "samples512_all": "512",
        "samples1024_all": "1024",
        "samplesfull_all": "full",
    },
    "filtered": {
        "samples128_filtered": "128",
        "samples256_filtered": "256",
        "samples512_filtered": "512",
        "samples1024_filtered": "1024",
        "samplesfull_filtered": "full",
    },
}

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
titles = {"all": "SFT Validation Accuracy vs Step (all)", "filtered": "SFT Validation Accuracy vs Step (filtered)"}
outputs = {"all": "runs/sft_accuracy_plot.png", "filtered": "runs/sft_accuracy_plot_filtered.png"}

for ax, (split, label_map) in zip(axes, run_names.items()):
    for run_name, label in label_map.items():
        log_path = os.path.join(runs_dir, run_name, "log.jsonl")
        if not os.path.exists(log_path):
            continue
        steps, accs = [], []
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("type") == "eval_metrics":
                    steps.append(entry["step"])
                    accs.append(entry["metrics"]["eval/accuracy"])
        ax.plot(steps, accs, marker="o", label=label)
    ax.set_title(titles[split])
    ax.set_xlabel("Step")
    ax.set_ylabel("eval/accuracy")
    ax.legend(title="train_samples")
    ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
out = "runs/sft_accuracy_plot_combined.png"
plt.savefig(out)
print(f"Saved to {out}")

# Also save individual plots
for split, label_map in run_names.items():
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for run_name, label in label_map.items():
        log_path = os.path.join(runs_dir, run_name, "log.jsonl")
        if not os.path.exists(log_path):
            continue
        steps, accs = [], []
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("type") == "eval_metrics":
                    steps.append(entry["step"])
                    accs.append(entry["metrics"]["eval/accuracy"])
        ax2.plot(steps, accs, marker="o", label=label)
    ax2.set_title(titles[split])
    ax2.set_xlabel("Step")
    ax2.set_ylabel("eval/accuracy")
    ax2.legend(title="train_samples")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outputs[split])
    plt.close(fig2)
    print(f"Saved to {outputs[split]}")
