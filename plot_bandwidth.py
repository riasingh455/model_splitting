# plot_bandwidth.py
from pathlib import Path
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("logs/nw_exp/bramble-1-3")

EXPERIMENTS = [
    "resnet18_children_zmq_bs_5_bn_2",
    "resnet18_children_zmq_bs_2_bn_5",
]

def parse_folder_info(folder_name):
    m = re.search(r"bs_(\d+)_bn_(\d+)", folder_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def parse_filename_info(filename):
    m = re.search(r"_(\d+)_(\d+)\.log$", filename)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def parse_net_times(text):
    for line in text.splitlines():
        if line.startswith("net_times:"):
            raw = line.split("net_times:", 1)[1].strip()
            return ast.literal_eval(raw)
    return []

rows = []

for exp in EXPERIMENTS:
    exp_path = ROOT / exp
    batch_size, batch_num = parse_folder_info(exp)

    # iterate runs 1 → 10
    for run_folder in sorted(exp_path.iterdir()):
        if not run_folder.is_dir():
            continue

        for log_file in run_folder.glob("*.log"):
            world_size, run_id = parse_filename_info(log_file.name)

            if world_size is None:
                continue

            text = log_file.read_text(errors="ignore")
            net_times = parse_net_times(text)

            mbps_values = []

            for item in net_times:
                if len(item) < 4:
                    continue

                start, end, _, num_bytes = item[:4]
                duration = end - start

                if duration <= 0:
                    continue

                mbps = (num_bytes * 8) / duration / 1_000_000
                mbps_values.append(mbps)

            if not mbps_values:
                continue

            # avg per run
            avg_mbps = sum(mbps_values) / len(mbps_values)

            rows.append({
                "experiment": exp,
                "batch_size": batch_size,
                "batch_num": batch_num,
                "world_size": world_size,
                "run_folder": run_folder.name,
                "avg_mbps": avg_mbps,
            })

df = pd.DataFrame(rows)

if df.empty:
    print("No data found")
    exit()

# average across 10 runs
summary = (
    df.groupby(["experiment", "batch_size", "batch_num", "world_size"])
      .agg(avg_mbps=("avg_mbps", "mean"))
      .reset_index()
)

print(summary)

plt.figure(figsize=(9, 5))

for exp, group in summary.groupby("experiment"):
    plt.plot(group["world_size"], group["avg_mbps"], marker="o", label=exp)

plt.xlabel("World size / number of splits")
plt.ylabel("Average bandwidth Mbps")
plt.title("Bandwidth comparison (avg over 10 runs)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bandwidth_comparison.png", dpi=200)
plt.show()