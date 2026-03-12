import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Each file is one MPI rank (unique_id) for one run:
    # res/scatter_times_<nb_clients>_<data_size>_<run_id>_<unique_id>.csv
    csv_files = sorted(glob.glob(os.path.join("res", "scatter_times_*_*_*_*.csv")))

    # (nb_clients, data_size, run_id) -> {unique_id: times_1d}
    per_run_per_rank = defaultdict(dict)

    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        stem = basename.removeprefix("scatter_times_").removesuffix(".csv")
        parts = stem.split("_")
        if len(parts) != 4:
            continue

        try:
            nb_clients, data_size, run_id, unique_id = map(int, parts)
        except ValueError:
            continue

        times = np.loadtxt(csv_file, delimiter=",")
        times = np.atleast_1d(times)
        per_run_per_rank[(nb_clients, data_size, run_id)][unique_id] = times

    # Step 1: compute ONE scalar per run:
    # run_scalar = average over unique_ids of (scatter latency for that unique_id)
    # "1 value from a file" => take the FIRST recorded scatter latency (ns)
    run_scalars = {}  # (nb_clients, data_size, run_id) -> float

    for (nb_clients, data_size, run_id), rank_map in per_run_per_rank.items():
        if not rank_map:
            continue

        rank_latencies_ns = []
        for t in rank_map.values():
            # one scatter latency value from this file (ns)
            rank_latencies_ns.append(float(t[0]))

        run_scalars[(nb_clients, data_size, run_id)] = float(np.mean(rank_latencies_ns))

    # Step 2: one bar per data_size:
    # bar height = mean(run_scalar over runs)
    # error bar  = std(run_scalar over runs)
    by_nbclients_datasize = defaultdict(lambda: defaultdict(list))  # nb_clients -> data_size -> [run_scalar]

    for (nb_clients, data_size, run_id), run_scalar in run_scalars.items():
        by_nbclients_datasize[nb_clients][data_size].append(run_scalar)

    os.makedirs("res", exist_ok=True)

    # One figure per nb_clients (keeps things clean if you benchmark different client counts)
    for nb_clients in sorted(by_nbclients_datasize.keys()):
        data_sizes = sorted(by_nbclients_datasize[nb_clients].keys())
        if not data_sizes:
            continue

        means = []
        stds = []
        n_runs_list = []

        for ds in data_sizes:
            vals = np.asarray(by_nbclients_datasize[nb_clients][ds], dtype=float)
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
            n_runs_list.append(int(vals.size))

        x = np.arange(len(data_sizes), dtype=float)

        plt.figure(figsize=(10, 6))
        plt.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
        )

        plt.xticks(x, [str(ds) for ds in data_sizes])
        plt.xlabel("Data size")
        plt.ylabel("Scatter latency (ns)")
        plt.yscale("log")
        plt.title(
            f"Scatter latency vs data size (nb_clients={nb_clients})\n"
            "One bar per data size: mean over runs; each run = mean over unique_ids (first scatter only)"
        )
        plt.grid(True, axis="y", which="both", alpha=0.3)

        for xi, m, n_runs in zip(x, means, n_runs_list):
            plt.text(xi, m, f"n={n_runs}", ha="center", va="bottom", fontsize=9)

        out_path = os.path.join("res", f"scatter_bar_vs_datasize_nbclients_{nb_clients}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    exit(0)
