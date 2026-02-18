import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Files:
    # res/scatter_times_<nb_clients>_<data_size>_<run_id>_<unique_id>.csv
    #
    # nb_clients is now in the filename and should be used directly.
    csv_files = sorted(glob.glob(os.path.join("res", "scatter_times_*_*_*_*.csv")))

    # (nb_clients, data_size, run_id) -> {unique_id: first_latency_ns}
    per_run_latencies = defaultdict(dict)

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

        # "1 value from a file" => first scatter latency, in ns
        per_run_latencies[(nb_clients, data_size, run_id)][unique_id] = float(times[0])

    # Step 1: ONE scalar per run (per-client average for that run):
    # run_scalar = mean(latency over unique_id/clients) (ns)
    run_scalars = {}  # (data_size, nb_clients, run_id) -> float

    for (nb_clients, data_size, run_id), lat_by_uid in per_run_latencies.items():
        if not lat_by_uid:
            continue

        # Average latency per client for this run
        run_scalar = float(np.mean(list(lat_by_uid.values())))
        run_scalars[(data_size, nb_clients, run_id)] = run_scalar

    # Step 2: aggregate run scalars into bars:
    # For each (data_size, nb_clients):
    #   bar height = mean over runs
    #   error bar  = std over runs
    by_datasize_clients = defaultdict(lambda: defaultdict(list))  # data_size -> nb_clients -> [run_scalar]
    for (data_size, nb_clients, run_id), run_scalar in run_scalars.items():
        by_datasize_clients[data_size][nb_clients].append(run_scalar)

    os.makedirs("res", exist_ok=True)

    data_sizes = sorted(by_datasize_clients.keys())
    if not data_sizes:
        raise SystemExit("No matching CSV files found in res/")

    # Subplots: side-by-side (single row)
    n_plots = len(data_sizes)
    nrows = 1
    ncols = n_plots

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.8 * ncols, 4.8),
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.ravel()

    for ax, data_size in zip(axes_flat, data_sizes):
        clients_list = sorted(by_datasize_clients[data_size].keys())
        if not clients_list:
            ax.set_visible(False)
            continue

        means = []
        stds = []
        n_runs_list = []

        for nb_clients in clients_list:
            vals = np.asarray(by_datasize_clients[data_size][nb_clients], dtype=float)
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
            n_runs_list.append(int(vals.size))

        x = np.arange(len(clients_list), dtype=float)

        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
        )

        ax.set_xticks(x, [str(nc) for nc in clients_list])
        ax.set_xlabel("Number of clients")
        ax.set_ylabel("Scatter latency (ns)")
        #ax.set_yscale("log")
        ax.set_title(f"data_size={data_size}")
        ax.grid(True, axis="y", which="both", alpha=0.3)

        for xi, m, n_runs in zip(x, means, n_runs_list):
            ax.text(xi, m, f"n={n_runs}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Avg scatter latency per client vs number of clients\n"
        "Each subplot: one data_size • Each bar: mean over runs • Each run: mean over clients (first scatter only)",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_path = os.path.join("scatter_multipanel_clients_vs_latency.png")
    fig.savefig(out_path)
    plt.close(fig)

    exit(0)
