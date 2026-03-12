import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Files:
    # res/scatter_times_<nb_clients>_<data_size>_<run_id>_<unique_id>_<scatter_type>.csv
    #
    # scatter_type is either "original" or "optim1".
    csv_files = sorted(glob.glob(os.path.join("res", "scatter_times_*_*_*_*_*.csv")))

    # (scatter_type, nb_clients, data_size, run_id) -> {unique_id: first_latency_ns}
    per_run_latencies = defaultdict(dict)

    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        stem = basename.removeprefix("scatter_times_").removesuffix(".csv")
        parts = stem.split("_")
        if len(parts) != 5:
            continue

        nb_clients_str, data_size_str, run_id_str, unique_id_str, scatter_type = parts
        if scatter_type not in {"original", "optim1"}:
            continue

        try:
            nb_clients = int(nb_clients_str)
            data_size = int(data_size_str)
            run_id = int(run_id_str)
            unique_id = int(unique_id_str)
        except ValueError:
            continue

        times = np.loadtxt(csv_file, delimiter=",")
        times = np.atleast_1d(times)

        # "1 value from a file" => first scatter latency, in ns
        per_run_latencies[(scatter_type, nb_clients, data_size, run_id)][unique_id] = float(times[0])

    # Step 1: ONE scalar per run (per-client average for that run)
    run_scalars = {}  # (scatter_type, data_size, nb_clients, run_id) -> float

    for (scatter_type, nb_clients, data_size, run_id), lat_by_uid in per_run_latencies.items():
        if not lat_by_uid:
            continue

        run_scalar = float(np.mean(list(lat_by_uid.values())))
        run_scalars[(scatter_type, data_size, nb_clients, run_id)] = run_scalar

    # Step 2: aggregate run scalars by line:
    # (scatter_type, data_size) -> nb_clients -> [run_scalar]
    by_line = defaultdict(lambda: defaultdict(list))
    for (scatter_type, data_size, nb_clients, run_id), run_scalar in run_scalars.items():
        by_line[(scatter_type, data_size)][nb_clients].append(run_scalar)

    os.makedirs("res", exist_ok=True)

    if not by_line:
        raise SystemExit("No matching CSV files found in res/")

    plt.figure(figsize=(10, 6))

    for (scatter_type, data_size) in sorted(by_line.keys()):
        clients_list = sorted(by_line[(scatter_type, data_size)].keys())
        if not clients_list:
            continue

        means = []
        stds = []

        for nb_clients in clients_list:
            vals = np.asarray(by_line[(scatter_type, data_size)][nb_clients], dtype=float)
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)

        label = f"{scatter_type}, data_size={data_size}"
        plt.plot(
            clients_list,
            means,
            marker=".",
            linewidth=1.8,
            label=label,
        )
        plt.errorbar(
            clients_list,
            means,
            yerr=stds,
            fmt="none",
            capsize=2,
            alpha=0.4,
        )

    plt.xlabel("Number of clients")
    plt.ylabel("Scatter latency (ns)")
    plt.yscale("log")
    plt.title(
        "Scatter latency vs number of clients\n"
        "One line per scatter_type and data_size • Mean over runs • Each run: mean over clients (first scatter only)"
    )
    plt.grid(True, axis="both", which="both", alpha=0.3)
    plt.legend()

    out_path = os.path.join("res", "scatter_lines_clients_vs_latency.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    exit(0)
