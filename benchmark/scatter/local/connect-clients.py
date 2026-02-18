import os
import sys
import time
from typing import List

import numpy as np
from dask.distributed import Client


def connect_clients(scheduler_file, nb_clients) -> List[Client]:
    clients = []
    print(f"Connecting {nb_clients} clients to scheduler.")
    start = time.time_ns()
    for i in range(nb_clients):
        clients.append(Client(scheduler_file=scheduler_file))
    stop = time.time_ns()
    print(f"Connected clients in {stop - start} ns.")
    return clients


def disconnect_clients(clients: List[Client]):
    for c in clients: c.close()


def do_scatter(clients, workers) -> List[float]:
    times = []
    data = np.zeros((data_size, data_size, data_size), dtype=np.float64)

    for i, c in enumerate(clients):
        start = time.time_ns()
        c.scatter(data, direct=True, workers=workers)
        stop = time.time_ns()
        times.append(stop - start)

        if len(times) > 2: print(f"delta to previous={times[-2] - times[-1]}")
        if i % 10 == 0:
            print(f">>> [{unique_id}] times={times}")

    return times


if __name__ == '__main__':
    scheduler_file = sys.argv[1]
    nb_clients = int(sys.argv[2])
    data_size = int(sys.argv[3])
    run_id = int(sys.argv[4])
    unique_id = int(sys.argv[5])

    clients = connect_clients(scheduler_file, 1)
    workers = list(clients[0].scheduler_info()["workers"].keys())

    time.sleep(2)

    times = do_scatter(clients, workers)

    disconnect_clients(clients)

    print(f">>> [{unique_id}] run={run_id}, size={data_size}, times={times}")

    # create res directory
    os.makedirs("res", exist_ok=True)

    # write times to csv file
    np.savetxt(f"res/scatter_times_{nb_clients}_{data_size}_{run_id}_{unique_id}.csv", times, delimiter=",")

    exit(0)
