import os
import sys
import time
from typing import List

import numpy as np
from dask.distributed import Client

from deisa.dask import Bridge, get_connection_info


def connect_clients(scheduler_file, nb_clients) -> List[Client]:
    clients = []
    print(f"Connecting {nb_clients} clients to scheduler.", flush=True)
    start = time.time_ns()
    for i in range(nb_clients):
        clients.append(Client(scheduler_file=scheduler_file))
    stop = time.time_ns()
    print(f"Connected clients in {stop - start} ns.", flush=True)
    return clients


def connect_bridges(scheduler_file: str, id: int, max: int, size: tuple) -> List[Bridge]:
    print(f"Connecting bridge {id} to scheduler.", flush=True)
    b = []
    start = time.time_ns()
    b.append(Bridge(id=id,
                    arrays_metadata={
                        'my_array': {
                            'size': size,
                            'subsize': (0, 0, 0)  # not used but currently needed
                        }
                    },
                    system_metadata={'connection': get_connection_info(scheduler_file),
                                     'nb_bridges': max},
                    wait_for_go=False))
    stop = time.time_ns()
    print(f"Connected bridge {id} in {stop - start} ns.", flush=True)
    return b


def disconnect_clients(clients: List[Client]):
    for c in clients: c.close()


def disconnect_bridges(bridges: List[Bridge]):
    for b in bridges:
        b.close()


def do_scatter(clients, workers) -> List[float]:
    times = []
    data = np.zeros((data_size, data_size, data_size), dtype=np.float64)

    for i, c in enumerate(clients):
        start = time.time_ns()
        c.scatter(data, direct=True, workers=workers)
        stop = time.time_ns()
        times.append(stop - start)

        if len(times) > 2: print(f"delta to previous={times[-2] - times[-1]}", flush=True)
        if i % 10 == 0:
            print(f">>> [{unique_id}] times={times}", flush=True)

    return times


def do_scatter_bridge(bridges: List[Bridge]) -> List[float]:
    times = []
    data = np.zeros((data_size, data_size, data_size), dtype=np.float64)

    for i, b in enumerate(bridges):
        start = time.time_ns()
        # b.send('my_array', data=data, iteration=0)
        f = b._better_scatter(data)
        f.release()  # release memory on worker

        stop = time.time_ns()
        times.append(stop - start)

    return times


if __name__ == '__main__':
    scheduler_file = sys.argv[1]
    nb_clients = int(sys.argv[2])
    data_size = int(sys.argv[3])
    run_id = int(sys.argv[4])
    unique_id = int(sys.argv[5])
    scatter_type = sys.argv[6]

    times = []

    if scatter_type == "original":
        clients = connect_clients(scheduler_file, 1)
        workers = list(clients[0].scheduler_info()["workers"].keys())

        time.sleep(2)
        times = do_scatter(clients, workers)
        disconnect_clients(clients)
    elif scatter_type == "optim1":
        bridges = connect_bridges(scheduler_file, unique_id, nb_clients, (data_size, data_size, data_size))

        time.sleep(2)
        times = do_scatter_bridge(bridges)
        disconnect_bridges(bridges)
    else:
        raise ValueError(f"Unknown scatter type: {scatter_type}")

    print(f">>> [{unique_id}] run={run_id}, scatter_type={scatter_type}, size={data_size}, times={times}", flush=True)

    # create res directory
    os.makedirs("res", exist_ok=True)

    # write times to csv file
    print(f"[{unique_id}] Writing times to CSV file.", flush=True)
    np.savetxt(f"res/scatter_times_{nb_clients}_{data_size}_{run_id}_{unique_id}_{scatter_type}.csv",
               times,
               delimiter=",")

    print(f"[{unique_id}] Done.", flush=True)
    sys.exit(0)
