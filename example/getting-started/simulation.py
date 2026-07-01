import numpy as np
from mpi4py import MPI

from deisa.dask import Bridge

global_grid_size = (64, 64)
mpi_parallelism = (2, 2)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert size == 4, "This example requires 4 ranks"

print(f"[{rank}] Hello from simulation. comm size={size}")

dims = MPI.Compute_dims(size, len(global_grid_size))
cart_comm = comm.Create_cart(dims)
rank_coords = cart_comm.Get_coords(rank)

nx_local = global_grid_size[0] // mpi_parallelism[0]
ny_local = global_grid_size[1] // mpi_parallelism[1]

# indices for this rank
x_start = rank_coords[0] * nx_local
x_end = (rank_coords[0] + 1) * nx_local
y_start = rank_coords[1] * ny_local
y_end = (rank_coords[1] + 1) * ny_local

x_full = np.linspace(-1, 1, global_grid_size[0])
y_full = np.linspace(-1, 1, global_grid_size[1])

x_local = x_full[x_start:x_end]
y_local = y_full[y_start:y_end]

X, Y = np.meshgrid(x_local, y_local, indexing="ij")

# Start the Bridge
bridge = Bridge(
    comm=comm,
    arrays_metadata={
        "temperature": {
            "global_shape": global_grid_size,
            "chunk_shape": tuple(
                g // p for g, p in zip(global_grid_size, mpi_parallelism)
            ),
            "chunk_position": rank_coords,
        }
    },
)

for ts in range(5):
    print(f"[{rank}] Sending timestep {ts}", flush=True)
    sigma = 0.15 + ts * 0.05
    local_data = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    bridge.send("temperature", local_data, timestep=ts)

bridge.close(timestep=5)
