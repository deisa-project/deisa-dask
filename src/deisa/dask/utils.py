# =============================================================================
# Copyright (C) 2026 Commissariat a l'energie atomique et aux energies alternatives (CEA)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of CEA, nor the names of the contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written  permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# =============================================================================
import logging
import os

import dask.array as da
from deisa.core import ICommunicator, DeisaArray
from distributed import Client, Lock, Variable

logger = logging.getLogger(__name__)


def get_client(*args, **kwargs):
    addr = os.getenv("DEISA_DASK_SCHEDULER_ADDRESS", "tcp://127.0.0.1:8787")
    logger.info(f"get_client: DEISA_DASK_SCHEDULER_ADDRESS={addr}")
    return get_connection_info(addr, *args, **kwargs)


def get_mpi_comm_world(cart_coord_dims: int = 1) -> ICommunicator:
    """
    Computes and returns an MPI Cartesian communicator based on the number of desired
    Cartesian coordinate dimensions.

    This function uses the MPI library to calculate and create a Cartesian communicator
    from the global MPI communicator (MPI.COMM_WORLD). The dimensions of the Cartesian coordinate
    grid are determined dynamically based on the size of the Cartesian coordinate dimensions
    requested by the user and the size of the MPI communicator.


    ``:param cart_coord_dims:`` Number of Cartesian coordinate dimensions used to compute the
        grid layout for the Cartesian communicator. Default is 1.  
    ``:type cart_coord_dims:`` int    
    ``:return:`` A new Cartesian communicator created from the MPI world communicator based
        on the computed dimensions.  
    ``:rtype:`` mpi4py.MPI.Cartcomm  
    """
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    dims = MPI.Compute_dims(mpi_comm.Get_size(), dims=cart_coord_dims)
    return mpi_comm.Create_cart(dims)


def get_connection_info(dask_scheduler_address: str | Client, *args, **kwargs) -> Client:
    logger.info(f"get_connection_info: {dask_scheduler_address}")
    if isinstance(dask_scheduler_address, Client):
        client = dask_scheduler_address
    elif isinstance(dask_scheduler_address, str):
        try:
            client = Client(address=dask_scheduler_address, *args, **kwargs)
        except ValueError:
            # try scheduler_file
            if os.path.isfile(dask_scheduler_address):
                client = Client(scheduler_file=dask_scheduler_address, *args, **kwargs)
            else:
                raise ValueError(
                    "dask_scheduler_address must be a string containing the address of the scheduler, "
                    "or a string containing a file name to a dask scheduler file, or a Dask Client object.")
    else:
        raise ValueError(
            "dask_scheduler_address must be a string containing the address of the scheduler, "
            "or a string containing a file name to a dask scheduler file, or a Dask Client object.")

    return client


def _get_actor(client: Client, clazz, **kwargs):
    def check_variable(dask_scheduler, name):
        ext = dask_scheduler.extensions["variables"]
        v = ext.variables.get(name)
        return v is not None

    key = f"deisa_actor_{clazz}"

    with Lock(key):
        is_set = client.run_on_scheduler(check_variable, name=key)
        if is_set:
            return Variable(key, client=client).get().result()
        else:
            actor_future = client.submit(clazz, actor=True, **kwargs)
            Variable(key, client=client).set(actor_future)
            return actor_future.result()


def build_deisa_array(darr: da.Array, timestep: int) -> DeisaArray:
    return DeisaArray(t=timestep,
                      dask=darr.dask,
                      name=darr.name,
                      chunks=darr.chunks,
                      dtype=darr.dtype,
                      meta=darr._meta,
                      shape=darr.shape)
