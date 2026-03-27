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


import os

from distributed import Client, Lock, Variable


def get_connection_info(dask_scheduler_address: str | Client) -> Client:
    if isinstance(dask_scheduler_address, Client):
        client = dask_scheduler_address
    elif isinstance(dask_scheduler_address, str):
        try:
            client = Client(address=dask_scheduler_address)
        except ValueError:
            # try scheduler_file
            if os.path.isfile(dask_scheduler_address):
                client = Client(scheduler_file=dask_scheduler_address)
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