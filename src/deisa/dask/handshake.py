# =============================================================================
# Copyright (C) 2025 Commissariat a l'energie atomique et aux energies alternatives (CEA)
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

import asyncio

from dask.distributed import Variable
from distributed import Client, Future, get_client


class Handshake:
    DEISA_HANDSHAKE_ACTOR_FUTURE_VARIABLE = 'deisa_handshake_actor_future'
    DEISA_WAIT_FOR_GO_VARIABLE = 'deisa_handshake_wait_for_go'

    class HandshakeActor:
        bridges = []
        max_bridges = 0
        arrays_metadata = {}
        analytics_ready = False

        def __init__(self):
            self.bridges = []
            self.max_bridges = 0
            self.arrays_metadata = {}
            self.analytics_ready = False
            self.client = get_client()

        def add_bridge(self, id: int, max: int) -> None:
            if max == 0:
                raise ValueError('max cannot be 0.')
            elif self.max_bridges == 0:
                self.max_bridges = max
            elif self.max_bridges != max:
                raise ValueError(f'Value {max} for bridge {id} is unexpected. Expecting max={self.max_bridges}.')
            elif len(self.bridges) >= max:
                raise RuntimeError(f'add_bridge cannot be called more than {max} times.')

            self.bridges.append(id)

        def set_analytics_ready(self) -> None:
            self.analytics_ready = True
            if self.__are_bridges_ready():
                self.__go()

        def set_arrays_metadata(self, arrays_metadata: dict) -> None:
            self.arrays_metadata = arrays_metadata

        def get_arrays_metadata(self) -> dict | Future:
            return self.arrays_metadata

        def get_max_bridges(self) -> int | Future:
            return self.max_bridges

        def __are_bridges_ready(self) -> bool | Future:
            return self.max_bridges != 0 and len(self.bridges) == self.max_bridges

        def __go(self):
            Variable(Handshake.DEISA_WAIT_FOR_GO_VARIABLE, client=self.client).set(None)

    def __init__(self, who: str, client: Client, **kwargs):
        self.client = client
        # self.client.direct_to_workers() # TODO
        self.handshake_actor = self.__get_handshake_actor()
        assert self.handshake_actor is not None

        if who == 'bridge':
            self.start_bridge(**kwargs)
        elif who == 'deisa':
            self.start_deisa(**kwargs)
        else:
            raise ValueError("Expecting 'bridge' or 'deisa'.")

    def start_bridge(self, id: int, max: int, arrays_metadata: dict, wait_for_go=True) -> None:
        """
        Bridge must wait for analytics to be ready.
        """
        assert self.handshake_actor is not None
        self.handshake_actor.add_bridge(id, max)

        if id == 0:
            self.handshake_actor.set_arrays_metadata(arrays_metadata)

        # wait for go
        if wait_for_go:
            self.__wait_for_go()

    def start_deisa(self, wait_for_go=True) -> None:
        """
        When analytics is ready, notify all Bridges
        """
        assert self.handshake_actor is not None
        self.handshake_actor.set_analytics_ready()

        # wait for go
        if wait_for_go:
            self.__wait_for_go()

    def get_arrays_metadata(self) -> dict:
        assert self.handshake_actor is not None
        return self.handshake_actor.get_arrays_metadata().result()

    def get_nb_bridges(self) -> int:
        assert self.handshake_actor is not None
        return self.handshake_actor.get_max_bridges().result()

    def __get_handshake_actor(self) -> HandshakeActor:
        try:
            return Variable(Handshake.DEISA_HANDSHAKE_ACTOR_FUTURE_VARIABLE, client=self.client).get(timeout=0).result()
        except asyncio.exceptions.TimeoutError:
            actor_future = self.client.submit(Handshake.HandshakeActor, actor=True)
            Variable(Handshake.DEISA_HANDSHAKE_ACTOR_FUTURE_VARIABLE, client=self.client).set(actor_future)
            return actor_future.result()

    def __wait_for_go(self) -> None:
        Variable(Handshake.DEISA_WAIT_FOR_GO_VARIABLE, client=self.client).get()
