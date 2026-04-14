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
import time

from distributed import Client, Future, get_client, Event, Variable

from deisa.dask.utils import _get_actor

logger = logging.getLogger(__name__)


class Handshake:
    DEISA_HANDSHAKE_ACTOR_FUTURE_VARIABLE = 'deisa_handshake_actor_future'
    DEISA_WAIT_FOR_GO_EVENT = 'deisa_handshake_wait_for_go'

    class HandshakeActor:
        bridges = []
        max_bridges = 0
        arrays_metadata = {}
        analytics_ready = False

        def __init__(self):
            logger.debug('HandshakeActor.__init__()')
            self.bridges = []
            self.max_bridges = 0
            self.arrays_metadata = {}
            self.analytics_ready = False
            self.client = get_client()

        def add_bridge(self, id: int, max: int) -> None:
            print(f"add_bridge: {id} {max}", flush=True)
            if max == 0:
                raise ValueError('max cannot be 0.')
            if self.max_bridges == 0:
                self.max_bridges = max
            if self.max_bridges != max:
                raise ValueError(f'Value {max} for bridge {id} is unexpected. Expecting max={self.max_bridges}.')
            if len(self.bridges) >= max:
                raise RuntimeError(f'add_bridge cannot be called more than {max} times.')

            self.bridges.append(id)
            if self.__is_everyone_ready():
                self.__go()

        def set_analytics_ready(self) -> None:
            self.analytics_ready = True
            if self.__are_bridges_ready():
                self.__go()

        def set_arrays_metadata(self, arrays_metadata: dict) -> None | Future:
            self.arrays_metadata = arrays_metadata

        def get_arrays_metadata(self) -> dict | Future:
            return self.arrays_metadata

        def get_max_bridges(self) -> int | Future:
            return self.max_bridges

        def __are_bridges_ready(self) -> bool | Future:
            return self.max_bridges != 0 and len(self.bridges) == self.max_bridges

        def __is_everyone_ready(self) -> bool | Future:
            print(f"__is_everyone_ready: {self.analytics_ready} {len(self.bridges)}/{self.max_bridges}", flush=True)
            # print(f"__is_everyone_ready: {self.bridges}")
            return self.__are_bridges_ready() and self.analytics_ready

        def is_everyone_ready(self):
            return self.__is_everyone_ready()

        def __go(self):
            Event(Handshake.DEISA_WAIT_FOR_GO_EVENT, client=self.client).set()

    def __init__(self, who: str, client: Client, **kwargs):
        logger.debug(f"Handshake.__init__({who}, {client}, {kwargs})")
        self.client = client
        # self.client.direct_to_workers() # TODO
        self.handshake_actor = _get_actor(self.client, Handshake.HandshakeActor)
        assert self.handshake_actor is not None

        if who == 'bridge':
            self.start_bridge(**kwargs)
        elif who == 'deisa':
            self.start_deisa(**kwargs)
        else:
            raise ValueError("Expecting 'bridge' or 'deisa'.")

    def start_bridge(self, id: int, max: int, arrays_metadata: dict, wait_for_go=True, *args, **kwargs) -> None:
        """
        Bridge must wait for analytics to be ready.
        """
        assert self.handshake_actor is not None
        self.handshake_actor.add_bridge(id, max).result()

        # TODO: change this so that the check is not done on id=0
        if id == 0:
            self.handshake_actor.set_arrays_metadata(arrays_metadata).result()

        # wait for go
        if wait_for_go:
            self.__wait_for_go()

    def start_deisa(self, wait_for_go=True, *args, **kwargs) -> None:
        """
        When analytics is ready, notify all Bridges
        """
        assert self.handshake_actor is not None
        self.handshake_actor.set_analytics_ready().result()

        # wait for go
        if wait_for_go:
            self.__wait_for_go()

    def get_arrays_metadata(self) -> dict:
        assert self.handshake_actor is not None
        return self.handshake_actor.get_arrays_metadata().result()

    def get_nb_bridges(self) -> int:
        assert self.handshake_actor is not None
        return self.handshake_actor.get_max_bridges().result()

    def __wait_for_go(self) -> None:
        # Event(Handshake.DEISA_WAIT_FOR_GO_EVENT, client=self.client).wait()
        while True:
            if self.handshake_actor.is_everyone_ready().result():
                return
            time.sleep(1)
