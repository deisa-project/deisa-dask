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

from distributed import Client, Future, get_client, Event

from deisa.dask.utils import _get_actor

logger = logging.getLogger(__name__)


class Handshake:
    _DEISA_HANDSHAKE_ACTOR_FUTURE_VARIABLE = 'deisa_handshake_actor_future'
    _DEISA_WAIT_FOR_DONE_EVENT = 'deisa_handshake_done'
    _DEISA_WAIT_FOR_BRIDGE_READY_EVENT = 'deisa_handshake_bridge_ready'
    _DEISA_WAIT_FOR_BRIDGE_DONE_EVENT = 'deisa_handshake_bridge_done'
    _DEISA_WAIT_FOR_ANALYTICS_READY_EVENT = 'deisa_handshake_analytics_ready'

    class HandshakeActor:
        bridges_ready = []
        bridges_done = []
        max_bridges = 0
        arrays_metadata = {}

        def __init__(self):
            logger.debug('HandshakeActor.__init__()')
            self.bridges_ready = []
            self.bridges_done = []
            self.max_bridges = 0
            self.arrays_metadata = {}
            self.client = get_client()

        def add_bridge_ready(self, id: int, max: int) -> None:
            if max == 0:
                raise ValueError('max cannot be 0.')
            elif self.max_bridges == 0:
                self.max_bridges = max
            elif self.max_bridges != max:
                raise ValueError(f'Value {max} for bridge {id} is unexpected. Expecting max={self.max_bridges}.')
            elif len(self.bridges_ready) >= max:
                raise RuntimeError(f'add_bridge cannot be called more than {max} times.')

            self.bridges_ready.append(id)

        def add_bridge_done(self, id: int) -> None:
            self.bridges_done.append(id)
            if len(self.bridges_ready) == self.max_bridges:
                Event(Handshake._DEISA_WAIT_FOR_BRIDGE_DONE_EVENT, client=self.client).set()

        def set_arrays_metadata(self, arrays_metadata: dict) -> None | Future:
            self.arrays_metadata = arrays_metadata

        def get_arrays_metadata(self) -> dict | Future:
            return self.arrays_metadata

        def get_max_bridges(self) -> int | Future:
            return self.max_bridges

        def are_bridges_ready(self) -> bool | Future:
            return self.max_bridges != 0 and len(self.bridges_ready) == self.max_bridges

    def __init__(self, who: str, client: Client, **kwargs):
        self.client = client
        # self.client.direct_to_workers() # TODO
        self.__handshake_actor = _get_actor(self.client, Handshake.HandshakeActor)
        assert self.__handshake_actor is not None

        if who == 'bridge':
            self.start_bridge(**kwargs)
        elif who == 'deisa':
            pass
        else:
            raise ValueError("Expecting 'bridge' or 'deisa'.")

    def start_bridge(self, id: int, max: int, arrays_metadata: dict, *args, **kwargs) -> None:
        """
        Bridge must wait for analytics to be ready.
        """
        assert self.__handshake_actor is not None
        self.__handshake_actor.add_bridge_ready(id, max).result()

        # TODO: change this so that the check is not necessarily done on id=0
        if id == 0:
            self.__handshake_actor.set_arrays_metadata(arrays_metadata).result()
        # TODO: check that arrays_metadata is the same for all bridges

        if self.__handshake_actor.are_bridges_ready().result():
            Event(Handshake._DEISA_WAIT_FOR_BRIDGE_READY_EVENT, client=self.client).set()

        if kwargs.get('analytics_ready', False):
            # bridges wait for analytics to be ready
            Event(Handshake._DEISA_WAIT_FOR_ANALYTICS_READY_EVENT, client=self.client).wait()

    def get_arrays_metadata(self) -> dict:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_arrays_metadata().result()

    def get_nb_bridges(self) -> int:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_max_bridges().result()

    def deisa_ready(self):
        Event(Handshake._DEISA_WAIT_FOR_ANALYTICS_READY_EVENT, client=self.client).set()

    def stop_bridge(self, id: int) -> None:
        self.__handshake_actor.add_bridge_done(id).result()

    def wait_for_bridges_to_start(self):
        Event(Handshake._DEISA_WAIT_FOR_BRIDGE_READY_EVENT, client=self.client).wait()

    def wait_for_bridges_to_finish(self):
        Event(Handshake._DEISA_WAIT_FOR_BRIDGE_DONE_EVENT, client=self.client).wait()
