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
from typing import Optional

from distributed import Client, Future, get_client, Event

from deisa.dask.utils import _get_actor

logger = logging.getLogger(__name__)


class Handshake:
    _DEISA_HANDSHAKE_ACTOR_FUTURE_VARIABLE = 'deisa_handshake_actor_future'
    _DEISA_WAIT_FOR_DONE_EVENT = 'deisa_handshake_done'
    _DEISA_WAIT_FOR_BRIDGE_DONE_EVENT = 'deisa_handshake_bridge_done'
    _DEISA_WAIT_FOR_GO_EVENT = 'deisa_handshake_go'

    class HandshakeActor:
        bridges_ready = []
        bridges_done = []
        max_bridges = 0
        arrays_metadata = {}

        def __init__(self):
            logger.debug('HandshakeActor.__init__()')
            self.analytics_ready = False
            self.bridges_ready = []
            self.bridges_done = []
            self.max_bridges = 0
            self.arrays_metadata = {}
            self.feedback_queue_size: int = 1024
            self.client = get_client()

        def add_analytics_ready(self, feedback_queue_size: int) -> None | Future:
            self.analytics_ready = True
            self.feedback_queue_size = feedback_queue_size

            if self.are_bridges_ready():
                self.go()

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

            if self.is_everyone_ready():
                self.go()

        def add_bridge_done(self, id: int) -> None:
            self.bridges_done.append(id)
            if len(self.bridges_ready) == self.max_bridges:
                Event(Handshake._DEISA_WAIT_FOR_BRIDGE_DONE_EVENT, client=self.client).set()

        def set_arrays_metadata(self, arrays_metadata: dict) -> None | Future:
            self.arrays_metadata = arrays_metadata

        def get_arrays_metadata(self) -> dict | Future:
            return self.arrays_metadata

        def get_feedback_queue_size(self) -> int | Future:
            return self.feedback_queue_size

        def get_max_bridges(self) -> int | Future:
            return self.max_bridges

        def are_bridges_ready(self) -> bool | Future:
            return self.max_bridges != 0 and len(self.bridges_ready) == self.max_bridges

        def is_everyone_ready(self) -> bool | Future:
            return self.analytics_ready and self.are_bridges_ready()

        def go(self) -> None:
            Event(Handshake._DEISA_WAIT_FOR_GO_EVENT, client=self.client).set()

    def __init__(self, who: str, client: Optional[Client], feedback_queue_size=1024, **kwargs):
        self.client = client
        # self.client.direct_to_workers() # TODO

        if client:
            self.__handshake_actor = _get_actor(self.client, Handshake.HandshakeActor)
            assert self.__handshake_actor is not None

        if who == 'bridge':
            self.start_bridge(**kwargs)
        elif who == 'deisa':
            self.start_deisa(feedback_queue_size, **kwargs)
        else:
            raise ValueError("Expecting 'bridge' or 'deisa'.")

    def start_bridge(self, id: int, max: int, arrays_metadata: dict, *args, **kwargs) -> None:
        """
        Bridge must wait for analytics to be ready.
        """
        assert self.__handshake_actor is not None

        if self.client:
            assert id == 0, "only bridge 0 should have a valid client"

        self.__handshake_actor.add_bridge_ready(id, max).result()

        # TODO: change this so that the check is not necessarily done on id=0
        if id == 0:
            self.__handshake_actor.set_arrays_metadata(arrays_metadata).result()
        # TODO: check that arrays_metadata is the same for all bridges

        if kwargs.get('wait_for_go', True):
            self.wait_for_go()

    def start_deisa(self, feedback_queue_size: int, *args, **kwargs) -> None:
        self.__handshake_actor.add_analytics_ready(feedback_queue_size).result()

        if kwargs.get('wait_for_go', True):
            self.wait_for_go()

    def get_arrays_metadata(self) -> dict:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_arrays_metadata().result()

    def get_feedback_queue_size(self) -> int:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_feedback_queue_size().result()

    def get_nb_bridges(self) -> int:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_max_bridges().result()

    def stop_bridge(self, id: int) -> None:
        self.__handshake_actor.add_bridge_done(id).result()

    def wait_for_go(self):
        Event(Handshake._DEISA_WAIT_FOR_GO_EVENT, client=self.client).wait()

    def wait_for_bridges_to_finish(self):
        Event(Handshake._DEISA_WAIT_FOR_BRIDGE_DONE_EVENT, client=self.client).wait()
