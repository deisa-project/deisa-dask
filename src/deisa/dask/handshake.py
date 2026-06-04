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

from distributed import Future, get_client, Event, Client

from deisa.dask.constants import KEY_PREFIX
from deisa.dask.utils import _get_actor

logger = logging.getLogger(__name__)


class Handshake:
    _DEISA_HANDSHAKE_ACTOR_FUTURE_VARIABLE = 'deisa_handshake_actor_future'
    _DEISA_WAIT_FOR_DONE_EVENT = 'deisa_handshake_done'
    _DEISA_WAIT_FOR_BRIDGE_DONE_EVENT = 'deisa_handshake_bridge_done'
    _DEISA_WAIT_FOR_GO_EVENT = 'deisa_handshake_go'

    class HandshakeActor:
        def __init__(self):
            logger.debug('HandshakeActor.__init__()')
            self.analytics_ready = False
            self.client = get_client()
            self.nb_bridges = 0
            self.arrays_metadata = {}
            self.bridges_ready = False
            self.analytics_ready = False
            self.feedback_queue_size = 1024
            self.timestep: Optional[int] = None

        def set_bridges_ready(self, nb_bridges: int, arrays_metadata: dict) -> None:
            logger.debug(f"set_bridges_ready(): nb_bridges={nb_bridges}, arrays_metadata={arrays_metadata}, "
                         f"analytics_ready={self.analytics_ready}")
            self.nb_bridges = nb_bridges
            self.arrays_metadata = arrays_metadata
            self.bridges_ready = True
            if self.analytics_ready:
                self.__go()

        def set_bridges_done(self, timestep: int) -> None:
            logger.debug(f"set_bridges_done(): timestep={timestep}")
            self.timestep = timestep

        def set_analytics_ready(self, feedback_queue_size: int) -> None:
            logger.debug(f"set_analytics_ready(): bridges_ready={self.bridges_ready}")
            self.feedback_queue_size = feedback_queue_size
            self.analytics_ready = True
            if self.bridges_ready:
                self.__go()

        def get_arrays_metadata(self) -> dict | Future:
            return self.arrays_metadata

        def get_nb_bridges(self) -> int | Future:
            return self.nb_bridges

        def get_feedback_queue_size(self) -> int | Future:
            return self.feedback_queue_size

        def __go(self) -> None:
            logger.debug("Handshake go !")
            Event(Handshake._DEISA_WAIT_FOR_GO_EVENT, client=self.client).set()

    def __init__(self, client: Optional[Client] = None):
        self.client = client
        self.__handshake_actor: Optional[Handshake.HandshakeActor] = None
        if self.client:
            self.__handshake_actor = _get_actor(self.client, Handshake.HandshakeActor,
                                                key=f"{KEY_PREFIX}-HandshakeActor")
            assert self.__handshake_actor is not None

    def all_bridges_ready(self, nb_bridge: int, arrays_metadata: dict, wait_for_go=True) -> None:
        """
        Bridge must wait for analytics to be ready.
        """
        logger.debug(f"All bridges ready. nb_bridge={nb_bridge}, arrays_metadata={arrays_metadata}")
        assert self.__handshake_actor is not None
        self.__handshake_actor.set_bridges_ready(nb_bridge, arrays_metadata).result()

        # wait for go
        if wait_for_go:
            self.__wait_for_go()

    def deisa_ready(self, feedback_queue_size: int = 1024, wait_for_go=True, *args, **kwargs) -> None:
        """
        When analytics is ready, notify all Bridges
        """
        logger.debug("deisa ready.")
        assert self.__handshake_actor is not None
        self.__handshake_actor.set_analytics_ready(feedback_queue_size).result()

        # wait for go
        if wait_for_go:
            self.__wait_for_go()

    def get_arrays_metadata(self) -> dict:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_arrays_metadata().result()

    def get_feedback_queue_size(self) -> int:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_feedback_queue_size().result()

    def get_nb_bridges(self) -> int:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_nb_bridges().result()

    def __wait_for_go(self) -> None:
        Event(Handshake._DEISA_WAIT_FOR_GO_EVENT, client=self.client).wait()

    def wait_for_bridges(self):
        Event(Handshake._DEISA_WAIT_FOR_GO_EVENT, client=self.client).wait()

    def set_bridges_done(self, timestep: int):
        logger.debug(f"set_bridges_done(): timestep={timestep}")
        assert self.__handshake_actor is not None
        self.__handshake_actor.set_bridges_done(timestep).result()
        Event(Handshake._DEISA_WAIT_FOR_BRIDGE_DONE_EVENT, client=self.client).set()

    def wait_for_bridges_to_finish(self):
        Event(Handshake._DEISA_WAIT_FOR_BRIDGE_DONE_EVENT, client=self.client).wait()
