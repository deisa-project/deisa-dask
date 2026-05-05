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
import asyncio
import logging
from typing import Protocol

from distributed import Future, get_client, Event, Client

from deisa.dask.utils import _get_actor

logger = logging.getLogger(__name__)


class IHandshake(Protocol):
    def add_bridge_ready(self, *, id: int, max: int) -> None: ...

    def add_bridge_done(self, id: int) -> None: ...

    def set_analytics_ready(self) -> None: ...

    def get_arrays_metadata(self) -> dict: ...

    def get_max_bridges(self) -> int: ...

    def wait_for_go(self): ...

    def wait_for_done(self): ...


class HandshakeHandler(IHandshake):
    bridges_ready = []
    bridges_done = []
    max_bridges = 0
    arrays_metadata = {}
    analytics_ready = False

    def __init__(self):
        logger.debug('HandshakeActor.__init__()')
        self.bridges_ready = []
        self.bridges_done = []
        self.max_bridges = 0
        self.arrays_metadata = {}
        self.analytics_ready = False
        self.go_event = asyncio.Event()
        self.done_event = asyncio.Event()

        self.handlers = {
            'add_bridge_ready': self.add_bridge_ready,
            'add_bridge_done': self.add_bridge_done,
            'set_analytics_ready': self.set_analytics_ready,
            'set_arrays_metadata': self.set_arrays_metadata,
            'get_arrays_metadata': self.get_arrays_metadata,
            'get_max_bridges': self.get_max_bridges,
            'wait_for_go': self.wait_for_go(),
            'wait_for_done': self.wait_for_done()
        }

    def add_bridge_ready(self, *, id: int, max: int) -> None:
        if max == 0:
            raise ValueError('max cannot be 0.')
        elif self.max_bridges == 0:
            self.max_bridges = max
        elif self.max_bridges != max:
            raise ValueError(f'Value {max} for bridge {id} is unexpected. Expecting max={self.max_bridges}.')
        elif len(self.bridges_ready) >= max:
            raise RuntimeError(f'add_bridge cannot be called more than {max} times.')

        self.bridges_ready.append(id)
        if self.__is_everyone_ready():
            self.__go()

    def add_bridge_done(self, id: int) -> None:
        self.bridges_done.append(id)
        if len(self.bridges_ready) == self.max_bridges:
            self.done_event.set()

    def set_analytics_ready(self) -> None:
        self.analytics_ready = True
        if self.__are_bridges_ready():
            self.__go()

    def set_arrays_metadata(self, *, arrays_metadata: dict) -> None:
        self.arrays_metadata = arrays_metadata

    def get_arrays_metadata(self) -> dict:
        return self.arrays_metadata

    def get_max_bridges(self) -> int:
        return self.max_bridges

    def wait_for_go(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.go_event.wait())

    def wait_for_done(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.done_event.wait())

    def __are_bridges_ready(self) -> bool | Future:
        return self.max_bridges != 0 and len(self.bridges_ready) == self.max_bridges

    def __is_everyone_ready(self) -> bool | Future:
        return self.__are_bridges_ready() and self.analytics_ready

    def __go(self):
        self.go_event.set()


class Handshake:
    DEISA_HANDSHAKE_ACTOR_FUTURE_VARIABLE = 'deisa_handshake_actor_future'
    DEISA_WAIT_FOR_GO_EVENT = 'deisa_handshake_wait_for_go'
    DEISA_WAIT_FOR_DONE_EVENT = 'deisa_handshake_done'

    class HandshakeActor:
        def __init__(self):
            logger.debug('HandshakeActor.__init__()')
            self.client = get_client()
            self.nb_bridges = 0
            self.arrays_metadata = {}
            self.bridges_ready = False
            self.analytics_ready = False

        def set_bridges_ready(self, nb_bridges: int, arrays_metadata: dict) -> None:
            logger.debug(f"set_bridges_ready(): nb_bridges={nb_bridges}, arrays_metadata={arrays_metadata}, "
                         f"analytics_ready={self.analytics_ready}")
            self.nb_bridges = nb_bridges
            self.arrays_metadata = arrays_metadata
            self.bridges_ready = True
            if self.analytics_ready:
                self.__go()

        def set_analytics_ready(self) -> None:
            logger.debug(f"set_analytics_ready(): bridges_ready={self.bridges_ready}")
            self.analytics_ready = True
            if self.bridges_ready:
                self.__go()

        def get_arrays_metadata(self) -> dict | Future:
            return self.arrays_metadata

        def get_nb_bridges(self) -> int | Future:
            return self.nb_bridges

        def __go(self):
            logger.debug("Handshake go !")
            Event(Handshake.DEISA_WAIT_FOR_GO_EVENT, client=self.client).set()

    def __init__(self, client: Client):
        self.client = client
        # self.client.direct_to_workers() # TODO
        self.handshake_actor = _get_actor(self.client, Handshake.HandshakeActor)
        assert self.handshake_actor is not None

    def all_bridges_ready(self, nb_bridge: int, arrays_metadata: dict, wait_for_go=True) -> None:
        """
        Bridge must wait for analytics to be ready.
        """
        logger.debug(f"All bridges ready. nb_bridge={nb_bridge}, arrays_metadata={arrays_metadata}")
        assert self.handshake_actor is not None
        self.handshake_actor.set_bridges_ready(nb_bridge, arrays_metadata).result()

        # wait for go
        if wait_for_go:
            self.__wait_for_go()

    def deisa_ready(self, wait_for_go=True) -> None:
        """
        When analytics is ready, notify all Bridges
        """
        logger.debug("deisa ready.")
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
        return self.handshake_actor.get_nb_bridges().result()

    def __wait_for_go(self) -> None:
        Event(Handshake.DEISA_WAIT_FOR_GO_EVENT, client=self.client).wait()

    def wait_for_bridges(self):
        Event(Handshake.DEISA_WAIT_FOR_DONE_EVENT, client=self.client).wait()

    def set_bridges_done(self):
        logger.debug("set_bridges_done()")
        Event(Handshake.DEISA_WAIT_FOR_DONE_EVENT, client=self.client).set()
