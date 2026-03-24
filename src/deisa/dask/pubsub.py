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
from collections import defaultdict, deque
from typing import Dict, Tuple, Any
from typing import List

from dask.distributed import get_client, Client
from distributed import Lock, Variable


class PubSubActor:
    DEISA_PUBSUB_ACTOR_FUTURE_VARIABLE = 'deisa_pubsub_actor_future'

    def __init__(self, nb_bridges: int, maxlen: int = 10):
        print(f"[PubSubActor] __init__(nb_bridges={nb_bridges}, maxlen={maxlen})", flush=True)

        self.client: Client = get_client()
        self.nb_bridges = nb_bridges
        self.maxlen = maxlen

        self.kvs = defaultdict(None)

        self.partial_metadata = defaultdict(lambda: defaultdict(dict))
        self.completed = defaultdict(lambda: deque(maxlen=maxlen))
        self.done: Dict[Tuple[str, int], bool] = {}

        self.offsets = {}

        # topic -> {arrays, when}
        self.callbacks: Dict[str, dict] = {}

        self.available_iterations = defaultdict(set)
        self.emitted = set()
        self._conditions = defaultdict(asyncio.Condition)

    async def publish(self, msg: dict):
        print(f"[PubSubActor] publish(msg={msg})", flush=True)
        array_name = msg["array_name"]
        iteration = msg["iteration"]
        rank = msg["rank"]

        partials = self.partial_metadata[array_name][iteration]

        if rank in partials:
            return False

        if self.done.get((array_name, iteration), False):
            return False

        partials[rank] = msg

        if len(partials) == self.nb_bridges:
            self.done[(array_name, iteration)] = True

            aggregated = self._aggregate(array_name, iteration, partials)

            self.completed[array_name].append((iteration, aggregated))
            self.available_iterations[array_name].add(iteration)

            # notify waiters
            cond = self._conditions[array_name]
            async with cond:
                cond.notify_all()

            del self.partial_metadata[array_name][iteration]

            await self._maybe_trigger_callbacks(array_name, iteration)

            return True

        return False

    def _aggregate(self, array_name, iteration, partials):
        parts = [partials[r] for r in sorted(partials)]
        return {
            "array_name": array_name,
            "iteration": iteration,
            "parts": parts,
        }

    @staticmethod
    def make_topic(arrays, when) -> str:
        return f"{when}|" + "|".join(sorted(arrays))

    async def register_callback(self, arrays: List[str], when) -> str:
        """
        when can be none is arrays contains only one array
        """
        topic = self.make_topic(arrays, when)
        self.callbacks[topic] = {
            "arrays": sorted(arrays),
            "when": when,
        }
        return topic

    async def unregister_callback(self, topic: str):
        self.callbacks.pop(topic, None)

    async def _maybe_trigger_callbacks(self, array_name, iteration):
        for topic, cb in self.callbacks.items():
            arrays = cb["arrays"]
            when = cb["when"]

            key = (topic, iteration)
            if key in self.emitted:
                continue

            if when == "OR":
                if array_name in arrays:
                    await self._emit(topic, arrays, when, iteration)

            elif when == "AND":
                if all(iteration in self.available_iterations[a] for a in arrays):
                    await self._emit(topic, arrays, when, iteration)

    async def _emit(self, topic, arrays, when, iteration):
        self.emitted.add((topic, iteration))

        payload = {
            "iteration": iteration,
            "when": when,
            "arrays": {},
        }

        for a in arrays:
            for it, data in self.completed[a]:
                if it == iteration:
                    payload["arrays"][a] = data
                    break

        # send to the scheduler. This will trigger the topic handler, which contains the user's callback
        print(f"[PubSubActor] _emit() client.log_event topic={topic} iteration={iteration}", flush=True)
        await self.client.log_event(topic, payload)

    async def subscribe(self, topic: str, subscriber_id: str):
        if self.completed[topic]:
            last_iter = self.completed[topic][-1][0]
        else:
            last_iter = -1

        self.offsets[(topic, subscriber_id)] = last_iter + 1

    async def get_array(self, array_name: str, iteration: int = None):
        """
        Blocking get:
          - if iteration is provided → wait until that iteration is available
          - otherwise → wait for next completed iteration
        """

        if array_name not in self.completed:
            raise ValueError(f"array_name {array_name} not found")

        cond = self._conditions[array_name]

        # retrieve specific iteration
        if iteration is not None:
            async with cond:
                while True:
                    for it, data in self.completed[array_name]:
                        if it == iteration:
                            return data

                    # detect eviction
                    if self.completed[array_name]:
                        oldest = self.completed[array_name][0][0]
                        if iteration < oldest:
                            raise ValueError(f"Iteration {iteration} has been evicted (oldest={oldest})")

                    await cond.wait()

        # No iteration specified, wait for the next iteration
        async with cond:
            if self.completed[array_name]:
                last_seen = self.completed[array_name][-1][0]
            else:
                last_seen = -1

            while True:
                if self.completed[array_name]:
                    latest_it, data = self.completed[array_name][-1]

                    if latest_it > last_seen:
                        return data

                await cond.wait()

    async def set(self, key, value):
        self.kvs[key] = value

    async def get(self, key) -> Any:
        return self.kvs.get(key, None)

    async def delete(self, key):
        self.kvs.pop(key, None)

def get_pubsub_actor(client: Client, *args, **kwargs) -> PubSubActor:
    def check_variable(dask_scheduler, name):
        ext = dask_scheduler.extensions["variables"]
        v = ext.variables.get(name)
        return v is not None

    with Lock(PubSubActor.DEISA_PUBSUB_ACTOR_FUTURE_VARIABLE):
        is_set = client.run_on_scheduler(check_variable, name=PubSubActor.DEISA_PUBSUB_ACTOR_FUTURE_VARIABLE)
        if is_set:
            return Variable(PubSubActor.DEISA_PUBSUB_ACTOR_FUTURE_VARIABLE, client=client).get().result()
        else:
            actor_future = client.submit(PubSubActor, *args, **kwargs, actor=True)
            Variable(PubSubActor.DEISA_PUBSUB_ACTOR_FUTURE_VARIABLE, client=client).set(actor_future)
            return actor_future.result()
