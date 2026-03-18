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


from collections import defaultdict, deque
from typing import Dict, Tuple
from typing import List

from dask.distributed import get_client, Client
from distributed import Lock, Variable


class PubSubActor:
    DEISA_PUBSUB_ACTOR_FUTURE_VARIABLE = 'deisa_pubsub_actor_future'

    def __init__(self, nb_bridges: int, maxlen: int = 10):
        print(f"pubsub: init {nb_bridges} {maxlen}", flush=True)

        self.client: Client = get_client()
        self.nb_bridges = nb_bridges
        self.maxlen = maxlen

        self.partial_metadata = defaultdict(lambda: defaultdict(dict))
        self.completed = defaultdict(lambda: deque(maxlen=maxlen))
        self.done: Dict[Tuple[str, int], bool] = {}

        self.offsets = {}

        # topic -> {arrays, when}
        self.callbacks: Dict[str, dict] = {}

        self.available_iterations = defaultdict(set)
        self.emitted = set()

    # -------------------------
    # Topic helper
    # -------------------------
    def _make_topic(self, arrays, when) -> str:
        return f"{when}|" + "|".join(sorted(arrays))

    # -------------------------
    # Publish
    # -------------------------
    async def publish(self, msg: dict):
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

    # -------------------------
    # Register callbacks
    # -------------------------
    async def register_callback(self, arrays: List[str], when) -> str:
        """
        when can be none is arrays contains only one array
        """
        topic = self._make_topic(arrays, when)
        self.callbacks[topic] = {
            "arrays": sorted(arrays),
            "when": when,
        }
        return topic

    async def unregister_callback(self, topic: str):
        self.callbacks.pop(topic, None)

    # -------------------------
    # Callback logic
    # -------------------------
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

        print(f"log_event topic={topic} iteration={iteration}", flush=True)
        await self.client.log_event(topic, payload)

    # -------------------------
    # Polling API (unchanged)
    # -------------------------
    async def subscribe(self, topic: str, subscriber_id: str):
        if self.completed[topic]:
            last_iter = self.completed[topic][-1][0]
        else:
            last_iter = -1

        self.offsets[(topic, subscriber_id)] = last_iter + 1

    async def get_since(self, topic: str, subscriber_id: str):
        key = (topic, subscriber_id)

        if key not in self.offsets:
            raise ValueError("Not subscribed")

        last_iter = self.offsets[key]

        results = [
            data
            for it, data in self.completed[topic]
            if it >= last_iter
        ]

        if self.completed[topic]:
            self.offsets[key] = self.completed[topic][-1][0] + 1

        return results

    async def get_iteration(self, topic: str, iteration: int):
        for it, data in self.completed[topic]:
            if it == iteration:
                return data
        return None


def get_pubsub_actor(client: Client, *args, **kwargs) -> PubSubActor:
    def check_variable(dask_scheduler, name):
        ext = dask_scheduler.extensions["variables"]
        v = ext.variables.get(name)
        return v is not None

    with Lock(PubSubActor.DEISA_PUBSUB_ACTOR_FUTURE_VARIABLE, client=client):
        is_set = client.run_on_scheduler(check_variable, name=PubSubActor.DEISA_PUBSUB_ACTOR_FUTURE_VARIABLE)
        if is_set:
            return Variable(PubSubActor.DEISA_PUBSUB_ACTOR_FUTURE_VARIABLE, client=client).get().result()
        else:
            actor_future = client.submit(PubSubActor, *args, **kwargs, actor=True)
            Variable(PubSubActor.DEISA_PUBSUB_ACTOR_FUTURE_VARIABLE, client=client).set(actor_future)
            return actor_future.result()


def get_topic_key(array_names: List[str], when: str) -> str:
    return str({'arrays': sorted(array_names), 'when': when})
