import numpy as np
import pytest
from dask.distributed import Client, LocalCluster, Queue


@pytest.fixture(scope="module")
def env_setup():
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)
    yield client, cluster
    client.close()
    cluster.close()


@pytest.mark.parametrize("nb_puts", [10 ** i for i in range(5)])
def test_queue_put_per_op(nb_puts, benchmark, env_setup):
    client, cluster = env_setup
    q = Queue("Test", client=client)
    f = client.scatter(np.random.random((2, 2)), direct=True)

    def put_once():
        q.put(f)

    # Benchmark a single put, repeated nb_puts times
    benchmark.pedantic(put_once, iterations=nb_puts, rounds=1)
