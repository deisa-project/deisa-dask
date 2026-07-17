import threading

import pytest

from utils import FakeComm, run_on_all_ranks


def test_gather():
    state = FakeComm.State(size=4)
    comm = FakeComm(state, 0)

    results = run_on_all_ranks(
        lambda: comm,
        lambda c: c.gather(c.Get_rank())
    )

    assert results[0] == [0, 1, 2, 3]
    assert results[1:] == [None, None, None]


def test_bcast():
    state = FakeComm.State(size=4)
    comm = FakeComm(state, 0)

    results = run_on_all_ranks(
        lambda: comm,
        lambda c: c.bcast("hello" if c.Get_rank() == 0 else None)
    )

    assert results == ["hello"] * 4


def test_barrier():
    state = FakeComm.State(size=4)
    comm = FakeComm(state, 0)

    entered = []
    exited = []

    lock = threading.Lock()

    def worker(c):
        with lock:
            entered.append(c.Get_rank())

        c.barrier()

        with lock:
            exited.append(c.Get_rank())

    run_on_all_ranks(lambda: comm, worker)

    assert sorted(entered) == [0, 1, 2, 3]
    assert sorted(exited) == [0, 1, 2, 3]

    # Nobody can leave before everybody entered
    assert len(entered) == 4


def test_split():
    state = FakeComm.State(size=6)
    comm = FakeComm(state, 0)

    def worker(c):
        color = c.Get_rank() % 2
        return c.Split(color=color)

    subcomms = run_on_all_ranks(lambda: comm, worker)

    even = [subcomms[r] for r in (0, 2, 4)]
    odd = [subcomms[r] for r in (1, 3, 5)]

    assert all(c.Get_size() == 3 for c in even)
    assert all(c.Get_size() == 3 for c in odd)

    assert [c.Get_rank() for c in even] == [0, 1, 2]
    assert [c.Get_rank() for c in odd] == [0, 1, 2]


def test_split_key_order():
    state = FakeComm.State(size=4)
    comm = FakeComm(state, 0)

    keys = [5, 2, 2, 8]

    def worker(c):
        return c.Split(color=0, key=keys[c.Get_rank()])

    subcomms = run_on_all_ranks(lambda: comm, worker)

    # key ordering:
    # rank1(key2)
    # rank2(key2)
    # rank0(key5)
    # rank3(key8)

    expected = {
        1: 0,
        2: 1,
        0: 2,
        3: 3,
    }

    for world_rank, subcomm in enumerate(subcomms):
        assert subcomm.Get_rank() == expected[world_rank]
        assert subcomm.Get_size() == 4


def test_multiple_gathers():
    state = FakeComm.State(size=4)
    comm = FakeComm(state, 0)

    def worker(c):
        r1 = c.gather(c.Get_rank())
        r2 = c.gather(c.Get_rank() + 10)
        return r1, r2

    results = run_on_all_ranks(lambda: comm, worker)

    assert results[0][0] == [0, 1, 2, 3]
    assert results[0][1] == [10, 11, 12, 13]


def test_split_then_gather():
    state = FakeComm.State(size=6)
    comm = FakeComm(state, 0)

    def worker(c):
        sub = c.Split(color=c.Get_rank() % 2)
        return sub.gather(c.Get_rank())

    results = run_on_all_ranks(lambda: comm, worker)

    assert results[0] == [0, 2, 4]
    assert results[1] == [1, 3, 5]
    assert results[2] is None
    assert results[3] is None
    assert results[4] is None
    assert results[5] is None


def test_free_unblocks_all():
    state = FakeComm.State(size=4)
    comm = FakeComm(state, 0)

    def worker(c):
        c.Free()
        return True

    results = run_on_all_ranks(lambda: comm, worker)

    assert results == [True, True, True, True]
    assert state.freed is True


def test_free_is_collective():
    state = FakeComm.State(size=4)
    comm = FakeComm(state, 0)

    finished = []
    lock = threading.Lock()

    def worker(c):
        rank = c.Get_rank()
        c.Free()

        with lock:
            finished.append(rank)

    run_on_all_ranks(lambda: comm, worker)

    assert sorted(finished) == [0, 1, 2, 3]
    assert state.freed


def test_use_after_free_raises():
    state = FakeComm.State(size=2)
    comm = FakeComm(state, 0)

    def worker(c):
        c.Free()
        with pytest.raises(RuntimeError):
            c.gather(c.Get_rank())

    run_on_all_ranks(lambda: comm, worker)


def test_free_then_new_collective_safety():
    state = FakeComm.State(size=4)
    comm = FakeComm(state, 0)

    def worker(c):
        c.Free()

        # This should fail immediately, not hang
        with pytest.raises(RuntimeError):
            c.bcast(c.Get_rank())

    run_on_all_ranks(lambda: comm, worker)


def test_multiple_free_calls_safe():
    state = FakeComm.State(size=4)
    comm = FakeComm(state, 0)

    def worker(c):
        c.Free()
        c.Free()  # should not hang or crash

    run_on_all_ranks(lambda: comm, worker)
    assert state.freed is True
