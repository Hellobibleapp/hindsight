"""``reconcile_bank_vector_indexes``: per-bank vector indexes follow the configured row threshold.

What is pinned here is the decision — which of create/drop runs, for a given threshold, row count and
index state, and how long a count is reused. The DDL itself belongs to the ops layer, and building it
for real on the shared ``memory_units`` table would deadlock the other xdist workers.
"""

import uuid
from types import SimpleNamespace

import pytest

from hindsight_api import RequestContext
from hindsight_api.config import get_config
from hindsight_api.engine.bank_stats_cache import BankStatsCache
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import bank_utils
from hindsight_api.engine.retain.bank_utils import _IndexState


class _RecordingOps:
    """Stands in for the backend ops, recording the index calls instead of running them."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def create_bank_vector_indexes(self, *args, **kwargs) -> None:
        self.calls.append("create")

    async def drop_bank_vector_indexes(self, *args, **kwargs) -> None:
        self.calls.append("drop")


def _stub_config(monkeypatch, *, threshold: int) -> None:
    real = get_config()
    monkeypatch.setattr(
        "hindsight_api.engine.retain.bank_utils.get_config",
        lambda: SimpleNamespace(
            per_bank_vector_index_min_rows=threshold,
            per_bank_vector_index_cache_ttl_seconds=0,
            bank_stats_cache_max_entries=real.bank_stats_cache_max_entries,
            vector_extension=real.vector_extension,
        ),
    )


def _stub_count_cache(monkeypatch, *, ttl: int) -> None:
    """Give the reconciler a cache of its own, so one test's counts never reach another's."""
    monkeypatch.setattr(bank_utils, "_memory_count_cache_instance", BankStatsCache(ttl_seconds=ttl, max_entries=100))


def _stub_index_state(monkeypatch) -> dict[str, _IndexState]:
    """Report the bank's index state from the returned dict, which a test moves as it goes."""
    state = {"value": _IndexState.ABSENT}

    async def _state(*_args) -> _IndexState:
        return state["value"]

    monkeypatch.setattr(bank_utils, "_bank_indexes_state", _state)
    return state


async def _seed(conn, bank_id: str, count: int) -> None:
    for _ in range(count):
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, created_at, updated_at)
            VALUES ($1, $2, 'a fact', 'experience', NOW(), NOW(), NOW())
            """,
            uuid.uuid4(),
            bank_id,
        )


@pytest.mark.asyncio
async def test_indexes_follow_the_threshold_in_both_directions(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch
):
    bank_id = f"test-reconcile-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    state = _stub_index_state(monkeypatch)
    _stub_count_cache(monkeypatch, ttl=0)

    backend = await memory._get_backend()
    ops = _RecordingOps()

    async def reconcile(threshold: int) -> None:
        _stub_config(monkeypatch, threshold=threshold)
        await bank_utils.reconcile_bank_vector_indexes(backend, bank_id=bank_id, ops=ops)

    async with acquire_with_retry(backend) as conn:
        await _seed(conn, bank_id, 2)
    await reconcile(3)
    assert ops.calls == []  # below the threshold, and no indexes: nothing to do

    async with acquire_with_retry(backend) as conn:
        await _seed(conn, bank_id, 1)
    await reconcile(3)
    assert ops.calls == ["create"]

    state["value"] = _IndexState.READY
    await reconcile(3)
    assert ops.calls == ["create"]  # already in the right state: no DDL

    async with acquire_with_retry(backend) as conn:
        await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
    await reconcile(3)
    assert ops.calls == ["create", "drop"]

    state["value"] = _IndexState.UNUSABLE
    async with acquire_with_retry(backend) as conn:
        await _seed(conn, bank_id, 5)
    await reconcile(3)
    assert ops.calls == ["create", "drop"]  # a build in flight or a dead one: left alone

    state["value"] = _IndexState.ABSENT
    await reconcile(0)
    assert ops.calls == ["create", "drop"]  # disabled: no DDL, whatever the bank holds

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_disabled_threshold_costs_nothing(monkeypatch):
    """The default is 0, and it must cost a consolidation nothing — not even a pooled connection.

    Handed a backend that refuses to be acquired: reaching the database at all, for a mechanism that
    is switched off, would surface as this test rather than as pool pressure on a live deployment.
    """
    _stub_config(monkeypatch, threshold=0)

    class _RefusingBackend:
        backend_type = "postgresql"

        def acquire(self):
            raise AssertionError("a disabled threshold must not take a connection")

    await bank_utils.reconcile_bank_vector_indexes(_RefusingBackend(), bank_id="test-off", ops=_RecordingOps())


@pytest.mark.asyncio
async def test_a_cached_count_hides_a_growing_bank_until_it_expires(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch
):
    """The cost of not counting on every consolidation: growth lands one interval late."""
    bank_id = f"test-ttl-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    state = _stub_index_state(monkeypatch)
    _stub_count_cache(monkeypatch, ttl=3600)

    backend = await memory._get_backend()
    ops = _RecordingOps()

    async def reconcile(threshold: int) -> None:
        _stub_config(monkeypatch, threshold=threshold)
        await bank_utils.reconcile_bank_vector_indexes(backend, bank_id=bank_id, ops=ops)

    async with acquire_with_retry(backend) as conn:
        await _seed(conn, bank_id, 2)
    await reconcile(10)
    assert ops.calls == []  # 2 memories against a threshold of 10: nothing to do

    async with acquire_with_retry(backend) as conn:
        await _seed(conn, bank_id, 20)
    await reconcile(10)
    assert ops.calls == []  # 22 rows now, but the remembered count still says 2

    _stub_count_cache(monkeypatch, ttl=3600)  # a fresh cache stands in for the TTL elapsing
    await reconcile(10)
    assert ops.calls == ["create"]

    await memory.delete_bank(bank_id, request_context=request_context)
