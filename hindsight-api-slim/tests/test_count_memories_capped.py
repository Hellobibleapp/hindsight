"""``count_memories_capped``: the bounded count behind the per-bank index threshold.

Pins the two properties the threshold relies on: the count covers this bank's memories only, and
it never exceeds ``limit`` however large the bank is.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.schema import fq_table


async def _seed(conn, bank_id: str, fact_type: str = "experience") -> None:
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, created_at, updated_at)
        VALUES ($1, $2, 'a fact', $3, NOW(), NOW(), NOW())
        """,
        uuid.uuid4(),
        bank_id,
        fact_type,
    )


@pytest.mark.asyncio
async def test_counts_the_banks_own_memories_and_stops_at_the_limit(
    memory: MemoryEngine, request_context: RequestContext
):
    bank_id = f"test-facts-{uuid.uuid4().hex[:8]}"
    other_bank_id = f"test-facts-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    await memory.get_bank_profile(bank_id=other_bank_id, request_context=request_context)

    store = get_memories()
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        for _ in range(3):
            await _seed(conn, bank_id)
        await _seed(conn, bank_id, fact_type="world")  # every fact type counts towards the total
        await _seed(conn, other_bank_id)  # another bank's rows never do

        async def count(limit: int) -> int:
            return await store.count_memories_capped(conn=conn, fq_table=fq_table, bank_id=bank_id, limit=limit)

        assert await count(1000) == 4
        assert await count(4) == 4  # exactly at the limit
        assert await count(2) == 2  # capped: the scan stops there

    await memory.delete_bank(bank_id, request_context=request_context)
    await memory.delete_bank(other_bank_id, request_context=request_context)
