"""Consolidation is what drives the per-bank vector index threshold.

The threshold only converges if every consolidation reconciles the bank, whatever triggered it, and
a reconciliation failure must never surface as a failed consolidation.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.consolidation import consolidator
from hindsight_api.engine.memory_engine import MemoryEngine


@pytest.mark.asyncio
async def test_consolidation_reconciles_and_survives_a_failure(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch
):
    bank_id = f"test-consolidate-idx-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    await memory.retain_async(
        bank_id=bank_id, content="Alice works at Google.", context="Test", request_context=request_context
    )

    calls: list[str] = []

    async def _spy(conn, *, bank_id: str, ops) -> None:
        calls.append(bank_id)

    monkeypatch.setattr(consolidator, "reconcile_bank_vector_indexes", _spy)
    await consolidator.run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)
    assert calls == [bank_id]

    async def _boom(conn, *, bank_id: str, ops) -> None:
        raise RuntimeError("index build failed")

    monkeypatch.setattr(consolidator, "reconcile_bank_vector_indexes", _boom)
    result = await consolidator.run_consolidation_job(
        memory_engine=memory, bank_id=bank_id, request_context=request_context
    )
    assert result.get("status") != "error"

    await memory.delete_bank(bank_id, request_context=request_context)
