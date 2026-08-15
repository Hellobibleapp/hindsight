"""Bank profile utilities, and the per-bank vector indexes a bank's lifecycle keeps in step."""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, Field

from ..._vector_index import index_using_clause, uses_per_bank_vector_indexes
from ...config import get_config
from ...metrics import get_metrics_collector
from ..bank_stats_cache import BankStatsCache
from ..db_utils import acquire_with_retry, retry_with_backoff
from ..memory_engine import fq_table, get_current_schema
from ..response_models import DispositionTraits

logger = logging.getLogger(__name__)

# Fact types that get per-bank partial vector indexes, mapped to their 4-char index suffix.
_BANK_INDEX_FACT_TYPES: dict[str, str] = {
    "world": "worl",
    "experience": "expr",
    "observation": "obsv",
}


class _IndexState(Enum):
    """What a bank's set of per-(bank, fact_type) vector indexes looks like right now."""

    ABSENT = "absent"
    READY = "ready"
    #: Present but not all usable — a build in flight, or one that died. The two are indistinguishable
    #: from the catalog, so a caller can only leave the bank alone.
    UNUSABLE = "unusable"


# A bank's memory count is the one read here that scans the shared memory_units table, so it is the
# one worth not repeating for a bank that consolidates several times in a session. Built on first use
# because it needs the resolved config; bounded and coalesced by the cache itself.
_memory_count_cache_instance: BankStatsCache | None = None


def _memory_count_cache(config) -> BankStatsCache:
    global _memory_count_cache_instance
    if _memory_count_cache_instance is None:
        _memory_count_cache_instance = BankStatsCache(
            ttl_seconds=config.per_bank_vector_index_cache_ttl_seconds,
            # Same purpose as the stats cache's own bound, so it takes the same setting.
            max_entries=config.bank_stats_cache_max_entries,
        )
    return _memory_count_cache_instance


def _bank_index_name(ft: str, internal_id: str) -> str:
    """Deterministic, schema-safe vector index name for a (bank, fact_type) pair.

    Uses the first 16 hex chars of internal_id (8 bytes of entropy) — unique
    enough in practice, fits comfortably within PostgreSQL's 63-char identifier limit.
    """
    uid = str(internal_id).replace("-", "")[:16]
    return f"idx_mu_emb_{_BANK_INDEX_FACT_TYPES[ft]}_{uid}"


async def _bank_indexes_state(conn, schema: str, internal_id: str) -> _IndexState:
    """Read whether a bank's full set of vector indexes exists and is usable."""
    names = [_bank_index_name(ft, internal_id) for ft in _BANK_INDEX_FACT_TYPES]
    row = await conn.fetchrow(
        "SELECT COUNT(*) AS found, COUNT(*) FILTER (WHERE i.indisvalid) AS usable "
        "FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_index i ON i.indexrelid = c.oid "
        "WHERE n.nspname = $1 AND c.relname = ANY($2::text[])",
        schema,
        names,
    )
    found, usable = (int(row["found"]), int(row["usable"])) if row else (0, 0)
    if found == 0:
        return _IndexState.ABSENT
    if usable == len(names):
        return _IndexState.READY
    return _IndexState.UNUSABLE


def _vector_index_clause() -> str | None:
    """Return the USING clause for per-bank vector indexes, if this backend uses them."""
    ext = get_config().vector_extension
    if not uses_per_bank_vector_indexes(ext):
        return None
    return index_using_clause(ext)


async def create_bank_vector_indexes(
    conn, bank_id: str, internal_id: str, ops=None, *, rows: int = 0, concurrently: bool = False
) -> None:
    """Create per-(bank, fact_type) partial vector indexes for a bank holding ``rows`` memories.

    A bank gets its indexes once it holds ``per_bank_vector_index_min_rows`` of them. ``rows``
    defaults to 0 — a bank being created is empty — so at the default threshold of 0 every bank
    qualifies, and above it a new bank waits for the reconciler to find it past the threshold.

    Respects the HINDSIGHT_API_VECTOR_EXTENSION config to use the appropriate
    index type (HNSW for pgvector, DiskANN for pgvectorscale, vchordrq for vchord).

    AlloyDB ScaNN uses global vector indexes with filtered vector search; it
    cannot safely create per-bank indexes at bank-creation time because new
    banks have no embedding rows.
    bank_id is escaped for SQL literal safety (apostrophes doubled).

    On Oracle 23ai, this is a no-op — Oracle uses a single global vector index
    created during migrations. Partial indexes (WHERE clause) are not supported
    for Oracle vector indexes.
    """
    index_clause = _vector_index_clause()
    if index_clause is None:
        logger.debug("Skipping per-bank vector indexes for configured backend")
        return

    if rows < get_config().per_bank_vector_index_min_rows:
        return

    await ops.create_bank_vector_indexes(
        conn,
        fq_table("memory_units"),
        bank_id,
        internal_id,
        index_clause,
        _BANK_INDEX_FACT_TYPES,
        concurrently=concurrently,
    )


async def drop_bank_vector_indexes(conn, internal_id: str, ops=None) -> None:
    """Drop per-(bank, fact_type) partial vector indexes for a bank being deleted.

    Called before the bank row is deleted so internal_id is still known.
    Idempotent via DROP INDEX IF EXISTS.

    On Oracle, this is a no-op (uses single global vector index).
    """
    await ops.drop_bank_vector_indexes(
        conn,
        get_current_schema(),
        internal_id,
        _BANK_INDEX_FACT_TYPES,
    )


DEFAULT_DISPOSITION = {
    "skepticism": 3,
    "literalism": 3,
    "empathy": 3,
}


class BankProfile(TypedDict):
    """Type for bank profile data."""

    name: str
    disposition: DispositionTraits
    mission: str


@dataclass
class BankProfileResult:
    """Result of a get-or-create bank lookup.

    ``created`` is True when the bank row was freshly inserted on this call,
    which callers use to drive the one-time HINDSIGHT_API_DEFAULT_BANK_TEMPLATE hook.
    """

    profile: BankProfile
    created: bool


class MissionMergeResponse(BaseModel):
    """LLM response for mission merge."""

    mission: str = Field(description="Merged mission in first person perspective")


async def get_bank_profile(pool, bank_id: str) -> BankProfile:
    """
    Get bank profile (name, disposition + mission).
    Auto-creates bank with default values if not exists.

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier

    Returns:
        BankProfile with name, typed DispositionTraits, and mission
    """
    result = await get_or_create_bank_profile(pool, bank_id)
    return result.profile


async def get_bank_profile_if_exists(pool, bank_id: str) -> BankProfile | None:
    """
    Get bank profile (name, disposition + mission) without auto-creating.

    Returns None if the bank does not exist. This is the read-only variant
    of get_bank_profile, intended for read endpoints where a bank that
    doesn't exist should surface as 404 rather than be silently created.

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier

    Returns:
        BankProfile if the bank exists, otherwise None.
    """
    async with acquire_with_retry(pool) as conn:
        row = await conn.fetchrow(
            f"""
            SELECT name, disposition, mission
            FROM {fq_table("banks")} WHERE bank_id = $1
            """,
            bank_id,
        )
        if not row:
            return None
        disposition_data = row["disposition"]
        if isinstance(disposition_data, str):
            disposition_data = json.loads(disposition_data)
        return BankProfile(
            name=row["name"],
            disposition=DispositionTraits(**disposition_data),
            mission=row["mission"] or "",
        )


async def get_or_create_bank_profile(pool, bank_id: str) -> BankProfileResult:
    """
    Get bank profile, auto-creating with defaults if it doesn't exist.

    Same as get_bank_profile, but also reports whether the bank was freshly
    created on this call (``BankProfileResult.created``). Used by the memory
    engine to apply the HINDSIGHT_API_DEFAULT_BANK_TEMPLATE hook on first bank
    creation.

    Acquires its own connection. When the caller already holds a connection and
    wants the bank row to share its transaction (so the lazy bank-create commits
    or rolls back atomically with the caller's write), use
    ``get_or_create_bank_profile_on_conn`` instead.
    """

    # A fresh bank builds its per-(bank, fact_type) partial vector indexes with
    # a plain CREATE INDEX (it must — this runs inside the bank-create tx, and
    # CONCURRENTLY cannot). That CREATE takes a ShareLock on the shared
    # memory_units table, which can deadlock with concurrent writers. The build
    # is idempotent (INSERT ... ON CONFLICT + CREATE INDEX IF NOT EXISTS), so a
    # transient deadlock (40P01 / ORA-00060) is safe to retry as a whole tx.
    async def _create() -> BankProfileResult:
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                return await get_or_create_bank_profile_on_conn(conn, bank_id, ops=pool.ops)

    return await retry_with_backoff(_create)


async def get_or_create_bank_profile_on_conn(conn, bank_id: str, *, ops) -> BankProfileResult:
    """
    Connection-bound variant of ``get_or_create_bank_profile``.

    Runs the SELECT, the ``INSERT ... ON CONFLICT DO NOTHING`` and the per-bank
    vector index creation on the caller-supplied ``conn``. When ``conn`` is
    inside an open transaction, the lazy bank-create therefore commits (or rolls
    back) atomically with whatever bank-scoped write the caller performs on the
    same connection — closing the window where a freshly-created bank could
    outlive a write that ultimately failed.

    ``ops`` is the backend's dialect ops object (``backend.ops``), needed for
    per-bank vector index DDL.
    """
    # Try to get existing bank
    row = await conn.fetchrow(
        f"""
        SELECT name, disposition, mission
        FROM {fq_table("banks")} WHERE bank_id = $1
        """,
        bank_id,
    )

    if row:
        # asyncpg returns JSONB as a string, so parse it
        disposition_data = row["disposition"]
        if isinstance(disposition_data, str):
            disposition_data = json.loads(disposition_data)

        return BankProfileResult(
            profile=BankProfile(
                name=row["name"],
                disposition=DispositionTraits(**disposition_data),
                mission=row["mission"] or "",
            ),
            created=False,
        )

    # Bank doesn't exist, create with defaults.
    # Generate internal_id here so we control the value and can use it
    # immediately for vector index creation without a RETURNING round-trip.
    internal_id = uuid.uuid4()
    inserted = await conn.fetchval(
        f"""
        INSERT INTO {fq_table("banks")} (bank_id, name, disposition, mission, internal_id)
        VALUES ($1, $2, $3::jsonb, $4, $5)
        ON CONFLICT (bank_id) DO NOTHING
        RETURNING bank_id
        """,
        bank_id,
        bank_id,  # Default name is the bank_id
        json.dumps(DEFAULT_DISPOSITION),
        "",
        internal_id,
    )

    created = inserted is not None
    if created:
        # Fresh insert — create per-bank vector indexes (instant on empty bank)
        await create_bank_vector_indexes(conn, bank_id, str(internal_id), ops=ops)

    return BankProfileResult(
        profile=BankProfile(name=bank_id, disposition=DispositionTraits(**DEFAULT_DISPOSITION), mission=""),
        created=created,
    )


async def update_bank_disposition(pool, bank_id: str, disposition: dict[str, int]) -> None:
    """
    Update bank disposition traits.

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier
        disposition: Dict with skepticism, literalism, empathy (all 1-5)
    """
    # Ensure bank exists first
    await get_bank_profile(pool, bank_id)

    async with acquire_with_retry(pool) as conn:
        await conn.execute(
            f"""
            UPDATE {fq_table("banks")}
            SET disposition = $2::jsonb,
                updated_at = NOW()
            WHERE bank_id = $1
            """,
            bank_id,
            json.dumps(disposition),
        )


async def set_bank_mission(pool, bank_id: str, mission: str) -> None:
    """
    Set bank mission (replacing any existing mission).

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier
        mission: The mission text
    """
    # Ensure bank exists first
    await get_bank_profile(pool, bank_id)

    async with acquire_with_retry(pool) as conn:
        await conn.execute(
            f"""
            UPDATE {fq_table("banks")}
            SET mission = $2,
                updated_at = NOW()
            WHERE bank_id = $1
            """,
            bank_id,
            mission,
        )


async def merge_bank_mission(pool, llm_config, bank_id: str, new_info: str) -> dict:
    """
    Merge new mission information with existing mission using LLM.
    Normalizes to first person ("I") and resolves conflicts.

    Args:
        pool: Database connection pool
        llm_config: LLM configuration for mission merging
        bank_id: bank IDentifier
        new_info: New mission information to add/merge

    Returns:
        Dict with 'mission' (str) key
    """
    # Get current profile
    profile = await get_bank_profile(pool, bank_id)
    current_mission = profile["mission"]

    # Use LLM to merge missions
    result = await _llm_merge_mission(llm_config, current_mission, new_info)

    merged_mission = result["mission"]

    # Update in database
    async with acquire_with_retry(pool) as conn:
        await conn.execute(
            f"""
            UPDATE {fq_table("banks")}
            SET mission = $2,
                updated_at = NOW()
            WHERE bank_id = $1
            """,
            bank_id,
            merged_mission,
        )

    return {"mission": merged_mission}


async def _llm_merge_mission(llm_config, current: str, new_info: str) -> dict:
    """
    Use LLM to intelligently merge mission information.

    Args:
        llm_config: LLM configuration to use
        current: Current mission text
        new_info: New information to merge

    Returns:
        Dict with 'mission' (str) key
    """
    prompt = f"""You are helping maintain an agent's mission statement.

Current mission: {current if current else "(empty)"}

New information to add: {new_info}

Instructions:
1. Merge the new information with the current mission
2. If there are conflicts, the NEW information overwrites the old
3. Keep additions that don't conflict
4. Output in FIRST PERSON ("I") perspective
5. Be concise - keep it under 500 characters
6. Return ONLY the merged mission text, no explanations

Merged mission:"""

    try:
        messages = [{"role": "user", "content": prompt}]

        content = await llm_config.call(
            messages=messages, scope="bank_mission", temperature=0.3, max_completion_tokens=8192
        )

        logger.info(f"LLM response for mission merge (first 500 chars): {content[:500]}")

        merged = content.strip()
        if not merged or merged.lower() in ["(empty)", "none", "n/a"]:
            merged = new_info if new_info else ""
        return {"mission": merged}

    except Exception as e:
        logger.error(f"Error merging mission with LLM: {e}")
        # Fallback: just append new info
        if current:
            merged = f"{current} {new_info}".strip()
        else:
            merged = new_info

        return {"mission": merged}


# Sort floor for banks that have never been written to and carry no created_at.
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _as_utc(ts: datetime | None) -> datetime | None:
    """Normalize a DB timestamp to an aware UTC datetime so values stay comparable."""
    if ts is None:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


async def list_banks(pool) -> list:
    """
    List all banks in the system with summary stats.

    ``last_document_at`` is document *ingestion* time (when a document first
    landed), while ``last_write_at`` is the last time anything was written to
    the bank — a document re-retained/appended to, or a fact stored. Appending
    to a long-lived document does not move ``last_document_at``, which is why
    the two differ and why UIs showing "last write" must use ``last_write_at``.

    Args:
        pool: Database connection pool

    Returns:
        List of dicts with bank info and stats (fact_count, last_document_at, last_write_at),
        most recently written bank first.
    """
    banks_table = fq_table("banks")
    docs_table = fq_table("documents")
    mu_table = fq_table("memory_units")

    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            f"""
            SELECT
                b.bank_id, b.name, b.disposition, b.mission,
                b.created_at, b.updated_at,
                COALESCE(m.fact_count, 0) AS fact_count,
                d.last_document_at,
                d.last_document_write_at,
                m.last_fact_at
            FROM {banks_table} b
            LEFT JOIN (
                SELECT bank_id,
                       MAX(created_at) AS last_document_at,
                       MAX(updated_at) AS last_document_write_at
                FROM {docs_table}
                GROUP BY bank_id
            ) d ON d.bank_id = b.bank_id
            LEFT JOIN (
                SELECT bank_id,
                       COUNT(*) AS fact_count,
                       MAX(created_at) AS last_fact_at
                FROM {mu_table}
                GROUP BY bank_id
            ) m ON m.bank_id = b.bank_id
            ORDER BY b.bank_id
            """
        )

        result = []
        # Banks are ordered by last write in Python rather than SQL: GREATEST() has
        # different NULL semantics on PostgreSQL vs Oracle, and the bank list is small.
        sort_keys: dict[str, datetime] = {}
        # A store that keeps memories outside SQL leaves the memory_units join empty, so its
        # per-bank fact_count comes from the store instead (one live count per bank).
        from ..memories import get_memories

        _store = get_memories()

        for row in rows:
            disposition_data = row["disposition"]
            if isinstance(disposition_data, str):
                disposition_data = json.loads(disposition_data)

            last_doc = _as_utc(row["last_document_at"])
            created_at = _as_utc(row["created_at"])
            updated_at = _as_utc(row["updated_at"])
            # Last write = newest of "a document was (re-)retained" and "a fact was stored".
            # Appending to an existing document only bumps documents.updated_at, and facts
            # written outside a retain (consolidation, curation, import) only bump memory_units.
            write_times = [t for t in (_as_utc(row["last_document_write_at"]), _as_utc(row["last_fact_at"])) if t]
            last_write = max(write_times) if write_times else None

            fact_count = row["fact_count"]
            if not _store.writes_memory_rows_in_sql_for(row["bank_id"]):
                fact_count = sum(
                    (await _store.count_memories(conn=conn, fq_table=fq_table, bank_id=row["bank_id"])).values()
                )

            sort_keys[row["bank_id"]] = last_write or created_at or _UNIX_EPOCH
            result.append(
                {
                    "bank_id": row["bank_id"],
                    "name": row["name"],
                    "disposition": disposition_data,
                    "mission": row["mission"] or "",
                    "created_at": created_at.isoformat() if created_at else None,
                    "updated_at": updated_at.isoformat() if updated_at else None,
                    "fact_count": fact_count,
                    "last_document_at": last_doc.isoformat() if last_doc else None,
                    "last_write_at": last_write.isoformat() if last_write else None,
                }
            )

        result.sort(key=lambda bank: sort_keys[bank["bank_id"]], reverse=True)
        return result


async def reconcile_bank_vector_indexes(backend, *, bank_id: str, ops) -> None:
    """Bring one bank's vector indexes in line with the configured row threshold.

    Creates them once the bank holds ``per_bank_vector_index_min_rows`` memories and drops them
    once it holds fewer. A threshold of 0 disables the mechanism, leaving indexes to bank creation.

    One attempt, no retry: the next consolidation comes back to it. Nothing here judges whether an
    index is sound — a bank whose indexes are not usable is left alone, since a build in flight and
    a dead one look the same from here.

    The bank's memory count is cached, so a bank that grows past the threshold — or a threshold that
    moves — is picked up on the next look rather than at once. Its index state is read every time:
    that one is a catalog lookup, and takes none of the locks a scan of memory_units would.

    Takes its own connection rather than borrowing the caller's, and only once it has something to
    do: a disabled threshold costs nothing at all, and both the build and the drop need autocommit
    since CONCURRENTLY cannot run inside a transaction.
    """
    from ..memories import get_memories

    config = get_config()
    threshold = config.per_bank_vector_index_min_rows
    # The backend is checked on its own: the configured extension is a Postgres extension name
    # whatever runs underneath, so on Oracle it still reads as one that uses per-bank indexes.
    if threshold <= 0 or _vector_index_clause() is None or backend.backend_type != "postgresql":
        return

    schema = get_current_schema()
    async with acquire_with_retry(backend) as conn:
        internal_id = await conn.fetchval(f"SELECT internal_id FROM {fq_table('banks')} WHERE bank_id = $1", bank_id)
        if internal_id is None:
            return
        internal_id = str(internal_id)

        state = await _bank_indexes_state(conn, schema, internal_id)
        if state is _IndexState.UNUSABLE:
            get_metrics_collector().record_vector_index_reconciliation(bank_id, "unusable")
            return

        async def _count() -> dict:
            total = await get_memories().count_memories_capped(
                conn=conn, fq_table=fq_table, bank_id=bank_id, limit=threshold
            )
            return {"count": total}

        # Cached under internal_id, not bank_id: a bank deleted and recreated under the same name gets
        # a fresh internal_id, so it cannot inherit the previous one's count for the rest of the TTL.
        cached = await _memory_count_cache(config).get_or_load(schema, internal_id, _count)
        count = cached["count"]
        indexed = state is _IndexState.READY
        wanted = count >= threshold

        if wanted and not indexed:
            try:
                await create_bank_vector_indexes(conn, bank_id, internal_id, ops=ops, rows=count, concurrently=True)
            except Exception:
                # CONCURRENTLY cannot run in a transaction, so a failed build is not rolled back:
                # what it leaves behind keeps costing every write without serving a read. Take it.
                await drop_bank_vector_indexes(conn, internal_id, ops=ops)
                raise
            logger.info("Created per-bank vector indexes for %s: reached the %d-row threshold", bank_id, threshold)
            get_metrics_collector().record_vector_index_reconciliation(bank_id, "created")
        elif indexed and not wanted:
            await drop_bank_vector_indexes(conn, internal_id, ops=ops)
            logger.info("Dropped per-bank vector indexes for %s: below the %d-row threshold", bank_id, threshold)
            get_metrics_collector().record_vector_index_reconciliation(bank_id, "dropped")
