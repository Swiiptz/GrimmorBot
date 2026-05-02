"""Toutes les requêtes SQL applicatives."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .database import Database


async def log_command_usage(db: Database, user_id: int, guild_id: int, command: str) -> None:
    await db.conn.execute(
        "INSERT INTO usage_log (user_id, guild_id, command) VALUES (?, ?, ?)",
        (str(user_id), str(guild_id), command),
    )
    await db.conn.commit()


async def count_command_usage_today(db: Database, user_id: int, command: str) -> int:
    """Compte les usages depuis minuit UTC pour un (utilisateur, commande)."""
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    cur = await db.conn.execute(
        "SELECT COUNT(*) FROM usage_log "
        "WHERE user_id = ? AND command = ? AND used_at >= ?",
        (str(user_id), command, start.strftime("%Y-%m-%d %H:%M:%S")),
    )
    row = await cur.fetchone()
    await cur.close()
    return int(row[0]) if row else 0


async def last_command_usage(
    db: Database, user_id: int, command: str
) -> Optional[datetime]:
    cur = await db.conn.execute(
        "SELECT used_at FROM usage_log "
        "WHERE user_id = ? AND command = ? "
        "ORDER BY used_at DESC LIMIT 1",
        (str(user_id), command),
    )
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return None
    raw = row[0]
    if isinstance(raw, datetime):
        dt = raw
    else:
        dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


async def upsert_channel_activity(
    db: Database, channel_id: int, guild_id: int, when: datetime
) -> None:
    await db.conn.execute(
        """
        INSERT INTO channel_activity (channel_id, guild_id, last_message_at)
        VALUES (?, ?, ?)
        ON CONFLICT(channel_id) DO UPDATE SET
            last_message_at = excluded.last_message_at,
            guild_id = excluded.guild_id
        WHERE excluded.last_message_at > channel_activity.last_message_at
        """,
        (str(channel_id), str(guild_id), when.strftime("%Y-%m-%d %H:%M:%S")),
    )
    await db.conn.commit()


async def get_channel_last_activity(
    db: Database, channel_id: int
) -> Optional[datetime]:
    cur = await db.conn.execute(
        "SELECT last_message_at FROM channel_activity WHERE channel_id = ?",
        (str(channel_id),),
    )
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return None
    raw = row[0]
    if isinstance(raw, datetime):
        dt = raw
    else:
        dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


async def get_all_channel_activity(
    db: Database, guild_id: int
) -> dict[int, datetime]:
    cur = await db.conn.execute(
        "SELECT channel_id, last_message_at FROM channel_activity WHERE guild_id = ?",
        (str(guild_id),),
    )
    rows = await cur.fetchall()
    await cur.close()
    out: dict[int, datetime] = {}
    for row in rows:
        raw = row[1]
        dt = raw if isinstance(raw, datetime) else datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        out[int(row[0])] = dt
    return out


async def add_inactive_exclusion(
    db: Database,
    guild_id: int,
    target_type: str,
    target_id: int,
    target_name: str,
) -> None:
    await db.conn.execute(
        """
        INSERT INTO inactive_exclusions (guild_id, target_type, target_id, target_name)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(guild_id, target_type, target_id) DO UPDATE SET
            target_name = excluded.target_name
        """,
        (str(guild_id), target_type, str(target_id), target_name),
    )
    await db.conn.commit()


async def remove_inactive_exclusion(
    db: Database,
    guild_id: int,
    target_type: str,
    target_id: int,
) -> bool:
    cur = await db.conn.execute(
        """
        DELETE FROM inactive_exclusions
        WHERE guild_id = ? AND target_type = ? AND target_id = ?
        """,
        (str(guild_id), target_type, str(target_id)),
    )
    await db.conn.commit()
    removed = cur.rowcount > 0
    await cur.close()
    return removed


async def get_inactive_exclusions(db: Database, guild_id: int) -> list[dict[str, str]]:
    cur = await db.conn.execute(
        """
        SELECT target_type, target_id, target_name
        FROM inactive_exclusions
        WHERE guild_id = ?
        ORDER BY target_type ASC, target_name COLLATE NOCASE ASC
        """,
        (str(guild_id),),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [
        {
            "target_type": str(row["target_type"]),
            "target_id": str(row["target_id"]),
            "target_name": str(row["target_name"]),
        }
        for row in rows
    ]


def seconds_since(dt: datetime) -> float:
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).total_seconds()
