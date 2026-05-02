"""Initialisation et gestion de la connexion SQLite asynchrone."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    guild_id TEXT NOT NULL,
    command TEXT NOT NULL,
    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_usage_user_command_date
    ON usage_log(user_id, command, used_at);

CREATE TABLE IF NOT EXISTS channel_activity (
    channel_id TEXT PRIMARY KEY,
    guild_id TEXT NOT NULL,
    last_message_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_channel_activity_guild
    ON channel_activity(guild_id);

CREATE TABLE IF NOT EXISTS inactive_exclusions (
    guild_id TEXT NOT NULL,
    target_type TEXT NOT NULL CHECK (target_type IN ('category', 'channel')),
    target_id TEXT NOT NULL,
    target_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, target_type, target_id)
);

CREATE INDEX IF NOT EXISTS idx_inactive_exclusions_guild
    ON inactive_exclusions(guild_id);
"""


class Database:
    """Wrapper minimal autour d'aiosqlite avec une connexion partagée."""

    def __init__(self, path: Path):
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        if self._conn is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        logger.info("SQLite ouvert: %s", self.path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database non initialisée. Appelle connect() d'abord.")
        return self._conn


_DB_INSTANCE: Optional[Database] = None


async def get_db(path: Path | None = None) -> Database:
    """Singleton de connexion. Le premier appel doit fournir le `path`."""
    global _DB_INSTANCE
    if _DB_INSTANCE is None:
        if path is None:
            raise RuntimeError("Premier appel à get_db sans chemin.")
        _DB_INSTANCE = Database(path)
        await _DB_INSTANCE.connect()
    return _DB_INSTANCE
