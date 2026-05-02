"""Système de permissions par rôle pour les commandes du bot.

Charge un fichier ``permissions.json`` qui définit, pour chaque rôle Discord,
les commandes autorisées, le quota journalier et le cooldown.

Quand un utilisateur a plusieurs rôles configurés, les permissions sont
fusionnées de manière permissive (union des commandes, meilleur quota,
meilleur cooldown).
"""
from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

import discord

from db import get_db
from db.queries import (
    count_command_usage_today,
    last_command_usage,
    log_command_usage,
    seconds_since,
)
from utils.formatting import error_embed

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RolePermissions:
    """Permissions d'un rôle spécifique."""

    name: str
    commands: set[str]
    daily_quota: int = 0  # 0 = illimité
    cooldown_seconds: int = 0  # 0 = pas de cooldown
    command_daily_quotas: dict[str, int] = field(default_factory=dict)
    command_cooldowns: dict[str, int] = field(default_factory=dict)

    @property
    def has_all_commands(self) -> bool:
        return "*" in self.commands

    def can_use(self, command: str) -> bool:
        return self.has_all_commands or command in self.commands

    def daily_quota_for(self, command: str) -> int:
        return self.command_daily_quotas.get(command, self.daily_quota)

    def cooldown_for(self, command: str) -> int:
        return self.command_cooldowns.get(command, self.cooldown_seconds)


@dataclass
class ResolvedPermissions:
    """Permissions effectives d'un utilisateur (fusion de ses rôles)."""

    commands: set[str]
    daily_quota: int
    cooldown_seconds: int
    has_all_commands: bool = False
    sources: list[RolePermissions] = field(default_factory=list)

    def can_use(self, command: str) -> bool:
        return self.has_all_commands or command in self.commands

    def daily_quota_for(self, command: str) -> int:
        sources = [source for source in self.sources if source.can_use(command)]
        quotas = [source.daily_quota_for(command) for source in sources]
        if not quotas:
            quotas = [self.daily_quota]
        if 0 in quotas:
            return 0
        return max(quotas)

    def cooldown_for(self, command: str) -> int:
        sources = [source for source in self.sources if source.can_use(command)]
        cooldowns = [source.cooldown_for(command) for source in sources]
        if not cooldowns:
            cooldowns = [self.cooldown_seconds]
        return min(cooldowns)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_COMMANDS = {"ask", "archive", "pfc"}
DEFAULT_DAILY_QUOTA = 0
DEFAULT_COOLDOWN_SECONDS = 0
DEFAULT_COMMAND_DAILY_QUOTAS = {"ask": 10}
DEFAULT_COMMAND_COOLDOWNS = {"ask": 30}

_DEFAULT_ROLE = RolePermissions(
    name="default",
    commands=set(DEFAULT_COMMANDS),
    daily_quota=DEFAULT_DAILY_QUOTA,
    cooldown_seconds=DEFAULT_COOLDOWN_SECONDS,
    command_daily_quotas=dict(DEFAULT_COMMAND_DAILY_QUOTAS),
    command_cooldowns=dict(DEFAULT_COMMAND_COOLDOWNS),
)


@dataclass
class PermissionConfig:
    """Configuration complète des permissions."""

    roles: dict[int, RolePermissions] = field(default_factory=dict)
    default: RolePermissions = field(default_factory=lambda: RolePermissions(
        name="default",
        commands=set(DEFAULT_COMMANDS),
        daily_quota=DEFAULT_DAILY_QUOTA,
        cooldown_seconds=DEFAULT_COOLDOWN_SECONDS,
        command_daily_quotas=dict(DEFAULT_COMMAND_DAILY_QUOTAS),
        command_cooldowns=dict(DEFAULT_COMMAND_COOLDOWNS),
    ))
    _path: Optional[Path] = field(default=None, repr=False)

    def resolve(self, member: discord.Member) -> ResolvedPermissions:
        """Résout les permissions d'un membre (union de ses rôles)."""
        matching = [self.roles[r.id] for r in member.roles if r.id in self.roles]

        if not matching:
            return ResolvedPermissions(
                commands=set(self.default.commands),
                daily_quota=self.default.daily_quota,
                cooldown_seconds=self.default.cooldown_seconds,
                has_all_commands=self.default.has_all_commands,
                sources=[self.default],
            )

        all_cmds: set[str] = set()
        has_star = False
        best_quota = 0
        best_cooldown: int | float = 999_999

        for rp in matching:
            all_cmds.update(rp.commands)
            if rp.has_all_commands:
                has_star = True
            if rp.daily_quota == 0:
                best_quota = 0
            elif best_quota != 0:
                best_quota = max(best_quota, rp.daily_quota)
            best_cooldown = min(best_cooldown, rp.cooldown_seconds)

        return ResolvedPermissions(
            commands=all_cmds,
            daily_quota=best_quota,
            cooldown_seconds=int(best_cooldown),
            has_all_commands=has_star,
            sources=matching,
        )

    # ─── Sérialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        roles_out: dict[str, dict] = {}
        for role_id, rp in self.roles.items():
            roles_out[str(role_id)] = {
                "name": rp.name,
                "commands": sorted(rp.commands),
                "daily_quota": rp.daily_quota,
                "cooldown_seconds": rp.cooldown_seconds,
                "command_daily_quotas": dict(sorted(rp.command_daily_quotas.items())),
                "command_cooldowns": dict(sorted(rp.command_cooldowns.items())),
            }
        return {
            "roles": roles_out,
            "default": {
                "commands": sorted(self.default.commands),
                "daily_quota": self.default.daily_quota,
                "cooldown_seconds": self.default.cooldown_seconds,
                "command_daily_quotas": dict(sorted(self.default.command_daily_quotas.items())),
                "command_cooldowns": dict(sorted(self.default.command_cooldowns.items())),
            },
        }

    def save(self) -> None:
        if self._path is None:
            raise RuntimeError("Pas de chemin de fichier configuré.")
        self._path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Permissions sauvegardées : %s", self._path)

    # ─── Mutation (utilisé par la commande /permissions) ──────────────

    def set_role(
        self,
        role_id: int,
        name: str,
        commands: set[str],
        daily_quota: int,
        cooldown_seconds: int,
        command_daily_quotas: dict[str, int] | None = None,
        command_cooldowns: dict[str, int] | None = None,
    ) -> None:
        previous = self.roles.get(role_id)
        self.roles[role_id] = RolePermissions(
            name=name,
            commands=commands,
            daily_quota=daily_quota,
            cooldown_seconds=cooldown_seconds,
            command_daily_quotas=(
                command_daily_quotas
                if command_daily_quotas is not None
                else (dict(previous.command_daily_quotas) if previous else {})
            ),
            command_cooldowns=(
                command_cooldowns
                if command_cooldowns is not None
                else (dict(previous.command_cooldowns) if previous else {})
            ),
        )
        self.save()

    def remove_role(self, role_id: int) -> bool:
        if role_id not in self.roles:
            return False
        del self.roles[role_id]
        self.save()
        return True

    def set_default(
        self,
        commands: set[str],
        daily_quota: int,
        cooldown_seconds: int,
        command_daily_quotas: dict[str, int] | None = None,
        command_cooldowns: dict[str, int] | None = None,
    ) -> None:
        self.default = RolePermissions(
            name="default",
            commands=commands,
            daily_quota=daily_quota,
            cooldown_seconds=cooldown_seconds,
            command_daily_quotas=(
                command_daily_quotas
                if command_daily_quotas is not None
                else dict(self.default.command_daily_quotas)
            ),
            command_cooldowns=(
                command_cooldowns
                if command_cooldowns is not None
                else dict(self.default.command_cooldowns)
            ),
        )
        self.save()

    def set_role_command_limits(
        self,
        role_id: int,
        command: str,
        daily_quota: int | None,
        cooldown_seconds: int | None,
    ) -> bool:
        rp = self.roles.get(role_id)
        if rp is None:
            return False
        if daily_quota is not None:
            rp.command_daily_quotas[command] = daily_quota
        if cooldown_seconds is not None:
            rp.command_cooldowns[command] = cooldown_seconds
        self.save()
        return True

    def clear_role_command_limits(self, role_id: int, command: str) -> bool:
        rp = self.roles.get(role_id)
        if rp is None:
            return False
        removed = False
        removed = rp.command_daily_quotas.pop(command, None) is not None or removed
        removed = rp.command_cooldowns.pop(command, None) is not None or removed
        if removed:
            self.save()
        return removed

    def set_default_command_limits(
        self,
        command: str,
        daily_quota: int | None,
        cooldown_seconds: int | None,
    ) -> None:
        if daily_quota is not None:
            self.default.command_daily_quotas[command] = daily_quota
        if cooldown_seconds is not None:
            self.default.command_cooldowns[command] = cooldown_seconds
        self.save()

    def clear_default_command_limits(self, command: str) -> bool:
        removed = False
        removed = self.default.command_daily_quotas.pop(command, None) is not None or removed
        removed = self.default.command_cooldowns.pop(command, None) is not None or removed
        if removed:
            self.save()
        return removed


# ═══════════════════════════════════════════════════════════════════════════════
# Chargement
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_role(data: dict) -> RolePermissions:
    return RolePermissions(
        name=data.get("name", "?"),
        commands=set(data.get("commands", [])),
        daily_quota=data.get("daily_quota", 0),
        cooldown_seconds=data.get("cooldown_seconds", 0),
        command_daily_quotas={
            str(command): int(value)
            for command, value in data.get("command_daily_quotas", {}).items()
        },
        command_cooldowns={
            str(command): int(value)
            for command, value in data.get("command_cooldowns", {}).items()
        },
    )


def load_permissions(path: Path) -> PermissionConfig:
    """Charge les permissions depuis un fichier JSON."""
    if not path.exists():
        logger.warning("Fichier permissions introuvable (%s), création par défaut.", path)
        config = PermissionConfig(_path=path)
        config.save()
        return config

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.exception("Erreur lecture %s, config par défaut.", path)
        return PermissionConfig(_path=path)

    roles: dict[int, RolePermissions] = {}
    for role_id_str, role_data in raw.get("roles", {}).items():
        try:
            roles[int(role_id_str)] = _parse_role(role_data)
        except (ValueError, TypeError):
            logger.warning("Role ID invalide ignoré : %s", role_id_str)

    default_data = raw.get("default", {})
    default = RolePermissions(
        name="default",
        commands=set(default_data.get("commands", sorted(DEFAULT_COMMANDS))),
        daily_quota=default_data.get("daily_quota", DEFAULT_DAILY_QUOTA),
        cooldown_seconds=default_data.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS),
        command_daily_quotas={
            str(command): int(value)
            for command, value in default_data.get(
                "command_daily_quotas", DEFAULT_COMMAND_DAILY_QUOTAS
            ).items()
        },
        command_cooldowns={
            str(command): int(value)
            for command, value in default_data.get(
                "command_cooldowns", DEFAULT_COMMAND_COOLDOWNS
            ).items()
        },
    )

    return PermissionConfig(roles=roles, default=default, _path=path)


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_PERM_CONFIG: PermissionConfig | None = None


def get_permission_config() -> PermissionConfig:
    """Retourne le singleton de configuration des permissions."""
    global _PERM_CONFIG
    if _PERM_CONFIG is None:
        from config import get_config
        cfg = get_config()
        _PERM_CONFIG = load_permissions(cfg.permissions_path)
    return _PERM_CONFIG


def reload_permissions() -> PermissionConfig:
    """Recharge les permissions depuis le fichier JSON."""
    global _PERM_CONFIG
    from config import get_config
    cfg = get_config()
    _PERM_CONFIG = load_permissions(cfg.permissions_path)
    return _PERM_CONFIG


# ═══════════════════════════════════════════════════════════════════════════════
# Décorateur de vérification
# ═══════════════════════════════════════════════════════════════════════════════


async def _reply_error(interaction: discord.Interaction, message: str) -> None:
    embed = error_embed(message)
    try:
        if interaction.response.is_done():
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
            await interaction.response.send_message(embed=embed, ephemeral=True)
    except Exception:
        pass


async def _enforce_limits_and_log(
    interaction: discord.Interaction,
    member: discord.Member,
    command_name: str,
    resolved: ResolvedPermissions,
) -> bool:
    """Applique les limites effectives d'une commande, puis log l'usage."""
    db = await get_db()
    cooldown_seconds = resolved.cooldown_for(command_name)
    daily_quota = resolved.daily_quota_for(command_name)

    if cooldown_seconds > 0:
        last = await last_command_usage(db, member.id, command_name)
        if last is not None:
            elapsed = seconds_since(last)
            if elapsed < cooldown_seconds:
                wait = int(cooldown_seconds - elapsed) + 1
                await _reply_error(
                    interaction,
                    f"Patiente encore {wait}s avant de relancer cette commande.",
                )
                return False

    if daily_quota > 0:
        used = await count_command_usage_today(db, member.id, command_name)
        if used >= daily_quota:
            await _reply_error(
                interaction,
                f"Quota journalier atteint ({daily_quota}/{daily_quota}). Reviens demain !",
            )
            return False

    try:
        await log_command_usage(db, member.id, interaction.guild_id or 0, command_name)
    except Exception:
        logger.exception("Échec d'écriture du log d'usage")

    return True


def check_permissions(command_name: str):
    """Décorateur : vérifie permissions par rôle, quota, cooldown, puis log l'usage.

    Résolution par rôle : l'utilisateur obtient les permissions les plus
    permissives parmi tous ses rôles configurés (union des commandes,
    meilleur quota, meilleur cooldown).
    """

    def decorator(func: Callable[..., Awaitable[None]]):
        @functools.wraps(func)
        async def wrapper(self, interaction: discord.Interaction, *args, **kwargs):
            member = interaction.user
            if not isinstance(member, discord.Member):
                await _reply_error(
                    interaction,
                    "Cette commande doit être utilisée dans un serveur.",
                )
                return

            perm_config = get_permission_config()
            resolved = perm_config.resolve(member)

            if not resolved.can_use(command_name):
                await _reply_error(
                    interaction,
                    "Tu n'as pas la permission d'utiliser cette commande.",
                )
                return

            if not await _enforce_limits_and_log(
                interaction, member, command_name, resolved
            ):
                return

            return await func(self, interaction, *args, **kwargs)

            db = await get_db()

            # ─── Cooldown ─────────────────────────────────────────────
            if resolved.cooldown_seconds > 0:
                last = await last_command_usage(db, member.id, command_name)
                if last is not None:
                    elapsed = seconds_since(last)
                    if elapsed < resolved.cooldown_seconds:
                        wait = int(resolved.cooldown_seconds - elapsed) + 1
                        await _reply_error(
                            interaction,
                            f"Patiente encore {wait}s avant de relancer cette commande.",
                        )
                        return

            # ─── Quota ────────────────────────────────────────────────
            if resolved.daily_quota > 0:
                used = await count_command_usage_today(db, member.id, command_name)
                if used >= resolved.daily_quota:
                    await _reply_error(
                        interaction,
                        f"Quota journalier atteint ({resolved.daily_quota}/{resolved.daily_quota}). Reviens demain !",
                    )
                    return

            # ─── Log usage ────────────────────────────────────────────
            try:
                await log_command_usage(
                    db, member.id, interaction.guild_id or 0, command_name
                )
            except Exception:
                logger.exception("Échec d'écriture du log d'usage")

            return await func(self, interaction, *args, **kwargs)

        return wrapper

    return decorator


async def require_permission(
    interaction: discord.Interaction,
    command_name: str,
    *,
    enforce_limits: bool = True,
) -> discord.Member | None:
    """Version non-décorateur : vérifie la permission et retourne le member ou None.

    Utile quand le décorateur ne peut pas être appliqué directement
    (ex: méthode appelée depuis un callback de bouton).
    Ne vérifie PAS le quota/cooldown — seulement l'accès à la commande.
    """
    member = interaction.user
    if not isinstance(member, discord.Member):
        await _reply_error(
            interaction, "Cette commande doit être utilisée dans un serveur."
        )
        return None

    perm_config = get_permission_config()
    resolved = perm_config.resolve(member)

    if not resolved.can_use(command_name):
        await _reply_error(
            interaction, "Tu n'as pas la permission d'utiliser cette commande."
        )
        return None

    if enforce_limits and not await _enforce_limits_and_log(
        interaction, member, command_name, resolved
    ):
        return None

    return member
