"""Commande /permissions : gestion des permissions du bot via Discord."""
from __future__ import annotations

import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from utils.formatting import GRIMMOR_PURPLE, error_embed, success_embed, truncate
from utils.permissions import get_permission_config, reload_permissions

logger = logging.getLogger(__name__)

ALL_COMMANDS = [
    "ask", "pfc", "inactifs", "inactifs_ignore",
    "archive", "ping", "grimmor_info", "grimmor_resync", "permissions",
]
COMMAND_CHOICES = [
    app_commands.Choice(name=f"/{command}", value=command)
    for command in ALL_COMMANDS
]


def _fmt_commands(cmds: set[str]) -> str:
    if "*" in cmds:
        return "`*` (toutes)"
    return ", ".join(f"`{c}`" for c in sorted(cmds)) or "_aucune_"


def _fmt_quota(q: int) -> str:
    return "illimité" if q == 0 else str(q)


def _fmt_cooldown(c: int) -> str:
    return "aucun" if c == 0 else f"{c}s"


def _fmt_command_limits(rp) -> str:
    commands = sorted(set(rp.command_daily_quotas) | set(rp.command_cooldowns))
    if not commands:
        return ""
    lines = ["Limites par commande :"]
    for command in commands:
        quota = rp.command_daily_quotas.get(command, rp.daily_quota)
        cooldown = rp.command_cooldowns.get(command, rp.cooldown_seconds)
        lines.append(
            f"- `/{command}` : quota {_fmt_quota(quota)}/jour, cooldown {_fmt_cooldown(cooldown)}"
        )
    return "\n" + "\n".join(lines)


class PermissionsCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="permissions",
        description="Gère les permissions du bot par rôle.",
    )
    @app_commands.describe(
        action="Action à effectuer",
        role="Rôle Discord à configurer",
        commandes="Commandes autorisées (séparées par virgules, * = toutes)",
        quota="Quota journalier (0 = illimité)",
        cooldown="Cooldown en secondes (0 = aucun)",
        commande_limitee="Commande précise à limiter, ex: /ask",
    )
    @app_commands.choices(
        action=[
            app_commands.Choice(name="Voir les permissions", value="list"),
            app_commands.Choice(name="Configurer un rôle", value="set_role"),
            app_commands.Choice(name="Retirer un rôle", value="remove_role"),
            app_commands.Choice(name="Configurer le défaut", value="set_default"),
            app_commands.Choice(name="Limiter commande d'un rôle", value="set_role_command"),
            app_commands.Choice(name="Retirer limite commande d'un rôle", value="clear_role_command"),
            app_commands.Choice(name="Limiter commande par défaut", value="set_default_command"),
            app_commands.Choice(name="Retirer limite commande par défaut", value="clear_default_command"),
            app_commands.Choice(name="Recharger depuis le fichier", value="reload"),
        ],
        commande_limitee=COMMAND_CHOICES,
    )
    async def permissions(
        self,
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
        role: Optional[discord.Role] = None,
        commandes: Optional[str] = None,
        quota: Optional[int] = None,
        cooldown: Optional[int] = None,
        commande_limitee: Optional[app_commands.Choice[str]] = None,
    ) -> None:
        member = interaction.user
        if not isinstance(member, discord.Member):
            await interaction.response.send_message(
                embed=error_embed("Cette commande doit être utilisée dans un serveur."),
                ephemeral=True,
            )
            return

        # Seuls les administrateurs Discord peuvent gérer les permissions
        perm_config = get_permission_config()
        can_manage_permissions = (
            member.guild_permissions.administrator
            or perm_config.resolve(member).can_use("permissions")
        )

        if not can_manage_permissions:
            denied_message = (
                "Seuls les administrateurs du serveur ou les roles autorises "
                "peuvent gerer les permissions."
            )
            await interaction.response.send_message(
                embed=error_embed(
                    "Seuls les **administrateurs du serveur** peuvent gérer les permissions."
                ),
                ephemeral=True,
            )
            return

        act = action.value

        # ─── Voir ────────────────────────────────────────────────────
        if act == "list":
            await self._list_permissions(interaction)
            return

        # ─── Recharger ───────────────────────────────────────────────
        if act == "reload":
            reload_permissions()
            await interaction.response.send_message(
                embed=success_embed(
                    "Permissions rechargées depuis le fichier.",
                    title="Rechargé",
                ),
                ephemeral=True,
            )
            return

        # ─── Configurer un rôle ──────────────────────────────────────
        if act == "set_role":
            if role is None:
                await interaction.response.send_message(
                    embed=error_embed("Choisis un rôle avec le champ `role`."),
                    ephemeral=True,
                )
                return
            if commandes is None:
                await interaction.response.send_message(
                    embed=error_embed(
                        "Précise les commandes autorisées avec le champ `commandes`.\n"
                        f"Commandes disponibles : {', '.join(f'`{c}`' for c in ALL_COMMANDS)}, `*`"
                    ),
                    ephemeral=True,
                )
                return

            cmd_set = {c.strip().lower() for c in commandes.split(",") if c.strip()}
            role_quota = quota if quota is not None else 0
            role_cooldown = cooldown if cooldown is not None else 0

            perm_config = get_permission_config()
            perm_config.set_role(
                role_id=role.id,
                name=role.name,
                commands=cmd_set,
                daily_quota=role_quota,
                cooldown_seconds=role_cooldown,
            )

            await interaction.response.send_message(
                embed=success_embed(
                    f"**{role.mention}** configuré :\n"
                    f"Commandes : {_fmt_commands(cmd_set)}\n"
                    f"Quota : {_fmt_quota(role_quota)}/jour\n"
                    f"Cooldown : {_fmt_cooldown(role_cooldown)}"
                    f"{_fmt_command_limits(perm_config.roles[role.id])}",
                    title="Rôle configuré",
                ),
                ephemeral=True,
            )
            return

        # ─── Retirer un rôle ─────────────────────────────────────────
        if act == "remove_role":
            if role is None:
                await interaction.response.send_message(
                    embed=error_embed("Choisis un rôle avec le champ `role`."),
                    ephemeral=True,
                )
                return

            perm_config = get_permission_config()
            removed = perm_config.remove_role(role.id)
            if removed:
                await interaction.response.send_message(
                    embed=success_embed(
                        f"{role.mention} retiré des permissions. "
                        f"Les membres avec ce rôle utiliseront les permissions par défaut.",
                        title="Rôle retiré",
                    ),
                    ephemeral=True,
                )
            else:
                await interaction.response.send_message(
                    embed=error_embed(f"{role.mention} n'était pas configuré."),
                    ephemeral=True,
                )
            return

        # ─── Configurer le défaut ────────────────────────────────────
        if act in {"set_role_command", "clear_role_command"}:
            if role is None:
                await interaction.response.send_message(
                    embed=error_embed("Choisis un rôle avec le champ `role`."),
                    ephemeral=True,
                )
                return
            if commande_limitee is None:
                await interaction.response.send_message(
                    embed=error_embed("Choisis une commande avec le champ `commande_limitee`."),
                    ephemeral=True,
                )
                return

            perm_config = get_permission_config()
            command_name = commande_limitee.value

            if act == "clear_role_command":
                removed = perm_config.clear_role_command_limits(role.id, command_name)
                message = (
                    f"Limite spécifique retirée pour `/{command_name}` sur {role.mention}."
                    if removed
                    else f"Aucune limite spécifique `/{command_name}` n'était configurée sur {role.mention}."
                )
                await interaction.response.send_message(
                    embed=success_embed(message, title="Limite retirée"),
                    ephemeral=True,
                )
                return

            role_quota = quota if quota is not None else 0
            role_cooldown = cooldown if cooldown is not None else 0
            updated = perm_config.set_role_command_limits(
                role.id,
                command_name,
                daily_quota=role_quota,
                cooldown_seconds=role_cooldown,
            )
            if not updated:
                await interaction.response.send_message(
                    embed=error_embed(
                        f"{role.mention} n'est pas encore configuré. "
                        "Configure d'abord le rôle avec `Configurer un rôle`."
                    ),
                    ephemeral=True,
                )
                return
            await interaction.response.send_message(
                embed=success_embed(
                    f"Limite `/{command_name}` pour {role.mention} :\n"
                    f"Quota : {_fmt_quota(role_quota)}/jour\n"
                    f"Cooldown : {_fmt_cooldown(role_cooldown)}",
                    title="Limite configurée",
                ),
                ephemeral=True,
            )
            return

        if act in {"set_default_command", "clear_default_command"}:
            if commande_limitee is None:
                await interaction.response.send_message(
                    embed=error_embed("Choisis une commande avec le champ `commande_limitee`."),
                    ephemeral=True,
                )
                return

            perm_config = get_permission_config()
            command_name = commande_limitee.value

            if act == "clear_default_command":
                removed = perm_config.clear_default_command_limits(command_name)
                message = (
                    f"Limite spécifique par défaut retirée pour `/{command_name}`."
                    if removed
                    else f"Aucune limite spécifique par défaut n'était configurée pour `/{command_name}`."
                )
                await interaction.response.send_message(
                    embed=success_embed(message, title="Limite retirée"),
                    ephemeral=True,
                )
                return

            def_quota = quota if quota is not None else 0
            def_cooldown = cooldown if cooldown is not None else 0
            perm_config.set_default_command_limits(
                command_name,
                daily_quota=def_quota,
                cooldown_seconds=def_cooldown,
            )
            await interaction.response.send_message(
                embed=success_embed(
                    f"Limite par défaut `/{command_name}` :\n"
                    f"Quota : {_fmt_quota(def_quota)}/jour\n"
                    f"Cooldown : {_fmt_cooldown(def_cooldown)}",
                    title="Limite configurée",
                ),
                ephemeral=True,
            )
            return

        if act == "set_default":
            if commandes is None:
                await interaction.response.send_message(
                    embed=error_embed(
                        "Précise les commandes par défaut avec le champ `commandes`.\n"
                        f"Commandes disponibles : {', '.join(f'`{c}`' for c in ALL_COMMANDS)}, `*`"
                    ),
                    ephemeral=True,
                )
                return

            cmd_set = {c.strip().lower() for c in commandes.split(",") if c.strip()}
            def_quota = quota if quota is not None else 5
            def_cooldown = cooldown if cooldown is not None else 60

            perm_config = get_permission_config()
            perm_config.set_default(cmd_set, def_quota, def_cooldown)

            await interaction.response.send_message(
                embed=success_embed(
                    f"Permissions par défaut mises à jour :\n"
                    f"Commandes : {_fmt_commands(cmd_set)}\n"
                    f"Quota : {_fmt_quota(def_quota)}/jour\n"
                    f"Cooldown : {_fmt_cooldown(def_cooldown)}"
                    f"{_fmt_command_limits(perm_config.default)}",
                    title="Défaut configuré",
                ),
                ephemeral=True,
            )

    async def _list_permissions(self, interaction: discord.Interaction) -> None:
        perm_config = get_permission_config()
        guild = interaction.guild

        embed = discord.Embed(
            title="Permissions du bot",
            color=GRIMMOR_PURPLE,
        )

        # Défaut
        d = perm_config.default
        embed.add_field(
            name="📋 Par défaut (aucun rôle configuré)",
            value=(
                f"Commandes : {_fmt_commands(d.commands)}\n"
                f"Quota : {_fmt_quota(d.daily_quota)}/jour\n"
                f"Cooldown : {_fmt_cooldown(d.cooldown_seconds)}"
                f"{_fmt_command_limits(d)}"
            ),
            inline=False,
        )

        # Rôles configurés
        if not perm_config.roles:
            embed.add_field(
                name="Rôles",
                value="_Aucun rôle configuré. Utilise `/permissions action:Configurer un rôle`._",
                inline=False,
            )
        else:
            for role_id, rp in perm_config.roles.items():
                role_obj = guild.get_role(role_id) if guild else None
                role_label = role_obj.mention if role_obj else f"`{rp.name}` ({role_id}, introuvable)"
                embed.add_field(
                    name=truncate(f"🔑 {rp.name}", 256),
                    value=truncate(
                        f"Rôle : {role_label}\n"
                        f"Commandes : {_fmt_commands(rp.commands)}\n"
                        f"Quota : {_fmt_quota(rp.daily_quota)}/jour\n"
                        f"Cooldown : {_fmt_cooldown(rp.cooldown_seconds)}"
                        f"{_fmt_command_limits(rp)}",
                        1024,
                    ),
                    inline=False,
                )

        embed.set_footer(
            text="Les administrateurs Discord peuvent toujours utiliser /permissions."
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(PermissionsCog(bot))
