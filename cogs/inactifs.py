"""Commande /inactifs : revue interactive des salons sans activité récente."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from db import get_db
from db.queries import (
    add_inactive_exclusion,
    get_all_channel_activity,
    get_inactive_exclusions,
    remove_inactive_exclusion,
    upsert_channel_activity,
)
from utils.archiver import archive_channel
from utils.formatting import (
    GRIMMOR_PURPLE,
    error_embed,
    format_duration,
    success_embed,
    truncate,
)
from utils.permissions import require_permission

logger = logging.getLogger(__name__)

PAGE_SIZE = 10
VIEW_TIMEOUT_SECONDS = 600.0
DEFAULT_DELETION_CATEGORY = "À supprimer"
DELETION_NOTICE = "\n**Ce salon est susceptible d'être supprimé dans les mois à venir. Merci.**"


@dataclass(frozen=True)
class InactiveChannel:
    channel: discord.TextChannel
    last_activity: Optional[datetime]


def _parse_csv(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _ensure_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


async def _resolve_last_activity(
    channel: discord.TextChannel, db_data: dict[int, datetime]
) -> Optional[datetime]:
    """Tente d'obtenir la date du dernier message d'un salon.

    Priorité : SQLite (rapide) -> API Discord (fallback : 1 message historique).
    """
    cached = db_data.get(channel.id)
    if cached is not None:
        return _ensure_utc(cached)
    try:
        async for msg in channel.history(limit=1):
            return _ensure_utc(msg.created_at)
    except discord.Forbidden:
        return None
    except Exception:
        logger.exception("Erreur historique salon %s", channel.id)
        return None
    return None


def _category_name(channel: discord.TextChannel) -> str:
    return channel.category.name if channel.category else "_sans catégorie_"


def _last_activity_label(item: InactiveChannel) -> str:
    if item.last_activity is None:
        return "_jamais (aucun message connu)_"
    return f"<t:{int(item.last_activity.timestamp())}:f>"


def _inactive_duration(item: InactiveChannel, now: datetime) -> str:
    if item.last_activity is None:
        created_at = _ensure_utc(item.channel.created_at)
        return format_duration((now - created_at).total_seconds())
    return format_duration((now - item.last_activity).total_seconds())


def _candidate_sort_key(item: InactiveChannel) -> tuple[int, datetime]:
    if item.last_activity is None:
        return (0, _ensure_utc(item.channel.created_at))
    return (1, item.last_activity)


class InactiveChannelSelect(discord.ui.Select):
    def __init__(self, parent_view: "InactifsView"):
        self.parent_view = parent_view
        options: list[discord.SelectOption] = []
        now = datetime.now(timezone.utc)
        for item in parent_view.current_items:
            channel = item.channel
            options.append(
                discord.SelectOption(
                    label=truncate(f"#{channel.name}", 100),
                    description=truncate(
                        f"{_category_name(channel)} - inactif {_inactive_duration(item, now)}",
                        100,
                    ),
                    value=str(channel.id),
                )
            )

        super().__init__(
            placeholder="Choisis les salons à traiter sur cette page",
            min_values=1,
            max_values=max(1, len(options)),
            options=options,
            row=0,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        self.parent_view.selected_ids = {int(value) for value in self.values}
        await interaction.response.defer()


class InactifsView(discord.ui.View):
    def __init__(
        self,
        owner_id: int,
        guild: discord.Guild,
        items: list[InactiveChannel],
        jours: int,
        deletion_category_name: str,
        *,
        auto_archive: bool = False,
        archive_user: Optional[discord.Member] = None,
    ):
        super().__init__(timeout=VIEW_TIMEOUT_SECONDS)
        self.owner_id = owner_id
        self.guild = guild
        self.items = items
        self.jours = jours
        self.deletion_category_name = deletion_category_name
        self.auto_archive = auto_archive
        self.archive_user = archive_user
        self.index = 0
        self.selected_ids: set[int] = set()
        self.notice: str | None = None
        self._sync_components()

    @property
    def page_count(self) -> int:
        return max(1, (len(self.items) + PAGE_SIZE - 1) // PAGE_SIZE)

    @property
    def current_items(self) -> list[InactiveChannel]:
        start = self.index * PAGE_SIZE
        return self.items[start : start + PAGE_SIZE]

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.followup.send(
                "Seul l'auteur de la commande peut utiliser cette revue.", ephemeral=True
            )
            return False
        return True

    def _clamp_index(self) -> None:
        self.index = min(max(self.index, 0), self.page_count - 1)

    def _selected_current_items(self) -> list[InactiveChannel]:
        current_by_id = {item.channel.id: item for item in self.current_items}
        return [
            current_by_id[channel_id]
            for channel_id in self.selected_ids
            if channel_id in current_by_id
        ]

    def _sync_components(self) -> None:
        self.clear_items()
        self._clamp_index()

        if self.current_items:
            self.add_item(InactiveChannelSelect(self))

        previous_button = discord.ui.Button(
            label="Précédent",
            style=discord.ButtonStyle.secondary,
            disabled=self.index <= 0,
            row=1,
        )
        previous_button.callback = self._previous_page
        self.add_item(previous_button)

        next_button = discord.ui.Button(
            label="Suivant",
            style=discord.ButtonStyle.secondary,
            disabled=self.index >= self.page_count - 1,
            row=1,
        )
        next_button.callback = self._next_page
        self.add_item(next_button)

        keep_button = discord.ui.Button(
            label="Garder sélection",
            style=discord.ButtonStyle.success,
            row=2,
        )
        keep_button.callback = self._keep_selected
        self.add_item(keep_button)

        delete_button = discord.ui.Button(
            label="À supprimer sélection",
            style=discord.ButtonStyle.danger,
            row=2,
        )
        delete_button.callback = self._move_selected_to_deletion_category
        self.add_item(delete_button)

        quit_button = discord.ui.Button(
            label="Quitter",
            style=discord.ButtonStyle.secondary,
            row=2,
        )
        quit_button.callback = self._quit_review
        self.add_item(quit_button)

    def build_embed(self) -> discord.Embed:
        total = len(self.items)
        embed = discord.Embed(
            title=f"Salons inactifs (>= {self.jours} jours)",
            color=GRIMMOR_PURPLE,
            description=(
                f"Page {self.index + 1}/{self.page_count} - {total} salon(s) restant(s).\n"
                "Les salons sont cliquables. Sélectionne ceux de cette page, puis choisis quoi faire."
            ),
        )
        if self.notice:
            embed.description += f"\n\n**Dernière action :** {self.notice}"

        now = datetime.now(timezone.utc)
        for item in self.current_items:
            channel = item.channel
            value = (
                f"Salon : {channel.mention}\n"
                f"Catégorie : {_category_name(channel)}\n"
                f"Dernier message : {_last_activity_label(item)}\n"
                f"Inactif depuis : **{_inactive_duration(item, now)}**"
            )
            embed.add_field(
                name=truncate(f"#{channel.name}", 256),
                value=truncate(value, 1024),
                inline=False,
            )

        embed.set_footer(
            text=(
                f"Action 'Garder' = retire de cette revue. "
                f"Action 'À supprimer' = déplace vers '{self.deletion_category_name}'."
            )
        )
        return embed

    async def _previous_page(self, interaction: discord.Interaction) -> None:
        if self.index > 0:
            self.index -= 1
        self.selected_ids.clear()
        self._sync_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def _next_page(self, interaction: discord.Interaction) -> None:
        if self.index < self.page_count - 1:
            self.index += 1
        self.selected_ids.clear()
        self._sync_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def _quit_review(self, interaction: discord.Interaction) -> None:
        self.stop()
        self.clear_items()
        embed = self.build_embed()
        embed.description += "\n\n**Revue fermée.** Tu peux relancer `/inactifs` quand tu veux."
        await interaction.response.edit_message(embed=embed, view=None)

    async def _keep_selected(self, interaction: discord.Interaction) -> None:
        selected = self._selected_current_items()
        if not selected:
            await interaction.followup.send(
                "Sélectionne d'abord un ou plusieurs salons sur cette page.", ephemeral=True
            )
            return

        selected_ids = {item.channel.id for item in selected}
        self.items = [item for item in self.items if item.channel.id not in selected_ids]
        self.selected_ids.clear()

        names = ", ".join(item.channel.mention for item in selected[:5])
        if len(selected) > 5:
            names += f", +{len(selected) - 5}"
        self.notice = f"{len(selected)} salon(s) gardé(s) : {names}"

        if not self.items:
            self.clear_items()
            await interaction.response.edit_message(
                embed=success_embed(
                    "Tous les salons de cette revue ont été traités.",
                    title="Revue terminée",
                ),
                view=None,
            )
            return

        self._sync_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def _move_selected_to_deletion_category(
        self, interaction: discord.Interaction
    ) -> None:
        selected = self._selected_current_items()
        if not selected:
            await interaction.followup.send(
                "Sélectionne d'abord un ou plusieurs salons sur cette page.", ephemeral=True
            )
            return

        bot_member = self._get_bot_member(interaction)
        if bot_member is not None and not bot_member.guild_permissions.manage_channels:
            await interaction.followup.send(
                embed=error_embed(
                    "Il me manque la permission **Gérer les salons** pour déplacer les salons."
                ),
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        try:
            category = await self._ensure_deletion_category()
        except discord.Forbidden:
            await interaction.followup.send(
                embed=error_embed(
                    "Je n'ai pas la permission de créer la catégorie de suppression."
                ),
                ephemeral=True,
            )
            return
        except Exception:
            logger.exception("Échec création catégorie suppression")
            await interaction.followup.send(
                embed=error_embed("Impossible de créer ou trouver la catégorie de suppression."),
                ephemeral=True,
            )
            return

        moved: list[InactiveChannel] = []
        archived: list[str] = []
        failed: list[str] = []
        for item in selected:
            channel = item.channel

            # Auto-archivage avant déplacement
            if self.auto_archive and self.archive_user is not None:
                try:
                    filepath, msg_count = await archive_channel(
                        channel, self.archive_user
                    )
                    archive_file = discord.File(str(filepath), filename=filepath.name)
                    await channel.send(
                        f"📋 **Archive automatique** — {msg_count:,} messages archivés "
                        f"par {self.archive_user.mention}.",
                        file=archive_file,
                    )
                    archived.append(channel.mention)
                except Exception:
                    logger.exception("Échec archivage auto salon %s", channel.id)
                    failed.append(f"{channel.mention} (archivage échoué)")

            try:
                await channel.edit(
                    category=category,
                    sync_permissions=False,
                    reason="Salon marqué comme susceptible d'être supprimé via /inactifs.",
                )
            except discord.Forbidden:
                failed.append(f"{channel.mention} (déplacement refusé)")
                continue
            except Exception:
                logger.exception("Échec déplacement salon inactif %s", channel.id)
                failed.append(f"{channel.mention} (erreur déplacement)")
                continue

            moved.append(item)
            try:
                await channel.send(DELETION_NOTICE)
            except discord.Forbidden:
                failed.append(f"{channel.mention} (déplacé, message non envoyé)")
            except Exception:
                logger.exception("Échec message suppression salon %s", channel.id)
                failed.append(f"{channel.mention} (déplacé, erreur message)")

        if moved:
            moved_ids = {item.channel.id for item in moved}
            self.items = [item for item in self.items if item.channel.id not in moved_ids]
        self.selected_ids.clear()

        moved_names = ", ".join(item.channel.mention for item in moved[:5])
        if len(moved) > 5:
            moved_names += f", +{len(moved) - 5}"
        pieces: list[str] = []
        if moved:
            pieces.append(f"{len(moved)} salon(s) déplacé(s) : {moved_names}")
        if archived:
            pieces.append(f"{len(archived)} salon(s) archivé(s)")
        if failed:
            pieces.append("Échecs : " + truncate(", ".join(failed), 500))
        self.notice = " | ".join(pieces) if pieces else "Aucun salon déplacé."

        if not self.items:
            self.clear_items()
            await interaction.edit_original_response(
                embed=success_embed(
                    "Tous les salons de cette revue ont été traités.",
                    title="Revue terminée",
                ),
                view=None,
            )
            return

        self._sync_components()
        await interaction.edit_original_response(embed=self.build_embed(), view=self)

    def _get_bot_member(self, interaction: discord.Interaction) -> discord.Member | None:
        bot_user = interaction.client.user
        if bot_user is None:
            return None
        return self.guild.get_member(bot_user.id)

    async def _ensure_deletion_category(self) -> discord.CategoryChannel:
        existing = discord.utils.get(self.guild.categories, name=self.deletion_category_name)
        if existing is not None:
            return existing
        return await self.guild.create_category(
            name=self.deletion_category_name,
            reason="Création automatique via /inactifs.",
        )


class InactifsCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @staticmethod
    def _exclusion_label(guild: discord.Guild, exclusion: dict[str, str]) -> str:
        target_id = int(exclusion["target_id"])
        target = guild.get_channel(target_id)
        if target is not None:
            return target.mention
        return f"{exclusion['target_name']} (`{target_id}`, introuvable)"

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.guild is None:
            return
        if not isinstance(message.channel, discord.TextChannel):
            return
        try:
            db = await get_db()
            await upsert_channel_activity(
                db,
                channel_id=message.channel.id,
                guild_id=message.guild.id,
                when=message.created_at or datetime.now(timezone.utc),
            )
        except Exception:
            logger.exception("Échec tracking activité salon %s", message.channel.id)

    @app_commands.command(
        name="inactifs_ignore",
        description="Configure les catégories/salons ignorés par défaut par /inactifs.",
    )
    @app_commands.describe(
        action="Ce que tu veux faire",
        categorie="Catégorie à ajouter ou retirer des exclusions",
        salon="Salon à ajouter ou retirer des exclusions",
    )
    @app_commands.choices(
        action=[
            app_commands.Choice(name="Lister", value="list"),
            app_commands.Choice(name="Ajouter catégorie", value="add_category"),
            app_commands.Choice(name="Retirer catégorie", value="remove_category"),
            app_commands.Choice(name="Ajouter salon", value="add_channel"),
            app_commands.Choice(name="Retirer salon", value="remove_channel"),
        ]
    )
    async def inactifs_ignore(
        self,
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
        categorie: Optional[discord.CategoryChannel] = None,
        salon: Optional[discord.TextChannel] = None,
    ) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)

        member = await require_permission(interaction, "inactifs_ignore")
        if member is None:
            return

        guild = interaction.guild
        if guild is None:
            await interaction.followup.send(
                embed=error_embed("Serveur introuvable."),
                ephemeral=True,
            )
            return

        try:
            db = await get_db()
        except Exception:
            logger.exception("Échec accès SQLite pour exclusions inactifs")
            await interaction.followup.send(
                embed=error_embed("Impossible d'accéder à la configuration des exclusions."),
                ephemeral=True,
            )
            return

        action_value = action.value
        if action_value == "list":
            exclusions = await get_inactive_exclusions(db, guild.id)
            categories = [
                self._exclusion_label(guild, item)
                for item in exclusions
                if item["target_type"] == "category"
            ]
            channels = [
                self._exclusion_label(guild, item)
                for item in exclusions
                if item["target_type"] == "channel"
            ]
            embed = discord.Embed(
                title="Exclusions /inactifs",
                color=GRIMMOR_PURPLE,
                description="Ces éléments sont ignorés automatiquement à chaque `/inactifs`.",
            )
            embed.add_field(
                name="Catégories ignorées",
                value=truncate("\n".join(categories) if categories else "_aucune_", 1024),
                inline=False,
            )
            embed.add_field(
                name="Salons ignorés",
                value=truncate("\n".join(channels) if channels else "_aucun_", 1024),
                inline=False,
            )
            await interaction.followup.send(embed=embed, ephemeral=True)
            return

        if action_value in {"add_category", "remove_category"}:
            if categorie is None:
                await interaction.followup.send(
                    embed=error_embed("Choisis une catégorie avec le champ `categorie`."),
                    ephemeral=True,
                )
                return
            if action_value == "add_category":
                await add_inactive_exclusion(
                    db,
                    guild_id=guild.id,
                    target_type="category",
                    target_id=categorie.id,
                    target_name=categorie.name,
                )
                await interaction.followup.send(
                    embed=success_embed(
                        f"{categorie.mention} sera ignorée par défaut dans `/inactifs`.",
                        title="Catégorie ajoutée",
                    ),
                    ephemeral=True,
                )
                return

            removed = await remove_inactive_exclusion(
                db,
                guild_id=guild.id,
                target_type="category",
                target_id=categorie.id,
            )
            message = (
                f"{categorie.mention} n'est plus ignorée par défaut."
                if removed
                else f"{categorie.mention} n'était pas dans les exclusions."
            )
            await interaction.followup.send(
                embed=success_embed(message, title="Catégorie retirée"),
                ephemeral=True,
            )
            return

        if action_value in {"add_channel", "remove_channel"}:
            if salon is None:
                await interaction.followup.send(
                    embed=error_embed("Choisis un salon avec le champ `salon`."),
                    ephemeral=True,
                )
                return
            if action_value == "add_channel":
                await add_inactive_exclusion(
                    db,
                    guild_id=guild.id,
                    target_type="channel",
                    target_id=salon.id,
                    target_name=salon.name,
                )
                await interaction.followup.send(
                    embed=success_embed(
                        f"{salon.mention} sera ignoré par défaut dans `/inactifs`.",
                        title="Salon ajouté",
                    ),
                    ephemeral=True,
                )
                return

            removed = await remove_inactive_exclusion(
                db,
                guild_id=guild.id,
                target_type="channel",
                target_id=salon.id,
            )
            message = (
                f"{salon.mention} n'est plus ignoré par défaut."
                if removed
                else f"{salon.mention} n'était pas dans les exclusions."
            )
            await interaction.followup.send(
                embed=success_embed(message, title="Salon retiré"),
                ephemeral=True,
            )

    @app_commands.command(
        name="inactifs",
        description="Liste les salons sans activité récente.",
    )
    @app_commands.describe(
        jours="Seuil d'inactivité en jours (défaut 30, max 730)",
        exclure_categories="Noms de catégories à ignorer, séparés par des virgules",
        exclure_salons="Noms ou IDs de salons à ignorer, séparés par des virgules",
        categorie_suppression="Catégorie où déplacer les salons non gardés (défaut : À supprimer)",
        archiver="Archiver automatiquement les salons déplacés vers 'À supprimer' (fichier HTML)",
    )
    async def inactifs(
        self,
        interaction: discord.Interaction,
        jours: app_commands.Range[int, 1, 730] = 30,
        exclure_categories: Optional[str] = None,
        exclure_salons: Optional[str] = None,
        categorie_suppression: Optional[str] = DEFAULT_DELETION_CATEGORY,
        archiver: bool = False,
    ) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)

        member = await require_permission(interaction, "inactifs")
        if member is None:
            return

        deletion_category_name = (categorie_suppression or DEFAULT_DELETION_CATEGORY).strip()
        if not deletion_category_name:
            deletion_category_name = DEFAULT_DELETION_CATEGORY
        if len(deletion_category_name) > 100:
            await interaction.followup.send(
                embed=error_embed("Le nom de catégorie doit faire 100 caractères maximum."),
                ephemeral=True,
            )
            return

        guild = interaction.guild
        if guild is None:
            await interaction.followup.send(
                embed=error_embed("Serveur introuvable."), ephemeral=True
            )
            return

        excl_cats = {category.lower() for category in _parse_csv(exclure_categories)}
        excl_cats.add(deletion_category_name.lower())
        excl_chans_raw = _parse_csv(exclure_salons)
        excl_chans_names = {channel.lower() for channel in excl_chans_raw if not channel.isdigit()}
        excl_chans_ids = {int(channel) for channel in excl_chans_raw if channel.isdigit()}

        try:
            db = await get_db()
            db_data = await get_all_channel_activity(db, guild.id)
            persistent_exclusions = await get_inactive_exclusions(db, guild.id)
        except Exception:
            logger.exception("Échec lecture activité salon")
            db_data = {}
            persistent_exclusions = []

        threshold = datetime.now(timezone.utc) - timedelta(days=jours)
        persistent_category_ids = {
            int(item["target_id"])
            for item in persistent_exclusions
            if item["target_type"] == "category"
        }
        persistent_channel_ids = {
            int(item["target_id"])
            for item in persistent_exclusions
            if item["target_type"] == "channel"
        }

        candidates: list[InactiveChannel] = []
        for channel in guild.text_channels:
            if channel.id in persistent_channel_ids:
                continue
            if channel.id in excl_chans_ids:
                continue
            if channel.name.lower() in excl_chans_names:
                continue
            category = channel.category
            if category is not None and category.id in persistent_category_ids:
                continue
            if category is not None and category.name.lower() in excl_cats:
                continue

            last = await _resolve_last_activity(channel, db_data)
            if last is None:
                created_at = _ensure_utc(channel.created_at)
                if created_at > threshold:
                    continue
                candidates.append(InactiveChannel(channel=channel, last_activity=None))
            elif last <= threshold:
                candidates.append(InactiveChannel(channel=channel, last_activity=last))

        candidates.sort(key=_candidate_sort_key)

        if not candidates:
            await interaction.followup.send(
                embed=success_embed(
                    f"Aucun salon inactif depuis plus de {jours} jours. Bravo !",
                    title="Tout est vivant",
                ),
                ephemeral=True,
            )
            return

        view = InactifsView(
            owner_id=member.id,
            guild=guild,
            items=candidates,
            jours=jours,
            deletion_category_name=deletion_category_name,
            auto_archive=archiver,
            archive_user=member,
        )
        await interaction.followup.send(embed=view.build_embed(), view=view, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(InactifsCog(bot))
