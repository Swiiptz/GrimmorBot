"""Commande /archive : archivage d'un channel Discord en HTML."""
from __future__ import annotations

import logging
import shutil

import discord
from discord import app_commands
from discord.ext import commands

from utils.archiver import archive_channel
from utils.formatting import error_embed
from utils.permissions import require_permission

logger = logging.getLogger(__name__)


class ArchiveCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="archive",
        description="Archive ce salon ou ce fil en un fichier HTML consultable.",
    )
    async def archive(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)

        member = await require_permission(interaction, "archive")
        if member is None:
            return

        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            await interaction.followup.send(
                embed=error_embed(
                    "Cette commande fonctionne dans un salon textuel ou dans un fil."
                ),
                ephemeral=True,
            )
            return

        try:
            filepath, total = await archive_channel(channel, member, save_to_disk=False)
        except Exception:
            logger.exception("Échec archivage de #%s (%s)", channel.name, channel.id)
            await interaction.followup.send(
                embed=error_embed(
                    "Une erreur est survenue lors de l'archivage. Les sorciers ont été prévenus."
                ),
            )
            return

        # Envoi du fichier seul (pas d'embed, pas d'aperçu)
        try:
            file = discord.File(str(filepath), filename=filepath.name)
            await interaction.followup.send(file=file)
        except discord.HTTPException as exc:
            if exc.status == 413:
                await interaction.followup.send(
                    embed=error_embed(
                        "Le fichier d'archive est trop volumineux pour Discord."
                    ),
                )
            else:
                raise
        finally:
            # Nettoyage du fichier temporaire
            try:
                shutil.rmtree(filepath.parent)
            except Exception:
                pass


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ArchiveCog(bot))
