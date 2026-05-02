"""Commande /pfc : tirage aléatoire pierre / feuille / ciseaux, N fois."""
from __future__ import annotations

import random

import discord
from discord import app_commands
from discord.ext import commands

from utils.formatting import GRIMMOR_PURPLE, truncate
from utils.permissions import require_permission

CHOICES = ["pierre", "feuille", "ciseaux"]
EMOJIS = {"pierre": "🪨", "feuille": "📄", "ciseaux": "✂️"}


class PfcCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="pfc",
        description="Tire aléatoirement pierre, feuille ou ciseaux (N fois).",
    )
    @app_commands.describe(tours="Nombre de tirages (1 à 10, défaut 1)")
    async def pfc(
        self,
        interaction: discord.Interaction,
        tours: app_commands.Range[int, 1, 10] = 1,
    ) -> None:
        member = await require_permission(interaction, "pfc")
        if member is None:
            return

        results = [random.choice(CHOICES) for _ in range(tours)]

        embed = discord.Embed(title="Pierre · Feuille · Ciseaux", color=GRIMMOR_PURPLE)

        if tours == 1:
            choice = results[0]
            embed.description = f"{EMOJIS[choice]}  **{choice}**"
        else:
            lines = [
                f"`{i:2d}`  {EMOJIS[c]}  **{c}**"
                for i, c in enumerate(results, start=1)
            ]
            embed.description = truncate("\n".join(lines), 4000)
            counts = {c: results.count(c) for c in CHOICES}
            embed.set_footer(
                text=f"{tours} tirages • "
                f"{EMOJIS['pierre']} {counts['pierre']} · "
                f"{EMOJIS['feuille']} {counts['feuille']} · "
                f"{EMOJIS['ciseaux']} {counts['ciseaux']}"
            )

        await interaction.response.send_message(embed=embed)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(PfcCog(bot))
