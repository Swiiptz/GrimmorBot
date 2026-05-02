"""Commandes admin diverses : ping, info RAG, resync."""
from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from config import get_config
from rag.retriever import get_retriever
from utils.formatting import GRIMMOR_PURPLE, error_embed, success_embed
from utils.permissions import require_permission

logger = logging.getLogger(__name__)


class AdminCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="ping", description="Vérifie que Grimmor est en vie.")
    async def ping(self, interaction: discord.Interaction) -> None:
        member = await require_permission(interaction, "ping")
        if member is None:
            return
        latency_ms = int(self.bot.latency * 1000)
        await interaction.response.send_message(
            f"Pong. Latence WebSocket : **{latency_ms} ms**.", ephemeral=True
        )

    @app_commands.command(
        name="grimmor_info",
        description="État du système RAG (chunks indexés, modèle, etc.).",
    )
    async def grimmor_info(self, interaction: discord.Interaction) -> None:
        member = await require_permission(interaction, "grimmor_info")
        if member is None:
            return
        cfg = get_config()
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            retriever = await get_retriever()
            chroma_count = retriever._collection.count() if retriever._collection else 0
            bm25_count = len(retriever._bm25.chunk_ids) if retriever._bm25 else 0
        except Exception:
            logger.exception("État RAG indisponible")
            await interaction.followup.send(
                embed=error_embed("Système RAG non initialisé."), ephemeral=True
            )
            return

        embed = discord.Embed(title="Grimmor — État RAG", color=GRIMMOR_PURPLE)
        embed.add_field(name="Chunks ChromaDB", value=str(chroma_count))
        embed.add_field(name="Chunks BM25", value=str(bm25_count))
        embed.add_field(
            name="Top-k (dense / sparse / final)",
            value=f"{cfg.rag_top_k_dense} / {cfg.rag_top_k_sparse} / {cfg.rag_top_k_final}",
            inline=False,
        )
        embed.add_field(
            name="Seuil RRF",
            value=f"{cfg.rag_min_rrf_score} (k = {cfg.rag_rrf_k})",
            inline=False,
        )
        embed.add_field(name="Modèle LLM", value="gemini-1.5-flash", inline=False)
        embed.add_field(name="Embeddings", value="all-MiniLM-L6-v2", inline=False)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(
        name="grimmor_resync",
        description="(Admin) Resynchronise les slash commands sur ce serveur.",
    )
    async def grimmor_resync(self, interaction: discord.Interaction) -> None:
        member = await require_permission(interaction, "grimmor_resync")
        if member is None:
            return
        cfg = get_config()
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            guild = discord.Object(id=cfg.guild_id)
            synced = await self.bot.tree.sync(guild=guild)
            await interaction.followup.send(
                embed=success_embed(f"{len(synced)} commandes resynchronisées."),
                ephemeral=True,
            )
        except Exception:
            logger.exception("Échec resync")
            await interaction.followup.send(
                embed=error_embed("Échec de la resynchronisation."), ephemeral=True
            )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AdminCog(bot))

