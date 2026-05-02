"""Commande /ask : RAG NotebookLM-style sur le corpus de l'association.

L'embed renvoyé contient :
- la réponse synthétique avec citations [1] [2] inline ;
- une section "Sources" numérotée avec extrait verbatim sous chaque source ;
- des boutons de questions de suivi cliquables (re-pose la question via /ask).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from rag.generator import LLMRateLimitError, NO_ANSWER_SENTINEL, get_generator, get_llm_gate
from rag.graph import get_graph
from rag.retriever import get_retriever
from utils.formatting import (
    GRIMMOR_PURPLE,
    error_embed,
    format_citation_field,
    truncate,
    warning_embed,
)
from utils.permissions import check_permissions

logger = logging.getLogger(__name__)

MAX_AUTOCOMPLETE_RESULTS = 25
FOLLOW_UP_BUTTON_LABEL_MAX = 80


# ─── Vue de questions de suivi ───────────────────────────────────────────────


class FollowUpButton(discord.ui.Button):
    def __init__(self, question: str, source_filter: Optional[str]):
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label=truncate(question, FOLLOW_UP_BUTTON_LABEL_MAX),
            emoji="🔮",
        )
        self.question = question
        self.source_filter = source_filter

    async def callback(self, interaction: discord.Interaction) -> None:
        cog: Optional[AskCog] = interaction.client.get_cog("AskCog")  # type: ignore[assignment]
        if cog is None:
            await interaction.response.send_message(
                embed=error_embed("Le cog /ask n'est pas chargé."), ephemeral=True
            )
            return
        await cog.run_ask(interaction, self.question, self.source_filter)


class FollowUpView(discord.ui.View):
    def __init__(
        self,
        questions: list[str],
        source_filter: Optional[str],
        timeout: float = 300.0,
    ):
        super().__init__(timeout=timeout)
        for q in questions[:3]:
            if q.strip():
                self.add_item(FollowUpButton(q.strip(), source_filter))


# ─── Cog ─────────────────────────────────────────────────────────────────────


class AskCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # ─── Slash command ────────────────────────────────────────────────

    @app_commands.command(
        name="ask",
        description="Pose une question à Grimmor. Il ne répond qu'à partir des règles indexées.",
    )
    @app_commands.describe(
        question="Ta question (formule-la précisément)",
        source="(optionnel) limite la recherche à un fichier précis",
    )
    async def ask(
        self,
        interaction: discord.Interaction,
        question: str,
        source: Optional[str] = None,
    ) -> None:
        await self.run_ask(interaction, question, source)

    @ask.autocomplete("source")
    async def _source_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        try:
            retriever = await get_retriever()
            sources = retriever.list_sources()
        except Exception:
            return []
        current_lc = (current or "").lower()
        matches = [s for s in sources if current_lc in s.lower()]
        matches = matches[:MAX_AUTOCOMPLETE_RESULTS]
        return [app_commands.Choice(name=truncate(s, 100), value=s) for s in matches]

    # ─── Pipeline factorisé (réutilisé par les boutons follow-up) ────

    @check_permissions("ask")
    async def run_ask(
        self,
        interaction: discord.Interaction,
        question: str,
        source_filter: Optional[str],
    ) -> None:
        question = (question or "").strip()
        if not question:
            await interaction.response.send_message(
                embed=error_embed("Tu dois poser une question."), ephemeral=True
            )
            return

        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True)

        # Si la file LLM est encombrée, prévient discrètement l'utilisateur.
        gate = get_llm_gate()
        if gate is not None and gate.queue_position >= 1:
            try:
                await interaction.followup.send(
                    f"⏳ Grimmor a {gate.queue_position} requête(s) avant la tienne. "
                    "Patiente quelques secondes…",
                    ephemeral=True,
                )
            except Exception:
                pass

        # ─── Init RAG ─────────────────────────────────────────────────
        try:
            retriever = await get_retriever()
            generator = get_generator()
        except Exception:
            logger.exception("Initialisation RAG impossible")
            await self._send(
                interaction,
                embed=error_embed(
                    "Le système de règles n'est pas encore prêt. Réessaie dans quelques instants."
                ),
            )
            return

        # ─── Vérification du filtre source ────────────────────────────
        if source_filter:
            available = retriever.list_sources()
            if source_filter not in available:
                await self._send(
                    interaction,
                    embed=warning_embed(
                        f"Le fichier `{truncate(source_filter, 100)}` n'est pas indexé.\n"
                        f"Sources disponibles : {', '.join(available[:10]) or '_(aucune)_'}",
                        title="Source inconnue",
                    ),
                )
                return

        # ─── Couche auxiliaire : résolution d'entités via graphe ─────
        graph = get_graph()
        resolved_entities = []
        entity_cards: list[str] = []
        if graph is not None:
            try:
                resolved_entities = graph.resolve(question, max_results=4)
                if resolved_entities:
                    entity_cards = [graph.entity_card(r.entity) for r in resolved_entities]
                    logger.info(
                        "Graphe : %d entités résolues : %s",
                        len(resolved_entities),
                        ", ".join(r.entity.get("name", "?") for r in resolved_entities),
                    )
            except Exception:
                logger.exception("Échec résolution graphe (continue sans)")

        # ─── Multi-query expansion par LLM (optionnelle) ─────────────
        from config import get_config
        cfg = get_config()
        variants: list[str] = []
        if cfg.rag_use_multi_query_llm:
            try:
                variants = await generator.generate_query_variants(question)
            except Exception:
                logger.exception("Échec génération de variantes")
                variants = []

        # Requetes de recherche : question brute + entites resolues depuis le PDF.
        # Pas de reformulations hardcodees ici.
        all_queries = [question]
        for ent in resolved_entities:
            name = ent.entity.get("name")
            if name and name.lower() != question.lower() and name not in all_queries:
                all_queries.append(name)
        for variant in variants:
            if variant not in all_queries:
                all_queries.append(variant)
        if variants:
            logger.info("Multi-query : %d variantes LLM générées", len(variants))

        # ─── Recherche hybride multi-query ───────────────────────────
        try:
            chunks = await retriever.search_multi(all_queries, source_filter=source_filter)
        except Exception:
            logger.exception("Échec retriever.search_multi")
            await self._send(
                interaction,
                embed=error_embed(
                    "Une erreur est survenue lors de la recherche. Réessaie plus tard."
                ),
            )
            return

        # ─── Chunks bonus issus du graphe (sections d'origine des entités) ─
        if graph is not None and resolved_entities:
            try:
                bonus = graph.bonus_chunks_for(resolved_entities, max_chunks=3)
                # Évite les doublons par source+page+section
                existing = {(c.get("metadata", {}).get("source"),
                             c.get("metadata", {}).get("page"),
                             c.get("metadata", {}).get("section"))
                            for c in chunks}
                for b in bonus:
                    key = (b.get("metadata", {}).get("source"),
                           b.get("metadata", {}).get("page"),
                           b.get("metadata", {}).get("section"))
                    if key not in existing:
                        chunks.append(b)
                        existing.add(key)
            except Exception:
                logger.exception("Échec ajout chunks bonus du graphe")

        if not chunks:
            await self._send(
                interaction,
                embed=warning_embed(
                    "Je n'ai trouvé aucun extrait pertinent dans mes grimoires pour cette question.\n"
                    "Essaie de reformuler ou utilise d'autres mots-clés.",
                    title="Rien trouvé",
                ),
            )
            return

        # ─── Génération JSON structurée ──────────────────────────────
        try:
            result = await generator.generate(question, chunks, entity_cards=entity_cards)
        except asyncio.TimeoutError:
            await self._send(
                interaction,
                embed=error_embed(
                    "Le modèle de langage a mis trop de temps à répondre. Réessaie dans un instant."
                ),
            )
            return
        except LLMRateLimitError as exc:
            retry = exc.retry_after_seconds
            suffix = ""
            if retry is not None:
                suffix = f" Attends environ {int(retry) + 1}s avant de reposer une question."
            await self._send(
                interaction,
                embed=error_embed(
                    "Limite temporaire du modele LLM atteinte." + suffix
                ),
            )
            return
        except Exception:
            logger.exception("Échec génération LLM")
            await self._send(
                interaction,
                embed=error_embed("Impossible de générer la réponse. Réessaie plus tard."),
            )
            return

        embed = self._build_embed(question, source_filter, result, chunks)
        view = self._build_view(result.get("follow_ups", []), source_filter)
        await self._send(interaction, embed=embed, view=view)

    # ─── Construction de l'embed NotebookLM ──────────────────────────

    def _build_embed(
        self,
        question: str,
        source_filter: Optional[str],
        result: dict,
        chunks: list[dict],
    ) -> discord.Embed:
        answer = result.get("answer", "").strip() or NO_ANSWER_SENTINEL
        citations = result.get("citations", []) or []

        title_prefix = ""
        if source_filter:
            title_prefix = f"[{truncate(source_filter, 30)}] "

        embed = discord.Embed(
            title=truncate(title_prefix + question, 256),
            description=truncate(answer, 4000),
            color=GRIMMOR_PURPLE,
        )

        if citations:
            for cit in citations:
                meta = cit.get("metadata", {}) or {}
                quote = cit.get("quote") or cit.get("chunk_text") or ""
                name, value = format_citation_field(cit["id"], meta, quote)
                embed.add_field(name=name, value=value, inline=False)
        elif answer != NO_ANSWER_SENTINEL:
            # Aucune citation explicite mais réponse non-vide : on liste au moins
            # les chunks consultés en repli, pour transparence.
            from utils.formatting import short_source_label

            seen: set[tuple] = set()
            lines: list[str] = []
            for c in chunks:
                meta = c.get("metadata", {}) or {}
                key = (meta.get("source"), meta.get("page"), meta.get("section"))
                if key in seen:
                    continue
                seen.add(key)
                lines.append("• " + short_source_label(meta))
                if len(lines) >= 8:
                    break
            if lines:
                embed.add_field(
                    name="Extraits consultés",
                    value=truncate("\n".join(lines), 1024),
                    inline=False,
                )

        footer_bits = [f"{result.get('chunks_used', 0)} extraits consultés"]
        if source_filter:
            footer_bits.append(f"source : {source_filter}")
        embed.set_footer(text="Grimmor • " + " • ".join(footer_bits))
        return embed

    def _build_view(
        self, follow_ups: list[str], source_filter: Optional[str]
    ) -> Optional[FollowUpView]:
        questions = [q for q in follow_ups if q and q.strip()]
        if not questions:
            return None
        return FollowUpView(questions, source_filter)

    # ─── Helper d'envoi (gère defer / followup) ──────────────────────

    @staticmethod
    async def _send(
        interaction: discord.Interaction,
        *,
        embed: discord.Embed,
        view: Optional[discord.ui.View] = None,
    ) -> None:
        kwargs: dict = {"embed": embed}
        if view is not None:
            kwargs["view"] = view
        try:
            if interaction.response.is_done():
                await interaction.followup.send(**kwargs)
            else:
                await interaction.response.send_message(**kwargs)
        except Exception:
            logger.exception("Échec envoi de la réponse /ask")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AskCog(bot))
