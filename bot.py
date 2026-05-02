"""Point d'entrée principal du bot Grimmor."""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import traceback

import discord
from discord.ext import commands

from config import get_config
from db import get_db
from rag.graph import get_graph
from rag.retriever import get_retriever

logger = logging.getLogger("grimmor")

COGS = ["cogs.ask", "cogs.pfc", "cogs.inactifs", "cogs.admin", "cogs.archive", "cogs.permissions_cog"]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )
    # Discord est très bavard en debug, on garde INFO max
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("discord.http").setLevel(logging.WARNING)


class Grimmor(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True
        intents.messages = True
        super().__init__(command_prefix="!grimmor!", intents=intents, help_command=None)

    async def setup_hook(self) -> None:
        cfg = get_config()

        # ─── Initialisation DB ─────────────────────────────────────────
        await get_db(cfg.sqlite_path)

        # ─── Chargement des cogs ──────────────────────────────────────
        for ext in COGS:
            try:
                await self.load_extension(ext)
                logger.info("Cog chargé: %s", ext)
            except Exception:
                logger.exception("Échec chargement cog %s", ext)

        # ─── Handler d'erreur slash commands ──────────────────────────
        self.tree.on_error = self._on_tree_error  # type: ignore[assignment]

        # ─── Préchauffe du RAG (peut prendre 10-30s la 1re fois) ──────
        try:
            await get_retriever()
        except Exception:
            logger.exception(
                "Préchauffe RAG impossible (le bot démarre quand même, /ask retournera une erreur)"
            )

        # ─── Préchauffe du graphe consolidé (couche auxiliaire) ───────
        try:
            graph = get_graph()
            if graph is None:
                logger.warning("Graphe consolidé indisponible — couche auxiliaire désactivée.")
        except Exception:
            logger.exception("Préchauffe du graphe impossible (couche auxiliaire désactivée)")

        # ─── Sync des slash commands sur le guild ─────────────────────
        guild = discord.Object(id=cfg.guild_id)
        try:
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            logger.info("Slash commands synchronisées: %d", len(synced))
        except Exception:
            logger.exception("Échec sync slash commands")

    async def on_ready(self) -> None:
        logger.info(
            "Connecté en tant que %s (id=%s) — guilds: %d",
            self.user,
            self.user.id if self.user else "?",
            len(self.guilds),
        )
        await self._leave_unauthorized_guilds()

    async def on_guild_join(self, guild: discord.Guild) -> None:
        cfg = get_config()
        if guild.id == cfg.guild_id:
            return
        logger.warning(
            "Serveur non autorise rejoint: %s (%s). Depart automatique.",
            guild.name,
            guild.id,
        )
        try:
            await guild.leave()
        except Exception:
            logger.exception("Impossible de quitter le serveur non autorise %s", guild.id)

    async def _leave_unauthorized_guilds(self) -> None:
        cfg = get_config()
        for guild in list(self.guilds):
            if guild.id == cfg.guild_id:
                continue
            logger.warning(
                "Serveur non autorise detecte au demarrage: %s (%s). Depart automatique.",
                guild.name,
                guild.id,
            )
            try:
                await guild.leave()
            except Exception:
                logger.exception("Impossible de quitter le serveur non autorise %s", guild.id)

    async def _on_tree_error(
        self,
        interaction: discord.Interaction,
        error: discord.app_commands.AppCommandError,
    ) -> None:
        logger.exception("Erreur slash command: %s", error)
        await self._log_to_channel(f"⚠️ Erreur slash : `{type(error).__name__}: {error}`")
        msg = "Une erreur interne est survenue. Les sorciers ont été prévenus."
        try:
            if interaction.response.is_done():
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=True)
        except Exception:
            pass

    async def on_error(self, event_method: str, /, *args, **kwargs) -> None:
        logger.exception("Erreur non gérée dans event %s", event_method)
        tb = traceback.format_exc()
        await self._log_to_channel(
            f"⚠️ Erreur dans `{event_method}`:\n```{tb[-1500:]}```"
        )

    async def _log_to_channel(self, message: str) -> None:
        cfg = get_config()
        if cfg.log_channel_id is None:
            return
        ch = self.get_channel(cfg.log_channel_id)
        if ch is None:
            try:
                ch = await self.fetch_channel(cfg.log_channel_id)
            except Exception:
                return
        try:
            await ch.send(message[:1900])
        except Exception:
            pass


async def _main() -> None:
    _setup_logging()
    cfg = get_config()
    bot = Grimmor()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    has_signal_handlers = False

    def _request_stop() -> None:
        logger.info("Signal d'arrêt reçu, fermeture en cours…")
        stop_event.set()

    # Windows ne supporte pas add_signal_handler pour SIGTERM ; on tente quand même.
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _request_stop)
            has_signal_handlers = True
        except NotImplementedError:
            pass

    try:
        if not has_signal_handlers:
            # Windows ne permet souvent pas add_signal_handler(). Dans ce cas,
            # asyncio.run() annule _main() sur Ctrl+C ; le finally ci-dessous
            # ferme Discord avant que la boucle d'evenements ne disparaisse.
            await bot.start(cfg.discord_token)
            return

        bot_task = asyncio.create_task(bot.start(cfg.discord_token))
        stop_task = asyncio.create_task(stop_event.wait())

        done, pending = await asyncio.wait(
            {bot_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )

        if stop_task in done and not bot.is_closed():
            await bot.close()

        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        if bot_task in done:
            exc = bot_task.exception()
            if exc:
                raise exc
    finally:
        if not bot.is_closed():
            logger.info("Fermeture Discord en cours...")
            await bot.close()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
