"""Moteur d'archivage de channel Discord en fichier HTML standalone."""
from __future__ import annotations

import html
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import discord

from config import get_config

logger = logging.getLogger(__name__)

MAX_MESSAGES = 10_000
ArchivableChannel = discord.TextChannel | discord.Thread


def _esc(text: str) -> str:
    """Échappe le HTML."""
    return html.escape(text, quote=True)


def _avatar_url(user: discord.User | discord.Member) -> str:
    """URL de l'avatar ou avatar par défaut."""
    if user.avatar:
        return str(user.avatar.replace(size=64, format="webp"))
    return str(user.default_avatar)


def _role_color(member: discord.User | discord.Member) -> str:
    """Couleur hexadécimale du rôle le plus élevé ayant une couleur."""
    if isinstance(member, discord.Member) and member.color and member.color.value != 0:
        return f"#{member.color.value:06x}"
    return "#ffffff"


def _format_timestamp(dt: datetime) -> str:
    """Formate un datetime en date/heure lisible."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%d/%m/%Y %H:%M")


def _safe_filename_part(name: str) -> str:
    """Transforme un nom Discord en morceau de fichier sûr."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    safe = safe.strip(".-_")
    return (safe or "channel")[:80]


def _archive_context(channel: ArchivableChannel) -> tuple[str, str, str]:
    """Retourne (type, catégorie, parent/topic) pour salon ou fil."""
    if isinstance(channel, discord.Thread):
        parent = channel.parent
        parent_name = f"#{parent.name}" if parent is not None else "parent introuvable"
        category = getattr(parent, "category", None) if parent is not None else None
        category_name = category.name if category is not None else "Sans catégorie"

        details = [f"Fil rattaché à {parent_name}"]
        if channel.archived:
            details.append("fil archivé")
        if channel.locked:
            details.append("fil verrouillé")
        return "Fil", category_name, " · ".join(details)

    return "Salon textuel", (
        channel.category.name if channel.category else "Sans catégorie"
    ), channel.topic or ""


def _render_content(content: str) -> str:
    """Convertit le contenu d'un message Discord en HTML basique."""
    if not content:
        return ""
    import re

    text = _esc(content)

    # Liens
    text = re.sub(
        r'(https?://[^\s<>&"\']+)',
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        text,
    )

    # Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic *text* ou _text_
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
    # Strikethrough ~~text~~
    text = re.sub(r'~~(.+?)~~', r'<del>\1</del>', text)
    # Inline code `text`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Code blocks ```text```
    text = re.sub(
        r'```(?:\w+)?\n?(.*?)```',
        r'<pre><code>\1</code></pre>',
        text,
        flags=re.DOTALL,
    )
    # Spoiler ||text||
    text = re.sub(
        r'\|\|(.+?)\|\|',
        r'<span class="spoiler" onclick="this.classList.toggle(\'revealed\')">\1</span>',
        text,
    )

    # Newlines
    text = text.replace("\n", "<br>")

    return text


def _render_attachments(attachments: list[discord.Attachment]) -> str:
    """Rend les pièces jointes en HTML (images inline, liens pour le reste)."""
    if not attachments:
        return ""
    parts: list[str] = []
    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
    video_exts = {".mp4", ".webm", ".mov"}

    for att in attachments:
        ext = Path(att.filename).suffix.lower()
        url = _esc(att.url)
        fname = _esc(att.filename)

        if ext in image_exts or (att.content_type and att.content_type.startswith("image/")):
            parts.append(
                f'<div class="attachment-img">'
                f'<a href="{url}" target="_blank"><img src="{url}" alt="{fname}" loading="lazy"></a>'
                f'</div>'
            )
        elif ext in video_exts or (att.content_type and att.content_type.startswith("video/")):
            parts.append(
                f'<div class="attachment-video">'
                f'<video controls preload="metadata" src="{url}">Vidéo : {fname}</video>'
                f'</div>'
            )
        else:
            size_kb = att.size / 1024 if att.size else 0
            parts.append(
                f'<div class="attachment-file">'
                f'📎 <a href="{url}" target="_blank">{fname}</a>'
                f' <span class="file-size">({size_kb:.1f} Ko)</span>'
                f'</div>'
            )
    return "\n".join(parts)


def _render_embeds(embeds: list[discord.Embed]) -> str:
    """Rend les embeds Discord en HTML."""
    if not embeds:
        return ""
    parts: list[str] = []
    for embed in embeds:
        color = f"#{embed.color.value:06x}" if embed.color else "#4f545c"
        inner: list[str] = []
        if embed.author and embed.author.name:
            author_icon = ""
            if embed.author.icon_url:
                author_icon = f'<img src="{_esc(str(embed.author.icon_url))}" class="embed-author-icon">'
            inner.append(f'<div class="embed-author">{author_icon}{_esc(embed.author.name)}</div>')
        if embed.title:
            title_text = _esc(embed.title)
            if embed.url:
                title_text = f'<a href="{_esc(str(embed.url))}" target="_blank">{title_text}</a>'
            inner.append(f'<div class="embed-title">{title_text}</div>')
        if embed.description:
            inner.append(f'<div class="embed-description">{_render_content(embed.description)}</div>')
        for field in embed.fields:
            inline_class = " inline" if field.inline else ""
            inner.append(
                f'<div class="embed-field{inline_class}">'
                f'<div class="embed-field-name">{_esc(field.name)}</div>'
                f'<div class="embed-field-value">{_render_content(field.value)}</div>'
                f'</div>'
            )
        if embed.image and embed.image.url:
            inner.append(
                f'<div class="embed-image">'
                f'<img src="{_esc(str(embed.image.url))}" loading="lazy">'
                f'</div>'
            )
        if embed.thumbnail and embed.thumbnail.url:
            inner.insert(0,
                f'<div class="embed-thumbnail">'
                f'<img src="{_esc(str(embed.thumbnail.url))}" loading="lazy">'
                f'</div>'
            )
        if embed.footer and embed.footer.text:
            footer_icon = ""
            if embed.footer.icon_url:
                footer_icon = f'<img src="{_esc(str(embed.footer.icon_url))}" class="embed-footer-icon">'
            inner.append(f'<div class="embed-footer">{footer_icon}{_esc(embed.footer.text)}</div>')

        parts.append(
            f'<div class="embed" style="border-left-color: {color};">'
            + "\n".join(inner)
            + '</div>'
        )
    return "\n".join(parts)


def _render_reactions(reactions: list[discord.Reaction]) -> str:
    """Rend les réactions en badges."""
    if not reactions:
        return ""
    parts: list[str] = []
    for reaction in reactions:
        emoji = _esc(str(reaction.emoji))
        parts.append(f'<span class="reaction">{emoji} {reaction.count}</span>')
    return '<div class="reactions">' + " ".join(parts) + "</div>"


def _render_reply(ref: discord.MessageReference, resolved: Optional[discord.Message]) -> str:
    """Rend l'indicateur de réponse."""
    if resolved and resolved.author:
        author = _esc(resolved.author.display_name)
        preview = _esc(resolved.content[:80]) if resolved.content else "_pièce jointe_"
        return (
            f'<div class="reply-indicator">'
            f'<span class="reply-arrow">↩</span> '
            f'<span class="reply-author">{author}</span> '
            f'<span class="reply-preview">{preview}</span>'
            f'</div>'
        )
    return (
        '<div class="reply-indicator">'
        '<span class="reply-arrow">↩</span> '
        '<span class="reply-preview"><em>message original supprimé ou inaccessible</em></span>'
        '</div>'
    )


def _should_group(prev: Optional[discord.Message], msg: discord.Message) -> bool:
    """Détermine si un message doit être groupé avec le précédent (même auteur, < 7 min)."""
    if prev is None:
        return False
    if prev.author.id != msg.author.id:
        return False
    delta = (msg.created_at - prev.created_at).total_seconds()
    return delta < 420  # 7 minutes


async def archive_channel(
    channel: ArchivableChannel,
    archived_by: discord.User | discord.Member,
    *,
    save_to_disk: bool = True,
) -> tuple[Path, int]:
    """Archive un salon ou un fil Discord en fichier HTML.

    Returns:
        Tuple (chemin_du_fichier, nombre_de_messages).
    """
    cfg = get_config()
    now = datetime.now(timezone.utc)
    guild = channel.guild

    # ─── Récupération des messages ────────────────────────────────────
    messages: list[discord.Message] = []
    try:
        async for msg in channel.history(limit=MAX_MESSAGES, oldest_first=True):
            messages.append(msg)
    except discord.Forbidden:
        logger.warning("Pas d'accès à l'historique de #%s (%s)", channel.name, channel.id)
    except Exception:
        logger.exception("Erreur récupération historique #%s", channel.name)

    total = len(messages)

    # ─── Construction du HTML ─────────────────────────────────────────
    html_parts: list[str] = []

    # Header
    channel_name = _esc(channel.name)
    guild_name = _esc(guild.name)
    channel_kind, category_raw, topic_raw = _archive_context(channel)
    channel_kind = _esc(channel_kind)
    topic = _esc(topic_raw)
    category_name = _esc(category_raw)
    archived_by_name = _esc(archived_by.display_name)
    archived_at = _format_timestamp(now)

    html_parts.append(_HTML_HEAD.format(
        channel_name=channel_name,
        guild_name=guild_name,
    ))

    html_parts.append(_HTML_HEADER.format(
        channel_name=channel_name,
        guild_name=guild_name,
        channel_kind=channel_kind,
        category_name=category_name,
        topic=topic if topic else "<em>aucun topic</em>",
        archived_by=archived_by_name,
        archived_at=archived_at,
        total_messages=total,
    ))

    if total >= MAX_MESSAGES:
        html_parts.append(
            '<div class="limit-warning">'
            f'⚠️ Seuls les {MAX_MESSAGES:,} derniers messages ont été archivés.'
            '</div>'
        )

    # Messages
    html_parts.append('<div class="messages">')
    prev_msg: Optional[discord.Message] = None

    for msg in messages:
        grouped = _should_group(prev_msg, msg)

        # Réponse
        reply_html = ""
        if msg.reference and msg.reference.message_id:
            reply_html = _render_reply(msg.reference, msg.reference.resolved)

        author_name = _esc(msg.author.display_name)
        color = _role_color(msg.author)
        avatar = _esc(_avatar_url(msg.author))
        timestamp = _format_timestamp(msg.created_at)
        content_html = _render_content(msg.content)
        attachments_html = _render_attachments(list(msg.attachments))
        embeds_html = _render_embeds(list(msg.embeds))
        reactions_html = _render_reactions(list(msg.reactions))

        # Badges système
        badges = ""
        if msg.author.bot:
            badges = ' <span class="bot-badge">BOT</span>'
        if msg.pinned:
            badges += ' <span class="pin-badge">📌</span>'

        if grouped:
            html_parts.append(
                f'<div class="message grouped">'
                f'<div class="message-timestamp-hover">{timestamp}</div>'
                f'<div class="message-body">'
                f'{reply_html}'
                f'{f"<div class=message-content>{content_html}</div>" if content_html else ""}'
                f'{attachments_html}'
                f'{embeds_html}'
                f'{reactions_html}'
                f'</div>'
                f'</div>'
            )
        else:
            html_parts.append(
                f'<div class="message">'
                f'<div class="avatar-col">'
                f'<img src="{avatar}" class="avatar" loading="lazy">'
                f'</div>'
                f'<div class="message-body">'
                f'{reply_html}'
                f'<div class="message-header">'
                f'<span class="author" style="color: {color};">{author_name}</span>'
                f'{badges}'
                f'<span class="timestamp">{timestamp}</span>'
                f'</div>'
                f'{f"<div class=message-content>{content_html}</div>" if content_html else ""}'
                f'{attachments_html}'
                f'{embeds_html}'
                f'{reactions_html}'
                f'</div>'
                f'</div>'
            )

        prev_msg = msg

    html_parts.append('</div>')  # .messages

    # Footer
    html_parts.append(_HTML_FOOTER.format(
        total_messages=total,
        archived_at=archived_at,
        archived_by=archived_by_name,
    ))

    html_parts.append('</body></html>')

    full_html = "\n".join(html_parts)

    # ─── Sauvegarde sur disque ────────────────────────────────────────
    safe_name = _safe_filename_part(channel.name)
    filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{safe_name}.html"

    if save_to_disk:
        archive_dir = cfg.archive_path
        archive_dir.mkdir(parents=True, exist_ok=True)
        filepath = archive_dir / filename
        filepath.write_text(full_html, encoding="utf-8")
        logger.info("Archive sauvegardée : %s (%d messages)", filepath, total)
    else:
        # Fichier temporaire (le caller gère le nettoyage)
        tmp_dir = Path(tempfile.mkdtemp(prefix="grimmor_archive_"))
        filepath = tmp_dir / filename
        filepath.write_text(full_html, encoding="utf-8")
        logger.info("Archive temporaire : %s (%d messages)", filepath, total)

    return filepath, total


# ══════════════════════════════════════════════════════════════════════════════
# Templates HTML
# ══════════════════════════════════════════════════════════════════════════════

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Archive #{channel_name} — {guild_name}</title>
<style>
/* ─── Reset & base ──────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    background: #36393f;
    color: #dcddde;
    font-family: 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 15px;
    line-height: 1.375;
    padding: 0;
}}

a {{ color: #00aff4; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

img {{ max-width: 100%; }}

code {{
    background: #2f3136;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.875em;
    font-family: 'Consolas', 'Monaco', monospace;
}}

pre {{
    background: #2f3136;
    border: 1px solid #202225;
    border-radius: 4px;
    padding: 12px;
    margin: 6px 0;
    overflow-x: auto;
}}

pre code {{
    background: none;
    padding: 0;
}}

/* ─── Header ────────────────────────────────────────────────────────── */
.archive-header {{
    background: #2f3136;
    border-bottom: 2px solid #7b2fbe;
    padding: 24px 32px;
    margin-bottom: 0;
}}

.archive-header h1 {{
    font-size: 1.5rem;
    color: #ffffff;
    margin-bottom: 8px;
}}

.archive-header h1 .hash {{
    color: #72767d;
    margin-right: 4px;
}}

.archive-meta {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px 24px;
    margin-top: 12px;
    font-size: 0.875rem;
    color: #96989d;
}}

.archive-meta .label {{
    color: #72767d;
}}

.archive-meta .value {{
    color: #dcddde;
}}

/* ─── Limit warning ────────────────────────────────────────────────── */
.limit-warning {{
    background: #faa61a20;
    border: 1px solid #faa61a;
    color: #faa61a;
    padding: 10px 32px;
    font-size: 0.875rem;
}}

/* ─── Messages container ────────────────────────────────────────────── */
.messages {{
    padding: 16px 0;
}}

/* ─── Single message ────────────────────────────────────────────────── */
.message {{
    display: flex;
    padding: 4px 32px;
    gap: 16px;
    position: relative;
}}

.message:hover {{
    background: #32353b;
}}

.message:not(.grouped) {{
    margin-top: 16px;
}}

.message.grouped {{
    padding-left: 80px;  /* 32px padding + 40px avatar + 8px gap */
}}

.message.grouped .message-timestamp-hover {{
    position: absolute;
    left: 32px;
    top: 6px;
    font-size: 0.6875rem;
    color: #72767d;
    opacity: 0;
    width: 40px;
    text-align: center;
}}

.message.grouped:hover .message-timestamp-hover {{
    opacity: 1;
}}

/* ─── Avatar ────────────────────────────────────────────────────────── */
.avatar-col {{
    flex-shrink: 0;
    width: 40px;
    padding-top: 2px;
}}

.avatar {{
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
}}

/* ─── Message body ──────────────────────────────────────────────────── */
.message-body {{
    flex: 1;
    min-width: 0;
}}

.message-header {{
    display: flex;
    align-items: baseline;
    gap: 8px;
    flex-wrap: wrap;
}}

.author {{
    font-weight: 600;
    font-size: 1rem;
    cursor: default;
}}

.timestamp {{
    font-size: 0.75rem;
    color: #72767d;
}}

.bot-badge {{
    background: #5865f2;
    color: #fff;
    font-size: 0.625rem;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
    text-transform: uppercase;
    vertical-align: middle;
}}

.pin-badge {{
    font-size: 0.75rem;
    vertical-align: middle;
}}

.message-content {{
    margin-top: 2px;
    word-wrap: break-word;
    overflow-wrap: anywhere;
}}

/* ─── Reply indicator ───────────────────────────────────────────────── */
.reply-indicator {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8125rem;
    color: #96989d;
    margin-bottom: 2px;
    padding: 2px 0;
}}

.reply-arrow {{
    color: #4f545c;
    font-weight: bold;
}}

.reply-author {{
    font-weight: 600;
    color: #b9bbbe;
}}

.reply-preview {{
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    max-width: 300px;
}}

/* ─── Attachments ───────────────────────────────────────────────────── */
.attachment-img {{
    margin: 6px 0;
}}

.attachment-img img {{
    max-width: 400px;
    max-height: 350px;
    border-radius: 8px;
    cursor: pointer;
}}

.attachment-video {{
    margin: 6px 0;
}}

.attachment-video video {{
    max-width: 400px;
    max-height: 350px;
    border-radius: 8px;
}}

.attachment-file {{
    background: #2f3136;
    border: 1px solid #202225;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    display: inline-block;
}}

.file-size {{
    color: #72767d;
    font-size: 0.75rem;
}}

/* ─── Embeds ────────────────────────────────────────────────────────── */
.embed {{
    background: #2f3136;
    border-left: 4px solid #4f545c;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 520px;
    position: relative;
}}

.embed-author {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8125rem;
    font-weight: 600;
    margin-bottom: 4px;
}}

.embed-author-icon {{
    width: 24px;
    height: 24px;
    border-radius: 50%;
}}

.embed-title {{
    font-weight: 700;
    margin-bottom: 4px;
    color: #ffffff;
}}

.embed-title a {{
    color: #00aff4;
}}

.embed-description {{
    font-size: 0.875rem;
    color: #dcddde;
    margin-bottom: 8px;
}}

.embed-field {{
    margin-bottom: 8px;
}}

.embed-field.inline {{
    display: inline-block;
    width: 45%;
    vertical-align: top;
    margin-right: 4%;
}}

.embed-field-name {{
    font-weight: 700;
    font-size: 0.8125rem;
    color: #ffffff;
    margin-bottom: 2px;
}}

.embed-field-value {{
    font-size: 0.8125rem;
    color: #dcddde;
}}

.embed-image {{
    margin-top: 8px;
}}

.embed-image img {{
    max-width: 100%;
    max-height: 300px;
    border-radius: 4px;
}}

.embed-thumbnail {{
    float: right;
    margin-left: 16px;
    margin-bottom: 8px;
}}

.embed-thumbnail img {{
    width: 80px;
    height: 80px;
    border-radius: 4px;
    object-fit: cover;
}}

.embed-footer {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: #72767d;
    margin-top: 8px;
}}

.embed-footer-icon {{
    width: 20px;
    height: 20px;
    border-radius: 50%;
}}

/* ─── Reactions ─────────────────────────────────────────────────────── */
.reactions {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 4px;
}}

.reaction {{
    background: #2f3136;
    border: 1px solid #4f545c;
    border-radius: 8px;
    padding: 2px 8px;
    font-size: 0.8125rem;
    cursor: default;
}}

/* ─── Spoiler ───────────────────────────────────────────────────────── */
.spoiler {{
    background: #202225;
    color: transparent;
    padding: 0 4px;
    border-radius: 3px;
    cursor: pointer;
    transition: background 0.1s, color 0.1s;
}}

.spoiler.revealed {{
    background: #4f545c80;
    color: #dcddde;
}}

/* ─── Footer ────────────────────────────────────────────────────────── */
.archive-footer {{
    background: #2f3136;
    border-top: 2px solid #7b2fbe;
    padding: 16px 32px;
    margin-top: 16px;
    text-align: center;
    font-size: 0.8125rem;
    color: #72767d;
}}

.archive-footer .brand {{
    color: #7b2fbe;
    font-weight: 700;
}}

/* ─── Responsive ────────────────────────────────────────────────────── */
@media (max-width: 600px) {{
    .message {{ padding: 4px 12px; }}
    .message.grouped {{ padding-left: 52px; }}
    .archive-header, .archive-footer {{ padding-left: 12px; padding-right: 12px; }}
    .attachment-img img {{ max-width: 100%; }}
}}
</style>
</head>
<body>
"""

_HTML_HEADER = """\
<div class="archive-header">
    <h1><span class="hash">#</span>{channel_name}</h1>
    <div class="archive-meta">
        <div><span class="label">Serveur :</span> <span class="value">{guild_name}</span></div>
        <div><span class="label">Type :</span> <span class="value">{channel_kind}</span></div>
        <div><span class="label">Catégorie :</span> <span class="value">{category_name}</span></div>
        <div><span class="label">Topic :</span> <span class="value">{topic}</span></div>
        <div><span class="label">Messages :</span> <span class="value">{total_messages}</span></div>
        <div><span class="label">Archivé par :</span> <span class="value">{archived_by}</span></div>
        <div><span class="label">Date d'archivage :</span> <span class="value">{archived_at}</span></div>
    </div>
</div>
"""

_HTML_FOOTER = """\
<div class="archive-footer">
    Archive générée par <span class="brand">Grimmor</span><br>
    {total_messages} messages · Archivé le {archived_at} par {archived_by}
</div>
"""
