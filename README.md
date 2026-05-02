# Grimmor

> Bot Discord **RAG NotebookLM-style** pour une association de jeu de rôle.
> Répond aux questions de règles à partir d'un PDF indexé, avec citations
> verbatim et zéro hallucination. Conçu pour tourner sur un VPS **2 Go RAM max**.

**Stack** : Python 3.11 · `discord.py` 2.x · ChromaDB (dense) + BM25 (sparse) · `sentence-transformers` (MiniLM) · LLM au choix (Gemini, Mistral, Groq, Cerebras) · SQLite (quotas/activité) · systemd

**Highlights** :
- 🧙 **`/ask`** : recherche hybride (dense + sparse RRF) + couche d'entités structurées + LLM avec sortie JSON contrainte → citations cliquables avec extrait verbatim
- 🛡️ **Anti-hallucination** : sentinel "Je n'ai pas trouvé cette règle" si rien dans le corpus
- 📜 **`/archive`** : export HTML standalone d'un channel (avatars, embeds, réactions, timestamps)
- 🧹 **`/inactifs`** : revue paginée + archivage auto + déplacement vers "À supprimer"
- 🔒 Permissions par rôle Discord (quotas journaliers, cooldowns par commande)
- ⏳ File d'attente LLM intégrée → pas de 429 quand plusieurs utilisateurs interrogent en même temps

---

## ⚠️ Avant de cloner / pousser sur git

**Fichiers à NE JAMAIS commiter** (déjà dans `.gitignore`) :
- `.env` — secrets (token Discord, clés API)
- `permissions.json` — IDs de rôles Discord réels
- `docs/` — PDF de règles sous copyright
- `chroma_db/`, `grimmor.db`, `archives/` — données utilisateur
- `rules_pipeline/output/` — artifacts dérivés du PDF (~15 MB, régénérables)

Pour configurer ton instance :
1. `cp .env.example .env` puis remplis tes secrets
2. `cp permissions_example.json permissions.json` puis remplace les IDs de rôles
3. Dépose ton PDF dans `docs/`
4. `python -m rag.ingest` puis lance le bot

---

## Fonctionnalités

| Commande | Description |
|---|---|
| `/ask` | Réponses RAG sur les règles de l'association (ChromaDB + BM25 + LLM) |
| `/pfc` | Tirage aléatoire pierre · feuille · ciseaux (1 à 10 fois) |
| `/inactifs` | Revue interactive des salons inactifs avec option d'archivage |
| `/inactifs_ignore` | Configure les catégories/salons exclus de la revue inactifs |
| `/archive` | Archive le channel courant en fichier HTML consultable |
| `/ping` | Vérifie que le bot est en vie (latence WebSocket) |
| `/grimmor_info` | État du système RAG (chunks, modèle, etc.) |
| `/grimmor_resync` | Resynchronise les slash commands sur le serveur |

---

## 1. Prérequis

- Python **3.11+**
- `pip` et `venv`
- `systemd` (pour l'installation comme service)
- Accès à une **clé d'API LLM** (Gemini, Mistral, Groq ou Cerebras)
- Accès à une **application Discord** (`DISCORD_TOKEN`, `GUILD_ID`)

Le bot a besoin des **intents** suivants (à activer dans le Developer Portal) :
`SERVER MEMBERS INTENT` et `MESSAGE CONTENT INTENT`.

---

## 2. Installation

```bash
git clone <repo> grimmor
cd grimmor

python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Configuration

```bash
cp .env.example .env
# Édite .env avec ton éditeur préféré
```

### Variables principales

| Variable | Description | Défaut |
|---|---|---|
| `DISCORD_TOKEN` | Token du bot Discord | *(requis)* |
| `GUILD_ID` | ID du serveur Discord cible | *(requis)* |
| `LLM_PROVIDER` | Fournisseur LLM (`gemini`, `mistral`, `groq`, `cerebras`) | `gemini` |
| `GEMINI_API_KEY` | Clé API Google Gemini | |
| `MISTRAL_API_KEY` | Clé API Mistral | |
| `GROQ_API_KEY` | Clé API Groq | |
| `CEREBRAS_API_KEY` | Clé API Cerebras | |

### Permissions et limites

Les permissions, quotas journaliers et cooldowns sont gérés par rôle via le fichier `permissions.json`.
Par défaut, le chemin est :
| Variable | Description | Défaut |
|---|---|---|
| `PERMISSIONS_PATH` | Fichier JSON de configuration des permissions par rôle | `./permissions.json` |

### Stockage

| Variable | Description | Défaut |
|---|---|---|
| `CHROMA_PATH` | Dossier persistant ChromaDB | `./chroma_db` |
| `DOCS_PATH` | Dossier des documents à ingérer | `./docs` |
| `SQLITE_PATH` | Base SQLite (quotas, activité) | `./grimmor.db` |
| `ARCHIVE_PATH` | Dossier de sauvegarde des archives HTML | `./archives` |
| `LOG_CHANNEL_ID` | Salon Discord où loguer les erreurs (optionnel) | |

### RAG (Retrieval-Augmented Generation)

| Variable | Description | Défaut |
|---|---|---|
| `RAG_TOP_K_DENSE` | Nombre de chunks dense retrieval | `15` |
| `RAG_TOP_K_SPARSE` | Nombre de chunks sparse (BM25) | `15` |
| `RAG_TOP_K_FINAL` | Nombre de chunks final après fusion RRF | `8` |
| `RAG_MIN_RRF_SCORE` | Score RRF minimum pour garder un chunk | `0.015` |
| `RAG_RRF_K` | Constante RRF | `60` |
| `RAG_USE_MULTI_QUERY_LLM` | Expansion multi-query par LLM (coûte 1 appel) | `false` |
| `LLM_MAX_CONCURRENT` | Requêtes LLM simultanées max | `1` |
| `LLM_MIN_INTERVAL_SECONDS` | Intervalle minimum entre appels LLM | `0.0` |
| `ASSO_NAME` | Nom de ton association (utilisé dans le prompt) | `Notre Association JDR` |

---

## 4. Ingestion des documents

Dépose tes règles dans `docs/` (sous-dossiers libres).
Formats supportés : **`.md`**, **`.markdown`**, **`.txt`**, **`.pdf`**.

> Un sous-dossier `docs/lore/` marquera automatiquement ses chunks comme `type: lore`.

```bash
python -m rag.ingest
```

Le script :
1. Calcule le hash MD5 de chaque fichier
2. Saute ceux inchangés depuis le dernier run
3. Découpe en chunks sémantiques
4. Génère les embeddings (`all-MiniLM-L6-v2`)
5. Insère dans ChromaDB + reconstruit l'index BM25

À refaire à chaque ajout/modification de document.

---

## 5. Lancement

### Dev (local)

```bash
source venv/bin/activate  # Linux
# ou: .venv\Scripts\activate  # Windows
python bot.py
```

### Production (systemd)

```bash
sudo useradd -r -s /usr/sbin/nologin grimmor
sudo mkdir -p /opt/grimmor
sudo chown -R grimmor: /opt/grimmor

# Copie le code dans /opt/grimmor
# Copie .env dans /opt/grimmor/.env (mode 600)

sudo cp grimmor.service /etc/systemd/system/grimmor.service
sudo systemctl daemon-reload
sudo systemctl enable --now grimmor.service

# Vérifier
sudo systemctl status grimmor.service
sudo journalctl -u grimmor.service -f
```

---

## 6. Guide d'utilisation des commandes

### `/ask` — Questions sur les règles

Pose une question et Grimmor cherche la réponse dans les documents indexés (règles, lore, etc.).

```
/ask question: Comment fonctionne l'initiative ?
/ask question: Quels sont les types de dégâts ? source: regles.pdf
```

**Options :**
- `question` — La question (obligatoire)
- `source` — Filtre sur un fichier spécifique (optionnel, autocomplétion disponible)

**Réponse :** Embed avec la réponse, citations numérotées avec extraits, et boutons de questions de suivi.

**Permissions :** Définies dans `permissions.json` via la commande `ask`. Soumis au quota et au cooldown du rôle.

---

### `/archive` — Archiver un channel

Archive le channel courant en un fichier HTML standalone, avec un rendu fidèle à Discord.

```
/archive
```

Le fichier HTML contient :
- Tous les messages (max 10 000) avec avatars, pseudos colorés, timestamps
- Images, vidéos, fichiers joints (affichés inline)
- Embeds Discord (titre, description, champs, couleurs)
- Réactions en badges
- Indicateurs de réponse
- Header avec infos du serveur, channel, qui a archivé et quand

Le fichier est envoyé dans le chat (**pas sauvegardé** sur le serveur).

**Permissions :** Définies dans `permissions.json` via la commande `archive`.

---

### `/inactifs` — Revue des salons inactifs

Lance une revue interactive paginée des salons texte sans activité récente.

```
/inactifs
/inactifs jours: 60
/inactifs jours: 90 archiver: True
/inactifs exclure_categories: Général, Administration
```

**Options :**
- `jours` — Seuil d'inactivité en jours (défaut 30, max 730)
- `exclure_categories` — Catégories à ignorer (noms séparés par virgules)
- `exclure_salons` — Salons à ignorer (noms ou IDs séparés par virgules)
- `categorie_suppression` — Nom de la catégorie de destination (défaut « À supprimer »)
- `archiver` — `True` pour archiver automatiquement les salons avant de les déplacer

**Interface interactive :**
- Sélectionne les salons sur la page courante
- **Garder** — Retire de la revue (ne touche à rien)
- **À supprimer** — Déplace vers la catégorie de suppression

**Archivage automatique** (`archiver: True`) :
Avant chaque déplacement vers « À supprimer », le bot :
1. Archive le channel en HTML
2. Envoie le fichier d'archive **dans le channel** concerné
3. Sauvegarde le fichier dans le dossier `ARCHIVE_PATH` du serveur (`YYYY-MM-DD_nom-du-channel.html`)
4. Puis déplace le channel et envoie le message d'avertissement

**Permissions :** Définies dans `permissions.json` via la commande `inactifs`.

---

### `/inactifs_ignore` — Exclusions permanentes

Configure les catégories et salons toujours ignorés par `/inactifs`.

```
/inactifs_ignore action: Lister
/inactifs_ignore action: Ajouter catégorie categorie: Général
/inactifs_ignore action: Retirer catégorie categorie: Général
/inactifs_ignore action: Ajouter salon salon: #bienvenue
/inactifs_ignore action: Retirer salon salon: #bienvenue
```

**Permissions :** Définies dans `permissions.json` via la commande `inactifs_ignore`.

---

### `/pfc` — Pierre · Feuille · Ciseaux

```
/pfc
/pfc tours: 5
```

**Permissions :** Définies dans `permissions.json` via la commande `pfc` (par défaut autorisé pour tous).

---

### `/ping`, `/grimmor_info`, `/grimmor_resync`

| Commande | Description | Permission |
|---|---|---|
| `/ping` | Latence WebSocket | Commande `ping` dans `permissions.json` |
| `/grimmor_info` | État RAG (chunks, modèle) | Commande `grimmor_info` dans `permissions.json` |
| `/grimmor_resync` | Force la resync des slash commands | Commande `grimmor_resync` dans `permissions.json` |
| `/permissions` | Configure les permissions par rôle (quota, cooldown) | **Seulement les Administrateurs du serveur Discord** |

---

## 7. Système de permissions

### Vue d'ensemble

Le bot utilise un fichier **`permissions.json`** pour gérer l'accès aux commandes, ainsi que des limites d'utilisation de manière très fine. Chaque rôle Discord peut avoir ses propres permissions. Si un membre a plusieurs rôles, le bot prend la combinaison la plus avantageuse (union des commandes, le quota le plus haut, le cooldown le plus court).

La commande `/permissions` est réservée aux **administrateurs du serveur Discord** et permet de gérer cette configuration directement dans Discord.

### Attributs d'une permission

| Attribut | Description |
|---|---|
| **Commandes** | Liste des commandes autorisées (`ask`, `archive`, `pfc`, etc.), ou `*` pour toutes. |
| **Quota journalier** | Nombre max d'utilisations par jour par utilisateur pour une commande donnée. `0` = illimité. |
| **Cooldown** | Temps d'attente minimum en secondes entre deux utilisations d'une commande. `0` = aucun. |
| **Limites par commande** | `command_daily_quotas` et `command_cooldowns` permettent de limiter `/ask` sans ralentir `/pfc` ou `/archive`. |

### Commandes pour gérer les permissions

- `/permissions action:Voir les permissions` : Affiche l'état actuel.
- `/permissions action:Configurer un rôle role:@Role commandes:ask,pfc quota:10 cooldown:30` : Configure un rôle.
- `/permissions action:Configurer le défaut commandes:ask,archive,pfc quota:0 cooldown:0` : Règle les permissions pour les membres n'ayant aucun rôle configuré.
- `/permissions action:Limiter commande par défaut commande_limitee:/ask quota:10 cooldown:30` : Limite seulement `/ask` par défaut.
- `/permissions action:Limiter commande d'un rôle role:@MJ commande_limitee:/ask quota:100 cooldown:5` : Limite seulement `/ask` pour un rôle.
- `/permissions action:Retirer un rôle role:@Role` : Supprime la config d'un rôle.

> **Note :** La configuration est immédiatement sauvegardée dans le fichier `permissions.json` de votre serveur/VPS.

### Exemple de configuration (`permissions.json`)

```json
{
  "roles": {
    "123456789": {
      "name": "Admin",
      "commands": ["*"],
      "daily_quota": 0,
      "cooldown_seconds": 0
    },
    "987654321": {
      "name": "MJ",
      "commands": ["ask", "archive", "inactifs", "inactifs_ignore", "pfc"],
      "daily_quota": 100,
      "cooldown_seconds": 5
    }
  },
  "default": {
    "commands": ["ask", "archive", "pfc"],
    "daily_quota": 0,
    "cooldown_seconds": 0,
    "command_daily_quotas": {
      "ask": 10
    },
    "command_cooldowns": {
      "ask": 30
    }
  }
}
```

---

## 8. Ajouter de nouveaux documents

1. Dépose les fichiers dans `docs/` (Markdown ou PDF)
2. Relance `python -m rag.ingest`
3. Redémarre le bot : `sudo systemctl restart grimmor.service`

---

## 9. Architecture

```
grimmor/
├── bot.py               # Point d'entrée, classe Grimmor
├── config.py             # Chargement .env, dataclass Config
├── cogs/
│   ├── ask.py            # /ask — RAG NotebookLM-style
│   ├── pfc.py            # /pfc — Pierre Feuille Ciseaux
│   ├── inactifs.py       # /inactifs + /inactifs_ignore
│   ├── archive.py        # /archive — Archivage HTML
│   └── admin.py          # /ping, /grimmor_info, /grimmor_resync
├── db/
│   ├── database.py       # Wrapper aiosqlite, schéma
│   └── queries.py        # Requêtes SQL (quotas, activité, exclusions)
├── rag/
│   ├── retriever.py      # Recherche hybride ChromaDB + BM25
│   ├── generator.py      # Appel LLM (Gemini/Mistral/Groq/Cerebras)
│   ├── graph.py          # Graphe d'entités (couche auxiliaire)
│   └── ingest.py         # Ingestion de documents
├── utils/
│   ├── archiver.py       # Moteur d'archivage HTML
│   ├── formatting.py     # Embeds, couleurs, truncation
│   ├── permissions.py    # Décorateurs check_permissions / require_roles
│   └── text_cleaning.py  # Nettoyage texte PDF
├── archives/             # Fichiers d'archive HTML (auto-archive /inactifs)
├── chroma_db/            # Base vectorielle ChromaDB
├── docs/                 # Documents source à indexer
└── grimmor.db            # Base SQLite (quotas, activité)
```

---

## Notes techniques

### RAG hybride (le cœur de `/ask`)

```
question → graph.resolve() ──┐
       ↓                      │
 multi-query (optionnel) ──┐  │
       ↓                    │  │
 dense (ChromaDB MiniLM) ─┐│  │
 sparse (BM25 tokenizé)  ─┤│  ├→ entity_cards
                          ↓↓  │
                  fusion RRF + reranking
                          ↓
                 8 chunks + cards → LLM (JSON schema)
                          ↓
        {answer, citations[{quote verbatim}], follow_ups}
                          ↓
              Embed Discord NotebookLM-style
```

- ChromaDB en mode **PersistentClient** (pas en mémoire)
- Embedding et BM25 chargés **une seule fois** au démarrage (singletons)
- L'embedding de chaque question `/ask` tourne dans un `ThreadPoolExecutor`
- Tokenizer BM25 reconnaît les bullets `•` (notation des niveaux JDR)
- File d'attente LLM (`LLM_MAX_CONCURRENT`) → pas de 429 si plusieurs utilisateurs

### Couche graphe (couche auxiliaire de précision)

Le pipeline `rules_pipeline/` extrait depuis le PDF un **graphe consolidé**
(`03_consolidated_graph.json`) avec ~466 entités (powers, atouts, clans,
focus, …) et ~177 relations. Chargé en mémoire au boot (~5 MB), il fournit :
- résolution d'entités fuzzy (nom + alias + jaccard)
- inférence de discipline par adjacence des sections
- cartes structurées (level, prérequis, coûts, mécaniques) injectées dans
  le prompt LLM comme vérité de référence

→ permet à `/ask` de répondre correctement à "détaille le niveau 5 d'animalisme"
même si le PDF ne contient pas cette formulation littérale (le graphe sait
que niveau 5 d'Animalisme = "Conquérir la Bête").

Pour régénérer le graphe (lourd, ~30 min, consomme du quota LLM) :
```bash
python -m rules_pipeline.main
```

### Autres notes

- Le tracking d'activité des salons est mis à jour à chaque message reçu
- Les archives HTML sont standalone (CSS intégré, thème sombre Discord)
- Les images dans les archives utilisent les URLs Discord (peuvent expirer)
- `google-generativeai` est marqué deprecated par Google ; migration vers
  `google-genai` à prévoir mais pas urgent

---

## Licence

À définir par l'auteur. Le PDF de règles dans `docs/` reste la propriété
exclusive de la Fédération Camarilla Française et n'est pas distribué avec
ce code.
