# dipsik

Discord bot with AI, powered by the DeepSeek API. Talks casually when
mentioned or replied to, answers with web search when it needs current
info, and ships a handful of fun/utility slash commands.

## Features

- Chats when mentioned, replied to, in DMs, or in configured "open" channels
- Per-channel conversation history with automatic trimming
- Optional web search (via `ddgs`) that the model can call on its own
- Per-user notes (`/remember`, `/forget`, `/memory`)
- Scheduled daily tips per channel
- Customizable personality — see [Personality](#personality)

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A Discord bot token ([Discord Developer Portal](https://discord.com/developers/applications))
- A [DeepSeek API key](https://platform.deepseek.com/)

## Setup

1. Copy the example environment file and fill in your values:

   ```bash
   cp .env.example .env
   ```

   Only `DISCORD_TOKEN` and `DEEPSEEK_API_KEY` are required; everything
   else has a sane default. See `.env.example` for the full list.

2. Install dependencies and run:

   ```bash
   uv sync
   uv run bot.py
   ```

## Running with Docker

```bash
docker compose up -d --build
```

This builds the image from the `Dockerfile` and reads your `.env` file
(see `docker-compose.yml`).

## Commands

| Command | Description |
|---|---|
| `/ping` | Check if the bot is alive |
| `/dice [sides]` | Roll a die (6 sides by default) |
| `/coin` | Flip a coin |
| `/eight_ball <question>` | Ask the magic eight ball |
| `/choose <opt1 \| opt2 \| ...>` | Pick between options |
| `/search <topic>` | Search the web and summarize with sources |
| `/tip` | Get a tip right now |
| `/daily_tip <HH:MM \| off>` | Schedule/disable the daily tip in a channel |
| `/remember <note>` | Save a note about you |
| `/memory` | Show what the bot remembers about you |
| `/forget [text]` | Forget a note, or all of them |
| `/reset` | Clear this channel's conversation history |
| `/history` | How many messages are in memory for this channel |
| `/status` | Technical status |
| `/help` | List all commands |

All commands also work with the `!` prefix (configurable via `PREFIX`) if
slash commands aren't available.

## Personality

The bot's character — tone, language, and the system prompts sent to the
model — lives entirely in `personality.py`, separate from the bot logic in
`bot.py`. Edit `BASE_SYSTEM_PROMPT` there to change how the bot talks, and
`DAILY_TIP_TOPICS` to change what it gives tips about. The shipped prompt
is just an example persona (a casual Spanish-speaking bot) — swap it for
whatever fits your server.

## Data

Conversation history, per-user notes, and daily-tip schedules are stored
as JSON files under `data/`, which is gitignored. Delete that folder to
reset the bot's memory.

## License

_Not yet specified._
