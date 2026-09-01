"""
dipsik, Discord bot with AI (DeepSeek API).
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import AsyncOpenAI

import personality

# Optional web search: if the package isn't installed, the bot keeps
# working and just says it has no internet access.
try:
    from ddgs import DDGS

    DDGS_AVAILABLE = True
except ImportError:
    DDGS = None  # type: ignore[assignment]
    DDGS_AVAILABLE = False


log = logging.getLogger("dipsik")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ddgs uses primp as its internal HTTP client and logs every request at INFO.
logging.getLogger("primp").setLevel(logging.WARNING)

load_dotenv()


# -----------------configuration-----------------------------

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
PREFIX = os.getenv("PREFIX", "!")


def _read_number(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        log.warning("%s is not a valid number; using %s", name, default)
        return default


TEMPERATURE = _read_number("TEMPERATURE", 0.8)
MAX_HISTORY = int(_read_number("MAX_HISTORY", 24))
MAX_TOKENS_RESPONSE = 800
HISTORY_TOKEN_LIMIT = 8000

# "auto" (default) lets the model search the internet when it needs to.
WEB_SEARCH = os.getenv("WEB_SEARCH", "auto").strip().lower() not in {
    "no",
    "off",
    "0",
    "false",
    "desactivada",
    "desactivado",
}

# Channels where the bot responds to every message, without needing a mention.
OPEN_CHANNELS = {
    int(channel_id)
    for channel_id in os.getenv("OPEN_CHANNELS", "").split(",")
    if channel_id.strip().isdigit()
}

# Server ID: if set, slash commands sync only there (they show up instantly,
# instead of waiting for global propagation).
GUILD_ID = os.getenv("GUILD_ID", "").strip()

# Timezone for the daily tips (e.g. "America/Mexico_City").
# If not set, the server's local time is used.
TIMEZONE = os.getenv("TIMEZONE", "").strip()

try:
    LOCAL_TIMEZONE = ZoneInfo(TIMEZONE) if TIMEZONE else None
except Exception:
    log.warning(
        "Invalid timezone %r; using the server's local time",
        TIMEZONE,
    )
    LOCAL_TIMEZONE = None


def get_local_time() -> datetime:
    """Current time in the configured zone (or the server's local time)."""
    return datetime.now(LOCAL_TIMEZONE) if LOCAL_TIMEZONE else datetime.now()


if not DISCORD_TOKEN or not DEEPSEEK_API_KEY:
    raise ValueError(
        "Missing DISCORD_TOKEN or DEEPSEEK_API_KEY in the .env file "
        "(see .env.example for all the options)."
    )


deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    max_retries=2,
    timeout=60,
)


# -------------------persistent data--------------------------

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"

HISTORY_FILE = DATA_DIRECTORY / "history.json"
NOTES_FILE = DATA_DIRECTORY / "notes.json"
SETTINGS_FILE = DATA_DIRECTORY / "settings.json"


def load_json(file: Path, default):
    try:
        if file.exists():
            return json.loads(file.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read %s: %s", file, exc)

    return default


async def save_json(file: Path, data) -> None:
    """Saves data atomically without blocking the event loop."""

    def write_file() -> None:
        DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

        temporary_file = file.with_suffix(".tmp")

        temporary_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        os.replace(temporary_file, file)

    try:
        await asyncio.to_thread(write_file)
    except Exception as exc:
        log.warning("Could not save %s: %s", file, exc)


try:
    history_by_channel = {
        int(key): value
        for key, value in load_json(HISTORY_FILE, {}).items()
    }

    notes_by_user = {
        int(key): value
        for key, value in load_json(NOTES_FILE, {}).items()
    }

    settings = load_json(SETTINGS_FILE, {})

except Exception as exc:
    log.warning(
        "Saved data is corrupted; starting fresh (%s)",
        exc,
    )

    history_by_channel = {}
    notes_by_user = {}
    settings = {}


MAX_NOTES_PER_USER = 30


# Daily tip schedule: {"<channel_id>": {"time": "HH:MM", "last_sent": ...}}
settings.setdefault("daily_tips", {})


def is_valid_time(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:[01]?\d|2[0-3]):[0-5]\d",
            value.strip(),
        )
    )


def normalize_time(value: str) -> str:
    hours, minutes = value.strip().split(":")
    return f"{int(hours):02d}:{int(minutes):02d}"


# If .env defines DAILY_TIP_CHANNEL and DAILY_TIP_TIME, they're seeded as
# initial values (a later /daily_tip command takes priority).
daily_tip_channel_env = os.getenv("DAILY_TIP_CHANNEL", "").strip()
daily_tip_time_env = os.getenv("DAILY_TIP_TIME", "").strip()

if (
    daily_tip_channel_env.isdigit()
    and is_valid_time(daily_tip_time_env)
):
    settings["daily_tips"].setdefault(
        daily_tip_channel_env,
        {},
    ).setdefault(
        "time",
        normalize_time(daily_tip_time_env),
    )


def estimate_tokens(text: str) -> int:
    """Quick token estimate (enough for trimming history)."""
    return len(text) // 4


def trim_history(history: list[dict]) -> list[dict]:
    """Limits history by message count and by token budget."""

    while len(history) > MAX_HISTORY:
        history.pop(0)

    total_tokens = sum(
        estimate_tokens(message["content"])
        for message in history
    )

    while (
        total_tokens > HISTORY_TOKEN_LIMIT
        and len(history) > 2
    ):
        total_tokens -= estimate_tokens(
            history.pop(0)["content"]
        )

    return history


for channel_id, history in list(history_by_channel.items()):
    history_by_channel[channel_id] = trim_history(history)


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Searches the internet for current information (news, weather, "
            "prices, recent events, technical data...). Use it when the "
            "question requires recent information or something you don't know."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query, in Spanish or English."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


# The bot's only fixed texts: only for cases where no AI is available (the
# API failed or returned empty text). Everything else is generated by the
# model with its personality.
API_ERROR_MESSAGE = (
    "I couldn't reach DeepSeek right now. "
    "Try again in a moment."
)

EMPTY_RESPONSE_MESSAGE = "I drew a blank on that one."


# ----------------- web search ---------------------------

def _search_sync(query: str):
    """Synchronous DuckDuckGo search (runs in a thread)."""

    with DDGS() as ddgs:
        return ddgs.text(
            query,
            max_results=6,
        )


async def search_web(query: str) -> str:
    """Searches DuckDuckGo and returns the first results as plain text.

    Returns a string starting with "ERROR:" if the search isn't possible,
    so the model can react honestly.
    """

    if not DDGS_AVAILABLE:
        return (
            "ERROR: web search isn't available in this deployment."
        )

    try:
        results = await asyncio.to_thread(
            _search_sync,
            query,
        )
    except Exception as exc:
        log.warning(
            "Search failed (%r): %s",
            query,
            exc,
        )

        return f"ERROR: the search could not be performed: {exc}"

    if not results:
        return "The search returned no results."

    blocks = []

    for index, result in enumerate(results, 1):
        title = (result.get("title") or "").strip()
        url = (result.get("href") or "").strip()
        body = (result.get("body") or "").strip()

        blocks.append(
            f"{index}. {title}\n"
            f"   {url}\n"
            f"   {body}"
        )

    return "\n\n".join(blocks)


# ---------------text generation--------------------------------

async def generate_response(
    messages: list[dict],
    *,
    use_tools: bool = True,
    max_tokens: Optional[int] = None,
) -> str:
    """Requests a streamed response from DeepSeek.

    - If the model requests `search_web`, the search runs and the call
      is repeated with the results (max 2 rounds).
    """

    tools = (
        [WEB_SEARCH_TOOL]
        if (
            use_tools
            and WEB_SEARCH
            and DDGS_AVAILABLE
        )
        else None
    )

    for _round in range(2):
        try:
            stream = await deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=max_tokens or MAX_TOKENS_RESPONSE,
                tools=tools,
                stream=True,
                timeout=120,
            )

        except Exception as exc:
            log.error(
                "Error contacting DeepSeek: %s",
                exc,
            )
            return API_ERROR_MESSAGE

        text = ""
        tool_calls: dict[int, dict] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta is None:
                continue

            if delta.content:
                text += delta.content

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    index = tool_call.index or 0

                    entry = tool_calls.setdefault(
                        index,
                        {
                            "id": None,
                            "name": None,
                            "arguments": "",
                        },
                    )

                    if tool_call.id:
                        entry["id"] = tool_call.id

                    if (
                        tool_call.function
                        and tool_call.function.name
                    ):
                        entry["name"] = tool_call.function.name

                    if (
                        tool_call.function
                        and tool_call.function.arguments
                    ):
                        entry["arguments"] += (
                            tool_call.function.arguments
                        )

        if tool_calls:
            # The model wants to search the internet: log and execute it.
            formatted_tool_calls = [
                {
                    "id": data["id"] or f"tool_call_{index}",
                    "type": "function",
                    "function": {
                        "name": data["name"] or "search_web",
                        "arguments": data["arguments"] or "{}",
                    },
                }
                for index, data in sorted(tool_calls.items())
            ]

            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": formatted_tool_calls,
                }
            )

            for tool_call in formatted_tool_calls:
                try:
                    arguments = json.loads(
                        tool_call["function"]["arguments"] or "{}"
                    )
                except json.JSONDecodeError:
                    arguments = {}

                query = (
                    str(arguments.get("query", "")).strip()
                    or "recent information"
                )

                result = await search_web(query)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                )

            # Second round: respond using the search results.
            continue

        if text.strip():
            return text.strip()

        return EMPTY_RESPONSE_MESSAGE

    # The model kept requesting searches without responding: return whatever
    # we have.
    return text.strip() or EMPTY_RESPONSE_MESSAGE


async def generate_ai_phrase(
    user_message: str,
    max_tokens: int = 100,
) -> str:
    """Generates a short phrase with dipsik's personality (for fun commands
    and reactions). No tools or channel history."""

    messages = [
        {
            "role": "system",
            "content": personality.build_system_prompt(
                get_local_time(),
                web_search_available=False,
            ),
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    return await generate_response(
        messages,
        use_tools=False,
        max_tokens=max_tokens,
    )


async def send_response(
    channel: discord.abc.Messageable,
    text: str,
    reply_to: Optional[discord.Message] = None,
) -> None:

    chunks = [
        text[index:index + 2000]
        for index in range(0, len(text), 2000)
    ]

    for index, chunk in enumerate(chunks):
        try:
            if index == 0 and reply_to is not None:
                try:
                    await reply_to.reply(
                        chunk,
                        mention_author=False,
                    )
                    continue
                except discord.HTTPException:
                    pass  # no permission to reply

            await channel.send(chunk)

        except discord.HTTPException as exc:
            log.warning(
                "Could not send response: %s",
                exc,
            )
            break


# ----------------------- daily tips ----------------------------------------

async def generate_daily_tip() -> Optional[str]:
    """Generates a daily tip with dipsik's personality.

    Returns None if the API isn't available; in that case nothing is made up.
    """

    topic = random.choice(personality.DAILY_TIP_TOPICS)

    messages = [
        {
            "role": "system",
            "content": personality.build_daily_tip_system_prompt(),
        },
        {
            "role": "user",
            "content": (
                f"Today's topic: {topic}. Give me your daily tip."
            ),
        },
    ]

    response = await generate_response(
        messages,
        use_tools=False,
    )

    if response in (
        API_ERROR_MESSAGE,
        EMPTY_RESPONSE_MESSAGE,
    ):
        return None

    return response


async def daily_tip_task() -> None:
    """Checks every minute whether a daily tip is due in any channel."""

    while True:
        # Wake up right after each minute changes.
        await asyncio.sleep(
            61 - (time.monotonic() % 60)
        )

        current_time = get_local_time()

        current_minute = current_time.strftime("%H:%M")
        current_key = current_time.strftime(
            "%Y-%m-%d %H:%M"
        )

        for channel_id, data in list(
            settings["daily_tips"].items()
        ):
            if data.get("time") != current_minute:
                continue

            if data.get("last_sent") == current_key:
                continue  # already sent during this exact minute

            channel = bot.get_channel(int(channel_id))

            if channel is None:
                continue

            tip = await generate_daily_tip()

            if tip is None:
                tip = await generate_daily_tip()  # one retry

            if tip is None:
                log.warning(
                    "Daily tip for channel %s not sent "
                    "(API unavailable)",
                    channel_id,
                )
                continue

            try:
                await channel.send(
                    f"**Tip of the day**\n{tip}"
                )

            except discord.HTTPException as exc:
                log.warning(
                    "Could not send daily tip to %s: %s",
                    channel_id,
                    exc,
                )
                continue

            data["last_sent"] = current_key

            await save_json(
                SETTINGS_FILE,
                settings,
            )

            log.info(
                "Daily tip sent to channel %s at %s",
                channel_id,
                current_minute,
            )


# -------------------bot----------------------------

intents = discord.Intents.default()
intents.message_content = True


class DipsikBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_time = time.time()
        self._status_task: Optional[asyncio.Task] = None
        self._daily_tip_task: Optional[asyncio.Task] = None

    async def setup_hook(self) -> None:
        guild = (
            discord.Object(id=int(GUILD_ID))
            if GUILD_ID.isdigit()
            else None
        )

        try:
            await self.tree.sync(guild=guild)

            log.info(
                "Slash commands synced%s",
                (
                    f" for guild {GUILD_ID}"
                    if guild
                    else " globally"
                ),
            )

        except Exception as exc:
            log.warning(
                "Could not sync slash commands: %s",
                exc,
            )


bot = DipsikBot(
    command_prefix=PREFIX,
    intents=intents,
    help_command=None,
)


BOT_STATUSES = [
    ("listening", "your next question"),
    ("playing", "at being human"),
    ("watching", "the chat like it's nothing"),
    ("listening", "lo-fi and thinking"),
    ("watching", f"{PREFIX}help"),
]


async def rotate_status() -> None:
    """Changes the bot's status every 10 minutes."""

    while True:
        for activity_type, activity_name in BOT_STATUSES:
            activity = discord.Activity(
                type=getattr(
                    discord.ActivityType,
                    activity_type,
                ),
                name=activity_name,
            )

            try:
                await bot.change_presence(
                    activity=activity
                )
            except discord.HTTPException:
                pass

            await asyncio.sleep(600)


@bot.event
async def on_ready():
    log.info(
        "Bot connected as %s (ID: %s)",
        bot.user,
        bot.user.id,
    )

    if (
        bot._status_task is None
        or bot._status_task.done()
    ):
        bot._status_task = bot.loop.create_task(
            rotate_status()
        )

    if (
        bot._daily_tip_task is None
        or bot._daily_tip_task.done()
    ):
        bot._daily_tip_task = bot.loop.create_task(
            daily_tip_task()
        )

    scheduled_tips = len(
        settings["daily_tips"]
    )

    if scheduled_tips:
        log.info(
            "%d daily tip(s) scheduled",
            scheduled_tips,
        )


@bot.event
async def on_command_error(ctx, error):
    if isinstance(
        error,
        commands.CommandNotFound,
    ):
        return

    if isinstance(
        error,
        commands.MissingRequiredArgument,
    ):
        await ctx.reply(
            f"You're missing an argument. Usage: `{ctx.prefix}{ctx.command} "
            f"{ctx.command.signature}`",
            mention_author=False,
        )
        return

    if isinstance(
        error,
        commands.BadArgument,
    ):
        await ctx.reply(
            "That argument doesn't add up. "
            "Check the data type.",
            mention_author=False,
        )
        return

    log.error(
        "Error in command %s: %s",
        ctx.command,
        error,
    )

    try:
        await ctx.reply(
            "Oops, something broke here. "
            "Try again.",
            mention_author=False,
        )
    except discord.HTTPException:
        pass


# ---------------conversation--------------------------------

CHANNEL_COOLDOWNS: dict[int, float] = {}

MINIMUM_RESPONSE_INTERVAL = 1.0  # minimum seconds between replies in the same channel


async def is_reply_to_bot(
    message: discord.Message,
) -> bool:
    """True if the message is a reply to one of the bot's messages."""

    if message.reference is None:
        return False

    try:
        referenced_message = message.reference.resolved

        if referenced_message is None:
            referenced_message = (
                await message.channel.fetch_message(
                    message.reference.message_id
                )
            )

        return referenced_message.author.id == bot.user.id

    except (
        discord.NotFound,
        discord.HTTPException,
    ):
        return False


def clean_mentions(message: discord.Message) -> str:
    """Strips mentions and leaves display names, so the model can see who
    it's talking to."""

    content = message.content

    for mention in message.mentions:
        display_name = f"@{mention.display_name}"

        content = (
            content
            .replace(
                f"<@{mention.id}>",
                display_name,
            )
            .replace(
                f"<@!{mention.id}>",
                display_name,
            )
        )

    content = re.sub(
        r"<@&?\d+>",
        "",
        content,
    )  # role mentions

    return content.strip()


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Commands (slash and prefix) take priority over conversation.
    context = await bot.get_context(message)

    if context.valid:
        await bot.invoke(context)
        return

    is_dm = message.guild is None
    is_mentioned = bot.user in message.mentions
    is_reply = await is_reply_to_bot(message)

    is_open_channel = (
        is_dm
        or message.channel.id in OPEN_CHANNELS
    )

    if not (
        is_dm
        or is_mentioned
        or is_reply
        or is_open_channel
    ):
        return

    # Small per-channel anti-spam cooldown.
    current_time = time.monotonic()

    if (
        current_time
        - CHANNEL_COOLDOWNS.get(
            message.channel.id,
            0,
        )
        < MINIMUM_RESPONSE_INTERVAL
    ):
        return

    CHANNEL_COOLDOWNS[message.channel.id] = current_time

    content = clean_mentions(message)

    if message.attachments:
        attachments = ", ".join(
            attachment.filename
            for attachment in message.attachments
        )

        content = (
            f"{content}\n[Attachments: {attachments}]"
            if content
            else f"[Attachments: {attachments}]"
        )

    if not content:
        content = (
            "The user mentioned you without writing any text. "
            "Greet them briefly and naturally."
        )

    history = history_by_channel.get(
        message.channel.id,
        [],
    )

    messages = [
        {
            "role": "system",
            "content": personality.build_system_prompt(
                get_local_time(),
                web_search_available=(WEB_SEARCH and DDGS_AVAILABLE),
                user_notes=notes_by_user.get(message.author.id),
            ),
        }
    ]

    messages += history

    messages.append(
        {
            "role": "user",
            "content": (
                f"@{message.author.display_name}: "
                f"{content}"
            ),
        }
    )

    try:
        async with message.channel.typing():
            response = await generate_response(
                messages
            )

    except Exception as exc:
        log.error(
            "Unexpected error while responding: %s",
            exc,
        )
        response = API_ERROR_MESSAGE

    # Save the conversation (always, even if sending fails).
    history.append(
        {
            "role": "user",
            "content": (
                f"@{message.author.display_name}: "
                f"{content}"
            ),
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    history_by_channel[
        message.channel.id
    ] = trim_history(history)

    await save_json(
        HISTORY_FILE,
        history_by_channel,
    )

    await send_response(
        message.channel,
        response,
        reply_to=message,
    )


# ------------ commands -------------------

@bot.hybrid_command(
    name="ping",
    description="Check if I'm still alive",
)
async def cmd_ping(ctx):
    # Intentionally static: this is a connectivity check that must work
    # even if the DeepSeek API is down.
    await ctx.reply(
        f"pong — {round(bot.latency * 1000)} ms",
        mention_author=False,
    )


@bot.hybrid_command(
    name="dice",
    description="Roll an N-sided die (6 by default)",
)
async def cmd_dice(
    ctx,
    sides: int = 6,
):
    await ctx.defer()

    sides = max(
        2,
        min(sides, 1000),
    )

    result = random.randint(
        1,
        sides,
    )  # randomness is the function; the phrase comes from the AI

    reaction = await generate_ai_phrase(
        f"The user just rolled a {sides}-sided die "
        f"and got **{result}**. "
        "React in ONE short sentence, "
        "with your personality and humor."
    )

    await ctx.reply(
        reaction,
        mention_author=False,
    )


@bot.hybrid_command(
    name="coin",
    description="Flip a coin: heads or tails",
)
async def cmd_coin(ctx):
    await ctx.defer()

    result = random.choice(
        ["heads", "tails"]
    )

    reaction = await generate_ai_phrase(
        f"The user flipped a coin and got **{result}**. "
        "React in ONE short sentence, "
        "with your personality and humor."
    )

    await ctx.reply(
        reaction,
        mention_author=False,
    )


@bot.hybrid_command(
    name="eight_ball",
    aliases=["8ball"],
    description="Ask the magic eight ball",
)
async def cmd_eight_ball(
    ctx,
    *,
    question: str,
):
    await ctx.defer()

    response = await generate_ai_phrase(
        f"The user asked the magic eight ball: "
        f'"{question}". Answer like the magic eight ball with your '
        "personality: ONE short sentence; randomly mix "
        "yes, no, maybe, or creative variants."
    )

    await ctx.reply(
        response,
        mention_author=False,
    )


@bot.hybrid_command(
    name="choose",
    description="Pick between options separated by | or commas",
)
async def cmd_choose(
    ctx,
    *,
    options: str,
):
    option_list = [
        option.strip()
        for option in re.split(
            r"\||,",
            options,
        )
        if option.strip()
    ]

    if len(option_list) < 2:
        await ctx.reply(
            "Give me at least two options separated by `|`, please.",
            mention_author=False,
        )
        return

    await ctx.defer()

    selected = random.choice(
        option_list
    )

    reaction = await generate_ai_phrase(
        f"The user asked you to choose between these options: "
        f"{', '.join(option_list)}. "
        f'You picked "{selected}". '
        "Announce it in ONE short sentence, "
        "with your personality and humor."
    )

    if reaction in (
        API_ERROR_MESSAGE,
        EMPTY_RESPONSE_MESSAGE,
    ):
        await ctx.reply(
            f"I picked: **{selected}**",
            mention_author=False,
        )
    else:
        await ctx.reply(
            reaction,
            mention_author=False,
        )


@bot.hybrid_command(
    name="search",
    description="Search the internet and summarize with sources",
)
async def cmd_search(
    ctx,
    *,
    query: str,
):
    await ctx.defer()

    results = await search_web(query)

    if (
        results.startswith("ERROR")
        or results == "The search returned no results."
    ):
        await ctx.reply(
            "I couldn't search the internet right now. "
            "Check that the server has a connection.",
            mention_author=False,
        )
        return

    messages = [
        {
            "role": "system",
            "content": personality.build_search_summary_system_prompt(),
        },
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"Results:\n{results}"
            ),
        },
    ]

    response = await generate_response(
        messages,
        use_tools=False,
    )

    await send_response(
        ctx.channel,
        response,
        reply_to=ctx.message,
    )


@bot.hybrid_command(
    name="tip",
    description="Give you a tip right now",
)
async def cmd_tip(ctx):
    await ctx.defer()

    tip = await generate_daily_tip()

    if tip is None:
        await ctx.reply(
            API_ERROR_MESSAGE,
            mention_author=False,
        )
        return

    await ctx.reply(
        f"**Tip of the day**\n{tip}",
        mention_author=False,
    )


@bot.hybrid_command(
    name="daily_tip",
    description="Schedule the daily tip in this channel (e.g. 09:00, or 'off' to disable it)",
)
async def cmd_daily_tip(
    ctx,
    *,
    time_value: Optional[str] = None,
):
    channel_key = str(
        ctx.channel.id
    )

    timezone_text = (
        f" (timezone: {LOCAL_TIMEZONE.key})"
        if LOCAL_TIMEZONE
        else " (server time)"
    )

    if time_value is None:
        scheduled = settings[
            "daily_tips"
        ].get(channel_key)

        if scheduled:
            last_sent = scheduled.get(
                "last_sent",
                "never",
            )

            await ctx.reply(
                f"Daily tip scheduled in this channel at "
                f"**{scheduled['time']}**{timezone_text}.\n"
                f"Last sent: {last_sent}. "
                f"To change it: "
                f"`{PREFIX}daily_tip HH:MM`.",
                mention_author=False,
            )
        else:
            await ctx.reply(
                f"No daily tip is scheduled in this channel. "
                f"Schedule one with "
                f"`{PREFIX}daily_tip HH:MM` "
                f"(e.g. `{PREFIX}daily_tip 09:00`).",
                mention_author=False,
            )

        return

    if time_value.strip().lower() in {
        "off",
        "disable",
        "none",
        "no",
    }:
        if settings["daily_tips"].pop(
            channel_key,
            None,
        ):
            await save_json(
                SETTINGS_FILE,
                settings,
            )

            await ctx.reply(
                "Daily tip disabled in this channel.",
                mention_author=False,
            )
        else:
            await ctx.reply(
                "There was no tip scheduled here.",
                mention_author=False,
            )

        return

    if not is_valid_time(time_value):
        await ctx.reply(
            "Invalid time format. Use HH:MM, "
            "for example `09:00` or `18:30`.",
            mention_author=False,
        )
        return

    entry = settings[
        "daily_tips"
    ].setdefault(
        channel_key,
        {},
    )

    entry["time"] = normalize_time(
        time_value
    )

    # allow it to be sent today with the new time
    entry.pop(
        "last_sent",
        None,
    )

    await save_json(
        SETTINGS_FILE,
        settings,
    )

    await ctx.reply(
        f"Done! I'll send a daily tip here every day at "
        f"**{entry['time']}**{timezone_text}.",
        mention_author=False,
    )


@bot.hybrid_command(
    name="remember",
    description="Save a note about you for future conversations",
)
async def cmd_remember(
    ctx,
    *,
    note: str,
):
    notes = notes_by_user.setdefault(
        ctx.author.id,
        [],
    )

    if len(notes) >= MAX_NOTES_PER_USER:
        notes.pop(0)

    notes.append(
        note.strip()
    )

    await save_json(
        NOTES_FILE,
        notes_by_user,
    )

    await ctx.reply(
        f"Noted: *{note.strip()}*",
        mention_author=False,
    )


@bot.hybrid_command(
    name="forget",
    description="Forget a note (or all of them, if you don't type anything)",
)
async def cmd_forget(
    ctx,
    *,
    text: Optional[str] = None,
):
    notes = notes_by_user.get(
        ctx.author.id,
        [],
    )

    if not notes:
        await ctx.reply(
            "I don't have anything noted about you, "
            "so there's nothing to forget.",
            mention_author=False,
        )
        return

    if text is None:
        notes_by_user[
            ctx.author.id
        ] = []

        await save_json(
            NOTES_FILE,
            notes_by_user,
        )

        await ctx.reply(
            "Done, clean slate.",
            mention_author=False,
        )
        return

    searched_text = text.strip().lower()

    remaining_notes = [
        note
        for note in notes
        if searched_text not in note.lower()
    ]

    deleted_count = (
        len(notes)
        - len(remaining_notes)
    )

    notes_by_user[
        ctx.author.id
    ] = remaining_notes

    await save_json(
        NOTES_FILE,
        notes_by_user,
    )

    if deleted_count == 0:
        await ctx.reply(
            f"I couldn't find any note mentioning "
            f'"{text.strip()}".',
            mention_author=False,
        )
    else:
        await ctx.reply(
            f"Forgot {deleted_count} note(s).",
            mention_author=False,
        )


@bot.hybrid_command(
    name="memory",
    description="Show what I remember about you",
)
async def cmd_memory(ctx):
    notes = notes_by_user.get(
        ctx.author.id,
        [],
    )

    if not notes:
        await ctx.reply(
            "I don't have any notes about you yet. "
            f"Use `{PREFIX}remember <something>` and I'll save it.",
            mention_author=False,
        )
        return

    note_list = "\n".join(
        f"• {note}"
        for note in notes
    )

    await ctx.reply(
        f"Here's what I remember about you:\n{note_list}",
        mention_author=False,
    )


@bot.hybrid_command(
    name="reset",
    description="Clear this channel's conversation history",
)
async def cmd_reset(ctx):
    history_by_channel.pop(
        ctx.channel.id,
        None,
    )

    await save_json(
        HISTORY_FILE,
        history_by_channel,
    )

    await ctx.reply(
        "This channel's memory has been cleared. "
        "Starting fresh.",
        mention_author=False,
    )


@bot.hybrid_command(
    name="history",
    description="How many messages I remember in this channel",
)
async def cmd_history(ctx):
    message_count = len(
        history_by_channel.get(
            ctx.channel.id,
            [],
        )
    )

    await ctx.reply(
        f"I have {message_count} messages in memory for this channel "
        f"(max {MAX_HISTORY}).",
        mention_author=False,
    )


@bot.hybrid_command(
    name="status",
    description="Bot's technical status",
)
async def cmd_status(ctx):
    uptime_seconds = int(
        time.time()
        - bot.start_time
    )

    hours, remainder = divmod(
        uptime_seconds,
        3600,
    )

    minutes, _ = divmod(
        remainder,
        60,
    )

    embed = discord.Embed(
        title="dipsik Status",
        color=0x5865F2,
    )

    embed.add_field(
        name="Model",
        value=f"`{DEEPSEEK_MODEL}`",
        inline=True,
    )

    embed.add_field(
        name="Temperature",
        value=str(TEMPERATURE),
        inline=True,
    )

    embed.add_field(
        name="Web search",
        value=(
            "Available"
            if (
                WEB_SEARCH
                and DDGS_AVAILABLE
            )
            else "Disabled"
        ),
        inline=True,
    )

    embed.add_field(
        name="History per channel",
        value=f"{MAX_HISTORY} messages",
        inline=True,
    )

    embed.add_field(
        name="Saved notes",
        value=(
            f"{sum(len(value) for value in notes_by_user.values())} "
            f"across {len(notes_by_user)} user(s)"
        ),
        inline=True,
    )

    embed.add_field(
        name="Daily tips",
        value=(
            f"{len(settings['daily_tips'])} scheduled"
        ),
        inline=True,
    )

    embed.add_field(
        name="Latency",
        value=f"{round(bot.latency * 1000)} ms",
        inline=True,
    )

    embed.set_footer(
        text=f"Online for {hours}h {minutes}m"
    )

    await ctx.reply(
        embed=embed,
        mention_author=False,
    )


@bot.hybrid_command(
    name="help",
    description="Show all commands",
)
async def cmd_help(ctx):
    embed = discord.Embed(
        title="Need something?",
        description=(
            "For **regular chat**, just mention me or reply to me. "
            f"I also use the `{PREFIX}` prefix."
        ),
        color=0x5865F2,
    )

    embed.add_field(
        name="Commands",
        value=(
            "`/ping` — am I alive?\n"
            "`/dice [sides]` — roll a die\n"
            "`/coin` — heads or tails\n"
            "`/eight_ball <question>` — the magic eight ball\n"
            "`/choose <opt1 | opt2 | ...>` — I'll pick for you\n"
            "`/search <topic>` — I'll search the internet and summarize\n"
            "`/tip` — a tip right now\n"
            "`/daily_tip <HH:MM | off>` — daily tip in this channel\n"
            "`/remember <note>` — I'll save something about you\n"
            "`/memory` — what I remember about you\n"
            "`/forget [text]` — forget a note or all of them\n"
            "`/reset` — clear this channel's history\n"
            "`/history` — how many messages I remember\n"
            "`/status` — technical status\n"
            "`/help` — this right here"
        ),
        inline=False,
    )

    await ctx.reply(
        embed=embed,
        mention_author=False,
    )


# ---------------------startup---------------------

if __name__ == "__main__":
    bot.run(
        DISCORD_TOKEN,
        log_handler=None,
    )
