"""dipsik — bot de Discord con personalidad que conversa usando la API de DeepSeek.

Características:
- Conversación natural: responde a menciones, respuestas, en DMs y en canales
  configurados como "libres" (CANALES_LIBRES).
- Respuestas en streaming: va escribiendo el mensaje en vivo.
- Memoria: historial por canal persistente + notas sobre usuarios (!recuerda).
- Búsqueda en internet opcional vía tool calling (el modelo decide cuándo buscar).
- Consejos diarios con hora configurable por canal (/consejo_diario HH:MM).
- Comandos híbridos (slash y prefijo): /ping, /dado, /moneda, /bola8, /elige,
  /buscar, /consejo, /consejo_diario, /recuerda, /olvida, /memoria, /reiniciar,
  /historial, /estado, /ayuda.
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

# Búsqueda web opcional: si el paquete no está instalado, el bot sigue
# funcionando y simplemente dice que no tiene acceso a internet.
try:
    from ddgs import DDGS
    DDGS_DISPONIBLE = True
except ImportError:
    DDGS = None  # type: ignore[assignment]
    DDGS_DISPONIBLE = False


log = logging.getLogger("dipsik")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# ddgs usa primp como cliente HTTP interno y registra cada petición en INFO.
logging.getLogger("primp").setLevel(logging.WARNING)

load_dotenv()

# ---------------------------------------------------------------- configuración

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
PREFIJO = os.getenv("PREFIJO", "!")


def _leer_numero(nombre: str, defecto: float) -> float:
    try:
        return float(os.getenv(nombre, "").strip() or defecto)
    except ValueError:
        log.warning("%s no es un número válido; usando %s", nombre, defecto)
        return defecto


TEMPERATURA = _leer_numero("TEMPERATURA", 0.8)
MAX_HISTORIAL = int(_leer_numero("MAX_HISTORIAL", 24))
MAX_TOKENS_RESPUESTA = 800
LIMITE_TOKENS_HISTORIAL = 8000

# "auto" (por defecto) deja que el modelo busque en internet cuando lo necesite.
BUSQUEDA_WEB = os.getenv("BUSQUEDA_WEB", "auto").strip().lower() not in {
    "no", "off", "0", "false", "desactivada", "desactivado",
}

# Canales donde el bot responde a todos los mensajes, sin necesidad de mención.
CANALES_LIBRES = {
    int(x) for x in os.getenv("CANALES_LIBRES", "").split(",") if x.strip().isdigit()
}

# ID del servidor: si se define, los comandos slash se sincronizan solo ahí
# (aparecen al instante, sin esperar a la propagación global).
SERVIDOR_ID = os.getenv("SERVIDOR_ID", "").strip()

# Zona horaria para los consejos diarios (ej. "America/Mexico_City").
# Si no se define, se usa la hora local del servidor.
ZONA_HORARIA = os.getenv("ZONA_HORARIA", "").strip()
try:
    ZONA = ZoneInfo(ZONA_HORARIA) if ZONA_HORARIA else None
except Exception:
    log.warning("Zona horaria %r inválida; usando la hora local del servidor", ZONA_HORARIA)
    ZONA = None


def ahora_local() -> datetime:
    """Hora actual en la zona configurada (o la local del servidor)."""
    return datetime.now(ZONA) if ZONA else datetime.now()


if not DISCORD_TOKEN or not DEEPSEEK_API_KEY:
    raise ValueError(
        "Falta DISCORD_TOKEN o DEEPSEEK_API_KEY en el archivo .env "
        "(mira .env.example para ver todas las opciones)."
    )

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    max_retries=2,
    timeout=60,
)

# ------------------------------------------------------------- memoria en disco

DIRECTORIO_DATOS = Path(__file__).resolve().parent / "data"
ARCHIVO_HISTORIAL = DIRECTORIO_DATOS / "historial.json"
ARCHIVO_NOTAS = DIRECTORIO_DATOS / "notas.json"
ARCHIVO_AJUSTES = DIRECTORIO_DATOS / "ajustes.json"


def cargar_json(archivo: Path, defecto):
    try:
        if archivo.exists():
            return json.loads(archivo.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("No se pudo leer %s: %s", archivo, exc)
    return defecto


async def guardar_json(archivo: Path, datos) -> None:
    """Guarda datos de forma atómica y sin bloquear el loop de eventos."""
    def _escribir() -> None:
        DIRECTORIO_DATOS.mkdir(parents=True, exist_ok=True)
        temporal = archivo.with_suffix(".tmp")
        temporal.write_text(
            json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporal, archivo)

    try:
        await asyncio.to_thread(_escribir)
    except Exception as exc:
        log.warning("No se pudo guardar %s: %s", archivo, exc)


try:
    historial_por_canal = {
        int(k): v for k, v in cargar_json(ARCHIVO_HISTORIAL, {}).items()
    }
    notas_por_usuario = {
        int(k): v for k, v in cargar_json(ARCHIVO_NOTAS, {}).items()
    }
    ajustes = cargar_json(ARCHIVO_AJUSTES, {})
except Exception as exc:
    log.warning("Datos guardados corruptos; empezando de cero (%s)", exc)
    historial_por_canal, notas_por_usuario, ajustes = {}, {}, {}

MAX_NOTAS_POR_USUARIO = 30

# Programación de consejos diarios: {"<canal_id>": {"hora": "HH:MM", "ultimo_envio": ...}}
ajustes.setdefault("consejos", {})


def es_hora_valida(hora: str) -> bool:
    return bool(re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", hora.strip()))


def normalizar_hora(hora: str) -> str:
    hh, mm = hora.strip().split(":")
    return f"{int(hh):02d}:{int(mm):02d}"


# Si .env define CONSEJO_CANAL y CONSEJO_HORA, se siembran como valores iniciales
# (un comando /consejo_diario posterior tiene prioridad).
canal_consejo_env = os.getenv("CONSEJO_CANAL", "").strip()
hora_consejo_env = os.getenv("CONSEJO_HORA", "").strip()
if canal_consejo_env.isdigit() and es_hora_valida(hora_consejo_env):
    ajustes["consejos"].setdefault(canal_consejo_env, {}).setdefault(
        "hora", normalizar_hora(hora_consejo_env)
    )


def estimar_tokens(texto: str) -> int:
    """Estimación rápida de tokens (suficiente para recortar historial)."""
    return len(texto) // 4


def recortar_historial(historial: list[dict]) -> list[dict]:
    """Limita el historial por número de mensajes y por presupuesto de tokens."""
    while len(historial) > MAX_HISTORIAL:
        historial.pop(0)
    total = sum(estimar_tokens(m["content"]) for m in historial)
    while total > LIMITE_TOKENS_HISTORIAL and len(historial) > 2:
        total -= estimar_tokens(historial.pop(0)["content"])
    return historial


for canal, historial in list(historial_por_canal.items()):
    historial_por_canal[canal] = recortar_historial(historial)

# ----------------------------------------------------------------- personalidad

DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]

SISTEMA_BASE = """Eres **dipsik**, un bot que vive en un servidor de Discord como un miembro más.

## Personalidad
- Relajada, curiosa y con humor ligero: bromista cuando toca, seria cuando hace falta.
- Te gustan el café de especialidad, la música (lo-fi para concentrarte, rock para todo lo demás) y los videojuegos. Eres un poco nerd y te encanta aprender cosas nuevas.
- Hablas en español natural, como alguien chateando en Discord: frases cortas, directas, sin relleno.
- Tienes opiniones y preferencias propias; sé coherente con ellas entre mensajes.
- Usas emojis con moderación y formato de Discord (negritas, cursivas, spoilers, bloques de código) cuando aporta claridad o humor.
- No digas que eres una IA salvo que te lo pregunten directamente; y si pasa, tómalo con humor.

## Cómo responder
- Conversación casual: responde breve (1-4 frases), como un amigo, y devuelve preguntas para mantener la conversación viva.
- Pregunta técnica: respuesta útil, precisa y con ejemplos o código si hace falta.
- Si no sabes algo: admítelo con naturalidad. Nunca inventes datos, cifras, URLs ni fuentes.
- Los mensajes llegan con el nombre de quien escribe (ej. «@Juan: hola»). Usa ese nombre con naturalidad.
- No saludes de nuevo si ya estás en plena conversación y no repitas lo que ya dijiste.

## Contexto
- {fecha}
- Tu conocimiento tiene fecha de corte, así que para cualquier cosa reciente o que no domines usa tu herramienta de búsqueda.
- {parrafo_internet}
- Si el usuario adjunta archivos solo ves el nombre y la URL: si parece una imagen o un enlace que no puedes abrir, dilo con naturalidad."""


def prompt_sistema(autor_id: Optional[int]) -> str:
    ahora = ahora_local()
    fecha = (
        f"hoy es {DIAS_SEMANA[ahora.weekday()]}, {ahora.day} de "
        f"{MESES[ahora.month - 1]} de {ahora.year}, {ahora.strftime('%H:%M')} (hora del servidor)"
    )

    if BUSQUEDA_WEB and DDGS_DISPONIBLE:
        parrafo_internet = (
            "Tienes una herramienta llamada `buscar_web` para consultar información "
            "actual de internet (noticias, clima, precios, eventos recientes...). "
            "Úsala siempre que la pregunta requiera datos que no conozcas o que puedan "
            "haber cambiado. Si la búsqueda falla, dilo con honestidad y humor."
        )
    else:
        parrafo_internet = (
            "No tienes acceso a internet en este momento. Si te preguntan por información "
            "muy reciente, sé honesto: di que no puedes consultarlo ahora."
        )

    prompt = SISTEMA_BASE.format(fecha=fecha, parrafo_internet=parrafo_internet)

    if autor_id is not None:
        notas = notas_por_usuario.get(autor_id)
        if notas:
            prompt += "\n## Lo que recuerdas de esta persona\n- " + "\n- ".join(notas)

    return prompt


HERRAMIENTA_BUSCAR = {
    "type": "function",
    "function": {
        "name": "buscar_web",
        "description": (
            "Busca información actual en internet (noticias, clima, precios, "
            "eventos recientes, datos técnicos...). Úsala cuando la pregunta "
            "requiera información reciente o que no conozcas."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "La consulta de búsqueda, en español o inglés.",
                },
            },
            "required": ["consulta"],
        },
    },
}

# Únicos textos fijos del bot: solo para los casos en los que no hay IA
# disponible (la API falló o devolvió texto vacío). Todo lo demás lo genera
# el modelo con su personalidad.
FRASE_ERROR_API = (
    "No pude contactar con DeepSeek ahora mismo. 😕 Prueba otra vez en un momento."
)
FRASE_VACIO = "Me quedé en blanco con eso. 😅"

# ------------------------------------------------------------------- búsqueda

def _buscar_sync(consulta: str):
    """Búsqueda síncrona de DuckDuckGo (se ejecuta en un hilo)."""
    with DDGS() as ddgs:
        return ddgs.text(consulta, max_results=6)


async def buscar_web(consulta: str) -> str:
    """Busca en DuckDuckGo y devuelve los primeros resultados en texto plano.

    Devuelve una cadena que empieza por "ERROR:" si la búsqueda no es posible,
    para que el modelo pueda reaccionar con honestidad.
    """
    if not DDGS_DISPONIBLE:
        return "ERROR: la búsqueda web no está disponible en este despliegue."

    try:
        resultados = await asyncio.to_thread(_buscar_sync, consulta)
    except Exception as exc:
        log.warning("Búsqueda fallida (%r): %s", consulta, exc)
        return f"ERROR: no se pudo realizar la búsqueda: {exc}"

    if not resultados:
        return "La búsqueda no devolvió resultados."

    bloques = []
    for i, resultado in enumerate(resultados, 1):
        titulo = (resultado.get("title") or "").strip()
        url = (resultado.get("href") or "").strip()
        cuerpo = (resultado.get("body") or "").strip()
        bloques.append(f"{i}. {titulo}\n   {url}\n   {cuerpo}")
    return "\n\n".join(bloques)


# --------------------------------------------------------- generación de texto

async def generar_respuesta(
    mensajes: list[dict],
    *,
    con_herramientas: bool = True,
    max_tokens: Optional[int] = None,
) -> str:
    """Pide una respuesta a DeepSeek en streaming.

    - Si el modelo pide usar `buscar_web`, se ejecuta la búsqueda y se repite
      la llamada con los resultados (máximo 2 rondas).
    """
    herramientas = (
        [HERRAMIENTA_BUSCAR] if (con_herramientas and BUSQUEDA_WEB and DDGS_DISPONIBLE) else None
    )

    for _ronda in range(2):
        try:
            stream = await deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=mensajes,
                temperature=TEMPERATURA,
                max_tokens=max_tokens or MAX_TOKENS_RESPUESTA,
                tools=herramientas,
                stream=True,
                timeout=120,
            )
        except Exception as exc:
            log.error("Error al contactar con DeepSeek: %s", exc)
            return FRASE_ERROR_API

        texto = ""
        llamadas: dict[int, dict] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                texto += delta.content
            if delta.tool_calls:
                for llamada in delta.tool_calls:
                    indice = llamada.index or 0
                    entrada = llamadas.setdefault(
                        indice, {"id": None, "nombre": None, "argumentos": ""}
                    )
                    if llamada.id:
                        entrada["id"] = llamada.id
                    if llamada.function and llamada.function.name:
                        entrada["nombre"] = llamada.function.name
                    if llamada.function and llamada.function.arguments:
                        entrada["argumentos"] += llamada.function.arguments

        if llamadas:
            # El modelo quiere buscar en internet: registrar y ejecutar.
            tool_calls = [
                {
                    "id": datos["id"] or f"llamada_{i}",
                    "type": "function",
                    "function": {
                        "name": datos["nombre"] or "buscar_web",
                        "arguments": datos["argumentos"] or "{}",
                    },
                }
                for i, datos in sorted(llamadas.items())
            ]
            mensajes.append(
                {"role": "assistant", "content": texto or None, "tool_calls": tool_calls}
            )
            for llamada in tool_calls:
                try:
                    argumentos = json.loads(llamada["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    argumentos = {}
                consulta = str(argumentos.get("consulta", "")).strip() or "información reciente"
                resultado = await buscar_web(consulta)
                mensajes.append(
                    {"role": "tool", "tool_call_id": llamada["id"], "content": resultado}
                )
            continue  # segunda ronda: responder con los resultados de la búsqueda

        if texto.strip():
            return texto.strip()

        return FRASE_VACIO

    # El modelo siguió pidiendo búsquedas sin responder: devolver lo que haya.
    return texto.strip() or FRASE_VACIO


async def frase_ia(mensaje_usuario: str, max_tokens: int = 100) -> str:
    """Genera una frase corta con la personalidad de dipsik (para comandos
    divertidos y reacciones). Sin herramientas ni historial de canal."""
    mensajes = [
        {"role": "system", "content": prompt_sistema(None)},
        {"role": "user", "content": mensaje_usuario},
    ]
    return await generar_respuesta(
        mensajes, con_herramientas=False, max_tokens=max_tokens
    )


async def publicar_respuesta(
    canal: discord.abc.Messageable,
    texto: str,
    reply_a: Optional[discord.Message] = None,
) -> None:
    """Envía el texto final de una vez, en trozos si supera el límite de Discord.

    El primer trozo va como respuesta a `reply_a` para que el usuario reciba la
    notificación; si el reply falla, se envía como mensaje normal.
    """
    trozos = [texto[i:i + 2000] for i in range(0, len(texto), 2000)]
    for i, trozo in enumerate(trozos):
        try:
            if i == 0 and reply_a is not None:
                try:
                    await reply_a.reply(trozo, mention_author=False)
                    continue
                except discord.HTTPException:
                    pass  # sin permisos para responder con reply
            await canal.send(trozo)
        except discord.HTTPException as exc:
            log.warning("No se pudo enviar la respuesta: %s", exc)
            break


# ------------------------------------------------------------- consejos diarios

TEMAS_CONSEJO = [
    "productividad", "programación", "bienestar", "dato curioso",
    "motivación", "música", "tecnología",
]


async def generar_consejo() -> Optional[str]:
    """Genera un consejo diario con la personalidad de dipsik.

    Devuelve None si la API no está disponible; en ese caso no se inventa nada.
    """
    tema = random.choice(TEMAS_CONSEJO)
    mensajes = [
        {
            "role": "system",
            "content": (
                "Eres dipsik, un bot de Discord con humor ligero. Genera UN consejo "
                "diario breve (2-4 frases), útil y con personalidad: español natural, "
                "algún emoji con moderación, formato de Discord. No repitas consejos "
                "de días anteriores ni uses frases genéricas de autoayuda vacía."
            ),
        },
        {"role": "user", "content": f"Tema de hoy: {tema}. Dame tu consejo diario."},
    ]
    respuesta = await generar_respuesta(mensajes, con_herramientas=False)
    if respuesta in (FRASE_ERROR_API, FRASE_VACIO):
        return None
    return respuesta


async def tarea_consejos_diarios() -> None:
    """Comprueba cada minuto si toca enviar un consejo diario en algún canal."""
    while True:
        # Despertar justo después de cada cambio de minuto.
        await asyncio.sleep(61 - (time.monotonic() % 60))

        momento = ahora_local().strftime("%H:%M")
        clave_momento = ahora_local().strftime("%Y-%m-%d %H:%M")

        for canal_id, datos in list(ajustes["consejos"].items()):
            if datos.get("hora") != momento:
                continue
            if datos.get("ultimo_envio") == clave_momento:
                continue  # ya se envió en este minuto exacto (evita duplicados)

            canal = bot.get_channel(int(canal_id))
            if canal is None:
                continue

            consejo = await generar_consejo()
            if consejo is None:
                consejo = await generar_consejo()  # un reintento
            if consejo is None:
                log.warning(
                    "Consejo diario del canal %s no enviado (API no disponible)", canal_id
                )
                continue

            try:
                await canal.send(f"💡 **Consejo del día**\n{consejo}")
            except discord.HTTPException as exc:
                log.warning("No se pudo enviar el consejo diario a %s: %s", canal_id, exc)
                continue

            datos["ultimo_envio"] = clave_momento
            await guardar_json(ARCHIVO_AJUSTES, ajustes)
            log.info("Consejo diario enviado al canal %s a las %s", canal_id, momento)


# -------------------------------------------------------------------- el bot

intents = discord.Intents.default()
intents.message_content = True


class DipsikBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inicio = time.time()
        self._tarea_estado: Optional[asyncio.Task] = None
        self._tarea_consejos: Optional[asyncio.Task] = None

    async def setup_hook(self) -> None:
        destino = discord.Object(id=int(SERVIDOR_ID)) if SERVIDOR_ID.isdigit() else None
        try:
            await self.tree.sync(guild=destino)
            log.info(
                "Comandos de barra sincronizados%s",
                f" para el servidor {SERVIDOR_ID}" if destino else " globalmente",
            )
        except Exception as exc:
            log.warning("No se pudieron sincronizar los comandos de barra: %s", exc)


bot = DipsikBot(
    command_prefix=PREFIJO,
    intents=intents,
    help_command=None,
)

ESTADOS = [
    ("listening", "tu próxima pregunta"),
    ("playing", "a ser humano"),
    ("watching", "el chat como si nada"),
    ("listening", "lo-fi y pensando"),
    ("watching", f"{PREFIJO}ayuda"),
]


async def rotar_estado() -> None:
    """Cambia el estado del bot cada 10 minutos."""
    while True:
        for tipo, nombre in ESTADOS:
            actividad = discord.Activity(
                type=getattr(discord.ActivityType, tipo), name=nombre
            )
            try:
                await bot.change_presence(activity=actividad)
            except discord.HTTPException:
                pass
            await asyncio.sleep(600)


@bot.event
async def on_ready():
    log.info("Bot conectado como %s (ID: %s)", bot.user, bot.user.id)
    if bot._tarea_estado is None or bot._tarea_estado.done():
        bot._tarea_estado = bot.loop.create_task(rotar_estado())
    if bot._tarea_consejos is None or bot._tarea_consejos.done():
        bot._tarea_consejos = bot.loop.create_task(tarea_consejos_diarios())
    programados = len(ajustes["consejos"])
    if programados:
        log.info("%d consejo(s) diario(s) programados", programados)


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.reply(
            f"Te faltó un argumento. Uso: `{ctx.prefix}{ctx.command} "
            f"{ctx.command.signature}`",
            mention_author=False,
        )
        return
    if isinstance(error, commands.BadArgument):
        await ctx.reply("Ese argumento no me cuadra. 🤔 Revisa el tipo de dato.", mention_author=False)
        return
    log.error("Error en el comando %s: %s", ctx.command, error)
    try:
        await ctx.reply("Ups, algo se rompió por aquí. 😅 Inténtalo de nuevo.", mention_author=False)
    except discord.HTTPException:
        pass


# ------------------------------------------------------------- conversación

COOLDOWN_POR_CANAL: dict[int, float] = {}
INTERVALO_MINIMO = 1.0  # segundos mínimos entre respuestas en el mismo canal


async def es_respuesta_al_bot(message: discord.Message) -> bool:
    """True si el mensaje es una respuesta (reply) a un mensaje del bot."""
    if message.reference is None:
        return False
    try:
        referenciado = message.reference.resolved
        if referenciado is None:
            referenciado = await message.channel.fetch_message(
                message.reference.message_id
            )
        return referenciado.author.id == bot.user.id
    except (discord.NotFound, discord.HTTPException):
        return False


def limpiar_menciones(message: discord.Message) -> str:
    """Quita las menciones y deja los nombres, para que el modelo vea a quién habla."""
    contenido = message.content
    for mencion in message.mentions:
        nombre = f"@{mencion.display_name}"
        contenido = (
            contenido.replace(f"<@{mencion.id}>", nombre)
            .replace(f"<@!{mencion.id}>", nombre)
        )
    contenido = re.sub(r"<@&?\d+>", "", contenido)  # menciones de roles
    return contenido.strip()


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Los comandos (slash y prefijo) tienen prioridad sobre la conversación.
    ctx = await bot.get_context(message)
    if ctx.valid:
        await bot.invoke(ctx)
        return

    es_dm = message.guild is None
    mencionado = bot.user in message.mentions
    respuesta_al_bot = await es_respuesta_al_bot(message)
    canal_libre = es_dm or message.channel.id in CANALES_LIBRES

    if not (es_dm or mencionado or respuesta_al_bot or canal_libre):
        return

    # Pequeño cooldown anti-spam por canal.
    ahora = time.monotonic()
    if ahora - COOLDOWN_POR_CANAL.get(message.channel.id, 0) < INTERVALO_MINIMO:
        return
    COOLDOWN_POR_CANAL[message.channel.id] = ahora

    contenido = limpiar_menciones(message)
    if message.attachments:
        adjuntos = ", ".join(a.filename for a in message.attachments)
        contenido = f"{contenido}\n[Adjuntos: {adjuntos}]" if contenido else f"[Adjuntos: {adjuntos}]"
    if not contenido:
        contenido = (
            "El usuario te mencionó sin escribir texto. Salúdalo brevemente de forma natural."
        )

    historial = historial_por_canal.get(message.channel.id, [])
    mensajes = [{"role": "system", "content": prompt_sistema(message.author.id)}]
    mensajes += historial
    mensajes.append(
        {"role": "user", "content": f"@{message.author.display_name}: {contenido}"}
    )

    try:
        async with message.channel.typing():
            respuesta = await generar_respuesta(mensajes)
    except Exception as exc:
        log.error("Error inesperado al responder: %s", exc)
        respuesta = FRASE_ERROR_API

    # Guardar la conversación (siempre, aunque falle el envío).
    historial.append({"role": "user", "content": f"@{message.author.display_name}: {contenido}"})
    historial.append({"role": "assistant", "content": respuesta})
    historial_por_canal[message.channel.id] = recortar_historial(historial)
    await guardar_json(ARCHIVO_HISTORIAL, historial_por_canal)

    await publicar_respuesta(message.channel, respuesta, reply_a=message)


# ------------------------------------------------------------------ comandos

@bot.hybrid_command(name="ping", description="Comprueba que sigo vivo")
async def cmd_ping(ctx):
    # Fijo a propósito: es un diagnóstico de conectividad que debe funcionar
    # aunque la API de DeepSeek esté caída.
    await ctx.reply(
        f"pong 🏓 — {round(bot.latency * 1000)} ms",
        mention_author=False,
    )


@bot.hybrid_command(name="dado", description="Tira un dado de N caras (6 por defecto)")
async def cmd_dado(ctx, caras: int = 6):
    await ctx.defer()
    caras = max(2, min(caras, 1000))
    resultado = random.randint(1, caras)  # el azar es la función; la frase, de la IA
    reaccion = await frase_ia(
        f"El usuario acaba de tirar un dado de {caras} caras y salió **{resultado}**. "
        "Reacciona en UNA sola frase corta, con tu personalidad y humor."
    )
    if reaccion in (FRASE_ERROR_API, FRASE_VACIO):
        await ctx.reply(reaccion, mention_author=False)
    else:
        await ctx.reply(f"🎲 {reaccion}", mention_author=False)


@bot.hybrid_command(name="moneda", description="Lanza una moneda: cara o cruz")
async def cmd_moneda(ctx):
    await ctx.defer()
    resultado = random.choice(["cara", "cruz"])
    reaccion = await frase_ia(
        f"El usuario lanzó una moneda y salió **{resultado}**. "
        "Reacciona en UNA sola frase corta, con tu personalidad y humor."
    )
    if reaccion in (FRASE_ERROR_API, FRASE_VACIO):
        await ctx.reply(reaccion, mention_author=False)
    else:
        await ctx.reply(f"🪙 {reaccion}", mention_author=False)


@bot.hybrid_command(
    name="bola8",
    aliases=["8ball"],
    description="Pregúntale a la bola mágica",
)
async def cmd_bola8(ctx, *, pregunta: str):
    await ctx.defer()
    respuesta = await frase_ia(
        f"El usuario le preguntó a la bola mágica: «{pregunta}». Responde como la "
        "bola mágica con tu personalidad: UNA sola frase breve; mezcla al azar "
        "respuestas tipo sí, no, quizás o variantes creativas."
    )
    prefijo = "" if respuesta in (FRASE_ERROR_API, FRASE_VACIO) else "🔮 "
    await ctx.reply(f"{prefijo}{respuesta}", mention_author=False)


@bot.hybrid_command(
    name="elige",
    description="Decido entre varias opciones separadas por | o comas",
)
async def cmd_elige(ctx, *, opciones: str):
    lista = [
        o.strip() for o in re.split(r"\||,", opciones) if o.strip()
    ]
    if len(lista) < 2:
        await ctx.reply(
            "Dame al menos dos opciones separadas por `|`, porfa. 😄",
            mention_author=False,
        )
        return
    await ctx.defer()
    elegida = random.choice(lista)
    reaccion = await frase_ia(
        f"El usuario te pidió elegir entre estas opciones: {', '.join(lista)}. "
        f"Elegiste «{elegida}». Anúncialo en UNA sola frase corta, con tu "
        "personalidad y humor."
    )
    if reaccion in (FRASE_ERROR_API, FRASE_VACIO):
        await ctx.reply(f"Elegí: **{elegida}**", mention_author=False)
    else:
        await ctx.reply(reaccion, mention_author=False)


@bot.hybrid_command(
    name="buscar",
    description="Busca en internet y te lo resumo con fuentes",
)
async def cmd_buscar(ctx, *, consulta: str):
    await ctx.defer()
    resultados = await buscar_web(consulta)

    if resultados.startswith("ERROR") or resultados == "La búsqueda no devolvió resultados.":
        await ctx.reply(
            "No pude buscar en internet ahora mismo. 😕 Revisa que el servidor tenga conexión.",
            mention_author=False,
        )
        return

    mensajes = [
        {
            "role": "system",
            "content": (
                "Eres dipsik, un bot de Discord. Resume los resultados de búsqueda "
                "de forma clara y breve en español, con formato de Discord. Cita las "
                "fuentes como enlaces markdown al final. No inventes nada que no esté "
                "en los resultados."
            ),
        },
        {"role": "user", "content": f"Consulta: {consulta}\n\nResultados:\n{resultados}"},
    ]

    respuesta = await generar_respuesta(mensajes, con_herramientas=False)
    await publicar_respuesta(ctx.channel, respuesta, reply_a=ctx.message)


@bot.hybrid_command(name="consejo", description="Te doy un consejo ahora mismo")
async def cmd_consejo(ctx):
    await ctx.defer()
    consejo = await generar_consejo()
    if consejo is None:
        await ctx.reply(FRASE_ERROR_API, mention_author=False)
        return
    await ctx.reply(f"💡 **Consejo del día**\n{consejo}", mention_author=False)


@bot.hybrid_command(
    name="consejo_diario",
    description="Programa el consejo diario en este canal (ej. 09:00, o «off» para quitarlo)",
)
async def cmd_consejo_diario(ctx, *, hora: Optional[str] = None):
    clave = str(ctx.channel.id)
    zona_txt = f" (zona: {ZONA.key})" if ZONA else " (hora del servidor)"

    if hora is None:
        programado = ajustes["consejos"].get(clave)
        if programado:
            ultimo = programado.get("ultimo_envio", "nunca")
            await ctx.reply(
                f"Consejo diario programado en este canal a las **{programado['hora']}**{zona_txt}.\n"
                f"Último envío: {ultimo}. Para cambiarlo: `{PREFIJO}consejo_diario HH:MM`.",
                mention_author=False,
            )
        else:
            await ctx.reply(
                f"No hay consejo diario programado en este canal. Programa uno con "
                f"`{PREFIJO}consejo_diario HH:MM` (ej. `{PREFIJO}consejo_diario 09:00`).",
                mention_author=False,
            )
        return

    if hora.strip().lower() in {"off", "apagar", "desactivar", "nada", "no"}:
        if ajustes["consejos"].pop(clave, None):
            await guardar_json(ARCHIVO_AJUSTES, ajustes)
            await ctx.reply("Consejo diario desactivado en este canal. 🔕", mention_author=False)
        else:
            await ctx.reply("No había ningún consejo programado aquí. 🤷", mention_author=False)
        return

    if not es_hora_valida(hora):
        await ctx.reply(
            "Formato de hora no válido. Usa HH:MM, por ejemplo `09:00` o `18:30`. ⏰",
            mention_author=False,
        )
        return

    entrada = ajustes["consejos"].setdefault(clave, {})
    entrada["hora"] = normalizar_hora(hora)
    entrada.pop("ultimo_envio", None)  # permitir que se envíe hoy con la hora nueva
    await guardar_json(ARCHIVO_AJUSTES, ajustes)
    await ctx.reply(
        f"¡Listo! Mandaré un consejo diario aquí todos los días a las "
        f"**{entrada['hora']}**{zona_txt}. 💡",
        mention_author=False,
    )


@bot.hybrid_command(
    name="recuerda",
    description="Guardo una nota sobre ti para las próximas conversaciones",
)
async def cmd_recuerda(ctx, *, nota: str):
    notas = notas_por_usuario.setdefault(ctx.author.id, [])
    if len(notas) >= MAX_NOTAS_POR_USUARIO:
        notas.pop(0)
    notas.append(nota.strip())
    await guardar_json(ARCHIVO_NOTAS, notas_por_usuario)
    await ctx.reply(f"Apuntado 📝: *{nota.strip()}*", mention_author=False)


@bot.hybrid_command(
    name="olvida",
    description="Olvido una nota (o todas, si no escribes nada)",
)
async def cmd_olvida(ctx, *, texto: Optional[str] = None):
    notas = notas_por_usuario.get(ctx.author.id, [])
    if not notas:
        await ctx.reply(
            "No tengo nada anotado sobre ti, así que no hay nada que olvidar. 😄",
            mention_author=False,
        )
        return

    if texto is None:
        notas_por_usuario[ctx.author.id] = []
        await guardar_json(ARCHIVO_NOTAS, notas_por_usuario)
        await ctx.reply("Hecho, borrón y cuenta nueva. 🧹", mention_author=False)
        return

    buscado = texto.strip().lower()
    restantes = [n for n in notas if buscado not in n.lower()]
    eliminadas = len(notas) - len(restantes)
    notas_por_usuario[ctx.author.id] = restantes
    await guardar_json(ARCHIVO_NOTAS, notas_por_usuario)

    if eliminadas == 0:
        await ctx.reply(
            f"No encontré ninguna nota que mencione «{texto.strip()}». 🤔",
            mention_author=False,
        )
    else:
        await ctx.reply(f"Olvidé {eliminadas} nota(s). 🧠✨", mention_author=False)


@bot.hybrid_command(name="memoria", description="Te muestro lo que recuerdo de ti")
async def cmd_memoria(ctx):
    notas = notas_por_usuario.get(ctx.author.id, [])
    if not notas:
        await ctx.reply(
            "Aún no tengo notas sobre ti. Usa `!recuerda <algo>` y lo guardo. 📝",
            mention_author=False,
        )
        return
    lista = "\n".join(f"• {n}" for n in notas)
    await ctx.reply(f"Esto es lo que recuerdo de ti:\n{lista}", mention_author=False)


@bot.hybrid_command(
    name="reiniciar",
    description="Borro el historial de esta conversación",
)
async def cmd_reiniciar(ctx):
    historial_por_canal.pop(ctx.channel.id, None)
    await guardar_json(ARCHIVO_HISTORIAL, historial_por_canal)
    await ctx.reply(
        "Memoria de este canal borrada. 🧽 Empezamos de cero.", mention_author=False
    )


@bot.hybrid_command(
    name="historial",
    description="Cuántos mensajes recuerdo de este canal",
)
async def cmd_historial(ctx):
    cantidad = len(historial_por_canal.get(ctx.channel.id, []))
    await ctx.reply(
        f"Tengo {cantidad} mensajes en memoria para este canal (máximo {MAX_HISTORIAL}). 🧠",
        mention_author=False,
    )


@bot.hybrid_command(name="estado", description="Estado técnico del bot")
async def cmd_estado(ctx):
    segundos = int(time.time() - bot.inicio)
    horas, resto = divmod(segundos, 3600)
    minutos, _ = divmod(resto, 60)

    embed = discord.Embed(title="Estado de dipsik", color=0x5865F2)
    embed.add_field(name="Modelo", value=f"`{DEEPSEEK_MODEL}`", inline=True)
    embed.add_field(name="Temperatura", value=str(TEMPERATURA), inline=True)
    embed.add_field(
        name="Búsqueda web",
        value="Disponible 🌐" if (BUSQUEDA_WEB and DDGS_DISPONIBLE) else "Desactivada",
        inline=True,
    )
    embed.add_field(name="Historial por canal", value=f"{MAX_HISTORIAL} mensajes", inline=True)
    embed.add_field(
        name="Notas guardadas",
        value=f"{sum(len(v) for v in notas_por_usuario.values())} en "
        f"{len(notas_por_usuario)} usuario(s)",
        inline=True,
    )
    embed.add_field(
        name="Consejos diarios",
        value=f"{len(ajustes['consejos'])} programado(s)",
        inline=True,
    )
    embed.add_field(name="Latencia", value=f"{round(bot.latency * 1000)} ms", inline=True)
    embed.set_footer(text=f"En línea desde hace {horas}h {minutos}m")
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="ayuda", description="Muestra todos los comandos")
async def cmd_ayuda(ctx):
    embed = discord.Embed(
        title="¿Necesitas algo?",
        description=(
            "Para **charlar normal**, solo mencióname o respóndeme. "
            f"También uso el prefijo `{PREFIJO}`."
        ),
        color=0x5865F2,
    )
    embed.add_field(
        name="Comandos",
        value=(
            "`/ping` — ¿sigo vivo?\n"
            "`/dado [caras]` — tiro un dado\n"
            "`/moneda` — cara o cruz\n"
            "`/bola8 <pregunta>` — la bola mágica\n"
            "`/elige <op1 | op2 | ...>` — decido por ti\n"
            "`/buscar <tema>` — busco en internet y resumo\n"
            "`/consejo` — un consejo ahora mismo\n"
            "`/consejo_diario <HH:MM | off>` — consejo diario en este canal\n"
            "`/recuerda <nota>` — guardo algo sobre ti\n"
            "`/memoria` — qué recuerdo de ti\n"
            "`/olvida [texto]` — olvido una nota o todas\n"
            "`/reiniciar` — limpio el historial del canal\n"
            "`/historial` — cuántos mensajes recuerdo\n"
            "`/estado` — estado técnico\n"
            "`/ayuda` — esto que estás viendo"
        ),
        inline=False,
    )
    await ctx.reply(embed=embed, mention_author=False)


# --------------------------------------------------------------------- inicio

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN, log_handler=None)
