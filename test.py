"""Prueba rápida de la API de DeepSeek sin Discord.

Comprueba:
1. Que la clave del .env funciona (chat normal).
2. Que el modelo usa la herramienta buscar_web cuando se le pide info actual.
3. Que el paquete de búsqueda ddgs está instalado.

Uso: uv run python test.py
"""

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODELO = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

HERRAMIENTA_BUSCAR = {
    "type": "function",
    "function": {
        "name": "buscar_web",
        "description": "Busca información actual en internet.",
        "parameters": {
            "type": "object",
            "properties": {
                "consulta": {"type": "string"},
            },
            "required": ["consulta"],
        },
    },
}


async def main():
    if not API_KEY:
        print("❌ Falta DEEPSEEK_API_KEY en el archivo .env")
        return

    client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

    # 1) Chat normal
    respuesta = await client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "user",
                "content": "Responde solamente: Hola, DeepSeek funciona.",
            },
        ],
    )
    print("1) Chat:", respuesta.choices[0].message.content)

    # 2) Tool calling: el modelo debería pedir buscar en internet
    respuesta = await client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "user",
                "content": "¿Qué clima hace hoy en Ciudad de México?",
            },
        ],
        tools=[HERRAMIENTA_BUSCAR],
    )
    mensaje = respuesta.choices[0].message
    if mensaje.tool_calls:
        print(
            "2) El modelo pidió buscar:",
            mensaje.tool_calls[0].function.name,
            mensaje.tool_calls[0].function.arguments,
        )
    else:
        print("2) Sin tool call:", mensaje.content)

    # 3) ¿Está instalado ddgs?
    try:
        from ddgs import DDGS
        print("3) Búsqueda web (ddgs): instalado ✅")
    except ImportError:
        print("3) Búsqueda web (ddgs): NO instalado ❌ (el bot funcionará sin búsqueda)")


asyncio.run(main())
