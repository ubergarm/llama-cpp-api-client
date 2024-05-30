#!/usr/bin/env python3

import asyncio
import json
import sys

from llama_cpp_api_client import chat_to_prompt, stream_response

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax


async def main(console: Console):
    system_prompt = "You are a Zen master and mystical poet."
    user_prompt = "Write a simple haiku about llamas."

    chat_thread = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Do you like llamas?"},
        {"role": "assistant", "content": "Yes I like llamas. What do you want to know about llamas?"},
        {"role": "user", "content": user_prompt},
    ]

    prompt = chat_to_prompt(chat_thread=chat_thread, format="Llama-3")
    options = {"prompt": prompt}
    headers = {"User-Agent": "Mozilla/3.01Gold (X11; I; SunOS 5.5.1 sun4m)"}

    total = ""
    async for response in stream_response(base_url="http://localhost:8080", options=options, headers=headers):
        if response.get("stop", False):
            console.print("\n>>> Timings")
            timings = response["timings"]
            timings = json.dumps(timings, sort_keys=True, indent=2)
            console.print(timings)
            console.print(f">>> Prompt:\n")
            md = Markdown("```\n"+response["prompt"]+"\n```")
            console.print(md)
            continue
        total += response["content"]
        console.print(response["content"], end="")
        sys.stdout.flush()

    console.print(f">>> Response:")
    md = Markdown(total)
    console.print(md)


if __name__ == "__main__":
    console = Console()
    asyncio.run(main(console))
