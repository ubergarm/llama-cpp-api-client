#!/usr/bin/env python3

import asyncio
import json
import sys

from llama_cpp_api_client import LlamaCppAPIClient

from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.layout import Layout


async def main():
    system_prompt = "You are a Zen master and mystical poet."
    user_prompt = "Write a simple haiku about llamas."

    chat_thread = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    headers = {"User-Agent": "Mozilla/3.01Gold (X11; I; SunOS 5.5.1 sun4m)"}
    options = {"n_predict": 128}
    client = LlamaCppAPIClient(base_url="http://localhost:8080", headers=headers, options=options)

    total = ""
    try:
        async for response in client.stream_completion(chat_thread=chat_thread, format="Llama-3"):
            if response.get("stop", False):
                print("\n>>> Timings")
                timings = response["timings"]
                timings = json.dumps(timings, sort_keys=True, indent=2)
                print(timings)
                print(f">>> Prompt:\n")
                md = Markdown("```\n" + response["prompt"] + "\n```")
                print(md)
                continue
            total += response["content"]
            print(response["content"], end="")
            sys.stdout.flush()
    except Exception as e:
        print(f"ERORR: {e}")
        sys.exit(1)

    print(f">>> Response:")
    md = Markdown(total)
    print(md)


if __name__ == "__main__":
    layout = Layout()
    # setup the RICH UI layout
    layout.split_column(Layout(name="upper"), Layout(name="lower"))
    layout["upper"].ratio = 4
    layout["lower"].update(">>>")
    print(layout)

    asyncio.run(main())
