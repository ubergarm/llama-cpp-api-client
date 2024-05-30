#!/usr/bin/env python3

import asyncio
import sys

from llama_cpp_api_client import chat_to_prompt, stream_response


async def main():
    system_prompt = "You are a Zen master and mystical poet."
    user_prompt = "Write a simple haiku about llamas."

    chat_thread = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = chat_to_prompt(chat_thread=chat_thread, format="Llama-3")
    options = {"prompt": prompt}

    async for response in stream_response(base_url="http://localhost:8080", options=options):
        if response.get("stop", False):
            continue
        print(response["content"], end="")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
