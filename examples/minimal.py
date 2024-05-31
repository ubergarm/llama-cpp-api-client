#!/usr/bin/env python3

import asyncio
import sys

from llama_cpp_api_client import LlamaCppAPIClient


async def main():
    system_prompt = "You are a Zen master and mystical poet."
    user_prompt = "Write a simple haiku about llamas."

    chat_thread = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    client = LlamaCppAPIClient()

    async for response in client.stream_completion(chat_thread):
        if response.get("stop", False):
            continue
        print(response["content"], end="")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
