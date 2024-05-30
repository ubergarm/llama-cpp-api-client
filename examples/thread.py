#!/usr/bin/env python3

import asyncio
import sys

from llama_cpp_api_client import chat_to_prompt, stream_response


async def main():
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
    try:
        async for response in stream_response(base_url="http://localhost:8080", options=options, headers=headers):
            if response.get("stop", False):
                print("")
                print(f">>> Timings:\n{response["timings"]}")
                print(f">>> Prompt:\n{response["prompt"]}")
                continue
            total += response["content"]
            print(response["content"], end="")
            sys.stdout.flush()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f">>> Response:\n{total}")


if __name__ == "__main__":
    asyncio.run(main())
