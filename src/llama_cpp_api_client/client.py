import asyncio
import json
from collections.abc import AsyncGenerator

import aiohttp
from aiohttp import ClientSession


async def stream_response(base_url: str, data: dict) -> AsyncGenerator[dict, None]:
    """
    Stream llamma.cpp server /completion API full responses.
          * try/catch/raise errors
    """
    async with ClientSession() as session:
        headers = {}
        headers.update({"User-Agent": "aiohttp"})
        headers.update({"Content-Type": "application/json"})
        if data["stream"] == True:
            headers.update({"Connection": "keep-alive"})
            headers.update({"Accept": "text/event-stream"})

        url = f"{base_url.rstrip("/")}/completion"
        print(f"API Endpoint: {url}")
        # FIXME: this next line does not work if it isn't already set before passing in the dict
        data["stream"] = True

        async with session.post(url, headers=headers, json=data) as response:
            if not response.status == 200:
                print(response.status)

            async for raw_line in response.content:
                if len(raw_line) == 1:
                    continue
                if raw_line[:6] != "data: ".encode("utf-8"):
                    print("Invalid data header...")
                    continue
                line = raw_line.decode("utf-8")[6:]
                yield json.loads(line)


async def main() -> None:

    # system_prompt = "You are a Zen master and mystical poet."
    # user_prompt = "Write a simple haiku about being Autistic."

    system_prompt = "You are an experienced Python software engineer working in a Linux development environment."
    user_prompt = "Suggest a good name for a Python library that providing a streaming API client to llama.cpp server."

    # >>> ChatML Prompt Template
    # "prompt": f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant\n",

    # >>> Llama-3-70B Prompt Template
    # "prompt": f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",

    base_url = "http://localhost:1234"
    data = {
        "prompt": f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "temperature": 0.8,
        "top_k": 45,
        "top_p": 0.95,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "n_predict": 768,
        "cache_prompt": False,
        "stop": ["<|eot_id|>", "<|im_end|>", "<|endoftext|>", "</s>"],
        "stream": True,
    }

    complete = ""
    async for response in stream_response(base_url=base_url, data=data):
        if response.get("stop", False):
            print(response["timings"])
            print(response["prompt"])
            continue
        complete += response["content"]
        print(response["content"])

    print(f">>> Response:\n{complete}")


if __name__ == "__main__":
    asyncio.run(main())
