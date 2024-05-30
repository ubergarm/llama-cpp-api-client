import asyncio
import json
from collections.abc import AsyncGenerator
import sys

from aiohttp import ClientSession

# Default Address and Port of your LLaMA.cpp HTTP Server
DEFAULT_BASE_URL = "http://localhost:8080"

# Default headers for HTTP POSTs/GETs
DEFAULT_HEADERS = {
    "User-Agent": "aiohttp",
    "Content-Type": "application/json",
    "Connection": "keep-alive",
    "Accept": "text/event-stream",
}

# Default options for POST /completion API endpoint
# https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints
DEFAULT_COMPLETION_OPTIONS = {
    # Llama-3 style prompt template shown in this example
    "prompt": f"<|start_header_id|>system<|end_header_id|>\n\nYou are a Zen master and mystical poet.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWrite a short haiku about llamas.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    # ChatML style prompt template shown below
    # "prompt": f"<|im_start|>system\nYou are a Zen master and mystical poet.\n<|im_end|>\n<|im_start|>user\nWrite a short haiku about llamas.\n<|im_end|>\n<|im_start|>assistant\n",
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
    "n_predict": -1,
    "seed": -1,
    "id_slot": -1,
    "cache_prompt": False,
    # Likely need to add more stop tokens below to support more model types.
    "stop": ["<|eot_id|>", "<|im_end|>", "<|endoftext|>", "</s>"],
    "stream": True,
}

# LLaMA.cpp HTTP Server Response Body Start String
# https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#result-json
DEFAULT_RESPONSE_BODY_START_STRING = "data: ".encode("utf-8")


async def stream_response(
    base_url: str = DEFAULT_BASE_URL, options: dict = {}, headers: dict = {}
) -> AsyncGenerator[dict, None]:
    """
    Stream LLaMA.cpp HTTP Server API POST /completion response
    """
    try:
        async with ClientSession() as session:
            # override defaults with whatever userland passes
            combined_headers = DEFAULT_HEADERS
            combined_headers.update(headers)
            combined_options = DEFAULT_COMPLETION_OPTIONS
            combined_options.update(options)

            url = f"{base_url.rstrip("/")}/completion"

            async with session.post(url=url, headers=combined_headers, json=combined_options) as response:
                if not response.status == 200:
                    print(f"HTTP Response: {response.status}")

                async for raw_line in response.content:
                    if len(raw_line) == 1:
                        continue
                    if raw_line[: len(DEFAULT_RESPONSE_BODY_START_STRING)] != DEFAULT_RESPONSE_BODY_START_STRING:
                        # FIXME: this is brittle code, not sure if another json decoder and skip the "data: " part...
                        print("Invalid response body starting string, unable to parse response...")
                        continue
                    line = raw_line.decode("utf-8")[len(DEFAULT_RESPONSE_BODY_START_STRING) :]
                    yield json.loads(line)
    except Exception as e:
        raise e


async def main() -> None:
    system_prompt = "You are a Zen master and mystical poet."
    user_prompt = "Write a simple haiku about llamas."

    # >>> ChatML Prompt Template
    # "prompt": f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant\n",

    # >>> Llama-3-70B Prompt Template
    # "prompt": f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",

    total = ""
    async for response in stream_response():
        if response.get("stop", False):
            print(f">>> Timings:\n{response["timings"]}")
            print(f">>> Prompt:\n{response["prompt"]}")
            continue
        total += response["content"]
        print(response["content"], end="")
        sys.stdout.flush()

    print(f">>> Response:\n{total}")


if __name__ == "__main__":
    asyncio.run(main())
