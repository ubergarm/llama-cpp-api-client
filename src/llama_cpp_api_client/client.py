import asyncio
import json
from collections.abc import AsyncGenerator

from aiohttp import ClientSession

# Default Address and Port of your LLaMA.cpp HTTP Server
DEFAULT_BASE_URL = "http://localhost:8080"

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

# Default headers for HTTP POSTs/GETs
DEFAULT_HEADERS = {
    "User-Agent": "aiohttp",
    "Content-Type": "application/json",
    "Connection": "keep-alive",
    "Accept": "text/event-stream",
}


async def stream_response(base_url: str = DEFAULT_BASE_URL, data: dict = {}) -> AsyncGenerator[dict, None]:
    """
    Stream llamma.cpp server /completion API full responses.
    TODO: try/catch/raise errors
    """
    async with ClientSession() as session:
        headers = DEFAULT_HEADERS
        data = DEFAULT_COMPLETION_OPTIONS

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

    complete = ""
    async for response in stream_response():
        if response.get("stop", False):
            print(f">>> Timings:\n{response["timings"]}")
            print(f">>> Prompt:\n{response["prompt"]}")
            continue
        complete += response["content"]
        print(response["content"])

    print(f">>> Response:\n{complete}")


if __name__ == "__main__":
    asyncio.run(main())
