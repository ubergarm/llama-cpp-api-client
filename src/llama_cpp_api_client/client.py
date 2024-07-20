import asyncio
import json
import sys
from collections.abc import AsyncGenerator

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
    "stop": ["<|eot_id|>", "<|im_end|>", "<|endoftext|>", "<|end|>", "</s>"],
    "stream": True,
}

# LLaMA.cpp HTTP Server Response Body Start String
# https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#result-json
DEFAULT_RESPONSE_BODY_START_STRING = "data: ".encode("utf-8")


class LlamaCppAPIClient:
    """headers and options can be overriden at constructions time or per inference call"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, headers: dict = {}, options: dict = {}):
        # override defaults with whatever userland passes into constructor
        self.base_url = base_url
        self.headers = DEFAULT_HEADERS
        self.headers.update(headers)
        self.options = DEFAULT_COMPLETION_OPTIONS
        self.options.update(options)

    async def stream_completion(
        self, chat_thread: list[dict] = [], format: str = "Llama-3"
    ) -> AsyncGenerator[dict, None]:
        """Stream LLaMA.cpp HTTP Server API POST /completion responses"""
        try:
            # convert chat_thread to a template formatted prompt string
            prompt = chat_to_prompt(chat_thread=chat_thread, format=format)

            # set the HTTP headers and /completion API options
            url = self.base_url.rstrip("/") + "/completion"
            combined_headers = self.headers
            combined_options = self.options
            combined_options.update({"prompt": prompt})

            async with ClientSession() as session:
                async with session.post(url=url, headers=combined_headers, json=combined_options) as response:
                    if not response.status == 200:
                        raise Exception(f"HTTP Response: {response.status}")

                    async for raw_line in response.content:
                        if len(raw_line) == 1:
                            continue
                        if raw_line[: len(DEFAULT_RESPONSE_BODY_START_STRING)] != DEFAULT_RESPONSE_BODY_START_STRING:
                            # FIXME: this is brittle code, not sure if another json decoder and skip the "data: " part...
                            raise Exception("Invalid response body starting string, unable to parse response...")
                        line = raw_line.decode("utf-8")[len(DEFAULT_RESPONSE_BODY_START_STRING) :]
                        yield json.loads(line)
        except Exception as e:
            raise e


def chat_to_prompt(chat_thread: list[dict], format: str) -> str:
    """Accepts a list of dicts in the OpenAI style chat thread and returns string with specified prompt template applied."""
    # There must be a better way to do this e.g.
    # https://github.com/ggerganov/llama.cpp/commit/8768b4f5ea1de69a4cace0481fdba70d89a47e47

    # Initialize result as empty string
    result = ""

    # Check if the chat is not empty or only contains system/user roles
    if len(chat_thread) == 0:
        raise ValueError("Chat thread cannot be empty.")

    for _, message in enumerate(chat_thread):
        # Error check to ensure 'role' and 'content' keys exist in each dict
        try:
            role = message["role"]
            content = message["content"]
        except KeyError as e:
            raise ValueError(f"Each chat thread item must contain both 'role' and 'content' keys: {e}")

        if role not in ["system", "user", "assistant"]:
            raise ValueError("Chat thread only supports 'system', 'user', and 'assistant' roles.")

        # TODO Apply chat template formats or jinja templates and return prompt string.
        # Could use jinja templates e.g. https://github.com/vllm-project/vllm/blob/main/examples/template_chatml.jinja
        # `u/Evening_Ad6637` is amazing: https://github.com/mounta11n/plusplus-camall/blob/plusplus/examples/server/public/prompt-formats.js
        # This is clunky hacky but gets a minimal PoC going quick...
        match format:
            # template["ChatML"] = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant\n"
            # Do not prepend the BOS as that seems to cause hallucinations...
            # llama_tokenize_internal: Added a BOS token to the prompt as specified by the model but the prompt also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. Are you sure this is what you want?
            case "ChatML":
                result += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
            # template["Llama-3"] =  f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            # Unclear if need to prepend BOS to start: https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/35 .. does not seem to hurt anything...
            # Don't add BOS, llama.cpp server side is doing that: llama_tokenize_internal: Added a BOS token to the prompt as specified by the model but the prompt also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. Are you sure this is what you want?
            case "Llama-3":
                result += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            # Phi-3 does not technically support a system prompt so could raise an error or fudge it in anyway e.g. "{{ bos_token }}{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + '<|end|>' }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + '<|end|>' }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + '<|end|>' }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
            # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct#chat-format
            # skip prepending BOS for now, haven't tested as much as above but seems fine without it...
            case "Phi-3":
                # if result == "":
                #     result += "<s>\n"
                result += f"<|{role}|>\n{content}<|end|>\n"
                if role == "system":
                    raise NotImplementedError(f"{format} models do not support {role} prompts. Please remove and try again.")
            case "Gemma2":
                # no system prompt here either
                # <start_of_turn>user
                # {prompt}<end_of_turn>
                # <start_of_turn>model
                result += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
                if role == "system":
                    raise NotImplementedError(f"{format} models do not support {role} prompts. Please remove and try again.")
            case "Mixtral":
                # NOTE: the spaces before strings are important. Seems like system prompt is folded into first user prompt.
                # NOTE2: no need to add the <s> BOS token as llamma.cpp server does that automatically
                # <s>[INST] You are a helpful assistant\nHello [/INST]Hi there</s>[INST] How are you? [/INST]
                result += f"[INST] {content}"
                if role == "system":
                    raise NotImplementedError(f"{format} models {role} prompt not yet implemented. Fold it into your first user prompt followed by \\n")
            case "Raw":
            # just concatanate all content fields if userland wants to pass raw string
                result += f"{content}"
            case _:
                raise NotImplementedError(f"{format} not in list of supported formats e.g. ChatML, Llama-3, Phi-3, Raw...")

    # chat threads must end by cueing the assistant to begin generation
    match format:
        case "ChatML":
            result += "<|im_start|>assistant\n"
        case "Llama-3":
            result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        case "Phi-3":
            result += "<|assistant|>\n"
        case "Mixtral":
            result += " [/INST]" # extra white space is on purpose
        case "Gemma2":
            result += "<start_of_turn>model\n"
    return result


async def main() -> None:
    system_prompt = "You are a Zen master and mystical poet."
    user_prompt = "Write a simple haiku about llamas."

    chat_thread = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Do you like llamas?"},
        {"role": "assistant", "content": "Yes I like llamas. What do you want to know about llamas?"},
        {"role": "user", "content": user_prompt},
    ]

    headers = {"User-Agent": "Mozilla/3.01Gold (X11; I; SunOS 5.5.1 sun4m)"}
    options = {"n_predict": 128}
    client = LlamaCppAPIClient(base_url="http://localhost:8080", headers=headers, options=options)

    total = ""
    try:
        async for response in client.stream_completion(chat_thread=chat_thread, format="Llama-3"):
            if response.get("stop", False):
                print("")
                # print(f">>> Timings:\n{response["timings"]}")
                # print(f">>> Prompt:\n{response["prompt"]}")
                continue
            total += response["content"]
            print(response["content"], end="")
            sys.stdout.flush()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f">>> Response:\n{total}")


if __name__ == "__main__":
    asyncio.run(main())
