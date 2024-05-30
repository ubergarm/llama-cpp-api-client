llama-cpp-api-client
===
LLaMA.cpp HTTP Server API Streaming Python Client

## What
A very thin python library providing async streaming inferencing to
LLaMA.cpp's HTTP Server via the API endpoints e.g. `/completion`.

## Quick Start
```bash
# install this jawn
pip install llama-cpp-api-client

# spin up your LLaMA.cpp HTTP Server
./server \
    --model "../models/mradermacher/Smaug-Llama-3-70B-Instruct-abliterated-v3-i1-GGUF/Smaug-Llama-3-70B-Instruct-abliterated-v3.i1-IQ4_XS.gguf" \
    --n-gpu-layers 44 \
    --ctx-size 4096 \
    --threads 24 \
    --flash-attn \
    --mlock \
    --n-predict -1 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --host 127.0.0.1 \
    --port 8080
```

# check out the examples
```python
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
```

## Why
While you could get up and running quickly using something like
[LiteLLM](https://github.com/BerriAI/litellm) or the official
[openai-python](https://github.com/openai/openai-python) client, neither
of those options seemed to provide enough flexibility regarding:

* Full control of exact prompt templates e.g. ChatML, Llama-3, etc...
* Return correct tokens/second speed for prompt and generation timings.
* Likely more...

Also, it seems like the built in LLaMA.cpp HTTP Server web app and
examples don't use the correct prompt template and stop tokens for many
newer Open LLM models which can degrade results and over-generate outputs
with the Assistant taking the User's turn or getting lots of `---` breaks.

* [completion.js](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/public/completion.js#L5)
* [chat.mjs](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/chat.mjs#L59)
* [chat.sh](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/chat.sh#L52)

So I wanted a minimalist way to interact with Open LLMs using all the
benefits of LLaMA.cpp for inferencing on a single 3090TI with full
control over all the knobs and prompts.

## Development
```bash
# create pip virtual environment
python3 -m venv ./venv
source ./venv/bin/activate
python3 -m ensurepip --upgrade

# install deps
pip install --upgrade -r requirements.txt
pip install --upgrade -r requirements-dev.txt

# install package
pip install .
pip show -f llama_cpp_api_client

# install from github
pip install git+https://github.com/ubergarm/llama-cpp-api-client

# testing
pip install .[dev]
python3 -m pytest --cov

# upgrade and check versions
python3 -m venv --upgrade ./venv
pip freeze
python3 -V

# formatting
black --line-length 120
isort
```

## TODO
- [x] initial commit
- [x] setup project to use as a library
- [x] Streamng inferencing
- [ ] Support ChatML and Llama-3 prompt formats
- [ ] add usage example with "rich" markdown CLI client.
- [ ] Possibly support additional API endpoints...
- [ ] publish to PyPI

## References
* [LLaMA.cpp HTTP Server Docs](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
* [denkiwakame/py-tiny-pkg PyPi Project Template](https://github.com/denkiwakame/py-tiny-pkg)
