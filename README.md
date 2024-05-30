llama-cpp-api-client
===
LLaMA.cpp HTTP Server API Streaming Python Client

## What
A very thin python library providing async streaming inferencing to
LLaMA.cpp's HTTP Server via the API endpoints e.g. `/completion`.

## Quick Start
```bash
pip install llama-cpp-api-client
```

```python
from llama_cpp_api_client import streaming_response
```

## Why
While you could get up and running quickly using something like
[LiteLLM](https://github.com/BerriAI/litellm) or the official
[openai-python](https://github.com/openai/openai-python) client, neither
of those options seemed to provide enough flexibility regarding:

* Full control of exact prompt templates.
* Return correct tokens/second prompt and generation speed.
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
- [ ] add usage example with "rich" markdown CLI client.
- [ ] Support ChatML and Llama-3 prompt formats
- [ ] Streamng inferencing
- [ ] Possibly support additional API endpoints...
- [ ] publish to PyPI

## References
* [LLaMA.cpp HTTP Server Docs](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
* [denkiwakame/py-tiny-pkg PyPi Project Template](https://github.com/denkiwakame/py-tiny-pkg)
