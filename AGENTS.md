This repository is called `speculative_decoding_speedup`. It is about speeding up speculative decoding for large language models. In this study, we are trying to build upon the "Learning to Draft" (LTD) paper to develop a more efficient speculative decoding method.

If this is your first time reading this repository, please read the README.md file before doing anything.

Some key philosophies in this project that affect how you write code:
1. Don't write unnecessary boilerplate code. Don't start functions with "maybe" in the name as it makes things confusing. Avoid using "_" at the front of function names as well. If you can inline a function, do it. We want to minimize levels of abstraction.
2. Write docstrings for all functions. Use the Google style. This is because many humans, and AI agents, will be reading your code. We want to make it easy for them to understand what your code does.
3. Use type hints.
4. Whenever you're entering a folder, you should always check any AGENTS.md and README.md files in that folder if present.
5. Don't use the `rg` tool to search for text. Not everyone has it installed. Don't assume that `gh` is installed as well.
6. Python virtual environments are located in the .venv/ directory in the project root. We always use `uv` for managing them. There may be multiple virtual environments in that folder. You should use the most sensible one based on the context/user's discussion. 