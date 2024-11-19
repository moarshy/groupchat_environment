# Group Chat

This is an environment with a groupchat mechanism. You can check out other examples of agent, orchestrator and environment modules using the CLI command with the Naptha SDK. 

## Running the Environment Module on a Naptha Node

### Pre-Requisites 

#### Install the Naptha SDK

Install the Naptha SDK using the [instructions here](https://github.com/NapthaAI/naptha-sdk).

#### (Optional) Run your own Naptha Node

You can run your own Naptha node using the [instructions here](https://github.com/NapthaAI/node).

### Run the Environment Module

Using the Naptha SDK:

```bash
naptha run environment:groupchat -p "function_name='get_global_state' function_input_data=None"
```

## Running the Environment Module Locally

### Pre-Requisites 

#### Install Poetry 

From the official poetry [docs](https://python-poetry.org/docs/#installing-with-the-official-installer):

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/$(whoami)/.local/bin:$PATH"
```

### Clone and Install the Environment Module

Clone the module using:

```bash
git clone https://github.com/NapthaAI/groupchat
cd groupchat
```

You can install the module using:

```bash
poetry install
```

### Running the Module

Before deploying to a Naptha node, you should iterate on improvements with the module locally. You can run the module using:

```bash
poetry run python groupchat/run.py
```
