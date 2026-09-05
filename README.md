# Qwen Local Coding Agent

A simple local coding agent powered by any AI model from Ollama.

It can read, create, edit, and inspect files in the current project directory and execute PowerShell commands.

## Features

- Local AI model
- Ollama backend
- Read files
- Create files
- Edit files
- List files
- Execute PowerShell commands
- Uses the current directory as the workspace
- No cloud AI API required

## Requirements

- Windows
- Python 3.12+
- Ollama
- any AI model from Ollama

## Installation

### 1. Install Ollama

Install Ollama from:

https://ollama.com/

### 2. Download any model from Ollama

```powershell
ollama pull <model_name>
```

Check that it is installed:

```powershell
ollama list
```

### 3. Get the Agent

Clone this repository:

```powershell
git clone https://github.com/YOUR_USERNAME/qwen-local-agent.git
cd qwen-local-agent
```

## How to Use

Open PowerShell in the project directory you want Qwen to work on.

For example:

```powershell
cd C:\Projects\MyProject
```

Then start the agent:

```powershell
python C:\Path\To\qwen_agent.py
```

The directory where you start the agent becomes the workspace.

You can then give Qwen instructions such as:

```text
Inspect this project and explain its structure.
```

```text
Create a README.md file for this project.
```

```text
Find and fix the errors in this project.
```

```text
Run the tests and fix any failures.
```

```text
Build the project and fix any build errors.
```

## Example

```powershell
cd C:\Projects\MyApp
python C:\Path\To\qwen_agent.py
```

Then:

```text
You > Inspect this project and tell me what needs to be fixed.
```

Qwen can inspect the files, make changes, and execute required PowerShell commands.

## Workspace

The agent uses the current directory as its workspace.

For example:

```powershell
cd C:\Projects\MyApp
python qwen_agent.py
```

Qwen works with:

```text
C:\Projects\MyApp
```

and its subdirectories.

## Commands

Inside the agent:

```text
/reset
```

Reset the current conversation.

```text
/exit
```

Exit the agent.

```text
/quit
```

Exit the agent.

## Available Tools

The agent provides the following tools:

- `list_files` - List files and folders
- `read_file` - Read a file
- `write_file` - Create or replace a file
- `edit_file` - Edit an existing file
- `run_powershell` - Execute PowerShell commands

## Configuration


Ollama runs locally at:

```text
http://127.0.0.1:11434
```

Thinking is disabled for the agent.

## Built With

- Python
- Ollama
- any AI model from Ollama
- PowerShell
