# Qwen Local Coding Agent

A lightweight, fully local AI coding agent powered by **Qwen3.5:9B** and **Ollama**.

Qwen Local Coding Agent allows a local Qwen model to interact with your project through tools for reading, creating, editing, and managing files, as well as executing PowerShell commands.

The goal is to provide a personal coding assistant that runs locally without requiring Gemini, OpenAI, Claude, or other cloud-based AI services.

---

## ✨ Features

- 🧠 **Qwen3.5:9B** as the AI model
- 🏠 Fully local inference through **Ollama**
- 🔒 No cloud AI API required
- ⚡ Thinking disabled for faster coding workflows
- 📂 List files and directories
- 📖 Read project files
- ✍️ Create new files
- 📝 Edit existing files
- 🔧 Replace specific sections of files
- 💻 Execute PowerShell commands
- 🧪 Build and test projects
- 🔄 Multi-step tool execution
- 📁 Uses the current directory as the workspace
- 🛡️ File operations are restricted to the workspace
- 🐍 Written in Python
- 🪶 Lightweight and easy to customize

---

## 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │      PowerShell      │
                         │                      │
                         │   python qwen_agent  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Qwen Local Agent    │
                         │     qwen_agent.py    │
                         └──────────┬───────────┘
                                    │
                                    │ HTTP API
                                    ▼
                         ┌──────────────────────┐
                         │        Ollama        │
                         │   localhost:11434    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Qwen3.5:9B      │
                         │    Thinking: OFF     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────┐
                 │               Tools                │
                 │                                    │
                 │  • list_files                      │
                 │  • read_file                       │
                 │  • write_file                      │
                 │  • edit_file                       │
                 │  • run_powershell                  │
                 └────────────────┬───────────────────┘
                                  │
                                  ▼
                 ┌────────────────────────────────────┐
                 │          Current Workspace         │
                 │                                    │
                 │  Source code                       │
                 │  Configuration files               │
                 │  Tests                              │
                 │  Assets                             │
                 │  Project files                      │
                 └────────────────────────────────────┘
