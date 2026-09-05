import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

# The directory where you launch this program is the workspace.
WORKSPACE = Path.cwd().resolve()

SYSTEM_PROMPT = f"""
You are a local coding agent running on Windows.
Your model is {MODEL}. Reasoning/thinking is disabled.

WORKSPACE:
{WORKSPACE}

You have tools for reading, writing, editing files and executing PowerShell commands.

Rules:
- Work primarily inside the current workspace.
- Before changing code, inspect relevant existing files.
- Use tools instead of merely describing commands or file contents.
- After making changes, verify them when practical.
- Do not invent file contents; read files when you need their actual contents.
- For project builds/tests, use the appropriate commands for the detected project.
- Keep responses concise and report what you actually did.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a workspace-relative directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path. Use '.' for the workspace root."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or completely replace a text file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact piece of text in a workspace file. Use this for targeted edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "replace_all": {"type": "boolean"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_powershell",
            "description": "Execute a PowerShell command with the workspace as the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    }
]


def api_chat(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.6
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {body}")
    except URLError as e:
        raise RuntimeError(
            "Cannot connect to Ollama at http://127.0.0.1:11434. "
            "Make sure Ollama is running."
        ) from e


def inside_workspace(path_text):
    p = Path(path_text)
    if not p.is_absolute():
        p = WORKSPACE / p
    p = p.resolve()

    try:
        p.relative_to(WORKSPACE)
    except ValueError:
        raise RuntimeError(f"Path is outside workspace: {path_text}")

    return p


def list_files(path):
    p = inside_workspace(path)
    if not p.exists():
        return f"Directory does not exist: {path}"
    if not p.is_dir():
        return f"Not a directory: {path}"

    entries = []
    for x in sorted(p.iterdir(), key=lambda z: (not z.is_dir(), z.name.lower())):
        entries.append(("[DIR] " if x.is_dir() else "[FILE] ") + x.name)

    return "\n".join(entries) if entries else "(empty directory)"


def read_file(path):
    p = inside_workspace(path)
    if not p.exists():
        raise RuntimeError(f"File does not exist: {path}")
    if not p.is_file():
        raise RuntimeError(f"Not a file: {path}")

    # Keep accidental huge files from consuming the model context.
    max_chars = 200_000
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED]"
    return text


def write_file(path, content):
    p = inside_workspace(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {p.relative_to(WORKSPACE)} ({len(content)} characters)."


def edit_file(path, old_text, new_text, replace_all=False):
    p = inside_workspace(path)
    if not p.exists():
        raise RuntimeError(f"File does not exist: {path}")

    text = p.read_text(encoding="utf-8", errors="replace")

    if old_text not in text:
        raise RuntimeError("old_text was not found in the file.")

    if replace_all:
        updated = text.replace(old_text, new_text)
        count = text.count(old_text)
    else:
        updated = text.replace(old_text, new_text, 1)
        count = 1

    p.write_text(updated, encoding="utf-8")
    return f"Edited {p.relative_to(WORKSPACE)} ({count} replacement(s))."


def run_powershell(command):
    print(f"\n[PowerShell]\n{command}\n")

    # Commands run with the current workspace as their working directory.
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += ("\n[stderr]\n" if output else "[stderr]\n") + result.stderr

    if len(output) > 50_000:
        output = output[:50_000] + "\n[OUTPUT TRUNCATED]"

    return f"Exit code: {result.returncode}\n{output}"


def execute_tool(name, args):
    if name == "list_files":
        return list_files(args["path"])
    if name == "read_file":
        return read_file(args["path"])
    if name == "write_file":
        return write_file(args["path"], args["content"])
    if name == "edit_file":
        return edit_file(
            args["path"],
            args["old_text"],
            args["new_text"],
            args.get("replace_all", False),
        )
    if name == "run_powershell":
        return run_powershell(args["command"])
    raise RuntimeError(f"Unknown tool: {name}")


def main():
    print("=" * 64)
    print("Qwen Local Coding Agent")
    print("=" * 64)
    print(f"Model:     {MODEL}")
    print("Thinking:  OFF")
    print(f"Workspace: {WORKSPACE}")
    print("Backend:   Ollama http://127.0.0.1:11434")
    print("Type /exit to quit, /reset to start a fresh conversation.")
    print("=" * 64)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "/quit"}:
            print("Bye.")
            break

        if user_input.lower() == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation reset.")
            continue

        messages.append({"role": "user", "content": user_input})

        # Give the model several tool rounds so it can inspect, modify, test and fix.
        for _ in range(20):
            try:
                response = api_chat(messages)
            except Exception as e:
                print(f"\n[ERROR] {e}")
                break

            msg = response.get("message", {})
            messages.append(msg)

            text = msg.get("content") or ""
            if text:
                print(f"\nQwen > {text}")

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break

            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name")
                args = fn.get("arguments") or {}

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                print(f"\n[Tool] {name} {json.dumps(args, ensure_ascii=False)}")

                try:
                    result = execute_tool(name, args)
                except subprocess.TimeoutExpired:
                    result = "PowerShell command timed out after 600 seconds."
                except Exception as e:
                    result = f"Tool error: {e}"

                print(f"[Tool result]\n{result[:5000]}")

                messages.append({
                    "role": "tool",
                    "content": result,
                })

        else:
            print("\n[Agent stopped after 20 tool rounds.]")

if __name__ == "__main__":
    main()
