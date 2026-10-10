import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ============================================================
# Main local agent
# ============================================================

MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/chat",
)

# OpenRouter is ONLY used for background context compression.
# It never performs the user's coding task.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# This is the context budget you want the local Qwen request to stay within.
# Change it if your local Ollama model has a different num_ctx.
CONTEXT_WINDOW_TOKENS = int(os.getenv("AGENT_CONTEXT_TOKENS", "32768"))
# Do not send a new Qwen request when estimated context is above this safe budget.
REQUEST_CONTEXT_LIMIT_TOKENS = int(os.getenv(
    "AGENT_REQUEST_CONTEXT_LIMIT", str(int(CONTEXT_WINDOW_TOKENS * 0.75))
))
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "3600"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "8192"))
POWERSHELL_TIMEOUT = int(os.getenv("POWERSHELL_TIMEOUT", "3600"))
STATE_FILE = Path(os.getenv(
    "AGENT_STATE_FILE", str(Path.home() / ".qwen_coding_agent_state.json")
)).expanduser()



# Start background compression before the local context gets dangerously full.
COMPACTION_TRIGGER_TOKENS = int(
    os.getenv(
        "AGENT_COMPACTION_TRIGGER",
        str(int(CONTEXT_WINDOW_TOKENS * 0.50)),
    )
)

# Always keep the most recent messages untouched.
RECENT_MESSAGES_TO_KEEP = int(os.getenv("AGENT_RECENT_MESSAGES", "8"))

# The summary is deliberately small. It is inserted back into the local
# agent's context, effectively returning the tokens that were reclaimed.
SUMMARY_MAX_CHARS = int(os.getenv("AGENT_SUMMARY_MAX_CHARS", "16000"))

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
- When a command must be run, call run_powershell; never just print the command.
- The run_powershell tool executes commands for real and returns the exit code/output.
- After making changes, verify them when practical.
- Do not invent file contents; read files when you need their actual contents.
- For project builds/tests, use the appropriate commands for the detected project.
- Keep responses concise and report what you actually did.

A rolling context summary may appear before recent conversation messages.
Treat that summary as useful memory, but do not assume it contains every detail.

A temporary JSON archive preserves original messages even after context compaction.
When an exact past detail, command, code fragment, error, or decision is needed:
1. Call search_past_conversation with distinctive keywords.
2. Use the returned JSON pointer or message ID with get_past_conversation.
3. For long records, request additional chunks using start_char.
Never claim to have checked an old detail unless you retrieved it or it is in context.
The archive is session-only and is deleted on normal agent shutdown.
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
                        "description": "Workspace-relative path. Use '.' for the workspace root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
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
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
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
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_past_conversation",
            "description": (
                "Search the temporary archive of original past conversation messages, "
                "including messages already compressed out of the active context. "
                "Returns message IDs and stable JSON pointers; use get_past_conversation "
                "to retrieve the exact record."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Distinctive words or phrase to find."},
                    "max_results": {"type": "integer", "description": "Maximum matches, 1-10."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_past_conversation",
            "description": (
                "Retrieve an exact archived conversation record by message_id or JSON pointer "
                "such as /archive/records/12. Use start_char and max_chars to page through "
                "long messages without flooding the active context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pointer": {"type": "string", "description": "JSON pointer or message ID returned by search."},
                    "start_char": {"type": "integer", "description": "Character offset, default 0."},
                    "max_chars": {"type": "integer", "description": "Characters to return, 1-16000."},
                },
                "required": ["pointer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_powershell",
            "description": (
                "EXECUTE the PowerShell command now in the workspace and return its "
                "real stdout/stderr and exit code. Do NOT merely print, explain, or "
                "describe the command. You MUST call this tool whenever the user asks "
                "you to run a PowerShell command, install something, start a server, "
                "build/test a project, or execute a shell command."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


# ============================================================
# Helpers
# ============================================================

def estimate_tokens(messages):
    """
    Cheap local estimate. It does not need to be exact; its purpose is to
    prevent sending the entire historical transcript to Qwen.
    """
    total_chars = 0

    for message in messages:
        total_chars += len(str(message.get("content") or ""))
        total_chars += len(str(message.get("tool_calls") or ""))

    # Rough English/code estimate.
    return max(1, total_chars // 4)


def compact_text_for_summary(text, max_chars=12000):
    text = str(text or "")
    if len(text) <= max_chars:
        return text

    half = max_chars // 2
    return (
        text[:half]
        + "\n...[middle omitted for background compressor]...\n"
        + text[-half:]
    )


def message_signature(message):
    """
    Used to check that the history prefix we compressed has not changed while
    the local agent was simultaneously working.
    """
    return json.dumps(message, sort_keys=True, ensure_ascii=False)


def api_chat(messages):
    """
    Main/local model call.
    This is the ONLY request that performs the coding-agent work.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.6,
            "num_ctx": CONTEXT_WINDOW_TOKENS,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }

    data = json.dumps(payload).encode("utf-8")

    req = Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=OLLAMA_REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))

    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {body}")

    except URLError as e:
        raise RuntimeError(
            "Cannot connect to Ollama at "
            "http://127.0.0.1:11434. Make sure Ollama is running."
        ) from e


def openrouter_compact(previous_summary, old_messages):
    """
    Background context compressor.

    OpenRouter is deliberately isolated from the main agent. It receives
    only the older conversation that we want to compress.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Set it before starting the agent."
        )

    transcript_parts = []

    if previous_summary:
        transcript_parts.append(
            "EXISTING ROLLING SUMMARY:\n"
            + compact_text_for_summary(previous_summary, 14000)
        )

    transcript_parts.append("OLDER CONVERSATION TO COMPRESS:")

    # Keep the compressor request bounded even if a tool returned a huge file.
    for index, message in enumerate(old_messages, 1):
        role = message.get("role", "unknown")
        content = compact_text_for_summary(
            message.get("content") or "",
            9000,
        )

        tool_calls = message.get("tool_calls")
        if tool_calls:
            content += (
                "\nTOOL_CALLS:\n"
                + compact_text_for_summary(json.dumps(tool_calls, ensure_ascii=False), 6000)
            )

        transcript_parts.append(
            f"\n--- MESSAGE {index} ({role}) ---\n{content}"
        )

    transcript = "\n".join(transcript_parts)

    # OpenRouter has a large free-context pool, but we still avoid sending
    # unnecessary data to the compressor.
    transcript = compact_text_for_summary(transcript, 90000)

    prompt = f"""
You are the background memory compressor for a coding agent.

Your job is NOT to solve the user's coding task.
Your job is to preserve the useful state of an ongoing coding session in a
compact rolling memory so another local agent can continue without receiving
the entire old transcript.

Create one concise but information-dense summary.

Preserve:
1. The user's current goals and requirements.
2. Important decisions and constraints.
3. Files, directories, classes, functions, APIs, ports, commands and configs
   that were actually discussed or changed.
4. Important code changes already made.
5. Errors, failures, test/build results and their causes if known.
6. Current implementation state.
7. Pending work and next actions.
8. Important preferences explicitly stated by the user.

Do NOT invent facts.
Do NOT repeat conversational filler.
Keep exact technical names where possible.
Prefer bullets and short sections.

Return ONLY the rolling summary.

Maximum output: approximately {SUMMARY_MAX_CHARS} characters.

{transcript}
"""

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You compress coding-agent conversation history. "
                    "Preserve state, not prose."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max(1000, SUMMARY_MAX_CHARS // 4),
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")

    req = Request(
        OPENROUTER_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Local Qwen Coding Agent Context Compressor",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}")

    except URLError as e:
        raise RuntimeError(f"Cannot connect to OpenRouter: {e}") from e

    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {result}")

    summary = (
        choices[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not summary:
        raise RuntimeError("OpenRouter returned an empty context summary.")

    return summary[:SUMMARY_MAX_CHARS]


# ============================================================
# Workspace tools
# ============================================================

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

    for x in sorted(
        p.iterdir(),
        key=lambda z: (not z.is_dir(), z.name.lower()),
    ):
        entries.append(
            ("[DIR] " if x.is_dir() else "[FILE] ") + x.name
        )

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

    return (
        f"Wrote {p.relative_to(WORKSPACE)} "
        f"({len(content)} characters)."
    )


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

    return (
        f"Edited {p.relative_to(WORKSPACE)} "
        f"({count} replacement(s))."
    )


def run_powershell(command):
    """Actually execute PowerShell in the workspace and return the result."""
    command = str(command or "").strip()
    if not command:
        raise RuntimeError("PowerShell command is empty.")

    print(f"\n[PowerShell EXECUTING]\n{command}\n", flush=True)

    # Windows PowerShell is preferred because the agent is designed for Windows.
    # pwsh is accepted as a fallback for machines that only have PowerShell 7.
    powershell = "powershell.exe"
    try:
        probe = subprocess.run(
            [powershell, "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        if probe.returncode != 0:
            powershell = "pwsh.exe"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        powershell = "pwsh.exe"

    try:
        result = subprocess.run(
            [
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=POWERSHELL_TIMEOUT,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "PowerShell was not found. Install Windows PowerShell or PowerShell 7."
        ) from e

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    output = stdout
    if stderr:
        output += ("\n[stderr]\n" if output else "[stderr]\n") + stderr

    if len(output) > 50_000:
        output = output[:50_000] + "\n[OUTPUT TRUNCATED]"

    print(
        f"[PowerShell FINISHED] exit_code={result.returncode}",
        flush=True,
    )

    return f"Exit code: {result.returncode}\n{output}"


def extract_shell_commands(text):
    """Extract obvious PowerShell/shell blocks when a model prints commands instead of using the tool."""
    if not text:
        return []

    import re
    commands = []

    # Prefer fenced powershell/shell/code blocks.
    for match in re.finditer(
        r"```(?:powershell|pwsh|ps1|shell|cmd|bat)?\s*\n(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        block = match.group(1).strip()
        if block:
            commands.append(block)

    return commands


def execute_tool(name, args, context=None):
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

    if name == "search_past_conversation":
        if context is None:
            raise RuntimeError("Conversation archive is unavailable.")
        return context.search_archive(
            args["query"], args.get("max_results", 5)
        )

    if name == "get_past_conversation":
        if context is None:
            raise RuntimeError("Conversation archive is unavailable.")
        return context.get_archive_record(
            args["pointer"],
            args.get("start_char", 0),
            args.get("max_chars", 12000),
        )

    raise RuntimeError(f"Unknown tool: {name}")


# ============================================================
# Rolling context manager
# ============================================================

class RollingContext:
    """
    Bounded conversation memory with persistent state and background compression.
    OpenRouter only summarizes older context; local Qwen remains the coding model.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.history = []
        self.rolling_summary = ""
        # Append-only records preserve original messages independently of compaction.
        # Each pointer is an RFC 6901-style JSON pointer into /archive/records/N.
        self.archive_records = []
        self.compactor_running = False
        self.compactor_future = None
        self._load_state()

    @staticmethod
    def _copy_message(message):
        # JSON round-trip creates a detached, serializable copy of tool-call payloads too.
        return json.loads(json.dumps(message, ensure_ascii=False))

    def _append_archive_locked(self, message, timestamp=None):
        index = len(self.archive_records)
        record = {
            "id": f"msg-{index + 1:08d}",
            "pointer": f"/archive/records/{index}",
            "timestamp": timestamp or __import__("datetime").datetime.now().isoformat(timespec="milliseconds"),
            "message": self._copy_message(message),
        }
        self.archive_records.append(record)
        return record

    def _load_state(self):
        """Restore a leftover state file after an abnormal shutdown, if one exists."""
        try:
            if not STATE_FILE.exists():
                return
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            history = data.get("history", [])
            summary = data.get("rolling_summary", "")
            archive = data.get("archive", {}).get("records", [])
            if isinstance(history, list) and all(isinstance(x, dict) for x in history):
                self.history = history
                self.rolling_summary = summary if isinstance(summary, str) else ""
                if isinstance(archive, list) and all(isinstance(x, dict) for x in archive):
                    self.archive_records = archive
                # Backward compatibility with state files written before archive support.
                if not self.archive_records:
                    for message in self.history:
                        self._append_archive_locked(message)
                # Normalize the pointer to the current stable array position.
                for index, record in enumerate(self.archive_records):
                    record["pointer"] = f"/archive/records/{index}"
                    record.setdefault("id", f"msg-{index + 1:08d}")
                self._save_state_locked()
                print(
                    f"[Context] Restored {len(self.history)} active messages, "
                    f"{len(self.archive_records)} archived records, and "
                    f"{len(self.rolling_summary):,} summary characters.",
                    flush=True,
                )
        except Exception as e:
            print(f"[Context] Could not restore saved state: {e}", flush=True)

    def _save_state_locked(self):
        """Atomically save active context and the full session archive to JSON."""
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            temp_path = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
            payload = {
                "schema_version": 2,
                "saved_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                "history": self.history,
                "rolling_summary": self.rolling_summary,
                "archive": {
                    "description": "Append-only full-message archive; pointers stay stable within this session.",
                    "records": self.archive_records,
                },
            }
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            temp_path.replace(STATE_FILE)
        except Exception as e:
            print(f"[Context] Warning: could not save state: {e}", flush=True)

    def append(self, message):
        with self.lock:
            detached = self._copy_message(message)
            self.history.append(detached)
            self._append_archive_locked(detached)
            self._save_state_locked()

    def reset(self):
        with self.lock:
            self.history.clear()
            self.rolling_summary = ""
            self.archive_records.clear()
            self.compactor_running = False
            self.compactor_future = None
            self._save_state_locked()

    def search_archive(self, query, max_results=5):
        """Search full archived records; results include retrievable JSON pointers."""
        query = str(query or "").strip().casefold()
        if not query:
            return "Search query is empty. Provide distinctive words or a phrase."
        try:
            limit = max(1, min(int(max_results), 10))
        except (TypeError, ValueError):
            limit = 5

        terms = [term for term in query.split() if term]
        hits = []
        with self.lock:
            records = list(self.archive_records)
        # Newest first so recent related details are found quickly.
        for record in reversed(records):
            message = record.get("message", {})
            searchable = "\n".join([
                str(message.get("content") or ""),
                json.dumps(message.get("tool_calls") or [], ensure_ascii=False),
            ]).casefold()
            if all(term in searchable for term in terms):
                content = str(message.get("content") or "")
                excerpts = []
                lower_content = content.casefold()
                for term in terms[:3]:
                    position = lower_content.find(term)
                    if position >= 0:
                        left = max(0, position - 120)
                        right = min(len(content), position + len(term) + 220)
                        excerpts.append(content[left:right].replace("\n", " "))
                hits.append({
                    "id": record.get("id"),
                    "pointer": record.get("pointer"),
                    "timestamp": record.get("timestamp"),
                    "role": message.get("role", "unknown"),
                    "excerpt": " ... ".join(excerpts)[:700] or "(match found in tool-call metadata)",
                })
                if len(hits) >= limit:
                    break
        if not hits:
            return f"No archived conversation records matched: {query}"
        return json.dumps({"query": query, "matches": hits}, ensure_ascii=False, indent=2)

    def get_archive_record(self, pointer, start_char=0, max_chars=12000):
        """Fetch an archive record by stable JSON pointer or message ID; supports paging."""
        pointer = str(pointer or "").strip()
        if pointer.startswith("#"):
            pointer = pointer[1:]
        try:
            start_char = max(0, int(start_char))
            max_chars = max(1, min(int(max_chars), 16000))
        except (TypeError, ValueError):
            return "start_char and max_chars must be integers."

        with self.lock:
            records = list(self.archive_records)
        record = None
        if pointer.startswith("/archive/records/"):
            suffix = pointer.rsplit("/", 1)[-1]
            try:
                index = int(suffix)
                if 0 <= index < len(records):
                    record = records[index]
            except ValueError:
                record = None
        else:
            record = next((r for r in records if r.get("id") == pointer), None)

        if record is None:
            return f"Archive pointer/message ID not found: {pointer}"

        message = record.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            content = json.dumps(message, ensure_ascii=False, indent=2)
        end_char = min(len(content), start_char + max_chars)
        return json.dumps({
            "id": record.get("id"),
            "pointer": record.get("pointer"),
            "timestamp": record.get("timestamp"),
            "role": message.get("role", "unknown"),
            "message_metadata": {
                key: value for key, value in message.items()
                if key not in {"content", "tool_calls"}
            },
            "tool_calls": message.get("tool_calls"),
            "content_length": len(content),
            "start_char": start_char,
            "returned_until_char": end_char,
            "has_more": end_char < len(content),
            "next_start_char": end_char if end_char < len(content) else None,
            "content": content[start_char:end_char],
        }, ensure_ascii=False, indent=2)

    def cleanup_session_file(self):
        """Delete temporary JSON state on graceful shutdown."""
        for path in (STATE_FILE, STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")):
            try:
                path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[Context] Warning: could not delete {path}: {e}", flush=True)
        print(f"[Context] Session JSON archive deleted: {STATE_FILE}", flush=True)

    def build_messages(self):
        with self.lock:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if self.rolling_summary:
                messages.append({
                    "role": "system",
                    "content": (
                        "ROLLING CONTEXT SUMMARY FROM OLDER CONVERSATION:\n\n"
                        + self.rolling_summary
                    ),
                })
            messages.extend(list(self.history))
            return messages

    def stats(self):
        messages = self.build_messages()
        used = estimate_tokens(messages)
        free = max(0, CONTEXT_WINDOW_TOKENS - used)
        with self.lock:
            compacting = self.compactor_running
            summary_tokens = estimate_tokens(
                [{"role": "system", "content": self.rolling_summary}]
            ) if self.rolling_summary else 0
        return {
            "used": used,
            "free": free,
            "summary_tokens": summary_tokens,
            "compacting": compacting,
        }

    @staticmethod
    def _safe_split(history, proposed):
        """
        Never split between an assistant tool-call message and its tool results.
        If the boundary lands there, move it past the related tool messages.
        """
        split = max(0, min(proposed, len(history)))
        if split and split < len(history):
            previous = history[split - 1]
            if previous.get("role") == "assistant" and previous.get("tool_calls"):
                while split < len(history) and history[split].get("role") == "tool":
                    split += 1
        return split

    def maybe_start_compaction(self, executor, force=False):
        with self.lock:
            if self.compactor_running:
                return False

            assembled = self.build_messages()
            used = estimate_tokens(assembled)
            if (used < COMPACTION_TRIGGER_TOKENS and not force) or len(self.history) < 2:
                return False

            # Keep a smaller tail, but never leave a tool call without its results.
            keep = min(RECENT_MESSAGES_TO_KEEP, max(2, len(self.history) // 3))
            split_at = self._safe_split(self.history, len(self.history) - keep)

            # If the safe boundary swallowed too much, compact any useful prefix.
            if split_at <= 0:
                split_at = self._safe_split(self.history, max(1, len(self.history) // 2))
            if split_at <= 0:
                return False

            old_messages = [dict(x) for x in self.history[:split_at]]
            previous_summary = self.rolling_summary
            frozen_signature = [message_signature(x) for x in old_messages]
            self.compactor_running = True

        print(
            f"\n[Context] {used:,} tokens estimated. "
            f"Compressing {len(old_messages)} older messages in background...",
            flush=True,
        )
        future = executor.submit(
            self._run_compaction,
            previous_summary,
            old_messages,
            frozen_signature,
        )
        with self.lock:
            self.compactor_future = future
        return True

    def _shrink_retained_history_locked(self):
        """Shorten active-context message bodies while the archive keeps originals."""
        changed = False
        last_two_start = max(0, len(self.history) - 2)
        for index, message in enumerate(self.history):
            content = message.get("content")
            if not isinstance(content, str):
                continue
            # Lower limits deliberately create a reserve for tool schemas and output.
            limit = 6000 if index >= last_two_start else 1800
            if len(content) > limit:
                message["content"] = compact_text_for_summary(content, limit)
                message["content"] += "\n[Active context shortened; full message remains in session archive]"
                changed = True

        if changed:
            self._save_state_locked()
        return changed

    def prepare_request(self, executor):
        """Do not send a near-full transcript to Ollama while compaction is pending."""
        used = self.stats()["used"]
        if used <= REQUEST_CONTEXT_LIMIT_TOKENS:
            return

        print(
            f"\n[Context] {used:,} estimated tokens exceeds the safe request budget "
            f"({REQUEST_CONTEXT_LIMIT_TOKENS:,}). Waiting for compaction before calling Qwen...",
            flush=True,
        )

        # Try to schedule compression even if an environment override moved the
        # ordinary trigger above the safe-request limit.
        self.maybe_start_compaction(executor, force=True)
        with self.lock:
            future = self.compactor_future

        if future is not None:
            try:
                future.result()  # Compressor has its own 180-second HTTP timeout.
            except Exception as e:
                print(f"\n[Context] Compaction task ended with an error: {e}", flush=True)

        used = self.stats()["used"]
        if used > REQUEST_CONTEXT_LIMIT_TOKENS:
            with self.lock:
                self._shrink_retained_history_locked()
                used = estimate_tokens(self.build_messages())

        if used > REQUEST_CONTEXT_LIMIT_TOKENS:
            raise RuntimeError(
                f"Context still too large after compaction ({used:,} estimated tokens; "
                f"safe request budget {REQUEST_CONTEXT_LIMIT_TOKENS:,}). "
                "The request was not sent to Ollama. The original messages remain in the "
                "session archive; inspect /context or search the archive before continuing."
            )

        print(
            f"[Context] Safe to continue: {used:,} estimated tokens; "
            f"{max(0, CONTEXT_WINDOW_TOKENS - used):,} estimated tokens free.",
            flush=True,
        )

    def _run_compaction(self, previous_summary, old_messages, frozen_signature):
        try:
            new_summary = openrouter_compact(previous_summary, old_messages)

            with self.lock:
                if len(self.history) < len(old_messages):
                    print(
                        "\n[Context] History changed unexpectedly; keeping original history.",
                        flush=True,
                    )
                    return

                current_prefix = self.history[:len(old_messages)]
                current_signature = [message_signature(x) for x in current_prefix]
                if current_signature != frozen_signature:
                    print(
                        "\n[Context] Compressed prefix changed; keeping original history.",
                        flush=True,
                    )
                    return

                before = estimate_tokens(self.build_messages())
                self.history = self.history[len(old_messages):]
                self.rolling_summary = new_summary
                after = estimate_tokens(self.build_messages())

                # Compression may not free enough space when retained messages are huge.
                if after >= before or after > int(CONTEXT_WINDOW_TOKENS * 0.65):
                    self._shrink_retained_history_locked()
                    after = estimate_tokens(self.build_messages())

                self._save_state_locked()
                reclaimed = max(0, before - after)
                print(
                    "\n[Context] OpenRouter compression complete.\n"
                    f"[Context] Reclaimed approximately {reclaimed:,} estimated tokens.\n"
                    f"[Context] New free context: "
                    f"{max(0, CONTEXT_WINDOW_TOKENS - after):,} / "
                    f"{CONTEXT_WINDOW_TOKENS:,} tokens.",
                    flush=True,
                )
                if after > int(CONTEXT_WINDOW_TOKENS * 0.70):
                    print(
                        "[Context] Warning: context is still high. "
                        "The next user/tool cycle will trigger another compaction.",
                        flush=True,
                    )

        except Exception as e:
            print(f"\n[Context] Background compression failed: {e}", flush=True)
        finally:
            with self.lock:
                self.compactor_running = False
                self.compactor_future = None


# ============================================================
# Main loop
# ============================================================

def main():
    print("=" * 64)
    print("Qwen Local Coding Agent + Background OpenRouter Compactor")
    print("=" * 64)
    print(f"Main model:       {MODEL}")
    print(f"Thinking:         OFF")
    print(f"Workspace:        {WORKSPACE}")
    print(f"Backend:          Ollama {OLLAMA_URL}")
    print(f"Context budget:   {CONTEXT_WINDOW_TOKENS:,} tokens")
    print(f"Request safe cap: {REQUEST_CONTEXT_LIMIT_TOKENS:,} tokens")
    print(f"Max output:       {OLLAMA_NUM_PREDICT:,} tokens/request")
    print(f"Ollama timeout:   {OLLAMA_REQUEST_TIMEOUT:,} seconds")
    print(f"PowerShell limit: {POWERSHELL_TIMEOUT:,} seconds")
    print(f"State file:       {STATE_FILE}")
    print(f"Compact trigger:  {COMPACTION_TRIGGER_TOKENS:,} tokens")
    print(f"OpenRouter model: {OPENROUTER_MODEL}")
    print()
    print("Main coding work stays on local Ollama.")
    print("OpenRouter is used only for background context compression.")
    print("Both can run in parallel.")
    print()
    print("Commands: /exit, /quit, /reset, /context")
    print("Past-history tools: search_past_conversation, get_past_conversation")
    print("=" * 64)

    if not OPENROUTER_API_KEY:
        print(
            "\n[WARNING] OPENROUTER_API_KEY is not set."
            "\nThe local agent will work, but automatic background "
            "context compression will be unavailable."
        )

    context = RollingContext()

    # A single background worker is intentional: only one compressor should
    # mutate rolling memory at a time.
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="openrouter-context",
    )

    try:
        while True:
            try:
                user_input = input("\nYou > ").strip()

            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue

            command = user_input.lower()

            if command in {"/exit", "/quit"}:
                print("Bye.")
                break

            if command == "/reset":
                context.reset()
                print("Conversation reset.")
                continue

            if command == "/context":
                stats = context.stats()
                print(
                    "\n[Context]"
                    f"\n  Used:              {stats['used']:,} tokens"
                    f"\n  Free:              {stats['free']:,} tokens"
                    f"\n  Rolling summary:   {stats['summary_tokens']:,} tokens"
                    f"\n  Archived records:  {len(context.archive_records):,}"
                    f"\n  Compaction active: {stats['compacting']}"
                    f"\n  Budget:            {CONTEXT_WINDOW_TOKENS:,} tokens"
                )
                continue

            context.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            # Start compression BEFORE/WHILE the local agent works.
            # This does not block api_chat().
            context.maybe_start_compaction(executor)

            # Give the model several tool rounds so it can inspect, modify,
            # test and fix.
            for _ in range(20):
                try:
                    context.prepare_request(executor)
                    request_messages = context.build_messages()
                    response = api_chat(request_messages)

                except Exception as e:
                    print(f"\n[ERROR] {e}")
                    break

                msg = response.get("message", {})

                # Add the assistant response to the shared conversation.
                context.append(msg)

                text = msg.get("content") or ""

                if text:
                    print(f"\nQwen > {text}")

                tool_calls = msg.get("tool_calls") or []

                # Some local models occasionally print a command instead of
                # emitting a structured tool call. Execute obvious fenced
                # PowerShell blocks as a safety net so the agent does not just
                # display the command and stop.
                if not tool_calls:
                    fallback_commands = extract_shell_commands(text)
                    if fallback_commands:
                        print(
                            "\n[Tool fallback] Model printed a shell command "
                            "instead of calling run_powershell. Executing it...",
                            flush=True,
                        )
                        for command in fallback_commands:
                            try:
                                result = run_powershell(command)
                            except subprocess.TimeoutExpired:
                                result = f"PowerShell command timed out after {POWERSHELL_TIMEOUT} seconds."
                            except Exception as e:
                                result = f"Tool error: {e}"

                            print(f"[Tool result]\n{result[:5000]}", flush=True)
                            context.append({
                                "role": "tool",
                                "content": result,
                            })
                        context.maybe_start_compaction(executor)
                        continue

                    break

                for call in tool_calls:
                    fn = call.get("function", {}) or {}
                    name = fn.get("name")
                    args = fn.get("arguments") or {}

                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError as e:
                            print(
                                f"\n[Tool error] Invalid JSON arguments for {name}: {e}",
                                flush=True,
                            )
                            args = {}

                    if not isinstance(args, dict):
                        args = {}

                    print(
                        f"\n[Tool] EXECUTING {name} "
                        f"{json.dumps(args, ensure_ascii=False)}",
                        flush=True,
                    )

                    try:
                        result = execute_tool(name, args, context=context)

                    except subprocess.TimeoutExpired:
                        result = (
                            "PowerShell command timed out after "
                            f"{POWERSHELL_TIMEOUT} seconds."
                        )

                    except Exception as e:
                        result = f"Tool error: {e}"

                    print(f"[Tool result]\n{result[:5000]}", flush=True)

                    # Ollama accepts tool results as role=tool. Include the
                    # call id when the model supplied one; this keeps the
                    # tool-call/result pairing correct for stricter models.
                    tool_message = {
                        "role": "tool",
                        "content": result,
                    }
                    if call.get("id"):
                        tool_message["tool_call_id"] = call["id"]

                    context.append(tool_message)

                    # If tool output pushed us over the threshold, start
                    # compression while Qwen is still doing subsequent work.
                    context.maybe_start_compaction(executor)

            else:
                print("\n[Agent stopped after 20 tool rounds.]")

            # If the user turn itself pushed the context over the threshold,
            # make sure compression gets scheduled for the next cycle.
            context.maybe_start_compaction(executor)

            stats = context.stats()

            print(
                f"\n[Context] "
                f"used≈{stats['used']:,} | "
                f"free≈{stats['free']:,} | "
                f"summary≈{stats['summary_tokens']:,} | "
                f"background_compactor="
                f"{'RUNNING' if stats['compacting'] else 'idle'}"
            )

    finally:
        # Let any in-flight compaction finish before deleting its state file, otherwise
        # a background worker could recreate the JSON after cleanup.
        executor.shutdown(wait=True, cancel_futures=False)
        context.cleanup_session_file()


if __name__ == "__main__":
    main()
