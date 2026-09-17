import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ============================================================
# CONFIG
# ============================================================

# Main coding model. Keep your current model here.
MODEL = "qwen3.5:9b"

# ============================================================
# COMPACTOR MODEL PLACEHOLDER
# ============================================================
#
# Put the SMALL model you choose here after testing it with Ollama.
# Examples (DO NOT uncomment blindly):
#     COMPACTOR_MODEL = "qwen3:1.7b"
#     COMPACTOR_MODEL = "qwen3:4b"
#     COMPACTOR_MODEL = "gemma3:1b"
#
# The compactor does NOT need tools or coding ability. It only needs to
# summarize context accurately and quickly.
#
COMPACTOR_MODEL = "YOUR_SMALL_COMPACTOR_MODEL_HERE"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
WORKSPACE = Path.cwd().resolve()

# Main model context window.
NUM_CTX = 12_000

# Keep a deliberate free reserve. Start preparing replacement memory early.
COMPACT_TRIGGER_TOKENS = 4_000

# Desired approximate size of the rebuilt main context after compaction.
# This is a target, not a hard tokenizer count.
COMPACT_TARGET_TOKENS = 5_000

# Last-resort local cleanup. The main model never waits for the compactor.
HARD_SAFETY_TOKENS = 9_000

# Recent complete user/tool interaction blocks kept verbatim-ish.
KEEP_RECENT_BLOCKS = 3

# Small model context is enough for summarization.
COMPACTOR_CTX = 4_000

# Limit generated text from both models so a single response cannot consume
# the entire context budget or run for an excessive amount of time.
MAIN_NUM_PREDICT = 1_500
COMPACTOR_NUM_PREDICT = 800

MAX_FILE_CHARS = 30_000
MAX_TOOL_RESULT_CHARS = 4_000
MAX_HISTORY_MESSAGE_CHARS = 3_000
MAX_SNAPSHOT_MESSAGE_CHARS = 1_500


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are a local coding agent running on Windows.
Your main model is {MODEL}. Reasoning/thinking is disabled.

WORKSPACE:
{WORKSPACE}

You have tools for reading, writing, editing files and executing PowerShell commands.

WORKING RULES:
- Work primarily inside the current workspace.
- Inspect only files directly relevant to the user's request.
- Do NOT scan, read, or reread the entire workspace unless explicitly asked.
- Use list_files only when you need to discover an unknown project structure.
- Treat files already read during this session as known.
- Do NOT call read_file merely because a new user prompt arrived.
- Do NOT reread a file you just created or edited.
- After write_file/edit_file succeeds, treat the resulting file as known.
- Use tools instead of merely describing commands or file contents.
- Do not invent file contents.
- Prefer targeted edits over rewriting unrelated files.
- Verify changes when practical.
- If a command fails, inspect the error and fix it.
- Continue working autonomously until the requested task is complete.
- Stop when the task is complete.

CONTEXT RULES:
- The workspace and file cache are the source of truth.
- Older conversation history may be replaced by compacted session memory.
- Never rescan the workspace simply because conversation history was compacted.
- Preserve the current user request and the latest work in every request.
- Keep responses concise.

PARALLEL COMPACTION:
- A separate small model compacts older generated context in the background.
- The main coding request must never wait for the compactor.
- When replacement compacted memory becomes ready, it is used on a subsequent
  main request for the same task.
- Compacted memory is REPLACEMENT memory. Do not accumulate old summaries forever.
"""


# ============================================================
# TOOLS
# ============================================================

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
            "description": "Read a relevant UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
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
            "description": "Replace an exact piece of text in a workspace file.",
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
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    }
]


# ============================================================
# OLLAMA
# ============================================================

def api_chat(messages, model=MODEL, use_tools=True, context=NUM_CTX,
             num_predict=MAIN_NUM_PREDICT, temperature=0.6):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": context,
            "num_predict": num_predict,
        }
    }

    if use_tools:
        payload["tools"] = TOOLS

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
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


# ============================================================
# FILE CACHE
# ============================================================

FILE_CACHE = {}
CACHE_LOCK = threading.Lock()


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


def cache_put(p, content):
    with CACHE_LOCK:
        FILE_CACHE[str(p)] = {
            "mtime_ns": p.stat().st_mtime_ns,
            "content": content
        }


def cache_get(p):
    with CACHE_LOCK:
        item = FILE_CACHE.get(str(p))
    if not item:
        return None
    try:
        current_mtime = p.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    if current_mtime != item["mtime_ns"]:
        return None
    return item["content"]


def cached_files_text():
    with CACHE_LOCK:
        paths = list(FILE_CACHE.keys())
    if not paths:
        return "Cached files: none"
    names = []
    for p in paths:
        try:
            names.append(str(Path(p).relative_to(WORKSPACE)))
        except ValueError:
            pass
    return "Cached files:\n- " + "\n- ".join(sorted(names))


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

    cached = cache_get(p)
    if cached is not None:
        return f"[CACHE HIT] {p.relative_to(WORKSPACE)}\n{cached}"

    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + "\n\n[TRUNCATED]"
    cache_put(p, text)
    return f"[READ FROM DISK] {p.relative_to(WORKSPACE)}\n{text}"


def write_file(path, content):
    p = inside_workspace(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    cache_put(p, content)
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
    cache_put(p, updated)
    return f"Edited {p.relative_to(WORKSPACE)} ({count} replacement(s))."


# ============================================================
# POWERSHELL
# ============================================================

def run_powershell(command):
    print(f"\n[PowerShell] {command}")
    env = os.environ.copy()
    env["CI"] = "1"
    env["npm_config_yes"] = "true"
    process = None

    try:
        process = subprocess.Popen(
            [
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-Command", command
            ],
            cwd=str(WORKSPACE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        output = []
        for line in process.stdout:
            print(line, end="", flush=True)
            output.append(line)

        code = process.wait(timeout=300)
        result = "".join(output)
        if len(result) > 20_000:
            result = result[:20_000] + "\n[OUTPUT TRUNCATED]"
        return f"Exit code: {code}\n{result}"

    except subprocess.TimeoutExpired:
        if process:
            process.kill()
            process.wait()
        return "PowerShell command timed out after 300 seconds."
    except KeyboardInterrupt:
        if process:
            process.kill()
            process.wait()
        return "PowerShell command interrupted."


def execute_tool(name, args):
    if name == "list_files":
        return list_files(args["path"])
    if name == "read_file":
        return read_file(args["path"])
    if name == "write_file":
        return write_file(args["path"], args["content"])
    if name == "edit_file":
        return edit_file(
            args["path"], args["old_text"], args["new_text"],
            args.get("replace_all", False)
        )
    if name == "run_powershell":
        return run_powershell(args["command"])
    raise RuntimeError(f"Unknown tool: {name}")


# ============================================================
# CONTEXT HELPERS
# ============================================================

def estimate_tokens(messages):
    payload = json.dumps(messages, ensure_ascii=False)
    return max(1, len(payload) // 4)


def truncate_text(text, limit):
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[TRUNCATED]"


def compact_message_for_snapshot(message):
    """Make an inexpensive snapshot without carrying giant source payloads."""
    role = message.get("role", "")

    if role == "tool":
        return {
            "role": "tool",
            "content": truncate_text(
                message.get("content", ""),
                MAX_SNAPSHOT_MESSAGE_CHARS
            )
        }

    result = {"role": role}
    content = message.get("content") or ""
    if content:
        result["content"] = truncate_text(content, MAX_SNAPSHOT_MESSAGE_CHARS)

    calls = message.get("tool_calls") or []
    if calls:
        safe_calls = []
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "unknown")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if name in {"write_file", "edit_file"}:
                safe_args = {"path": args.get("path", "")}
            elif name in {"read_file", "list_files"}:
                safe_args = {"path": args.get("path", ".")}
            elif name == "run_powershell":
                safe_args = {
                    "command": truncate_text(args.get("command", ""), 700)
                }
            else:
                safe_args = {}

            safe_calls.append({
                "function": {
                    "name": name,
                    "arguments": safe_args
                }
            })
        result["tool_calls"] = safe_calls

    return result


def compact_message_for_main_history(message):
    """Keep recent history small enough to preserve free context."""
    role = message.get("role", "")
    result = {"role": role}

    if role == "tool":
        result["content"] = truncate_text(
            message.get("content", ""),
            MAX_HISTORY_MESSAGE_CHARS
        )
        return result

    if message.get("content"):
        result["content"] = truncate_text(
            message.get("content", ""),
            MAX_HISTORY_MESSAGE_CHARS
        )

    if message.get("tool_calls"):
        result["tool_calls"] = message["tool_calls"]

    return result


def split_into_blocks(messages):
    """Group messages by user turn so tool messages are not orphaned."""
    blocks = []
    current = []

    for message in messages:
        if message.get("role") == "system":
            continue

        if message.get("role") == "user" and current:
            blocks.append(current)
            current = []

        current.append(message)

    if current:
        blocks.append(current)

    return blocks


def recent_blocks_for_main(blocks):
    selected = blocks[-KEEP_RECENT_BLOCKS:]
    return [
        [compact_message_for_main_history(m) for m in block]
        for block in selected
    ]


def build_compaction_snapshot(messages, previous_memory, main_memory):
    blocks = split_into_blocks(messages)
    old_blocks = blocks[:-KEEP_RECENT_BLOCKS] if len(blocks) > KEEP_RECENT_BLOCKS else []

    snapshot = []
    for block in old_blocks:
        for message in block:
            snapshot.append(compact_message_for_snapshot(message))

    # If there is not enough old history yet, compact the oldest part of the
    # current blocks too, but never remove the active recent blocks here.
    return {
        "previous_compacted_session_memory": truncate_text(previous_memory, 7000),
        "main_session_memory": truncate_text(main_memory, 4000),
        "older_generated_context": snapshot,
        "cached_files": cached_files_text(),
    }


# ============================================================
# BACKGROUND COMPACTOR
# ============================================================

class BackgroundCompactor:
    """
    One non-blocking background request at a time.

    The compactor uses COMPACTOR_MODEL, not the main 9B model.
    Its result is a REPLACEMENT rolling memory, never an accumulating stack.
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="context-compactor"
        )
        self.lock = threading.Lock()
        self.future = None
        self.generation = 0
        self.ready_generation = 0
        self.ready_memory = None
        self.main_session_memory = ""
        self.compacted_session_memory = ""
        self.last_snapshot_signature = None

    def get_main_session_memory(self):
        with self.lock:
            return self.main_session_memory

    def get_compacted_memory(self):
        with self.lock:
            return self.compacted_session_memory

    def _make_snapshot(self, messages):
        return build_compaction_snapshot(
            messages,
            self.compacted_session_memory,
            self.main_session_memory,
        )

    def _summarize(self, snapshot, generation):
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a background context compactor for a coding agent. "
                    "Create ONE concise REPLACEMENT COMPACTED SESSION MEMORY. "
                    "The main coding agent is working concurrently. Do not give "
                    "instructions and do not reproduce source code. Preserve only "
                    "facts needed to continue the same task: user goal, files "
                    "created/modified, architecture/implementation decisions, "
                    "dependencies, commands/tests, errors and fixes, current state, "
                    "constraints, and unresolved work. Merge the previous compacted "
                    "memory with the older generated context. Remove repetition and "
                    "obsolete details. Keep it short so the main model has plenty of "
                    "free context."
                )
            },
            {
                "role": "user",
                "content": (
                    "Produce the replacement memory from this snapshot:\n\n"
                    + json.dumps(snapshot, ensure_ascii=False)
                )
            }
        ]

        try:
            response = api_chat(
                prompt,
                model=COMPACTOR_MODEL,
                use_tools=False,
                context=COMPACTOR_CTX,
                num_predict=COMPACTOR_NUM_PREDICT,
                temperature=0.2,
            )
            summary = response.get("message", {}).get("content", "").strip()
            return generation, summary
        except Exception as exc:
            print(f"\n[Context] Background compactor error: {exc}")
            return generation, ""

    def start_if_needed(self, messages):
        if COMPACTOR_MODEL == "YOUR_SMALL_COMPACTOR_MODEL_HERE":
            return

        snapshot = self._make_snapshot(messages)
        signature = hash(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))

        with self.lock:
            if self.future is not None and not self.future.done():
                return
            if signature == self.last_snapshot_signature:
                return

            self.last_snapshot_signature = signature
            self.generation += 1
            generation = self.generation
            self.future = self.executor.submit(
                self._summarize,
                snapshot,
                generation
            )

        print(
            f"\n[Context] Background COMPACTOR started "
            f"(model={COMPACTOR_MODEL}, generation={generation})."
        )

    def collect_if_ready(self):
        with self.lock:
            future = self.future
            if future is None or not future.done():
                return False
            self.future = None

        try:
            generation, summary = future.result()
        except Exception as exc:
            print(f"\n[Context] Failed collecting compactor result: {exc}")
            return False

        if not summary:
            return False

        with self.lock:
            if generation >= self.ready_generation:
                self.ready_generation = generation
                self.ready_memory = summary

        print(
            f"\n[Context] Background compaction finished "
            f"(generation {generation}). Replacement memory is ready."
        )
        return True

    def consume_ready_memory(self):
        with self.lock:
            memory = self.ready_memory
            generation = self.ready_generation
            self.ready_memory = None
        return generation, memory

    def adopt_ready_memory(self):
        generation, memory = self.consume_ready_memory()
        if not memory:
            return False
        with self.lock:
            self.compacted_session_memory = memory
        print(
            f"[Context] Generation {generation} adopted. "
            f"Old raw context can now be discarded from the main workflow."
        )
        return True

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


# ============================================================
# MAIN CONTEXT BUILDER
# ============================================================

def build_main_context(messages, compactor):
    """
    Build the next main-model request as:

        SYSTEM PROMPT
        + MAIN SESSION MEMORY
        + ONE COMPACTED SESSION MEMORY
        + RECENT INTERACTIONS
        + CURRENT TASK

    The function is non-blocking and never waits for the compactor.
    """
    compactor.collect_if_ready()
    compactor.adopt_ready_memory()

    blocks = split_into_blocks(messages)
    recent = recent_blocks_for_main(blocks)

    rebuilt = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    main_memory = compactor.get_main_session_memory()
    compacted = compactor.get_compacted_memory()

    if main_memory:
        rebuilt.append({
            "role": "system",
            "content": "MAIN SESSION MEMORY:\n" + main_memory
        })

    if compacted:
        rebuilt.append({
            "role": "system",
            "content": (
                "COMPACTED SESSION MEMORY:\n"
                + compacted
                + "\n\nThis is a rolling replacement summary of older context. "
                  "The workspace/cache is the source of truth."
            )
        })

    # Recent blocks contain the current task. Keep the latest user message
    # fully enough to avoid losing what the user actually asked.
    for block in recent:
        rebuilt.extend(block)

    return rebuilt


def local_hard_safety(messages, compactor):
    """Deterministic fallback if background compaction is not ready."""
    blocks = split_into_blocks(messages)
    recent = recent_blocks_for_main(blocks)

    rebuilt = [{"role": "system", "content": SYSTEM_PROMPT}]

    main_memory = compactor.get_main_session_memory()
    compacted = compactor.get_compacted_memory()

    if main_memory:
        rebuilt.append({
            "role": "system",
            "content": "MAIN SESSION MEMORY:\n" + main_memory
        })

    if compacted:
        rebuilt.append({
            "role": "system",
            "content": "COMPACTED SESSION MEMORY:\n" + compacted
        })

    for block in recent:
        rebuilt.extend(block)

    print("[Context] HARD SAFETY: old raw history reduced without waiting.")
    return rebuilt


# ============================================================
# TOOL DISPLAY
# ============================================================

def print_tool_call(name, args):
    if name in {"write_file", "edit_file", "read_file"}:
        print(f"\n[Tool] {name} -> {args.get('path', 'unknown')}")
    elif name == "list_files":
        print(f"\n[Tool] list_files -> {args.get('path', '.')}")
    elif name == "run_powershell":
        print(f"\n[Tool] run_powershell -> {args.get('command', '')}")
    else:
        print(f"\n[Tool] {name}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("Qwen Local Coding Agent - Rolling Parallel Context")
    print("=" * 72)
    print(f"Main model:             {MODEL}")
    print(f"Compactor model:        {COMPACTOR_MODEL}")
    print(f"Main context:           {NUM_CTX}")
    print(f"Compaction starts:      ~{COMPACT_TRIGGER_TOKENS}")
    print(f"Compaction target:      ~{COMPACT_TARGET_TOKENS}")
    print(f"Hard safety:            ~{HARD_SAFETY_TOKENS}")
    print(f"Main max output:        {MAIN_NUM_PREDICT}")
    print(f"Compactor max output:   {COMPACTOR_NUM_PREDICT}")
    print("File cache:             ON")
    print("Parallel compaction:    ON")
    print("Type /exit to quit, /reset to reset the conversation.")
    print("=" * 72)

    if COMPACTOR_MODEL == "YOUR_SMALL_COMPACTOR_MODEL_HERE":
        print(
            "[Context] Compactor is DISABLED until you set "
            "COMPACTOR_MODEL to a local Ollama model."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    compactor = BackgroundCompactor()

    try:
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
                with compactor.lock:
                    compactor.ready_memory = None
                    compactor.ready_generation = 0
                    compactor.main_session_memory = ""
                    compactor.compacted_session_memory = ""
                    compactor.last_snapshot_signature = None
                print("Conversation and rolling memory reset. File cache preserved.")
                continue

            messages.append({"role": "user", "content": user_input})

            # Keep the user's current request available as explicit main-session
            # memory. It is intentionally small and survives rolling compaction.
            with compactor.lock:
                compactor.main_session_memory = truncate_text(
                    user_input,
                    3_000
                )

            for round_number in range(30):
                # Collect/adopt completed compaction without waiting.
                messages = build_main_context(messages, compactor)
                estimated = estimate_tokens(messages)

                # Proactively launch the small-model compactor before the main
                # context gets crowded. The main request continues immediately.
                if estimated >= COMPACT_TRIGGER_TOKENS:
                    compactor.start_if_needed(messages)

                # If we are approaching the ceiling, do deterministic local
                # reduction rather than waiting for the background request.
                if estimated >= HARD_SAFETY_TOKENS:
                    messages = local_hard_safety(messages, compactor)
                    estimated = estimate_tokens(messages)

                print(
                    f"\n[Context] Main request ~{estimated} tokens "
                    f"/ {NUM_CTX}; free reserve ~{max(0, NUM_CTX-estimated)} tokens."
                )

                try:
                    response = api_chat(
                        messages,
                        model=MODEL,
                        use_tools=True,
                        context=NUM_CTX,
                        num_predict=MAIN_NUM_PREDICT,
                        temperature=0.6,
                    )
                except Exception as exc:
                    print(f"\n[ERROR] {exc}")
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

                    print_tool_call(name, args)

                    try:
                        result = execute_tool(name, args)
                    except KeyboardInterrupt:
                        result = "Command interrupted by the user."
                    except Exception as exc:
                        result = f"Tool error: {exc}"

                    # Console output can be large, but the stored tool result is
                    # bounded too. This prevents giant npm/build logs from filling
                    # the model context.
                    result_for_context = truncate_text(
                        result,
                        MAX_TOOL_RESULT_CHARS
                    )

                    print(f"[Tool result]\n{truncate_text(result, 5000)}")

                    messages.append({
                        "role": "tool",
                        "content": result_for_context,
                    })

                    # Feed the background compactor after tool activity as soon
                    # as the threshold is reached. This is the key pipeline:
                    # main tool work -> snapshot -> small model compaction -> ready.
                    post_tool_tokens = estimate_tokens(messages)
                    if post_tool_tokens >= COMPACT_TRIGGER_TOKENS:
                        compactor.start_if_needed(messages)

            else:
                print("\n[Agent stopped after 30 tool rounds.] ")

    finally:
        compactor.shutdown()


if __name__ == "__main__":
    main()
