"""P0-P4: a small but real LLM agent, built from scratch.

P0: chat loop — raw HTTP to OpenRouter, conversation history in RAM,
    retry/backoff on rate limits.
P1: TOOL CALLING — the model can REQUEST that we run real Python
    functions, and we send the results back so it can answer better.
P2: THE AGENTIC LOOP — the model keeps calling tools across MULTIPLE
    rounds until it has enough information to answer ("plan -> act ->
    observe -> re-plan"), plus run_shell: a "computer command" tool that
    can run ANY PowerShell command (opt-in via ALLOW_SHELL=1).
P3: MCP (Model Context Protocol) — the agent becomes a CLIENT that can
    talk to EXTERNAL tool servers (separate processes) listed in mcp.json.
    We also BUILD two such servers in servers/. MCP tools join the same
    registry as builtins — the loop never cares where a tool came from.
P4: MEMORY — the agent stops forgetting:
    * session memory: transcripts saved to sessions/*.json (/save, /load,
      /resume, auto-save on quit) — survives restart WITHIN one story
    * long-term memory: memory.json + save_memory/search_memory TOOLS the
      model calls on its own; recent entries are injected into the system
      prompt so a fresh process "wakes up" knowing them — this is the part
      the model cannot know from its training data

THE BIG IDEA OF TOOL CALLING
============================
The model NEVER executes anything itself. It only emits JSON *requests*:

  1. We send:  messages + a list of tool schemas
               ("here are functions you may ask for")
  2. The model replies with EITHER:
       a. plain text                          → we just print it, done
       b. a message containing "tool_calls"   → it's REQUESTING tools, e.g.
            tool_calls: [{id: "call_1",
                          function: {name: "list_files",
                                     arguments: "{\"path\": \".\"}"}}]
  3. OUR code runs list_files() for real, then appends a new message:
       {"role": "tool", "tool_call_id": "call_1", "content": "agent.py\n..."}
  4. We call the LLM again (with everything). The model reads the raw result
     and decides its NEXT move: another tool? Or the final answer?

So: the model DECIDES (which tool, which arguments); Python EXECUTES.
That decision/execution split is the heart of every AI agent.

THE AGENTIC LOOP (P2's contribution)
====================================
P1 could do roughly "one tool call, then answer". A real task needs a CHAIN:

    you> "find where API_URL is defined and explain what protocol it uses"

    round 1: model -> search_files("API_URL")   <- result: agent.py:84
    round 2: model -> read_file("agent.py")     <- result: the source
    round 3: model -> plain text: "It uses https:// ..."

Each round's input depends on the previous round's OUTPUT — and nobody
hard-coded those three steps. The model chooses each move from what it just
saw. That is what "agentic" means: plan -> act -> observe -> re-plan,
repeated until done (or until MAX_ITERATIONS stops it).

SAFETY (why run_shell is gated)
===============================
run_shell executes WHATEVER the model types on this computer. So it is:
  - OFF unless you launch with:  $env:ALLOW_SHELL = "1"
  - jailed: cwd is always the project folder
  - time-limited: SHELL_TIMEOUT seconds, then killed
  - capped output: huge results are truncated before reaching the model
The other tools can only touch files inside the project folder.

WHAT IS A "TOOL SCHEMA"?
========================
A JSON description sent alongside the messages, telling the model:
  - the tool's name
  - a description (the model reads this to decide WHEN to use the tool!)
  - "parameters": a JSON Schema listing each argument's type and meaning
The format below is the OpenAI-compatible one that OpenRouter uses.

USAGE
=====
    $env:OPENROUTER_API_KEY = "sk-or-..."    # get one at https://openrouter.ai/keys
    python agent.py                          # safe: no shell access

    $env:ALLOW_SHELL = "1"                   # OPT-IN: enable the run_shell tool
    python agent.py                          # now the agent may run commands

Slash commands inside the REPL:
    /tools         list the tools the model may call (MCP tools are marked)
    /mcp           list connected MCP servers and their tools
    /model <id>    switch model, e.g. /model google/gemma-4-31b-it:free
    /history       show the conversation that is sent to the model
    /clear         forget the conversation (long-term memories are kept)
    /save <name>   save this transcript to sessions/<name>.json
    /load <name>   load a saved transcript (replaces the conversation)
    /resume        continue the most recently saved session
    /sessions      list saved transcripts
    /help          show this docstring
    /quit          exit (auto-saves the transcript first; Ctrl+C same)

P4 MEMORY — two layers:
    * sessions/*.json  — full transcripts (/save, /load, /resume, auto-save)
    * memory.json      — durable facts the model stores itself via the
      save_memory / search_memory tools; the most recent ones are injected
      into the system prompt at startup so a fresh process "wakes up"
      knowing them.

TIP: free models are auto-routed by "openrouter/free", and not every free
model supports tools. If you never see tool calls, pin a tool-capable one:
    /model google/gemma-4-31b-it:free
"""

import asyncio  # P3: the MCP SDK is async; we bridge it to our sync REPL
import contextlib  # P3: AsyncExitStack keeps each server connection open
import datetime
import json
import os
import random
import subprocess  # P2: run actual OS commands
import sys
import threading  # P3: background thread runs the asyncio loop
import time
from pathlib import Path  # Path = a much nicer way to handle file paths

import httpx  # a modern HTTP client library (like "requests", but faster)

# The MCP SDK is optional at import time: if it's not installed the agent
# still works with builtin tools — it just skips MCP servers (graceful
# degradation, the same policy we use for server crashes at connect time).
try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# ===========================================================================
# P0 — CONFIGURATION
# All tweakable from environment variables.
# ===========================================================================
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# The model to use. "openrouter/free" lets OpenRouter auto-pick any free model.
# Override before launching:  $env:AGENT_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "openrouter/free")

# The "system prompt" steers the model's personality/behavior. It is sent as
# the first message and never changes — it is the model's standing instruction.
# P4 NOTE: this is the BASE prompt. build_system_prompt() prepends long-term
# memories to it at startup (see the P4 section).
SYSTEM_PROMPT = """You are a helpful assistant working on the user's computer.

Report only what the latest tool calls and messages actually did. Never
describe actions you did not take this turn.

When tools are available, use them to look things up instead of guessing."""

REQUEST_TIMEOUT = 60.0  # give up on a request after this many seconds
MAX_RETRIES = 4  # how many times to retry on rate-limits (HTTP 429) or server errors

# --- P2: agentic loop settings ---
MAX_ITERATIONS = 10  # max chat->tool->chat rounds per user question (loop must terminate!)
TOOL_TIMEOUT = 30.0  # seconds allowed per run_shell call before it is killed

# --- P2: the dangerous tool is OPT-IN ---
# run_shell executes arbitrary commands, so it only exists if you explicitly
# enable it:  $env:ALLOW_SHELL = "1"
ALLOW_SHELL = os.environ.get("ALLOW_SHELL", "").strip().lower() in ("1", "true", "yes")


def get_api_key() -> str:
    """Read the API key from the environment, or exit with instructions.

    Environment variables keep secrets OUT of your source code.
    The key is read at runtime; it is never written into this file.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        sys.exit(  # sys.exit prints a message and terminates the program
            "Error: OPENROUTER_API_KEY is not set.\n"
            'PowerShell:  $env:OPENROUTER_API_KEY = "sk-or-..."\n'
            'cmd:         set OPENROUTER_API_KEY=sk-or-...\n'
            "Get a free key at https://openrouter.ai/keys"
        )
    return key


def chat(messages: list[dict], model: str, api_key: str, tools: list | None = None) -> dict:
    """Send a chat-completion request to OpenRouter and return the assistant message.

    Parameters
    ----------
    messages : the full conversation so far, e.g.
        [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user",      "content": "What's 2+2?"},
        ]
        Later turns may also contain assistant messages with "tool_calls"
        and messages with role "tool" — the model needs to see its own
        requests and their results to make sense of the conversation.
    model   : which model to use, e.g. "openrouter/free"
    api_key : your secret key (sent in the Authorization header)
    tools   : optional list of tool schemas (P1). When present, the model
              may reply with "tool_calls" instead of plain text.

    Returns
    -------
    The assistant's message dict. Check it for a "tool_calls" key!

    Why a retry loop? Free models return HTTP 429 ("too many requests") often.
    "Backoff" = wait longer after each failure (1s, 2s, 4s...) so we don't
    hammer the server.
    """
    # HTTP headers are extra key/value pairs sent with the request.
    # Authorization proves who we are; the rest are optional metadata.
    headers = {
        "Authorization": f"Bearer {api_key}",  # "Bearer <key>" is the standard auth format
        "Content-Type": "application/json",  # we are sending JSON in the body
        "HTTP-Referer": "http://localhost",  # optional: which site/app calls the API
        "X-Title": "firstagent",  # optional: app name shown on your OpenRouter dashboard
    }

    # The request body — this JSON is THE core of every LLM API call:
    # a model name + a list of messages (+ tools, when we offer any).
    payload = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools  # P1: advertise the tools the model may request

    for attempt in range(MAX_RETRIES):
        try:
            # The actual HTTP POST. httpx.post(url, json=...) sends JSON and
            # sets Content-Type automatically.
            resp = httpx.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except httpx.TimeoutException:
            # The server didn't answer in time. Transient — worth retrying.
            print(f"[network] request timed out after {REQUEST_TIMEOUT:.0f}s", file=sys.stderr)
            if attempt == MAX_RETRIES - 1:
                raise  # out of retries: re-raise the error to the caller
            time.sleep(2**attempt)  # sleep 1s, 2s, 4s...
            continue
        except httpx.HTTPError as exc:
            # Any other connection problem (DNS, refused, etc.)
            print(f"[network] {exc}", file=sys.stderr)
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)
            continue

        # ---- HTTP status codes tell us the outcome of the request ----
        if resp.status_code == 401:
            # 401 Unauthorized = bad or missing API key. Not worth retrying.
            sys.exit("Error: 401 Unauthorized — check OPENROUTER_API_KEY.")
        if resp.status_code == 429 or resp.status_code >= 500:
            # 429 = rate-limited (too many requests). 5xx = server-side bug.
            # Servers often send a "retry-after" header telling us how long to wait.
            wait = int(resp.headers.get("retry-after", 2**attempt))
            print(f"[rate-limit] HTTP {resp.status_code}, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            # Any other error (400 bad request, 402 out of credits, ...):
            # show the server's explanation so you can fix it.
            sys.exit(f"Error: HTTP {resp.status_code}\n{resp.text}")

        # ---- Success! Parse the JSON body. ----
        # The response looks like:
        # {
        #   "choices": [
        #       {"index": 0,
        #        "message": {"role": "assistant",
        #                     "content": "Hello!",              <- text (may be null!)
        #                     "tool_calls": [...]},             <- P1: may be present
        #        "finish_reason": "stop"}
        #   ],
        #   "usage": {"prompt_tokens": 12, "completion_tokens": 3, ...}
        # }
        data = resp.json()
        if "error" in data:  # some errors come back as 200 + an "error" field
            sys.exit(f"Error: {data['error']}")

        choice = (data.get("choices") or [{}])[0]  # first (usually only) choice
        message = choice.get("message")
        if not message:
            sys.exit(f"Error: unexpected response shape:\n{data}")

        # "Reasoning" models (e.g. DeepSeek-R1) may put their thinking in a
        # separate "reasoning" field with an empty "content". Show it anyway.
        if not message.get("content") and message.get("reasoning"):
            message["content"] = message["reasoning"]
        return message  # note: content may be None when the model calls tools

    sys.exit("Error: gave up after too many retries.")


# ===========================================================================
# P1 — THE TOOLS THEMSELVES
#
# Each tool is just a normal Python function whose:
#   - arguments come from the model's JSON (so they arrive as JSON types:
#     str, int, bool, list, dict — never fancy Python objects)
#   - return value gets turned into a STRING and handed back to the model
#
# The three tools below are our "built-in" tools. In P3, MCP servers will
# provide tools the same way — the loop won't care where a tool came from.
# ===========================================================================

# Our file tools are JAILed to the project folder: the model (or you) cannot
# read/write outside it with "..", absolute paths, etc. Security 101.
PROJECT_DIR = Path(__file__).resolve().parent  # folder containing this script
MAX_READ_CHARS = 10_000  # never send more than this to the model (save tokens)
MAX_SEARCH_RESULTS = 50
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules"}  # not worth searching


def safe_path(relative: str) -> Path:
    """Resolve a model-supplied path and refuse anything outside PROJECT_DIR.

    Raises ValueError (the caller turns it into an error string for the model).
    """
    # (PROJECT_DIR / relative) joins them; .resolve() collapses ".." etc.
    target = (PROJECT_DIR / relative).resolve()
    # is_relative_to: True if target is PROJECT_DIR or inside it.
    if not target.is_relative_to(PROJECT_DIR):
        raise ValueError(f"path escapes the project directory: {relative}")
    return target


def list_files(path: str = ".") -> str:
    """List files and folders inside the project directory."""
    target = safe_path(path)
    if not target.is_dir():
        return f"Error: not a directory: {path}"
    lines = []
    for entry in sorted(target.iterdir()):  # sorted = alphabetical, stable output
        if entry.is_dir():
            lines.append(f"  [dir]  {entry.name}/")
        else:
            lines.append(f"  [file] {entry.name} ({entry.stat().st_size:,} bytes)")
    if not lines:
        return f"{path} is empty."
    return f"{len(lines)} entries in '{path}':\n" + "\n".join(lines)


def read_file(path: str) -> str:
    """Read a text file from the project directory and return its content."""
    target = safe_path(path)
    if not target.is_file():
        return f"Error: file not found: {path}"
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Binary files (images, .exe, ...) aren't valid UTF-8 text.
        return f"Error: {path} looks like a binary file, not text."
    if len(text) > MAX_READ_CHARS:
        # Truncate: giant files would flood the model's context window.
        text = text[:MAX_READ_CHARS] + f"\n... [truncated at {MAX_READ_CHARS} characters]"
    return text


def get_current_time() -> str:
    """Return the computer's current local date and time."""
    # astimezone() attaches the local UTC offset automatically.
    now = datetime.datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")


def search_files(query: str, path: str = ".") -> str:
    """Search text files under 'path' for lines containing 'query' (case-insensitive).

    This is a mini grep — it lets the agent explore a codebase it's never
    seen: "where is the API URL defined?" -> search_files("API_URL").
    """
    root = safe_path(path)
    needle = query.lower()
    hits: list[str] = []
    for file in sorted(root.rglob("*")):  # rglob("*") = recurse into subfolders
        if len(hits) >= MAX_SEARCH_RESULTS:
            break
        if not file.is_file():
            continue
        # Skip junk folders anywhere in the relative path.
        rel = file.relative_to(PROJECT_DIR)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        try:
            if file.stat().st_size > 200_000:  # skip giant files
                continue
            text = file.read_text(encoding="utf-8")  # raises on binary files
        except (UnicodeDecodeError, OSError):
            continue  # not text / unreadable — just skip it
        for line_no, line in enumerate(text.splitlines(), start=1):
            if needle in line.lower():
                hits.append(f"{rel}:{line_no}: {line.strip()[:200]}")
                if len(hits) >= MAX_SEARCH_RESULTS:
                    break
    if not hits:
        return f"No lines containing '{query}' found under '{path}'."
    note = "" if len(hits) < MAX_SEARCH_RESULTS else f" (capped at {MAX_SEARCH_RESULTS})"
    return f"{len(hits)} match(es){note} for '{query}':\n" + "\n".join(hits)


def roll_dice(n: int = 1, sides: int = 6) -> str:
    """Roll n dice with the given number of sides and report the results."""
    # Validate! Model-generated arguments can be nonsense — never trust them.
    if not isinstance(n, int) or not 1 <= n <= 20:
        return "Error: n must be an integer between 1 and 20."
    if not isinstance(sides, int) or not 2 <= sides <= 1000:
        return "Error: sides must be an integer between 2 and 1000."
    rolls = [random.randint(1, sides) for _ in range(n)]
    return f"Rolled {n}d{sides}: {rolls} -> total {sum(rolls)}"


def run_shell(command: str) -> str:
    """Run an OS command and return its output. (P2 — the "hands" of the agent)

    This is what makes the agent able to DO things on the computer: run the
    project, check git status, ping a host... anything a terminal can do.

    DANGER: it executes whatever string the model produces. That's why it is
    triple-gated:
      1. it is only registered at all if ALLOW_SHELL=1 was set at launch
      2. it always runs with cwd=PROJECT_DIR (can't wander the filesystem)
      3. it is killed after TOOL_TIMEOUT seconds (no hanging processes)

    subprocess.run() = start a program, wait for it to finish, collect output.
    capture_output=True grabs stdout+stderr; text=True returns strings.
    """
    try:
        # Windows: PowerShell. (On macOS/Linux you'd use ["bash", "-c", command].)
        # -NoProfile skips your profile script (faster, more predictable).
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,  # grab stdout and stderr as strings
            text=True,
            timeout=TOOL_TIMEOUT,  # TimeoutExpired is raised if it takes too long
            cwd=PROJECT_DIR,  # JAIL: always run inside the project folder
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {TOOL_TIMEOUT:.0f}s: {command}"
    except OSError as exc:  # e.g. powershell not found
        return f"Error: could not run command: {exc}"

    # Show the model stdout, stderr AND the exit code — a failed command
    # (exit code != 0) usually explains itself in stderr.
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    parts = []
    if out:
        parts.append(out)
    if err:
        parts.append(f"[stderr]\n{err}")
    parts.append(f"[exit code {proc.returncode}]")
    result = "\n".join(parts)
    if len(result) > 8000:  # don't flood the model's context window
        result = result[:8000] + "\n... [output truncated]"
    return result


# ---------------------------------------------------------------------------
# Tool registry: name -> Python function.
# Registering a tool = adding one line here + one schema in TOOL_SCHEMAS.
# ---------------------------------------------------------------------------
TOOL_FUNCTIONS = {
    "list_files": list_files,
    "read_file": read_file,
    "get_current_time": get_current_time,
    "search_files": search_files,
    "roll_dice": roll_dice,
}

# TOOL_SCHEMAS is the JSON we SEND to the model so it knows which tools exist.
#
# Format (OpenAI-compatible, used by OpenRouter):
#   {
#     "type": "function",
#     "function": {
#       "name":        "list_files",              <- must match TOOL_FUNCTIONS
#       "description": "...",                     <- the model reads this to
#                                                    decide when to use the tool!
#       "parameters":  { JSON Schema of args }    <- "properties" = the args,
#                                                    "required" = which are needed
#     }
#   }
# Good descriptions matter: they are the model's only documentation.
#
# Step 1: the safe built-in tools.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files and folders inside a directory of the project. "
            "Use this to explore what files exist before reading or searching them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the project root. Default '.' (project root).",
                    }
                },
                "required": [],  # empty list = every argument is optional
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full text content of a file in the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project root, e.g. 'agent.py'.",
                    }
                },
                "required": ["path"],  # the model MUST provide this argument
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the computer's current local date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search all text files in the project for lines containing a word or phrase "
            "(case-insensitive grep). Use this to find where something is defined or mentioned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for."},
                    "path": {
                        "type": "string",
                        "description": "Directory to search under. Default '.' (project root).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": "Roll n dice with a given number of sides (for games or when the user asks).",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "How many dice (1-20). Default 1."},
                    "sides": {"type": "integer", "description": "Sides per die (2-1000). Default 6."},
                },
                "required": [],
            },
        },
    },
]

# Step 2 (P2): the DANGEROUS tool is only registered when explicitly enabled.
# This is the "opt-in" pattern: dangerous capability exists in the code, but
# is unreachable unless you launch with $env:ALLOW_SHELL = "1".
if ALLOW_SHELL:
    TOOL_FUNCTIONS["run_shell"] = run_shell
    TOOL_SCHEMAS.append(
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Run a PowerShell command on the computer and get its output "
                "(stdout, stderr and exit code). Use for anything a terminal can do: "
                "running scripts, checking installed tools, git, network checks, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The PowerShell command to run, e.g. 'python -c \"print(1+1)\"'.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    )

# Handy lookup: schema name -> schema dict (used for error messages).
# NOTE: built AFTER the conditional block so it sees run_shell too.
SCHEMAS_BY_NAME = {s["function"]["name"]: s for s in TOOL_SCHEMAS}


def run_tool(name: str, args_json: str) -> str:
    """Execute one tool call from the model. ALWAYS returns a string.

    The model's arguments arrive as a JSON *string* like '{"path": "."}'.
    The contract with the LLM is simple: results are text, errors are text
    too — an error string lets the model SEE what went wrong and self-correct
    ("that file doesn't exist — let me list the directory first").
    """
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:  # not a builtin — maybe it lives in an MCP server? (P3)
        if name in MCP_ROUTES and MCP is not None:
            try:
                args = json.loads(args_json) if args_json.strip() else {}
            except json.JSONDecodeError as exc:
                return f"Error: arguments were not valid JSON ({exc})."
            if not isinstance(args, dict):
                return 'Error: arguments must be a JSON object like {"key": "value"}.'
            try:
                return MCP.call_tool(MCP_ROUTES[name], name, args)
            except Exception as exc:  # transport crash (server died, timeout)
                return f"Error: MCP call to '{name}' failed: {type(exc).__name__}: {exc}"
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOL_FUNCTIONS)}."

    try:
        args = json.loads(args_json) if args_json.strip() else {}
    except json.JSONDecodeError as exc:
        return f"Error: arguments were not valid JSON ({exc})."
    if not isinstance(args, dict):
        return "Error: arguments must be a JSON object like {\"key\": \"value\"}."

    try:
        return str(fn(**args))  # **args unpacks the dict into keyword arguments
    except TypeError:
        # Usually means the model passed wrong/missing argument NAMES.
        schema = json.dumps(SCHEMAS_BY_NAME[name]["function"]["parameters"])
        return f"Error: bad arguments for '{name}'. Expected schema: {schema}."
    except Exception as exc:  # any crash inside the tool becomes feedback
        return f"Error: {type(exc).__name__}: {exc}"


def short(text: str, limit: int = 400) -> str:
    """Truncate long text for console display (the model still gets the full result)."""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars]"


# ===========================================================================
# P3 — MCP CLIENT: talk to external tool servers listed in mcp.json
#
# MCP (Model Context Protocol) is an open standard: any app can expose tools
# as a server process, and any agent can use them as a client. Transport here
# is "stdio": WE spawn the server as a subprocess and exchange JSON-RPC
# messages over its stdin/stdout.
#
#   agent.py (CLIENT)  ──spawn + JSON-RPC over stdio──▶  servers/calc_server.py
#
# THE SYNC/ASYNC BRIDGE (important!):
# The mcp SDK is fully async (asyncio). Our REPL is plain sync code.
# Solution: run ONE asyncio event loop in a background daemon thread, and
# ship coroutines to it with asyncio.run_coroutine_threadsafe(...), which
# returns a concurrent.futures.Future we can block on with .result().
# That is the standard way to call async libraries from sync code.
# ===========================================================================
MCP_CONFIG_FILE = PROJECT_DIR / "mcp.json"  # which servers to launch
MCP_CONNECT_TIMEOUT = 20.0  # seconds allowed for a server to start up

# tool name -> server name, e.g. {"add": "calc", "add_note": "notes"}.
# Filled at startup; lets run_tool() know a tool lives in MCP-land.
MCP_ROUTES: dict[str, str] = {}


class McpBridge:
    """Manages all MCP server connections behind a background asyncio loop.

    Lifecycle:  start() -> connect_all() -> [call_tool() many times] -> close()
    One background thread runs the event loop for the lifetime of the app.
    Each server = one AsyncExitStack (keeps its subprocess + session OPEN)
    + one ClientSession (the JSON-RPC conversation).
    """

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()  # a private event loop...
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="mcp-loop")
        self.stacks: dict[str, contextlib.AsyncExitStack] = {}  # server -> open contexts
        self.sessions: dict[str, ClientSession] = {}  # server -> session

    # ---- sync helpers ----------------------------------------------------
    def _run_loop(self) -> None:
        """Thread target: run the event loop forever (daemon=True = dies with the app)."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _submit(self, coro, timeout: float):
        """Ship a coroutine to the loop thread and BLOCK until it finishes."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    # ---- async work (runs ON the loop thread) ----------------------------
    async def _connect_one(self, command: str, args: list[str]) -> tuple:
        """Spawn one server subprocess and open a ClientSession to it.

        AsyncExitStack is the clean way to hold open context managers:
        both stdio_client() (owns the subprocess) and ClientSession() (owns
        the JSON-RPC handshake) stay open until we close the stack later.
        """
        stack = contextlib.AsyncExitStack()
        params = StdioServerParameters(command=command, args=args)
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()  # the MCP handshake: version + capabilities
        return stack, session

    # ---- public API (sync, safe to call from the REPL thread) ------------
    def start(self) -> None:
        self.thread.start()

    def connect_all(self) -> None:
        """Read mcp.json, launch every server, register its tools.

        GRACEFUL DEGRADATION: a server that crashes or times out is skipped
        with a warning — the agent keeps running with the other tools.
        """
        if not MCP_CONFIG_FILE.exists():
            return
        config = json.loads(MCP_CONFIG_FILE.read_text(encoding="utf-8"))
        for spec in config.get("servers", []):
            name, command, args = spec["name"], spec["command"], spec.get("args", [])
            try:
                stack, session = self._submit(self._connect_one(command, args), MCP_CONNECT_TIMEOUT)
            except Exception as exc:
                print(f"[mcp] server '{name}' failed to start: {exc}", file=sys.stderr)
                continue
            self.stacks[name] = stack
            self.sessions[name] = session
            self._register_tools(name, session)
            print(f"[mcp] connected: {name} ({len([t for t in MCP_ROUTES if MCP_ROUTES[t] == name])} tools)")

    def _register_tools(self, server: str, session: ClientSession) -> None:
        """Ask a server 'what tools do you have?' and add them to our registry.

        The beauty of MCP: the server's input_schema IS already a JSON Schema,
        so converting it to the OpenAI-compatible format our chat() sends is
        just re-wrapping it. No per-server glue code — that's the standard.
        """
        listed = self._submit(session.list_tools(), MCP_CONNECT_TIMEOUT)
        for tool in listed.tools:
            if tool.name in TOOL_FUNCTIONS or tool.name in MCP_ROUTES:
                print(f"[mcp] tool name collision, skipping: {tool.name}", file=sys.stderr)
                continue  # never shadow a builtin
            MCP_ROUTES[tool.name] = server
            TOOL_SCHEMAS.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or f"MCP tool from server '{server}'.",
                        "parameters": tool.input_schema,  # already JSON Schema!
                    },
                }
            )
            SCHEMAS_BY_NAME[tool.name] = TOOL_SCHEMAS[-1]

    def call_tool(self, server: str, name: str, args: dict) -> str:
        """Call one MCP tool and return its result as a plain string.

        MCP results are a LIST of content blocks ([TextContent(text='5'), ...])
        — we flatten them to text, because our whole agent (loop, history,
        model) speaks plain strings.
        """

        async def _call():
            return await self.sessions[server].call_tool(name, args)

        result = self._submit(_call(), TOOL_TIMEOUT)
        blocks = []
        for item in result.content or []:
            text = getattr(item, "text", None)
            blocks.append(text if text is not None else f"[{type(item).__name__}]")
        output = "\n".join(blocks) or "(no output)"
        # is_error=True means the SERVER reported a tool failure (vs. a
        # transport crash, which raises in _submit instead).
        return f"Error (reported by MCP server '{server}'): {output}" if result.is_error else output

    def close(self) -> None:
        """Shut down every server cleanly (kills the subprocesses)."""

        async def _close_all():
            for stack in self.stacks.values():
                await stack.aclose()  # exits stdio_client -> terminates server

        if self.sessions:
            try:
                self._submit(_close_all(), 5.0)
            except Exception:
                pass  # best-effort on shutdown
        self.loop.call_soon_threadsafe(self.loop.stop)  # stop the loop thread
        self.thread.join(timeout=2.0)


# The one bridge instance, shared by main() and run_tool(). None until started.
MCP: McpBridge | None = None


# ===========================================================================
# P4 — MEMORY
#
# Two layers, answering two different questions:
#
# 1) SESSION MEMORY: "what were we just talking about?"
#    The whole `messages` list is serialized to sessions/*.json. On restart,
#    /resume (or /load) puts it back and the model continues mid-story.
#    This is dumb persistence — no intelligence, just "don't lose the tape".
#
# 2) LONG-TERM MEMORY: "what should the agent know forever?"
#    memory.json holds facts the model CHOOSES to store via the save_memory
#    tool. Recent entries are injected into the system prompt at startup, so
#    a brand-new process "wakes up" knowing them. This is the P0 statelessness
#    problem finally solved: the API still remembers nothing, so WE carry the
#    important parts forward, on disk, in every request.
#
# KEY DESIGN POINT: memory is a TOOL, not magic. The model decides when to
# save ("user prefers metric units") and when to look things up
# (search_memory). We never silently extract facts — the model does, through
# the same tool-calling machinery as everything else.
# ===========================================================================
MEMORY_FILE = PROJECT_DIR / "memory.json"  # long-term facts live here
SESSIONS_DIR = PROJECT_DIR / "sessions"  # transcripts live here
MAX_PROMPT_MEMORIES = 10  # how many recent memories to inject into the prompt


def _load_memories() -> list[dict]:
    """Read memory.json -> list of {id, text, tags, created}. Missing file = []."""
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []  # corrupt file = start empty, don't crash the agent


def _save_memories(memories: list[dict]) -> None:
    MEMORY_FILE.write_text(json.dumps(memories, indent=2, ensure_ascii=False), encoding="utf-8")


def build_system_prompt() -> str:
    """BASE prompt + recent long-term memories.

    Called once at REPL start (and after /clear) — this is where the agent
    "wakes up" knowing what it stored in a previous session. Only the most
    recent MAX_PROMPT_MEMORIES entries are injected to keep the prompt small;
    older ones are still findable via the search_memory tool.
    """
    memories = _load_memories()
    if not memories:
        return SYSTEM_PROMPT
    lines = [SYSTEM_PROMPT, "", "Long-term memories (facts you stored in earlier sessions):"]
    for m in memories[-MAX_PROMPT_MEMORIES:]:
        tag_str = f" [{', '.join(m['tags'])}]" if m.get("tags") else ""
        lines.append(f"- ({m['created']}){tag_str} {m['text']}")
    return "\n".join(lines)


def save_memory(text: str, tags: str = "") -> str:
    """Tool: store a durable fact for future sessions."""
    if not text.strip():
        return "Error: memory text is empty."
    memories = _load_memories()
    memory_id = (max((m["id"] for m in memories), default=0)) + 1
    memories.append(
        {
            "id": memory_id,
            "text": text.strip(),
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    _save_memories(memories)
    return f"Saved memory #{memory_id}. It will be in your prompt from the next session on."


def search_memory(query: str) -> str:
    """Tool: search long-term memories by keyword (case-insensitive)."""
    memories = _load_memories()
    q = query.lower()
    hits = [m for m in memories if q in m["text"].lower() or q in " ".join(m.get("tags", [])).lower()]
    if not hits:
        return f"No memories matching '{query}'."
    lines = []
    for m in hits:
        tag_str = f" [{', '.join(m['tags'])}]" if m.get("tags") else ""
        lines.append(f"#{m['id']} ({m['created']}){tag_str} {m['text']}")
    return "\n".join(lines)


# Register the memory tools like any other builtin — same registry, same rules.
TOOL_FUNCTIONS["save_memory"] = save_memory
TOOL_FUNCTIONS["search_memory"] = search_memory
TOOL_SCHEMAS.extend(
    [
        {
            "type": "function",
            "function": {
                "name": "save_memory",
                "description": "Store a durable fact about the user, project, or preferences "
                "for future sessions (survives restarts). Use for things worth remembering: "
                "decisions, preferences, project state. Not for large data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The fact to remember, one clear sentence."},
                        "tags": {"type": "string", "description": "Optional comma-separated keywords, e.g. 'user,preference'."},
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "Search your long-term memories by keyword. Use when you need "
                "something you might have stored before (the prompt only shows the 10 most recent).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Keyword to search for."},
                    },
                    "required": ["query"],
                },
            },
        },
    ]
)
SCHEMAS_BY_NAME["save_memory"] = TOOL_SCHEMAS[-2]
SCHEMAS_BY_NAME["search_memory"] = TOOL_SCHEMAS[-1]


# ---- session persistence (transcripts) ------------------------------------
def save_session(messages: list[dict], name: str) -> Path:
    """Serialize the whole conversation to sessions/<name>.json.

    NOTE: messages contain only JSON-friendly values (str, list, dict) —
    that's why we could keep tool_calls in history in P1/P2. json.dumps
    just works, tool messages included.
    """
    SESSIONS_DIR.mkdir(exist_ok=True)  # create the folder if it doesn't exist
    path = SESSIONS_DIR / f"{name}.json"
    path.write_text(json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_session(name: str) -> list[dict] | None:
    """Read sessions/<name>.json back into a messages list. None if missing."""
    path = SESSIONS_DIR / f"{name}.json"
    if not path.exists():
        return None
    messages = json.loads(path.read_text(encoding="utf-8"))
    # Sanity check: it must be a list of {role, content} dicts.
    if not isinstance(messages, list) or not all(isinstance(m, dict) and "role" in m for m in messages):
        raise ValueError(f"session file {path.name} is not a valid transcript")
    return messages


def list_sessions() -> list[str]:
    if not SESSIONS_DIR.exists():
        return []
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.json"))


def sanitize_session_name(name: str) -> str:
    """Prevent path traversal: session names become filenames.

    Reject (don't silently fix!) anything containing characters outside
    [A-Za-z0-9-_] — a name like '../evil' must be an ERROR the user sees,
    not quietly renamed to 'evil'. Silently 'fixing' input is how security
    bugs hide; failing loudly is how they get fixed.
    """
    name = name.strip()
    if not name or not all(c.isalnum() or c in "-_" for c in name):
        raise ValueError("session name may only contain letters, digits, - and _")
    return name


def do_tool_rounds(messages: list[dict], model: str, api_key: str) -> None:
    """The AGENTIC LOOP: ask the model, run whatever tools it requests, repeat.

    P2 upgrade over P1's one-shot mini-loop. The cycle is:

        chat() -> has tool_calls?
                    no  -> print final answer, done
                    yes -> append the assistant message (WITH its tool_calls),
                           run each tool, append one role:"tool" message per
                           result, then loop back and chat() again.

    Because each round's tools depend on the previous round's results, the
    model can CHAIN steps it never planned in advance:

        you> "find where API_URL is defined and explain what protocol it uses"
        round 1: search_files("API_URL")  <- agent.py:84
        round 2: read_file("agent.py")    <- the source line
        round 3: plain text: "It uses https:// ..."

    Nothing here hard-codes those steps — the model picks each move from the
    result it just saw. plan -> act -> observe -> re-plan: that's "agentic".

    THREE HISTORY RULES that matter (study these!):
      1. When the model calls tools, its message has content=None — we must
         still append it, INCLUDING the raw "tool_calls" list, or the model
         won't know what it asked for.
      2. Every tool_call id gets exactly ONE {"role": "tool"} reply, in order,
         carrying the same "tool_call_id". Getting this wrong = API error.
      3. Tool ERRORS are sent back as ordinary strings — the model reads them
         and self-corrects ("wrong directory" -> list_files -> try again).
         Never crash the loop because one tool failed.
    """
    for round_no in range(1, MAX_ITERATIONS + 1):
        reply = chat(messages, model, api_key, tools=TOOL_SCHEMAS)
        content = (reply.get("content") or "").strip()
        tool_calls = reply.get("tool_calls") or []  # [] when it's plain text

        if not tool_calls:
            # Final answer — no tool requested. The loop ends by itself.
            print(f"agent> {content}")
            return

        # Keep the assistant turn in history, tool_calls and all.
        messages.append(
            {"role": "assistant", "content": reply.get("content"), "tool_calls": tool_calls}
        )
        if content:  # models sometimes write a little text alongside the calls
            print(f"agent> {content}")

        # Execute each requested tool and report the result back.
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "?")
            args_json = fn.get("arguments", "{}")
            print(f"  [{round_no}] -> {name}({short(args_json, 120)})")  # the request
            started = time.monotonic()
            try:
                result = run_tool(name, args_json)
            except KeyboardInterrupt:
                # Ctrl+C mid-tool: keep history VALID (rule 2) by still
                # answering the pending tool_call, then abandon the turn.
                result = "[interrupted by user]"
                print(f"  [{round_no}] <- {result}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": result,
                    }
                )
                print("[turn interrupted — press Enter to continue]")
                return
            elapsed = time.monotonic() - started
            # Show what really happened, with how long it took.
            print(f"  [{round_no}] <- {short(result)}  ({elapsed:.1f}s)")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),  # correlates with the request
                    "name": name,
                    "content": result,
                }
            )

    # Safety valve: the loop must always terminate, even if the model keeps
    # asking for tools forever (free models do this sometimes).
    print(f"agent> [stopped: hit MAX_ITERATIONS={MAX_ITERATIONS} without a final answer]")


def print_history(messages: list[dict]) -> None:
    """Pretty-print every message in the conversation (for the /history command)."""
    for msg in messages:
        role = msg["role"].upper()
        print(f"--- {role} ---")
        if msg.get("tool_calls"):
            # Tool-request turns have empty content but a tool_calls list.
            for call in msg["tool_calls"]:
                fn = call.get("function", {})
                print(f"[requested tool: {fn.get('name')}({short(fn.get('arguments', ''), 120)})]")
        if msg.get("content"):
            print(msg["content"])
        print()


def _quit(messages: list[dict]) -> None:
    """Exit path for /quit and Ctrl+C: auto-save the transcript first.

    Auto-saves go to sessions/autosave.json so /resume can pick them up
    (but /resume prefers explicitly named saves over autosave).
    Only bother if there's an actual conversation (more than the system msg).
    """
    if len(messages) > 1:
        try:
            save_session(messages, "autosave")
            print(f"[auto-saved {len(messages)} messages — /resume to continue later]")
        except OSError as exc:
            print(f"[warning: could not auto-save session: {exc}]")
    print("bye")


def repl(api_key: str) -> None:
    """The Read-Eval-Print Loop: read input, call the model, print the reply, repeat.

    "REPL" (Read-Eval-Print Loop) is the standard name for this kind of
    interactive loop — it's what makes a one-shot API call feel like a
    conversation.
    """
    model = DEFAULT_MODEL

    # The conversation history. Starts with just the system prompt —
    # P4: build_system_prompt() = base prompt + recent long-term memories,
    # so a fresh process "wakes up" knowing what previous sessions stored.
    messages: list[dict] = [{"role": "system", "content": build_system_prompt()}]

    print(f"Agent P4 — model: {model}  (type /help for commands)")
    if ALLOW_SHELL:
        print("run_shell: ENABLED — the agent CAN run commands on this computer.")
    else:
        print("run_shell: off (launch with $env:ALLOW_SHELL='1' to enable).")
    if MCP and MCP.sessions:
        print(f"MCP servers: {', '.join(MCP.sessions)}")
    n_memories = len(_load_memories())
    if n_memories:
        print(f"long-term memories loaded: {n_memories} (most recent are in the system prompt)")
    print()
    while True:
        # input() blocks until you press Enter. It raises EOFError on Ctrl+D
        # and KeyboardInterrupt on Ctrl+C — we treat both as "quit".
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            _quit(messages)  # P4: auto-save the transcript before exiting
            return
        if not user_input:  # ignore empty lines
            continue

        # ---- Slash commands: local actions that never reach the API ----
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)  # e.g. ["/model", "google/..."]
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                _quit(messages)  # P4: auto-save the transcript before exiting
                return
            elif cmd == "/help":
                print(__doc__)
            elif cmd == "/clear":
                # Forget the conversation — but keep long-term memories:
                # build_system_prompt() re-includes them, so /clear starts a
                # fresh conversation ON TOP of what the agent knows forever.
                messages = [{"role": "system", "content": build_system_prompt()}]
                print("[history cleared — long-term memories kept]")
            elif cmd == "/history":
                print_history(messages)
            elif cmd in ("/save", "/saveas"):
                # P4: persist the transcript to sessions/<name>.json
                try:
                    name = sanitize_session_name(arg) if arg else "autosave"
                    path = save_session(messages, name)
                    print(f"[saved {len(messages)} messages to {path.relative_to(PROJECT_DIR)}]")
                except ValueError as exc:
                    print(f"[error: {exc}]")
            elif cmd == "/load":
                # P4: restore a transcript (replaces current conversation)
                try:
                    name = sanitize_session_name(arg)
                except ValueError as exc:
                    print(f"[error: {exc}]")
                    continue
                loaded = load_session(name)
                if loaded is None:
                    print(f"[no session named '{name}' — available: {', '.join(list_sessions()) or 'none'}]")
                else:
                    messages = loaded
                    print(f"[loaded {len(messages)} messages from session '{name}']")
            elif cmd == "/resume":
                # P4: continue the most recently SAVED session (not auto-saves)
                saved = [s for s in list_sessions() if s != "autosave"]
                if not saved:
                    print("[no saved sessions — use /save <name> first]")
                else:
                    latest = max(saved, key=lambda s: (SESSIONS_DIR / f"{s}.json").stat().st_mtime)
                    messages = load_session(latest) or [{"role": "system", "content": build_system_prompt()}]
                    print(f"[resumed session '{latest}' — {len(messages)} messages]")
            elif cmd == "/sessions":
                print(f"sessions in {SESSIONS_DIR.name}/: {', '.join(list_sessions()) or 'none yet'}")
            elif cmd == "/tools":
                # Show the model's toolbox: name + the description IT reads.
                # P3: mark which tools come from MCP servers.
                print(f"{len(TOOL_SCHEMAS)} tools available:")
                for schema in TOOL_SCHEMAS:
                    fn = schema["function"]
                    required = fn["parameters"].get("required", [])
                    params = ", ".join(fn["parameters"]["properties"]) or "-"
                    req = f" [required: {', '.join(required)}]" if required else ""
                    origin = f" (mcp:{MCP_ROUTES[fn['name']]})" if fn["name"] in MCP_ROUTES else ""
                    print(f"  {fn['name']}({params}){req}{origin}")
                    print(f"      {fn['description']}")
            elif cmd == "/mcp":
                # P3: show which MCP servers are connected and what they offer.
                if not MCP or not MCP.sessions:
                    print("no MCP servers connected (need mcp.json + 'pip install mcp')")
                else:
                    for server, session in MCP.sessions.items():
                        tools = [n for n, s in MCP_ROUTES.items() if s == server]
                        print(f"  {server}: {', '.join(tools) or '(no tools)'}")
            elif cmd == "/model":
                if arg:
                    model = arg  # switch model on the fly
                    print(f"[model set to {model}]")
                else:
                    print(f"current model: {model}")
            else:
                print(f"unknown command: {cmd}  (try /help)")
            continue

        # ---- A normal turn ----
        # 1. Remember what the user said (so future calls include it)...
        messages.append({"role": "user", "content": user_input})
        # 2. Ask the model — and run any tools it requests (the P1/P2 loop).
        try:
            do_tool_rounds(messages, model, api_key)
        except Exception as exc:  # network errors after all retries
            print(f"[failed: {exc}]")
            messages.pop()  # drop the unanswered turn so history stays consistent


def main() -> None:
    """Program entry point: grab the key, start MCP servers, then the REPL."""
    global MCP
    api_key = get_api_key()

    # --- P3: connect the MCP servers listed in mcp.json ---
    if MCP_CONFIG_FILE.exists():
        if MCP_AVAILABLE:
            MCP = McpBridge()
            MCP.start()
            MCP.connect_all()  # failures are warnings, not crashes
        else:
            print("[mcp] mcp.json found but the 'mcp' package is not installed — skipping.")
            print("      install it with:  pip install mcp")

    try:
        repl(api_key)
    finally:
        # ALWAYS shut down MCP servers, even on Ctrl+C — otherwise the
        # subprocesses could linger. (They're daemons, so this is a
        # belt-and-suspenders measure.)
        if MCP is not None:
            MCP.close()


# This idiom means: "only run main() if this file is executed directly
# (python agent.py), not if it's imported by another file (import agent)."
if __name__ == "__main__":
    main()
