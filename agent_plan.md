# agent_plan.md — Building a simple but nearly-complete LLM agent from scratch

**Objective:** Code a simple, but nearly complete, LLM agent from scratch — raw HTTP chat,
tool / computer-command calling, MCP build & calling, and persistent memory.
No agent frameworks (no LangChain, no OpenAI SDK): everything hand-rolled on top of
`httpx` + the OpenRouter API so every moving part is visible and understandable.

## Ground rules / decisions

- **Language:** Python 3.14 (shared venv at `../venv`), deps: `httpx` (P0+), `mcp` (P3+).
- **API:** OpenRouter (`https://openrouter.ai/api/v1/chat/completions`), key from
  `OPENROUTER_API_KEY` env var — never hard-coded.
- **Default model:** `openrouter/free` (auto-routes to a free model), overridable via
  `AGENT_MODEL` env var or the `/model` REPL command.
- **Interface:** one CLI REPL (`python agent.py`) that grows capability with each phase.
- **Code style:** heavily commented for learners; single `agent.py` for as long as it
  stays readable (may split into a package if P3+ gets crowded).
- **Rule:** each phase must be **tested and confirmed by the user** before the next begins.

## Architecture

### Target shape (after all phases)

```
┌─────────────────────────────────────────────────────────────────┐
│  agent.py — CLI REPL                                            │
│                                                                 │
│  ┌──────────┐   user input    ┌──────────────┐    /model /clear │
│  │  stdin    │ ──────────────▶ │   repl()     │ ───────────────▶ │
│  └──────────┘                  │  (commands,  │    /history …    │
│                                │   history)   │                  │
│                                └──────┬───────┘                  │
│                                       │ messages[]              │
│                                       ▼                         │
│                                ┌──────────────┐                 │
│                                │   loop()     │ ◀── P2: iterate │
│                                │  agent loop  │     until no    │
│                                └──┬────────┬──┘     tool_calls  │
│                 chat reqs         │        │ tool_calls         │
│                                   ▼        ▼                    │
│                            ┌─────────┐  ┌───────────────┐       │
│                            │  llm()  │  │  dispatch()   │       │
│                            │ (httpx) │  │  tool router  │       │
│                            └────┬────┘  └──┬───────┬────┘       │
│                                 │          │       │            │
└─────────────────────────────────┼──────────┼───────┼────────────┘
                                  │          │       │
        OpenRouter API            │   builtin tools  │ MCP (stdio, P3)
   /chat/completions              │   (P1: list/read/    │
   ◀─────────────────────────────▶│    search/time/dice; │──▶ servers/calc_server.py
   model: openrouter/free         │    P2: shell)        │──▶ servers/notes_server.py
                                  │                  │──▶ any 3rd-party MCP server
                                  │           ┌──────┴──────┐
                                  │           │ memory.py   │ (P4)
                                  │           │ sessions/   │  save/load transcripts
                                  │           │ memory.json │  long-term facts
                                  │           └─────────────┘
```

### Component responsibilities (final)

| Component | Phase | Responsibility |
|---|---|---|
| `repl()` | P0 | I/O loop; parses slash commands; owns the `messages[]` history |
| `llm()` | P0 | One HTTP call to OpenRouter; retry/backoff; normalizes the response |
| tool registry + `dispatch()` | P1 | Maps tool name → (JSON schema, Python fn); executes and formats results |
| `loop()` | P2 | The "agentic" part: repeat tool calls until the model answers; iteration cap, error feedback |
| `mcp_client.py` logic | P3 | Spawns MCP servers from `mcp.json`; adapts their tools to the same registry format |
| `memory.py` logic | P4 | `sessions/*.json` transcripts; `memory.json` long-term store exposed as tools |

### Data flow of one user turn (after P2)

```
you> is 101 prime?
 1. messages.append({role: user})
 2. loop():
    a. llm(messages)            → reply with tool_calls: [is_prime(101)]
    b. dispatch(is_prime, 101)  → True          (or exception → error string)
    c. messages.append(assistant w/ tool_calls) + messages.append(role: tool)
    d. repeat from (a)
    e. llm(messages)            → plain text: "Yes, 101 is prime."
 3. print reply; messages.append(assistant)
```

### State

- **RAM:** `messages[]` — the only "memory" in P0–P3; resent to the API every call
  (the API is stateless).
- **Disk (from P3):** `mcp.json` (server config), `servers/notes.json` (notes data).
- **Disk (P4):** `sessions/*.json` (transcripts), `memory.json` (long-term facts).

### Key design decisions

1. **No framework, no SDK** — raw `httpx` so the wire format (request body, tool-call
   JSON, `role:"tool"` messages) is fully visible. This is the point of the exercise.
2. **One file while it fits** — a single `agent.py` keeps the whole agent readable
   top-to-bottom for a learner; split into a package only if P3+ gets crowded.
3. **One tool interface for everything** — builtin (P1), shell (P2), and MCP (P3) tools
   all normalize into the same `{name, schema, fn}` registry, so the loop never cares
   where a tool comes from. This mirrors how real agents abstract MCP.
4. **Dangerous tools are opt-in** — `run_shell` requires `ALLOW_SHELL=1`; memory writes
   go through the model choosing to call a tool, never silently.
5. **Statelessness is the lesson** — history is deliberately in-RAM until P4, to make
   the "you must resend everything" model explicit before adding persistence.

### Security boundaries

- API key: env var only, never in source or logs.
- Shell tool: env-gated, timeout-limited, cwd jailed to the project dir.
- File tools (notes server, memory): confined to their own data files; no path traversal.
- `calculator`: `ast`-based whitelist evaluator — no `eval`/`exec`.

---

## Phase status

| Phase | Topic | Status |
|-------|-------|--------|
| P0 | Calling LLMs (chat, history, retry) | ✅ done — built & user-tested |
| P1 | Tool calling (5 builtin tools) | ✅ done — built & user-tested |
| P2 | Agentic loop + run_shell | ✅ built — offline tests passed; live test assumed OK (user moved on) |
| P3 | MCP: build servers + call them | ✅ done — built, live-tested (scenario 1 OK) |
| P4 | Persistent memory | ✅ built, awaiting user test |
| P5 | (optional) Full harness build | ⬜ outline only |

---

## P0 — Calling LLMs ✅ built (awaiting user test)

Raw HTTP chat with OpenRouter. The model "remembers" only because we resend the
whole `messages` list on every call (the API is stateless).

- `agent.py`: `httpx` POST to `/chat/completions`, `messages` list as state.
- System prompt + in-memory conversation history.
- Error handling: missing key (fail fast), 401, 429/5xx with exponential backoff,
  timeouts, empty-response fallback, reasoning-field fallback.
- Slash commands: `/model`, `/history`, `/clear`, `/help`, `/quit`.
- History lives in RAM only — no persistence (that's P4).

**Test:** set key → `python agent.py` → basic question, follow-up (checks history),
`/history`, `/model <id>`, `/clear`, `/quit`.
**Result:** _pending user test_

---

## P1 — Tool calling ✅ built (awaiting user test)

Teach the model to request actions; we execute them and feed results back.
The model never executes anything — it only emits JSON `tool_calls` requests;
Python executes and returns `role:"tool"` strings (errors as strings too, so
the model can self-correct).

- Tool registry: `TOOL_FUNCTIONS` (name → Python fn) + `TOOL_SCHEMAS`
  (OpenAI-compatible JSON Schema list — the model reads the descriptions!).
- 5 builtin tools:
  - `list_files(path?)` — list project dir (optional param)
  - `read_file(path)` — text file, 10 KB cap, binary → friendly error (required param)
  - `get_current_time()` — local time (zero params)
  - `search_files(query, path?)` — mini-grep across project (required + optional)
  - `roll_dice(n?, sides?)` — typed/validated numeric params
- `run_tool()`: JSON args → fn call → always a string; wrong args, missing file,
  unknown tool all return helpful error text.
- Safety: file tools jailed to `PROJECT_DIR` via `safe_path()` (no `..` escapes).
- `do_tool_rounds()`: mini agent loop (P2 preview) — chat → tool_calls? → execute
  → append (assistant w/ raw tool_calls + role:"tool" per call) → chat again;
  capped at `MAX_TOOL_ROUNDS = 5`.
- Slash command `/tools` shows the toolbox.

**Test:** "what files are in this directory?" / "read agent.py, one-line summary" /
"what time is it?" / "where is API_URL defined?" (search) / "roll 2d20" /
"read ../../x.txt" (jail refusal).
**Result:** _pending user test (offline unit test of all 5 tools + error paths passed)_

---

## P2 — Agentic loop ✅ built (awaiting user test)

P1's one-shot mini-loop generalized into the engine: chat → execute tools →
chat again, up to `MAX_ITERATIONS = 10`. Each round's inputs depend on the
previous round's outputs, so the model can **chain steps it never planned in
advance** (plan → act → observe → re-plan). Nothing hard-codes the chain —
the model picks each move from the result it just saw.

- `run_shell(command)` — the "computer command" tool: PowerShell via
  `subprocess.run`, stdout+stderr+exit code returned to the model.
  - **Opt-in**: registered only when launched with `ALLOW_SHELL=1`.
  - Jailed: `cwd=PROJECT_DIR` always.
  - Timeout: `TOOL_TIMEOUT = 30s` then killed.
  - Output capped at 8 KB.
- Loop hardening: per-tool timing display `[1] -> run_shell(...)  (0.4s)`,
  Ctrl+C mid-tool keeps history valid (`[interrupted by user]` tool result),
  `MAX_ITERATIONS` safety valve, exceptions in tools already → error strings.
- Startup banner shows `run_shell: ENABLED/off`.

**Test:** see 2 scenarios below.
**Result:** _pending user test (offline: shell output/timeout/exit-code/opt-in gate verified)_

### Test scenario 1 — multi-step chain, no shell (safe mode)

```powershell
python agent.py          # no ALLOW_SHELL
```
```
you> figure out what this project is about and how far along it is — look at the
     files, read the plan, and tell me which phase we're in
```
**Expect:** 3+ numbered rounds — `list_files` → `read_file("agent_plan.md")` →
final answer mentioning the phase table (P2 built, awaiting test). Confirms
chaining where round 2's input depends on round 1's output.

### Test scenario 2 — agent runs the computer (shell mode)

```powershell
$env:ALLOW_SHELL = "1"
python agent.py
```
```
you> check whether this project has any Python syntax errors, fix nothing, just report
```
**Expect:** `run_shell` round running something like
`python -m py_compile agent.py`, then a summary ("no syntax errors, exit code 0").
Then a recovery test:
```
you> run the agent without an API key and tell me what happens
```
**Expect:** it runs `python agent.py` via shell, sees the 401/key error text in
tool output, and explains it exits cleanly with instructions — i.e. it read a
command's *failure* and reasoned about it.

---

## P3 — Simple MCP servers (build + call) ✅ built (awaiting user test)

Model Context Protocol: our agent becomes an MCP **client**; we also build two real
MCP **servers**. Transport: stdio — the client spawns each server as a subprocess
and exchanges JSON-RPC over its stdin/stdout. Config: `mcp.json` (same idea as
Claude Desktop's config).

- `McpBridge` in `agent.py`: **sync/async bridge** — one background thread runs an
  asyncio loop; coroutines shipped via `run_coroutine_threadsafe(...).result()`.
  Per server: `AsyncExitStack` (keeps subprocess + session open) + `ClientSession`.
- Connect flow: `initialize` handshake → `list_tools` → convert each tool's
  `input_schema` (already JSON Schema!) to OpenAI format → append to `TOOL_SCHEMAS`.
  **MCP tools join the same registry as builtins — the loop never cares where a
  tool came from.**
- `run_tool()`: builtin? → run it. Else in `MCP_ROUTES`? → `session.call_tool`,
  flatten `TextContent` blocks to one string; `is_error` → error string.
  Transport crash → error string. Model self-corrects either way.
- Graceful degradation: `mcp` package missing or a server crashing at startup =
  warning, agent keeps running with remaining tools. `/quit` kills servers.
- `/mcp` command lists connected servers + their tools; `/tools` marks MCP tools.
- Servers built with the official `mcp` SDK (**v2.x `MCPServer` API**):
  - `servers/calc_server.py` — `add`, `multiply`, `evaluate` (safe `ast` evaluator,
    whitelist: numbers + operators only — no `eval`)
  - `servers/notes_server.py` — `add_note`, `list_notes`, `delete_note`, persisted
    to `servers/notes.json` (server owns its data — state the LLM doesn't have)

**Design principle (agreed with user):** tools should be LOAD-BEARING — use them
when the model *can't* know the answer (state, files, time) — and a well-prompted
agent should NOT gratuitously call `add(2,3)` when it can just answer.

### Test scenarios (live, with API key)

**Scenario 1 — MCP + builtin tools in one task (no shell needed):**
```
you> list the files in this project, then use the calculator server to add up
     the byte sizes of the files you see
```
**Expect:** `list_files` (builtin) → `add` (calc server, possibly chained) → answer.
Confirms tools from two different sources cooperate in one agentic chain.

**Scenario 2 — stateful server across sessions (the P4 preview):**
```
you> save a note: "P3 built, MCP works" tagged project,mcp
you> what notes do I have?
```
then `/quit`, relaunch, and ask `what notes do I have?` again.
**Expect:** note persists — the data lives in `servers/notes.json`, not in the
model. Also inspect `servers/notes.json` on disk.

**Scenario 3 — anti-tool + degradation checks:**
- `what is 5+5?` → ideally NO tool call, just `10` (tools are load-bearing).
- Break `mcp.json` (bad command) → warning at startup, agent still chats with
  builtin tools.
- `evaluate the expression __import__('os').system('dir')` via the calc server →
  the AST whitelist refuses (offline-verified ✅).

**Result:** ✅ live-tested by user — scenario 1 passed (builtin + MCP tools chained,
correct total 63,789). Scenario 2 passed in-session (note saved & listed). **Finding:**
model hallucinated unrelated narration after a successful tool call (replayed the
previous turn's file listing) → fixed in P4 with a system-prompt honesty guardrail.

---

## P4 — Persistent memory ✅ built (awaiting user test)

Two layers, answering two different questions:

1. **Session memory** — "what were we just talking about?" Dumb persistence:
   - `save_session()` / `load_session()` — whole `messages` list → `sessions/<name>.json`
     (tool messages + `tool_calls` round-trip intact — verified offline).
   - Commands: `/save [name]`, `/load <name>`, `/resume` (most recent *named* save),
     `/sessions`, **auto-save on exit** (`/quit`/Ctrl+C → `sessions/autosave.json`).
   - `sanitize_session_name()` — session names become filenames, so path traversal
     (`../evil`) is **rejected loudly**, not silently renamed (offline-verified).
2. **Long-term memory** — "what should the agent know forever?"
   - `memory.json`: `{id, text, tags, created}` entries.
   - Model-facing tools `save_memory(text, tags)` / `search_memory(query)` — the
     model DECIDES what's worth remembering (same tool machinery as everything else).
   - `build_system_prompt()` injects the most recent `MAX_PROMPT_MEMORIES = 10`
     into the system prompt at startup → a fresh process "wakes up" knowing them.
     Older entries still reachable via `search_memory`.
   - `/clear` keeps long-term memories (clears only the conversation).
   - Corrupt/missing `memory.json` → treated as empty, never crashes (offline-verified).
3. **Honesty guardrail** (from the P3 hallucination finding): system prompt now says
   *"Report only what the latest tool calls and messages actually did. Never describe
   actions you did not take this turn."*

*(stretch, not built: auto-summarize old turns into memories when history grows long)*

### Test scenarios (live, with API key)

**Scenario 1 — the star: long-term memory across restarts:**
```
you> remember that my favorite editor is Neovim and that this project is a learning exercise
```
then `/quit` → relaunch `python agent.py` → ask `what do you know about me?`
**Expect:** the new process, with NO conversation history, answers from
`memory.json` (check the file on disk; check the startup banner shows
"long-term memories loaded: 2"). Also try `search_memory` depth: ask
`what editor do I prefer?` days later — the prompt only has the 10 most recent,
so search_memory covers the rest.

**Scenario 2 — session resume:**
Chat a bit (with a tool call in the mix), `/quit` (watch for
`[auto-saved N messages...]`), relaunch, `/resume`.
**Expect:** mid-story continuation — the model recalls the earlier turns
because the full transcript (incl. tool messages) was restored.

**Scenario 3 — layer separation:**
`/clear` after saving memories → ask `what do you know about me?`
**Expect:** conversation gone, memories still known. Then `/load <session>` →
conversation back, memories untouched.

**Result:** _pending user test (offline: memory tools, prompt injection, session
round-trip incl. tool_calls, traversal rejection, /resume ordering, corrupt-file
resilience all verified)_

---

## P5 (optional) — Full harness build

Outline only; decide after P4. Candidates: SSE streaming, `agent.toml` config file,
tool-approval prompts + sandboxing, token/cost accounting (`usage` field), subagents,
`rich`/`textual` TUI.

---

## Log

- **2026-10-12** — P0 `agent.py` written (chat + history + retry/backoff + slash commands);
  compiles, missing-key path verified. ~~Awaiting live test~~ → **user-tested OK**.
- **2026-10-12** — P1 added: 5-tool registry (`list_files`, `read_file`, `get_current_time`,
  `search_files`, `roll_dice`), `run_tool()` (errors-as-strings), `do_tool_rounds()` mini
  loop (cap 5), `/tools` command, path jail via `safe_path()`. Offline unit test of all
  tools + error paths passed. ~~Awaiting live test~~ → **user-tested OK**.
- **2026-10-12** — P2 added: loop generalized to `MAX_ITERATIONS=10` with round numbers +
  per-tool timing + Ctrl+C-safe history; `run_shell` (PowerShell, `cwd` jail, 30s timeout,
  8 KB cap) registered only when `ALLOW_SHELL=1`; startup banner shows shell status.
  Offline tests: shell off-by-default ✅, output/stderr/exit-code capture ✅,
  timeout kill ✅, opt-in registration ✅. User moved on to P3 without reporting
  live-test details.
- **2026-10-12** — P3 added: `mcp` SDK v2.2.0 installed (v2 renamed FastMCP →
  `MCPServer`; probed the API first — snake_case fields: `input_schema`, `is_error`).
  `McpBridge` (background asyncio thread + `run_coroutine_threadsafe`), `mcp.json`
  config, `/mcp` command, graceful degradation. Two real stdio servers built:
  `servers/calc_server.py` (add/multiply/evaluate with AST whitelist, no `eval`) and
  `servers/notes_server.py` (CRUD persisted to `servers/notes.json`).
  Offline test: both servers connect, all 6 MCP tools callable, 8 error paths return
  strings (incl. `__import__` rejected by AST jail), notes persist to disk, clean
  shutdown. Awaiting live test (3 scenarios above).
- **2026-10-12** — P3 live-tested by user: scenario 1 passed (builtin + MCP tools in one
  chain, correct total). **Bug found:** model hallucinated narration of the previous
  turn's task after a successful tool call → will fix with system-prompt guardrail in P4.
- **2026-10-12** — P4 added: session persistence (`/save` `/load` `/resume` `/sessions`,
  auto-save to `sessions/autosave.json` on quit; `sanitize_session_name` rejects path
  traversal loudly); long-term memory (`memory.json` + `save_memory`/`search_memory`
  tools; `build_system_prompt()` injects 10 most recent at startup; `/clear` keeps
  memories); honesty guardrail added to SYSTEM_PROMPT (fixes the P3 hallucination
  finding). Offline tests: memory tools + search ✅, prompt injection ✅, session
  round-trip incl. tool_calls ✅, traversal rejected ✅, /resume prefers named saves ✅,
  corrupt memory.json tolerated ✅. Awaiting live test (3 scenarios above).
