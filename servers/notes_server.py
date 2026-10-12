"""notes_server.py — a STATEFUL MCP server (P3 of our agent project).

Same mechanism as calc_server.py (tools over stdio JSON-RPC), but this one
OWNS DATA: it persists notes to servers/notes.json, so anything the agent
writes survives across tool calls, agent restarts, even a different client.

Why this server matters in our learning path:
  - calc_server proves the PROTOCOL plumbing works.
  - this server proves a tool can have STATE the LLM doesn't have —
    the model cannot guess what you noted yesterday; the tool is the
    only source of truth. (This is a preview of P4 memory!)
  - it also shows file I/O with a server-owned data file: the agent's
    path jail doesn't apply here — each server manages its own storage.

RUN IT STANDALONE (optional):
    python servers/notes_server.py

DEPENDENCY:  pip install mcp
"""

import json
from datetime import datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer

server = MCPServer("notes")

# The data file lives NEXT TO this server file — the server owns its storage.
NOTES_FILE = Path(__file__).resolve().parent / "notes.json"


def _load() -> list[dict]:
    """Read notes.json -> list of note dicts. Missing/corrupt file = empty list."""
    if not NOTES_FILE.exists():
        return []
    try:
        return json.loads(NOTES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(notes: list[dict]) -> None:
    """Write the note list back to notes.json (pretty-printed so humans can read it)."""
    NOTES_FILE.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")


@server.tool()
def add_note(text: str, tags: str = "") -> str:
    """Store a note for later. 'text' is the content; 'tags' is optional
    comma-separated keywords for searching, e.g. 'project,phase'."""
    if not text.strip():
        return "Error: note text is empty."
    notes = _load()
    note_id = (max((n["id"] for n in notes), default=0)) + 1  # 1, 2, 3, ...
    notes.append(
        {
            "id": note_id,
            "text": text.strip(),
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    _save(notes)
    return f"Saved note #{note_id} ({len(notes)} note(s) total)."


@server.tool()
def list_notes(query: str = "") -> str:
    """List saved notes. With 'query', only notes whose text or tags contain
    it (case-insensitive). Without it, list everything."""
    notes = _load()
    if query:
        q = query.lower()
        notes = [n for n in notes if q in n["text"].lower() or q in " ".join(n["tags"]).lower()]
    if not notes:
        return f"No notes found{' matching ' + repr(query) if query else ''}."
    lines = []
    for n in notes:
        tag_str = f"  [{', '.join(n['tags'])}]" if n["tags"] else ""
        lines.append(f"#{n['id']} ({n['created']}){tag_str} {n['text']}")
    return "\n".join(lines)


@server.tool()
def delete_note(note_id: int) -> str:
    """Delete the note with the given id. Use list_notes first to find ids."""
    notes = _load()
    before = len(notes)
    notes = [n for n in notes if n["id"] != note_id]
    if len(notes) == before:
        return f"Error: no note with id {note_id}."
    _save(notes)
    return f"Deleted note #{note_id} ({len(notes)} note(s) left)."


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
