"""
ai/tools.py — Tool registry for LLM tool-calling mode.

Initial tools:
- repl: evaluate small Python snippets (stateful locals within process)
- terminal: run shell commands with timeout
- web_search: lightweight DuckDuckGo instant-answer lookup
"""

from __future__ import annotations

import ast
from datetime import datetime
import json
import re
import subprocess
import traceback
import urllib.parse
import urllib.request
from typing import Any

from langchain_core.tools import tool

from config import (
    TOOL_REPL_ENABLED,
    TOOL_TERMINAL_ENABLED,
    TOOL_WEB_SEARCH_ENABLED,
    TOOL_TERMINAL_TIMEOUT_SECS,
    TOOL_WEB_SEARCH_MAX_RESULTS,
)


# Shared REPL namespace so follow-up tool calls can reuse variables.
_REPL_GLOBALS: dict[str, Any] = {"__builtins__": __builtins__}
_REPL_LOCALS: dict[str, Any] = {}


@tool
def current_datetime(_: str = "") -> str:
    """Return the current local date/time and timezone. Use this for questions about current year/date/time."""
    now = datetime.now().astimezone()
    return (
        f"local_iso={now.isoformat()}\n"
        f"date={now.strftime('%Y-%m-%d')}\n"
        f"time={now.strftime('%H:%M:%S')}\n"
        f"weekday={now.strftime('%A')}\n"
        f"timezone={now.tzname() or 'local'}\n"
        f"year={now.year}"
    )


@tool
def repl(code: str) -> str:
    """Run a short Python snippet and return result/output. Use for quick calculations and data shaping."""
    source = (code or "").strip()
    if not source:
        return "No code provided."

    try:
        parsed = ast.parse(source, mode="eval")
        value = eval(compile(parsed, "<tool-repl>", "eval"), _REPL_GLOBALS, _REPL_LOCALS)
        return repr(value)
    except SyntaxError:
        pass
    except Exception:
        return traceback.format_exc(limit=1).strip()

    try:
        parsed = ast.parse(source, mode="exec")
        exec(compile(parsed, "<tool-repl>", "exec"), _REPL_GLOBALS, _REPL_LOCALS)
        return "ok"
    except Exception:
        return traceback.format_exc(limit=1).strip()


@tool
def terminal(command: str) -> str:
    """Run a zsh command on the local machine with timeout. Useful for deterministic local checks like date/version/file lookups."""
    cmd = (command or "").strip()
    if not cmd:
        return "No command provided."

    try:
        proc = subprocess.run(
            ["zsh", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=TOOL_TERMINAL_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {TOOL_TERMINAL_TIMEOUT_SECS:.1f}s"
    except Exception as exc:
        return f"Command failed to start: {exc}"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if len(out) > 4000:
        out = out[:4000] + "\n...<truncated>"
    if len(err) > 2000:
        err = err[:2000] + "\n...<truncated>"

    return (
        f"exit_code={proc.returncode}\n"
        f"stdout:\n{out or '<empty>'}\n"
        f"stderr:\n{err or '<empty>'}"
    )


@tool
def web_search(query: str) -> str:
    """Search the web and return short snippets/URLs. Use this for current events, recent updates, and time-sensitive facts."""
    q = (query or "").strip()
    if not q:
        return "No query provided."

    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {
            "q": q,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
        }
    )

    req = urllib.request.Request(url, headers={"User-Agent": "TalkingMAC/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        return f"Web search failed: {exc}"

    lines: list[str] = []
    heading = (payload.get("Heading") or "").strip()
    abstract = (payload.get("AbstractText") or "").strip()
    abstract_url = (payload.get("AbstractURL") or "").strip()

    if heading:
        lines.append(f"{_friendly_source_name(abstract_url or heading)} says: {heading}")
    if abstract:
        lines.append(abstract)
    if abstract_url:
        lines.append(f"Source: {_friendly_source_name(abstract_url)}")

    related = payload.get("RelatedTopics") or []
    added = 0
    for item in related:
        if added >= TOOL_WEB_SEARCH_MAX_RESULTS:
            break
        if isinstance(item, dict) and "Text" in item and "FirstURL" in item:
            source = _friendly_source_name(item["FirstURL"])
            lines.append(f"{source} says: {item['Text']}")
            added += 1
        for nested in item.get("Topics", []) if isinstance(item, dict) else []:
            if added >= TOOL_WEB_SEARCH_MAX_RESULTS:
                break
            if isinstance(nested, dict) and "Text" in nested and "FirstURL" in nested:
                source = _friendly_source_name(nested["FirstURL"])
                lines.append(f"{source} says: {nested['Text']}")
                added += 1

    if not lines:
        return "No results found."
    return "\n".join(lines)


def build_langchain_tools() -> list:
    """Return enabled LangChain tools in deterministic order."""
    tools = [current_datetime]
    if TOOL_REPL_ENABLED:
        tools.append(repl)
    if TOOL_TERMINAL_ENABLED:
        tools.append(terminal)
    if TOOL_WEB_SEARCH_ENABLED:
        tools.append(web_search)
    return tools


def _friendly_source_name(url: str) -> str:
    """Convert a URL into a short human source label like 'The Verge'."""
    if not url:
        return "Unknown source"

    host = url.strip().lower()
    host = re.sub(r"^https?://", "", host)
    host = re.sub(r"^www\.", "", host)
    host = host.split("/", 1)[0]
    host = host.split("?", 1)[0]
    host = host.split("#", 1)[0]
    host = host.split(".", 1)[0] if host.count(".") == 1 else host

    parts = [p for p in re.split(r"[-_.]+", host) if p]
    if not parts:
        return "Unknown source"

    if parts[0] == "the" and len(parts) > 1:
        return "The " + " ".join(p.capitalize() for p in parts[1:])

    return " ".join(p.capitalize() for p in parts)


