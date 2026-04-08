"""
ai/mcp_client.py — LangChain MCP (Model Context Protocol) integration.

Wraps langchain-mcp to optionally connect to an MCP server and expose
its tools to the LLM.  If MCP_SERVER_CONFIG is None in config.py,
this module is a no-op.

Usage:
    client = MCPClient()
    await client.start()
    tools = client.get_tools()
    await client.stop()
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from config import MCP_SERVER_CONFIG

log = logging.getLogger(__name__)


class MCPClient:
    """
    Wraps langchain-mcp MultiServerMCPClient.
    If MCP_SERVER_CONFIG is None, all methods are safe no-ops.
    """

    def __init__(self):
        self._enabled = MCP_SERVER_CONFIG is not None
        self._client  = None
        self._tools: list = []

    async def start(self):
        if not self._enabled:
            log.info("MCP disabled (MCP_SERVER_CONFIG is None)")
            return
        try:
            from langchain_mcp import MultiServerMCPClient
            self._client = MultiServerMCPClient({"default": MCP_SERVER_CONFIG})
            await self._client.__aenter__()
            self._tools = self._client.get_tools()
            log.info("MCP connected — %d tools available", len(self._tools))
        except Exception as exc:
            log.warning("MCP startup failed (non-fatal): %s", exc)
            self._enabled = False

    async def stop(self):
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as exc:
                log.debug("MCP shutdown error: %s", exc)
            self._client = None

    def get_tools(self) -> list:
        """Return LangChain-compatible tool list (may be empty)."""
        return self._tools

    @property
    def enabled(self) -> bool:
        return self._enabled

    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]


# ── Convenience: run MCP lifecycle from sync code ────────────────────────────

def run_with_mcp(coro):
    """
    Run an async coroutine that receives an MCPClient as first arg.

    Example:
        async def my_task(mcp: MCPClient):
            tools = mcp.get_tools()
            ...

        run_with_mcp(my_task)
    """
    async def wrapper():
        mcp = MCPClient()
        await mcp.start()
        try:
            await coro(mcp)
        finally:
            await mcp.stop()

    asyncio.run(wrapper())
