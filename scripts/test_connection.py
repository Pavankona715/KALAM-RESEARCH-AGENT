"""
Connection Verification Script
================================
Tests connectivity to all required external services.
Run this after setting up .env to verify everything is reachable.

Usage: python scripts/test_connection.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table

console = Console()


async def test_postgres() -> tuple[bool, str]:
    try:
        import asyncpg
        from backend.config.settings import get_settings
        settings = get_settings()
        url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(url, timeout=5)
        await conn.close()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


async def test_redis() -> tuple[bool, str]:
    try:
        import redis.asyncio as aioredis
        from backend.config.settings import get_settings
        settings = get_settings()
        client = aioredis.from_url(settings.redis_url, socket_timeout=5)
        await client.ping()
        await client.aclose()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


async def test_qdrant() -> tuple[bool, str]:
    try:
        from qdrant_client import AsyncQdrantClient
        from backend.config.settings import get_settings
        settings = get_settings()
        client = AsyncQdrantClient(url=settings.qdrant_url, timeout=5)
        await client.get_collections()
        await client.close()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


async def test_openai() -> tuple[bool, str]:
    try:
        from backend.config.settings import get_settings
        settings = get_settings()
        if not settings.openai_api_key:
            return False, "OPENAI_API_KEY not set"
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        await client.models.list()
        return True, "API key valid"
    except Exception as e:
        return False, str(e)[:60]


async def test_anthropic() -> tuple[bool, str]:
    try:
        from backend.config.settings import get_settings
        settings = get_settings()
        if not settings.anthropic_api_key:
            return False, "ANTHROPIC_API_KEY not set"
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        await client.models.list()
        return True, "API key valid"
    except Exception as e:
        return False, str(e)[:60]


async def main():
    console.print("\n[bold blue]Universal AI Research Agent — Connection Check[/bold blue]\n")

    tests = [
        ("PostgreSQL", test_postgres()),
        ("Redis", test_redis()),
        ("Qdrant", test_qdrant()),
        ("OpenAI", test_openai()),
        ("Anthropic", test_anthropic()),
    ]

    results = await asyncio.gather(*[t[1] for t in tests], return_exceptions=True)

    table = Table(title="Service Connections")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    all_critical_pass = True
    critical_services = {"PostgreSQL", "Redis", "Qdrant"}

    for (name, _), result in zip(tests, results):
        if isinstance(result, Exception):
            ok, msg = False, str(result)
        else:
            ok, msg = result

        status = "[green]✅ OK[/green]" if ok else "[red]❌ FAIL[/red]"
        table.add_row(name, status, msg)

        if name in critical_services and not ok:
            all_critical_pass = False

    console.print(table)

    if all_critical_pass:
        console.print("\n[green]✅ All critical services connected. Ready to build![/green]\n")
    else:
        console.print("\n[red]❌ Some critical services failed. Check your .env and service status.[/red]\n")
        console.print("[yellow]Quick start commands:[/yellow]")
        console.print("  PostgreSQL: brew services start postgresql  (or: pg_ctl start)")
        console.print("  Redis:      brew services start redis")
        console.print("  Qdrant:     docker run -p 6333:6333 qdrant/qdrant")
        console.print("              OR: pip install qdrant-client && python -m qdrant_client.local\n")


if __name__ == "__main__":
    asyncio.run(main())