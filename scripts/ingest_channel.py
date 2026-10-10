"""
Thin CLI wrapper around yt_rag.ingest.ingest_channel.

Usage:
    .venv/bin/python scripts/ingest_channel.py <collection> @LewisWJackson [n=30]

The collection must already exist (create it via the create_collection MCP tool).
"""
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

from yt_rag.ingest import ingest_channel


def main():
    if len(sys.argv) < 3:
        print("usage: ingest_channel.py <collection> <channel_url_or_handle> [n=30]")
        sys.exit(1)
    collection, target = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    result = ingest_channel(target, collection, n=n)
    print(f"\nsummary: {result}")


if __name__ == "__main__":
    main()
