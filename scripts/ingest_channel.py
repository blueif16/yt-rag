"""
Thin CLI wrapper around yt_rag.ingest.ingest_channel.

Usage:
    .venv/bin/python scripts/ingest_channel.py @LewisWJackson [n=30] [namespace]
"""
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

from yt_rag.ingest import ingest_channel


def main():
    if len(sys.argv) < 2:
        print("usage: ingest_channel.py <channel_url_or_handle> [n=30] [namespace]")
        sys.exit(1)
    target = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    namespace = sys.argv[3] if len(sys.argv) > 3 else None
    result = ingest_channel(target, n=n, namespace=namespace)
    print(f"\nsummary: {result}")


if __name__ == "__main__":
    main()
