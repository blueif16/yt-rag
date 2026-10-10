"""
Thin CLI wrapper around yt_rag.ingest.ingest_videos.

Usage:
    .venv/bin/python scripts/ingest_videos.py <namespace> <url1> [url2] ...
"""
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

from yt_rag.ingest import ingest_videos


def main():
    if len(sys.argv) < 3:
        print("usage: ingest_videos.py <namespace> <url1> [url2] ...")
        sys.exit(1)
    namespace = sys.argv[1]
    urls = sys.argv[2:]
    result = ingest_videos(urls, namespace=namespace)
    print(f"\nsummary: {result}")


if __name__ == "__main__":
    main()
