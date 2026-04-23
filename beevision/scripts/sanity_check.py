"""CLI entrypoint for post-ingest sanity checks. Implemented in step 9."""
from __future__ import annotations

from beevision.data.sanity import main

if __name__ == "__main__":
    raise SystemExit(main())
