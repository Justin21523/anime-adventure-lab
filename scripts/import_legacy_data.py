from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.migration import import_legacy_data
from core.shared_cache import get_shared_cache


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import legacy WorldPack and Story JSON into Postgres"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Commit the import; default is dry-run"
    )
    parser.add_argument("--worldpacks", type=Path)
    parser.add_argument("--sessions", type=Path)
    args = parser.parse_args()

    cache = get_shared_cache()
    worldpacks = args.worldpacks or Path(cache.worldpacks_root)
    sessions = args.sessions or Path(cache.get_output_path("games")) / "story_sessions"
    report = import_legacy_data(
        worldpacks_dir=worldpacks,
        sessions_dir=sessions,
        dry_run=not args.apply,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if not report.skipped else 2


if __name__ == "__main__":
    raise SystemExit(main())
