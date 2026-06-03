#!/usr/bin/env python3
"""Upload swing_fold_towel part1-6 datasets to Hugging Face, one file at a time."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

BASE = Path("/home/ruihao/PiperX-ToolKit/lerobot_datasets/ruio248")
LOG = Path("/home/ruihao/PiperX-ToolKit/datasets/upload_swing_fold_towel_hf.log")
PARTS = range(1, 7)
MAX_RETRIES = 5
RETRY_DELAY_SEC = 30


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def local_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    return files


def remote_sizes(api: HfApi, repo_id: str) -> dict[str, int]:
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
    except HfHubHTTPError as exc:
        if exc.response.status_code == 404:
            return {}
        raise
    return {
        sibling.rfilename: sibling.size or 0
        for sibling in (info.siblings or [])
        if sibling.rfilename
    }


def upload_file_with_retry(
    api: HfApi,
    repo_id: str,
    rel_path: str,
    local_path: Path,
) -> None:
    size_mb = local_path.stat().st_size / 1024 / 1024
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"  upload {rel_path} ({size_mb:.1f}MB) attempt {attempt}/{MAX_RETRIES}")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=rel_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {rel_path}",
            )
            log(f"  ok {rel_path}")
            return
        except Exception as exc:
            log(f"  fail {rel_path}: {exc}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY_SEC * attempt)


def upload_part(api: HfApi, part: int) -> None:
    local_dir = BASE / f"swing_fold_towel_20260531_my_part{part}"
    repo_id = f"ruio248/swing_fold_towel_20260531_my_part{part}"

    if not local_dir.is_dir():
        log(f"part{part}: skip, missing {local_dir}")
        return

    files = local_files(local_dir)
    log(f"part{part}: {len(files)} local files -> {repo_id}")

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    remote = remote_sizes(api, repo_id)

    pending = []
    for rel_path, local_path in files.items():
        local_size = local_path.stat().st_size
        if remote.get(rel_path) == local_size:
            log(f"  skip {rel_path} (already on Hub, {local_size} bytes)")
            continue
        pending.append((rel_path, local_path))

    if not pending:
        log(f"part{part}: done (all files already on Hub)")
        return

    log(f"part{part}: uploading {len(pending)} file(s), one at a time")
    for rel_path, local_path in pending:
        upload_file_with_retry(api, repo_id, rel_path, local_path)

    log(f"part{part}: done")


def main() -> int:
    os.environ.pop("HF_ENDPOINT", None)
    os.environ.setdefault("http_proxy", "http://127.0.0.1:7897")
    os.environ.setdefault("https_proxy", "http://127.0.0.1:7897")
    os.environ.setdefault("all_proxy", "http://127.0.0.1:7897")

    api = HfApi(endpoint="https://huggingface.co")
    try:
        user = api.whoami()["name"]
    except Exception as exc:
        log(f"ERROR: HF auth failed: {exc}")
        return 1

    log(f"HF user: {user} (sequential upload, 1 file at a time)")

    for part in PARTS:
        upload_part(api, part)

    log("all uploads finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
