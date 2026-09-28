#!/usr/bin/env python3
"""Download Jev-Omni and its base model to a chosen cache (e.g. the Synology).

Skips `unified/` (23.9GB) and `assets/`: neither the model card's loader nor
`scripts/serve_jev_omni.py` reads them -- both assemble the classifier from
`backbone/` + `head.pt` + `runtime_buffers.pt` grafted into the base
multimodal model. Pass --everything to pull them anyway.

    HF_HOME=~/synology/hf-cache python scripts/fetch_jev_omni.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

MODEL_ID = "akhilaaa3/Jev-Omni"
BASE_ID = "google/gemma-4-12B-it"
SKIP = ["unified/*", "assets/*"]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def verify_repo(repo: str, local_dir: str, hub) -> list[str]:
    """Return the repo-relative paths whose local size differs from the Hub's."""
    import os

    info = hub.HfApi().model_info(repo, files_metadata=True)
    bad = []
    for sibling in info.siblings:
        expected = sibling.size
        if expected is None:
            continue
        path = os.path.join(local_dir, sibling.rfilename)
        if not os.path.exists(path):
            continue  # deliberately skipped by ignore_patterns
        actual = os.path.getsize(path)
        if actual != expected:
            bad.append(sibling.rfilename)
            print(f"    MISMATCH {sibling.rfilename}: {actual:,} != {expected:,}")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=os.environ.get("HF_HOME"),
                        help="Cache root (sets HF_HOME). Default: current HF_HOME.")
    parser.add_argument("--everything", action="store_true",
                        help="Also fetch unified/ and assets/ (+24GB, unused by the loader).")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare every local file against the Hub's size listing and re-fetch "
             "mismatches. Worth running after any download to a network share: "
             "small files have been observed landing as 0 bytes over CIFS.",
    )
    parser.add_argument(
        "--local-dir-root",
        help="Download into plain directories here instead of the hub cache. "
             "Required on CIFS/SMB: the classic hub layout symlinks snapshots to "
             "blobs, and os.symlink fails with EIO on a Synology share.",
    )
    args = parser.parse_args()

    if args.cache:
        os.environ["HF_HOME"] = os.path.expanduser(args.cache)
    # CIFS/SMB has no symlink support, so the hub copies blobs into snapshots
    # instead of linking them. That roughly doubles on-disk size; it is not an
    # error, and the warning is noise.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    from huggingface_hub import snapshot_download

    root = os.environ.get("HF_HOME", "~/.cache/huggingface")
    usage = shutil.disk_usage(os.path.expanduser(root))
    print(f"cache root : {root}")
    print(f"free space : {human(usage.free)}")
    print(f"symlinks   : off (CIFS) -- blobs are copied, expect ~2x on disk\n")

    jobs = []
    if not args.base_only:
        jobs.append((MODEL_ID, None if args.everything else SKIP))
    if not args.model_only:
        jobs.append((BASE_ID, None))

    for repo, ignore in jobs:
        print(f"==> {repo}" + (f"  (skipping {', '.join(ignore)})" if ignore else ""), flush=True)
        kwargs = {"ignore_patterns": ignore, "max_workers": 8}
        if args.local_dir_root:
            target = os.path.join(os.path.expanduser(args.local_dir_root), repo.split("/")[-1])
            os.makedirs(target, exist_ok=True)
            kwargs["local_dir"] = target
        started = time.perf_counter()
        path = snapshot_download(repo, **kwargs)
        elapsed = time.perf_counter() - started
        print(f"    done in {elapsed/60:.1f} min -> {path}\n", flush=True)

    if args.verify:
        import huggingface_hub as hub

        print("==> verifying file sizes against the Hub")
        ok = True
        for repo, _ in jobs:
            if args.local_dir_root:
                local = os.path.join(
                    os.path.expanduser(args.local_dir_root), repo.split("/")[-1]
                )
            else:
                local = hub.snapshot_download(repo, allow_patterns=["config.json"])
            print(f"  {repo} -> {local}")
            bad = verify_repo(repo, local, hub)
            if bad:
                ok = False
                print(f"    re-fetching {len(bad)} file(s)")
                for name in bad:
                    hub.hf_hub_download(
                        repo, name,
                        **({"local_dir": local} if args.local_dir_root else {}),
                        force_download=True,
                    )
                still = verify_repo(repo, local, hub)
                print("    repaired" if not still else f"    STILL BAD: {still}")
                ok = not still
            else:
                print("    all files match")
        if not ok:
            return 1

    usage = shutil.disk_usage(os.path.expanduser(root))
    print(f"free space now: {human(usage.free)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
