#!/usr/bin/env python3
"""Download Bilibili videos listed in urls.txt via yt-dlp.

Usage:
    pip install yt-dlp
    # ffmpeg recommended for stream merging
    python download_bilibili.py                       # read urls.txt, save into videos/
    python download_bilibili.py -o downloads -u my.txt
    python download_bilibili.py https://b23.tv/xxxxxx https://b23.tv/yyyyyy

Without bilibili login cookies, yt-dlp falls back to <=480p. To get HD/4K
quality, export your browser cookies and pass --cookies cookies.txt.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


OUTPUT_TEMPLATE = "%(playlist_index|)s%(playlist_index& - |)s%(title)s [%(id)s].%(ext)s"


def build_cmd(url: str, out_dir: Path, cookies: str | None, quality: str) -> list[str]:
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--no-check-certificates",
        "--no-playlist",
        "-f", quality,
        "--merge-output-format", "mp4",
        "-o", str(out_dir / OUTPUT_TEMPLATE),
        "--restrict-filenames",
        "--write-info-json",
        "--write-thumbnail",
    ]
    if cookies:
        cmd += ["--cookies", cookies]
    cmd.append(url)
    return cmd


def load_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="*", help="URLs (overrides --urls-file when given)")
    parser.add_argument("-u", "--urls-file", default="urls.txt", help="text file with one URL per line (default: urls.txt)")
    parser.add_argument("-o", "--output", default="videos", help="output directory (default: videos)")
    parser.add_argument("--cookies", help="Netscape-format cookies file for HD/login-required quality")
    parser.add_argument("-q", "--quality", default="bv*+ba/best", help="yt-dlp format selector")
    args = parser.parse_args()

    if shutil.which("yt-dlp") is None:
        print("error: yt-dlp not found. install with: pip install yt-dlp", file=sys.stderr)
        return 2

    urls = args.urls or load_urls(Path(args.urls_file))
    if not urls:
        print("error: no URLs supplied", file=sys.stderr)
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        rc = subprocess.call(build_cmd(url, out_dir, args.cookies, args.quality))
        if rc != 0:
            failures.append(url)
            print(f"  ! failed (exit {rc})", file=sys.stderr)

    print(f"\ndone: {len(urls) - len(failures)}/{len(urls)} succeeded")
    if failures:
        print("failed URLs:", file=sys.stderr)
        for u in failures:
            print(f"  {u}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
