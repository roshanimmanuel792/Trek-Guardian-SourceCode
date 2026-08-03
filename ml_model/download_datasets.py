#!/usr/bin/env python3
"""
Download real hypoxia physiology datasets from Figshare.

Sources:
  - Harespod (Scientific Data 2024): collection 6623344
    https://doi.org/10.6084/m9.figshare.c.6623344
  - HAPP (Scientific Data 2025): article 29947679
    https://doi.org/10.6084/m9.figshare.29947679
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import requests

FIGSHARE_API = "https://api.figshare.com/v2"

# Harespod: prefer altitude-segmented (Data_Disc) + continuous (Data_Cons)
HARESPOD_COLLECTION_ID = 6623344
HARESPOD_ARTICLES = {
    "Data_Disc": 22736447,  # altitude-split segments
    "Data_Cons": 22736432,  # continuous recordings
    "Instruction": 22736456,
}

HAPP_ARTICLE_ID = 29947679

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
RAW_DIR = ROOT_DIR / "dataset" / "raw"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "TrekGuardian-ML/1.0"})
    return s


def _download_file(url: str, dest: Path, session: requests.Session) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[SKIP] {dest.name} already present ({dest.stat().st_size} bytes)")
        return

    print(f"[GET]  {url}")
    print(f"      -> {dest}")
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".partial")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    print(f"[OK]   Saved {dest.name} ({dest.stat().st_size} bytes)")


def _extract_archive(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"[SKIP] Already extracted: {out_dir}")
        return

    print(f"[EXTRACT] {archive.name} -> {out_dir}")
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
    elif suffix == ".7z":
        try:
            import py7zr
        except ImportError as exc:
            raise SystemExit(
                "py7zr is required to extract Harespod archives. "
                "Install with: pip install py7zr"
            ) from exc
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=out_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive}")

    marker.write_text(json.dumps({"source": str(archive.name)}), encoding="utf-8")
    print(f"[OK]   Extracted to {out_dir}")


def download_happ(session: requests.Session, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    article = session.get(f"{FIGSHARE_API}/articles/{HAPP_ARTICLE_ID}", timeout=60).json()
    meta_path = out_dir / "article_meta.json"
    meta_path.write_text(json.dumps(article, indent=2), encoding="utf-8")

    for fmeta in article.get("files", []):
        name = fmeta["name"]
        if not name.lower().endswith(".csv"):
            continue
        _download_file(fmeta["download_url"], out_dir / name, session)

    print(f"[DONE] HAPP CSVs in {out_dir}")


def download_harespod(session: requests.Session, out_dir: Path, include_continuous: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = ["Data_Disc", "Instruction"]
    if include_continuous:
        targets.append("Data_Cons")

    for key in targets:
        article_id = HARESPOD_ARTICLES[key]
        article = session.get(f"{FIGSHARE_API}/articles/{article_id}", timeout=60).json()
        (out_dir / f"{key}_meta.json").write_text(json.dumps(article, indent=2), encoding="utf-8")

        for fmeta in article.get("files", []):
            dest = out_dir / fmeta["name"]
            _download_file(fmeta["download_url"], dest, session)
            if dest.suffix.lower() in {".7z", ".zip"}:
                extract_dir = out_dir / key
                _extract_archive(dest, extract_dir)

    print(f"[DONE] Harespod files in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Trek Guardian real datasets")
    parser.add_argument("--happ-only", action="store_true", help="Download only HAPP")
    parser.add_argument("--harespod-only", action="store_true", help="Download only Harespod")
    parser.add_argument(
        "--with-continuous",
        action="store_true",
        help="Also download Harespod Data_Cons (~77MB compressed)",
    )
    args = parser.parse_args()

    session = _session()
    do_happ = not args.harespod_only
    do_harespod = not args.happ_only

    print("=" * 60)
    print("Trek Guardian — Real Dataset Downloader")
    print("=" * 60)
    print(f"Raw directory: {RAW_DIR}")

    try:
        if do_happ:
            download_happ(session, RAW_DIR / "happ")
        if do_harespod:
            download_harespod(session, RAW_DIR / "harespod", include_continuous=args.with_continuous)
    except requests.RequestException as exc:
        print(f"[ERROR] Network failure: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nNext: python preprocess.py")


if __name__ == "__main__":
    main()
