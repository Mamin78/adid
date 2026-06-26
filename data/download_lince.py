"""Helper to walk the user through LinCE's registration-gated download.

LinCE requires a registration form per dataset; the resulting download link
is emailed and is not a stable, scriptable URL. This script does NOT bypass
that — it just makes the manual step less painful: it tells you where to go,
takes the link you paste, downloads the archive, verifies it, and unpacks it
into the expected layout for `data/lince.py`.

Usage:
    python -m data.download_lince [--dest ./datasets/lince]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REGISTRATION_URL = "https://ritual.uh.edu/lince/"
INSTRUCTIONS = f"""
LinCE datasets are gated behind a registration form. To get the LID Spa-Eng
archive:

  1. Open {REGISTRATION_URL} in a browser.
  2. Find the LID Spanish-English task and follow the dataset download link.
  3. Fill in the form (name, email, affiliation). You'll be shown — or emailed
     — a direct download URL for a zip file (typically named something like
     lid_spaeng.zip).
  4. Paste that URL below.

Note: the URL may be single-use or expire. If the download fails, just
re-request a fresh link.
""".strip()


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading -> {dest} ...")
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def _verify_zip(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise RuntimeError(
            f"Downloaded file at {path} is not a zip. "
            "The link may have expired or pointed somewhere unexpected."
        )
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(f"  sha256: {h.hexdigest()}")


def _unpack(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    print(f"  Unpacked into {dest_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch LinCE LID Spa-Eng dataset.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("./datasets/lince"),
        help="Where to put the unpacked dataset.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="If given, skip the prompt and use this URL directly.",
    )
    args = parser.parse_args(argv)

    print(INSTRUCTIONS)
    print()

    url = args.url
    if not url:
        try:
            url = input("Paste the LinCE LID Spa-Eng download URL: ").strip()
        except EOFError:
            print("No URL provided; aborting.", file=sys.stderr)
            return 1
    if not url:
        print("No URL provided; aborting.", file=sys.stderr)
        return 1

    zip_path = args.dest / "lid_spaeng.zip"
    _download(url, zip_path)
    _verify_zip(zip_path)
    _unpack(zip_path, args.dest)

    # Sanity check the expected files.
    expected = [
        args.dest / "lid_spaeng" / "train.conll",
        args.dest / "lid_spaeng" / "dev.conll",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        print(
            "  WARNING: expected file(s) not found after unpacking: "
            + ", ".join(missing)
            + ". The archive layout may differ from what data/lince.py expects; "
            "inspect the dest directory and adjust the loader paths if needed."
        )
    else:
        print("  Looks good. Try: python eval_lince.py --root", args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
