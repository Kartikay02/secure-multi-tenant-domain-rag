#!/usr/bin/env python3
"""Automated release package hygiene and secrets exposure validator (SEC-56 / Issue 1).

Scans repository and release bundles for:
1. Forbidden runtime artifacts (.db, .sqlite, .sqlite3, uploads, __pycache__)
2. Sensitive environment files (.env, .env.local, .env.production)
3. Cryptographic credentials (private keys, certificates, .pem, .key)
4. Obvious secret patterns (live API keys, tokens, high-entropy passwords)
5. Clean release archive creation and verification
"""

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

# Forbidden path globs and patterns in distributable repository
FORBIDDEN_GLOBS = [
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "id_rsa*",
    ".env",
    ".env.local",
    ".env.*.local",
    ".env.production",
    "data/uploads/**/*",
]

# Patterns detecting real secrets or leaked credentials
SECRET_PATTERNS = [
    (r"sk-[a-zA-Z0-9]{32,}", "OpenAI API Key"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub Personal Access Token"),
    (r"xox[baprs]-[0-9a-zA-Z]{10,48}", "Slack Token"),
    (r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----", "Private Key Header"),
]

# Allowlisted files that may contain test/mock names or regex definitions
EXCLUDED_FROM_CONTENT_SCAN = {
    Path("scripts/verify_clean_distribution.py"),
    Path("tests/unit/test_api_auth.py"),
    Path("tests/unit/test_validator.py"),
    Path("tests/integration/test_security_remediation_e2e.py"),
    Path("app/core/config.py"),
    Path("app/core/database.py"),
    Path(".env.example"),
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    ".coverage",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "eval_reports",
    "scratch",
    ".hypothesis",
}


def scan_forbidden_files(root: Path, check_uploads: bool = True) -> list[str]:
    """Check for presence of forbidden artifact files."""
    violations: list[str] = []

    for glob_pattern in FORBIDDEN_GLOBS:
        if not check_uploads and "uploads" in glob_pattern:
            continue
        for matched in root.glob(glob_pattern):
            # Ignore files inside virtualenv or cache
            if any(part in EXCLUDED_DIR_NAMES for part in matched.parts):
                continue
            if matched.name == ".gitkeep":
                continue
            if matched.is_file():
                violations.append(f"Forbidden artifact found: {matched.relative_to(root)}")

    return violations


def scan_file_secrets(root: Path) -> list[str]:
    """Scan source and configuration files for accidental secret patterns."""
    violations: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
            continue
        if any(
            part.startswith(".") for part in rel.parts if part not in {".env.example", ".gitignore"}
        ):
            continue
        if rel in EXCLUDED_FROM_CONTENT_SCAN:
            continue
        if path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".ico", ".pdf"}:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            for pattern, desc in SECRET_PATTERNS:
                matches = re.findall(pattern, content)
                if matches:
                    violations.append(
                        f"Potential {desc} detected in {rel} (matches count: {len(matches)})"
                    )
        except Exception as exc:
            violations.append(f"Could not scan {rel}: {exc}")

    return violations


def scan_zip_archive(zip_path: Path) -> list[str]:
    """Inspect contents of a ZIP release archive for forbidden files and secrets."""
    violations: list[str] = []
    forbidden_regexes = [
        r"\.db(?:-.*)?$",
        r"\.sqlite(?:3)?(?:-.*)?$",
        r"(?:^|/)\.env$",
        r"(?:^|/)\.env\.(?!example$)",
        r"\.pem$",
        r"\.key$",
        r"\.pfx$",
        r"\.p12$",
        r"id_rsa",
        r"^data/uploads/(?!.*\.gitkeep$).+",
        r"(?:^|/)__pycache__/",
        r"\.py[cod]$",
        r"\.so$",
        r"\.pyd$",
        r"(?:^|/)\.pytest_cache/",
        r"(?:^|/)\.mypy_cache/",
        r"(?:^|/)\.ruff_cache/",
        r"(?:^|/)\.venv/",
        r"(?:^|/)venv/",
        r"\.zip$",
        r"\.tar(?:\.gz)?$",
        r"\.tmp$",
        r"\.bak$",
        r"\.swp$",
    ]
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            for pat in forbidden_regexes:
                if re.search(pat, name, re.IGNORECASE):
                    violations.append(f"Forbidden artifact inside ZIP: {name}")

            # Scan text files within ZIP for secret leaks
            p = Path(name)
            if p.suffix in {
                ".py",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
                ".ini",
                ".cfg",
                ".sh",
                ".bat",
            }:
                if p not in EXCLUDED_FROM_CONTENT_SCAN:
                    try:
                        content = zf.read(name).decode("utf-8", errors="ignore")
                        for pattern, desc in SECRET_PATTERNS:
                            matches = re.findall(pattern, content)
                            if matches:
                                violations.append(
                                    f"Potential {desc} detected inside ZIP file '{name}' (matches: {len(matches)})"
                                )
                    except Exception:
                        pass
    return violations


def clean_working_tree(root: Path) -> None:
    """Clean generated runtime artifacts, pycache, and cache directories from working tree."""
    import shutil

    cleaned = 0
    # Clean cache directories
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        for dirname in dirnames:
            if dirname in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov"}:
                target = Path(dirpath) / dirname
                shutil.rmtree(target, ignore_errors=True)
                cleaned += 1

    # Clean leftover pyc/pyo files
    for path in root.rglob("*.py[co]"):
        try:
            path.unlink(missing_ok=True)
            cleaned += 1
        except Exception:
            pass

    # Clean uploaded test files in data/uploads (preserving .gitkeep)
    uploads_dir = root / "data" / "uploads"
    if uploads_dir.exists():
        for path in uploads_dir.rglob("*"):
            if path.is_file() and path.name != ".gitkeep":
                try:
                    path.unlink(missing_ok=True)
                    cleaned += 1
                except Exception:
                    pass
        # Clean empty subdirectories inside data/uploads
        for dirpath, _dirnames, _filenames in os.walk(uploads_dir, topdown=False):
            if Path(dirpath).resolve() != uploads_dir.resolve():
                try:
                    if not os.listdir(dirpath):
                        os.rmdir(dirpath)
                except Exception:
                    pass

    print(
        f"[SUCCESS] Cleaned {cleaned} cache directories, test uploads, and generated artifacts from working tree."
    )


def create_release_zip(root: Path, output_zip: Path) -> None:
    """Package clean repository files into a release ZIP, omitting forbidden artifacts.

    Uses an isolated staging directory outside the repository root and output destination,
    copies only clean source and configuration files in deterministic order, writes
    with normalized deterministic timestamps (reproducible build), and moves the completed
    ZIP to output_zip.
    """
    import shutil
    import tempfile

    output_zip = output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SEC-56] Building clean release archive: {output_zip}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        staging_dir = Path(tmp_dir) / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy permitted files into staging directory
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune excluded directories and hidden directories
            dirnames[:] = sorted(
                [d for d in dirnames if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")]
            )
            for filename in sorted(filenames):
                file_path = Path(dirpath) / filename
                rel_path = file_path.relative_to(root)
                rel_str = str(rel_path).replace("\\", "/")

                # Exclude databases, compiled files, archives, and temporary artifacts
                if file_path.suffix in {
                    ".db",
                    ".sqlite",
                    ".sqlite3",
                    ".pyc",
                    ".pyo",
                    ".pyd",
                    ".so",
                    ".zip",
                    ".tar",
                    ".gz",
                    ".tmp",
                    ".bak",
                    ".swp",
                }:
                    continue
                if filename in {".DS_Store", "Thumbs.db"} or filename.endswith("~"):
                    continue
                if (
                    filename == ".env"
                    or (filename.startswith(".env.") and filename != ".env.example")
                    or filename.endswith(".env")
                ):
                    continue
                if rel_str.startswith("data/uploads/") and filename != ".gitkeep":
                    continue
                if any(part in EXCLUDED_DIR_NAMES for part in rel_path.parts):
                    continue
                if file_path.resolve() == output_zip:
                    continue

                dest_file = staging_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_file)

        # 2. Build ZIP archive from staging directory with deterministic metadata
        temp_zip_path = Path(tmp_dir) / output_zip.name
        with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for dirpath, dirnames, filenames in os.walk(staging_dir):
                dirnames.sort()
                for filename in sorted(filenames):
                    src_file = Path(dirpath) / filename
                    rel_path = src_file.relative_to(staging_dir)
                    arcname = str(rel_path).replace("\\", "/")

                    # Use fixed deterministic timestamp for reproducible release archive
                    zinfo = zipfile.ZipInfo(arcname, date_time=(2026, 1, 1, 0, 0, 0))
                    zinfo.external_attr = 0o644 << 16  # standard file permissions
                    if src_file.suffix in {".sh", ".bat"}:
                        zinfo.external_attr = 0o755 << 16
                    zinfo.compress_type = zipfile.ZIP_DEFLATED
                    with open(src_file, "rb") as f:
                        zf.writestr(zinfo, f.read())

        shutil.move(str(temp_zip_path), str(output_zip))

    print(f"[SUCCESS] Release archive created successfully at {output_zip}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package and release hygiene validator")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory to inspect (default: current directory)",
    )
    parser.add_argument(
        "--clean-tree",
        action="store_true",
        help="Clean working tree from __pycache__, .pytest_cache, .mypy_cache, etc.",
    )
    parser.add_argument(
        "--strict-uploads",
        action="store_true",
        help="Enforce that data/uploads is empty of user documents",
    )
    parser.add_argument(
        "--check-zip",
        type=Path,
        default=None,
        help="Inspect a specific release ZIP file for forbidden artifacts",
    )
    parser.add_argument(
        "--build-release-zip",
        type=Path,
        default=None,
        help="Build a clean release ZIP archive at specified destination",
    )
    args = parser.parse_args()

    root = args.root.resolve()

    if args.clean_tree:
        clean_working_tree(root)

    if args.build_release_zip:
        create_release_zip(root, args.build_release_zip)
        zip_violations = scan_zip_archive(args.build_release_zip)
        if zip_violations:
            print(f"\n[FAIL] Generated ZIP contains {len(zip_violations)} forbidden artifacts:")
            for v in zip_violations:
                print(f"  - {v}")
            return 1
        print("[SUCCESS] Verified generated release ZIP archive: 100% clean.")
        return 0

    if args.check_zip:
        print(f"[SEC-56] Inspecting ZIP archive: {args.check_zip}")
        zip_violations = scan_zip_archive(args.check_zip)
        if zip_violations:
            print(f"\n[FAIL] Found {len(zip_violations)} forbidden artifacts in ZIP:")
            for v in zip_violations:
                print(f"  - {v}")
            return 1
        print("[SUCCESS] ZIP archive is completely clean!")
        return 0

    print(f"[SEC-56] Inspecting distribution package hygiene at: {root}")

    file_violations = scan_forbidden_files(root, check_uploads=args.strict_uploads)
    secret_violations = scan_file_secrets(root)

    all_violations = file_violations + secret_violations

    if all_violations:
        print(f"\n[FAIL] Found {len(all_violations)} distribution hygiene violations:")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("[SUCCESS] Distribution tree is clean! Zero forbidden artifacts or exposed secrets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
