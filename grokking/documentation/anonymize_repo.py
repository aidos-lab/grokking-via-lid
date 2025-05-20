import re
import json
from pathlib import Path


def load_replacements(
    mapping_path: Path,
) -> dict[str, str]:
    """Load anonymization replacements from a JSON file."""
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def anonymize_file(
    filepath: Path,
    replacements: dict[str, str],
) -> None:
    """Replace sensitive info in a file with anonymized placeholders."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Skip binary or non-UTF-8 files
        return
    original_text = text
    for pattern, replacement in replacements.items():
        text = re.sub(re.escape(pattern), replacement, text)
    if text != original_text:
        filepath.write_text(text, encoding="utf-8")


def process_repo(
    root: Path,
    replacements: dict[str, str],
    exclude_dirs: set[str] = {".git", ".venv", "data", "outputs", "wandb"},
) -> None:
    for filepath in root.rglob("*"):
        # Skip directories and certain excluded folders
        if filepath.is_dir():
            continue
        # Skip files in excluded directories or hidden files/folders
        if any(part.startswith(".") for part in filepath.parts):
            continue
        if any(part in exclude_dirs for part in filepath.parts):
            continue
        anonymize_file(filepath, replacements)


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    mapping_path = repo_root / "anonymization_map.json"
    replacements = load_replacements(mapping_path)
    process_repo(repo_root, replacements)
    print("Anonymization complete.")
