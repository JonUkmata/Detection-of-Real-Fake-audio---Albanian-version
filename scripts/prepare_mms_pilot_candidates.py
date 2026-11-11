"""Prepare local MMS-TTS review candidates; do not approve texts or generate audio.

Run from the repository root:
python -B -m scripts.prepare_mms_pilot_candidates

Outputs are ignored local files. Existing review files are never overwritten.
"""

import argparse
import csv
import json
import platform
import random
import subprocess
from collections import Counter
from pathlib import Path

from scripts.audit_common_voice import discover_root, normalized_text, read_tsv, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
# Fixed order: category, inclusive bounds, eventual approved quota, review count.
CATEGORIES = (
    ("short", 2, 6, 15, 25),
    ("medium", 7, 11, 20, 30),
    ("long", 12, 14, 15, 25),
)
CSV_FIELDS = [
    "candidate_id", "category", "rank_in_category", "original_text",
    "normalized_text", "unit_count", "has_\u00eb", "has_\u00e7", "has_digits",
    "review_status", "exclusion_reason",
]
OUTPUT_NAMES = ("candidates.csv", "provenance.json", "config.json")


def group_recordings(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group normalized texts and retain every associated recording row."""
    groups = {}
    for row in rows:
        text = normalized_text(row["sentence"])
        groups.setdefault(text, []).append(row)
    ordered = {}
    for text in sorted(groups):
        # Path is primary; remaining fields break ties without input-order bias.
        ordered[text] = sorted(groups[text], key=lambda row: (
            row["path"], row["sentence_id"], row["client_id"], row["sentence"]
        ))
    return ordered


def category_for_length(length: int) -> str | None:
    for name, lower, upper, _, _ in CATEGORIES:
        if lower <= length <= upper:
            return name
    return None


def select_candidates(rows: list[dict[str, str]]) -> tuple[list[dict], dict, dict]:
    """Sample sorted eligible groups with one RNG, in the fixed category order."""
    groups = group_recordings(rows)
    pools = {name: [] for name, *_ in CATEGORIES}
    excluded = Counter(outside_length_range=0, no_letters=0)
    for text in groups:
        category = category_for_length(len(text.split()))
        if category is None:
            excluded["outside_length_range"] += 1
        elif not any(character.isalpha() for character in text):
            excluded["no_letters"] += 1
        else:
            pools[category].append(text)

    rng = random.Random(SEED)
    candidates = []
    provenance = {}
    for category, _, _, _, review_count in CATEGORIES:
        if len(pools[category]) < review_count:
            raise ValueError(f"Insufficient eligible texts for {category}: {len(pools[category])}")
        for rank, text in enumerate(rng.sample(pools[category], review_count), start=1):
            candidate_id = f"mms-{category}-{rank:03d}"
            recordings = groups[text]
            candidates.append({
                "candidate_id": candidate_id, "category": category,
                "rank_in_category": rank, "original_text": recordings[0]["sentence"],
                "normalized_text": text, "unit_count": len(text.split()),
                "has_\u00eb": "\u00eb" in text, "has_\u00e7": "\u00e7" in text,
                "has_digits": any(character.isdigit() for character in text),
                "review_status": "pending", "exclusion_reason": "",
            })
            provenance[candidate_id] = {
                "representative_path": recordings[0]["path"],
                "recordings": [
                    {key: row[key] for key in ("sentence_id", "path", "client_id")}
                    for row in recordings
                ],
            }
    counts = {
        "input_rows": len(rows), "unique_normalized_texts": len(groups),
        "excluded_unique_texts": dict(excluded),
        "eligible_unique_texts": {name: len(pool) for name, pool in pools.items()},
        "review_candidates": dict(Counter(row["category"] for row in candidates)),
        "linked_recording_rows": sum(len(item["recordings"]) for item in provenance.values()),
    }
    return candidates, provenance, counts


def check_output_directory(output: Path) -> None:
    """Require ignored, untracked output paths outside raw data before writing."""
    output = output.resolve()
    if not output.is_relative_to(PROJECT_ROOT) or output.is_relative_to(PROJECT_ROOT / "data/raw"):
        raise ValueError("Output must be inside the repository and outside data/raw.")
    for name in OUTPUT_NAMES:
        path = output / name
        if path.exists():
            raise ValueError("Output files already exist; preserve the existing manual review.")
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", path.relative_to(PROJECT_ROOT).as_posix()],
            cwd=PROJECT_ROOT, capture_output=True,
        )
        if result.returncode != 0:
            raise ValueError("Every output file must be ignored and untracked in Git.")


def prepare(dataset_directory: Path, output: Path) -> dict:
    check_output_directory(output)
    root = discover_root(dataset_directory)
    input_path = root / "validated.tsv"
    input_hash = sha256(input_path)
    reference_path = PROJECT_ROOT / "reports/common_voice_27_sq_metadata_audit.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    if input_hash != reference["files"]["validated.tsv"]["sha256"]:
        raise ValueError("validated.tsv differs from the reference audit; stop for review.")
    _, rows = read_tsv(input_path)
    candidates, provenance, counts = select_candidates(rows)
    if sha256(input_path) != input_hash:
        raise ValueError("Input changed during candidate preparation.")
    config = {
        "purpose": "MMS-TTS candidates and reserves for manual review; no texts approved",
        "dataset_version": "cv-corpus-27.0-2026-09-11", "locale": "sq",
        "input_file": input_path.relative_to(PROJECT_ROOT).as_posix(),
        "input_sha256": input_hash, "reference_audit_sha256": sha256(reference_path),
        "script_sha256": sha256(Path(__file__)), "python_version": platform.python_version(),
        "normalization": "Existing normalized_text(): Unicode NFC, casefold, whitespace collapse; preserve punctuation and diacritics",
        "unit_count": "len(normalized_text.split())",
        "seed": SEED, "category_order": [item[0] for item in CATEGORIES],
        "categories": [
            {"name": name, "min_units": lower, "max_units": upper,
             "eventual_approved_quota": quota, "review_count": count}
            for name, lower, upper, quota, count in CATEGORIES
        ],
        "ordering": "Normalized groups sorted lexicographically by Python Unicode string order; eligible pools retain that order",
        "representative": "Original sentence from smallest (path, sentence_id, client_id, sentence); path is primary",
        "sampling": "One random.Random(42); sequential random.sample(pool, review_count) in category_order; retain returned order",
        "flags": "Computed on normalized text; has_digits uses str.isdigit()",
        "exclusions": "Outside 2-14 units first, then no Unicode letter (str.isalpha); mutually exclusive counts",
        "manual_review": "Albanian intelligibility, unclear errors, URLs and codes remain pending; no synthesis-quality exclusions",
        "counts": counts,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidates.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(candidates)
    for name, data in [("provenance.json", provenance), ("config.json", config)]:
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", type=Path,
                        default=PROJECT_ROOT / "data/raw/common_voice_27_sq")
    parser.add_argument("--output-directory", type=Path,
                        default=PROJECT_ROOT / "data/interim/mms_tts_pilot_candidates")
    args = parser.parse_args()
    try:
        counts = prepare(args.dataset_directory.resolve(), args.output_directory.resolve())
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Candidate preparation failed: {exc}\n")
    # Aggregate counts only: do not print texts or speaker identifiers.
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
