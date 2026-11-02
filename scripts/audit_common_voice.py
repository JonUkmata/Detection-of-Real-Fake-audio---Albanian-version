"""Read Common Voice metadata; emit a reproducible JSON audit to stdout.

Usage: python -B scripts/audit_common_voice.py data/raw/common_voice_27_sq
No audio is opened and no input or output files are written by this script.
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path


QUOTED_TABLES = {"validated_sentences.tsv", "unvalidated_sentences.tsv", "reported.tsv"}
CLIP_TABLES = {"validated.tsv", "invalidated.tsv", "other.tsv", "train.tsv", "dev.tsv", "test.tsv"}
REQUIRED = {"validated.tsv", "train.tsv", "dev.tsv", "test.tsv"}


def sha256(path: Path) -> str:
    """Hash input bytes without changing them."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def discover_root(base: Path) -> Path:
    candidates = sorted({p.parent for p in base.rglob("validated.tsv")})
    if len(candidates) != 1:
        raise ValueError(f"Expected one validated.tsv beneath {base}; found {len(candidates)}")
    root = candidates[0]
    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    if missing:
        raise ValueError(f"Missing required split metadata: {missing}")
    return root


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Clip manifests use literal quotes; sentence/report tables use CSV quoting.

    Reject unknown schemas and malformed row widths rather than silently losing
    metadata. Keep raw cell values; normalization is only used for comparisons.
    """
    if path.name not in CLIP_TABLES | QUOTED_TABLES | {"clip_durations.tsv"}:
        raise ValueError(f"No parser policy defined for {path.name}")
    quoting = csv.QUOTE_MINIMAL if path.name in QUOTED_TABLES else csv.QUOTE_NONE
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", quoting=quoting, strict=True)
        columns = next(reader, [])
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(f"Empty or duplicate header in {path}")
        required = {"client_id", "path", "sentence"} if path.name in CLIP_TABLES else set()
        if not required.issubset(columns):
            raise ValueError(f"Missing required clip columns in {path}")
        rows = []
        for cells in reader:
            if len(cells) != len(columns):
                raise ValueError(f"Unexpected field count in {path} at line {reader.line_num}")
            rows.append(dict(zip(columns, cells)))
    return columns, rows


def normalized_text(text: str) -> str:
    """NFC + casefold + Unicode whitespace collapse; retain accents/punctuation."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def nonempty_values(rows: list[dict[str, str]], column: str) -> set[str]:
    return {row[column] for row in rows if row[column].strip()}


def quote_diagnostic(path: Path) -> dict:
    """Reproduce the earlier default CSV behavior, retaining source line spans."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_MINIMAL, strict=False)
        next(reader)
        count = 0
        spans = []
        previous_line = reader.line_num
        for _ in reader:
            count += 1
            if reader.line_num > previous_line + 1:
                spans.append({"first_line": previous_line + 1, "last_line": reader.line_num,
                              "physical_lines": reader.line_num - previous_line})
            previous_line = reader.line_num
    return {"parser": "csv.QUOTE_MINIMAL", "strict": False, "rows": count,
            "multiline_record_spans": spans}


def audit(base: Path) -> dict:
    root = discover_root(base)
    paths = sorted(root.glob("*.tsv"))
    hashes = {p.name: sha256(p) for p in paths}
    tables = {}
    metadata = {}
    for path in paths:
        columns, rows = read_tsv(path)
        tables[path.name] = rows
        summary = {
            "sha256": hashes[path.name], "columns": columns, "rows": len(rows),
            "parser": {"encoding": "utf-8-sig", "newline": "", "delimiter": "\t",
                       "quoting": "csv.QUOTE_MINIMAL" if path.name in QUOTED_TABLES else "csv.QUOTE_NONE",
                       "quotechar": '"', "doublequote": True, "escapechar": None,
                       "strict": True, "skipinitialspace": False},
            "missing_blank": {c: sum(not r[c].strip() for r in rows) for c in columns},
            "unique_nonempty": {c: len(nonempty_values(rows, c))
                                for c in ("path", "client_id", "sentence") if c in columns},
            "duplicate_row_excess": len(rows) - len({tuple(r[c] for c in columns) for r in rows}),
        }
        if path.name in CLIP_TABLES:
            counts = Counter(r["client_id"] for r in rows if r["client_id"].strip())
            summary["speaker_concentration"] = {
                "clip_counts_descending": sorted(counts.values(), reverse=True),
                "largest_speaker_fraction_of_all_rows": max(counts.values(), default=0) / len(rows) if rows else None,
            }
        metadata[path.name] = summary
    overlaps = []
    for a, b in combinations(("train.tsv", "dev.tsv", "test.tsv"), 2):
        left, right = tables[a], tables[b]
        overlaps.append({"pair": [a, b], "shared_unique": {
            c: len(nonempty_values(left, c) & nonempty_values(right, c))
            for c in ("path", "client_id", "sentence")},
            "shared_normalized_texts": len(
                {normalized_text(s) for s in nonempty_values(left, "sentence")} &
                {normalized_text(s) for s in nonempty_values(right, "sentence")})})
    diagnostic = quote_diagnostic(root / "validated.tsv")
    literal_count = metadata["validated.tsv"]["rows"]
    reduction = sum(span["physical_lines"] - 1 for span in diagnostic["multiline_record_spans"])
    explained = literal_count - diagnostic["rows"] == reduction and reduction > 0
    for path in paths:
        if sha256(path) != hashes[path.name]:
            raise ValueError(f"Input changed while auditing: {path}")
    return {
        "dataset_root_relative_to_argument": root.relative_to(base).as_posix(),
        "python_version": platform.python_version(), "python_build": sys.version,
        "normalization": "Unicode NFC, casefold, whitespace collapse; retain punctuation and diacritics",
        "missing_definition": "Empty or whitespace-only; unique nonempty values otherwise retain raw spelling",
        "files": metadata, "split_overlaps": overlaps,
        "validated_quote_diagnostic": {**diagnostic, "literal_quote_rows": literal_count,
            "row_reduction": literal_count - diagnostic["rows"], "reduction_explained_by_line_spans": explained,
            "reproduces_reported_6374_vs_6344": literal_count == 6374 and diagnostic["rows"] == 6344,
            "interpretation": "Quote-aware parsing merges physical clip-manifest rows into multiline records."
                              if explained else "No demonstrated multiline explanation for a count difference.",
            "historical_limit": "This tests current bytes; no earlier dataset snapshot or hash is available to prove historical file identity."},
        "limitations": ["Metadata only: no audio decoding, quality checks, or waveform deduplication.",
                        "Speaker IDs are metadata identifiers, not independently verified identities.",
                        "Normalized equality does not detect all near-duplicate or semantic text overlap.",
                        "QUOTE_NONE is the observed clip-manifest policy; unknown TSV filenames are rejected."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.dataset_directory)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
