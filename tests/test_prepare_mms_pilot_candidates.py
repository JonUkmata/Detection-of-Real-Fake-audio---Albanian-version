"""Synthetic metadata only: no raw sentences or identifiers in tracked fixtures."""

import random
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_mms_pilot_candidates import (
    CATEGORIES, SEED, category_for_length, check_output_directory,
    group_recordings, select_candidates,
)


def recording(text: str, path: str, speaker: str = "synthetic-speaker") -> dict[str, str]:
    return {"sentence": text, "path": path, "sentence_id": "synthetic-" + path,
            "client_id": speaker}


def fixture_rows() -> list[dict[str, str]]:
    rows = []
    for category, lower, _, _, _ in CATEGORIES:
        for index in range(40):
            text = " ".join([f"synthetic{category}{index}"] + ["word"] * (lower - 1))
            rows.append(recording(text, f"{category}-{index:03d}.mp3"))
    return rows


class CandidateTests(unittest.TestCase):
    def test_normalization_and_all_recording_links(self) -> None:
        rows = [recording("  E\u0308SHT\u00cb  \u00c7UDI! ", "z.mp3", "synthetic-a"),
                recording("\u00ebsht\u00eb \u00e7udi!", "a.mp3", "synthetic-b")]
        groups = group_recordings(rows)
        self.assertEqual(list(groups), ["\u00ebsht\u00eb \u00e7udi!"])
        self.assertEqual([r["path"] for r in groups["\u00ebsht\u00eb \u00e7udi!"]], ["a.mp3", "z.mp3"])
        self.assertEqual(len(groups["\u00ebsht\u00eb \u00e7udi!"]), 2)
        self.assertNotIn("\u00ebsht\u00eb \u00e7udi", groups)  # punctuation remains significant

    def test_inclusive_category_boundaries(self) -> None:
        expected = {0: None, 1: None, 2: "short", 6: "short", 7: "medium",
                    11: "medium", 12: "long", 14: "long", 15: None}
        for length, category in expected.items():
            with self.subTest(length=length):
                self.assertEqual(category_for_length(length), category)

    def test_filters_only_range_and_presence_of_a_letter(self) -> None:
        rows = fixture_rows()
        rows += [recording("", "empty.mp3"), recording("one", "one.mp3"),
                 recording(" ".join(["word"] * 15), "fifteen.mp3"),
                 recording("123 !!!", "digits.mp3"), recording("... !!!", "punct.mp3")]
        _, _, counts = select_candidates(rows)
        self.assertEqual(counts["excluded_unique_texts"],
                         {"outside_length_range": 3, "no_letters": 2})
        # URLs/codes with letters are deliberately not filtered automatically.
        rows += [recording("https://synthetic.invalid code42", "url.mp3")]
        _, _, counts = select_candidates(rows)
        self.assertEqual(counts["eligible_unique_texts"]["short"], 41)

    def test_reproducible_for_shuffled_input_with_fixed_rng_order(self) -> None:
        rows = fixture_rows()
        expected = select_candidates(rows)
        shuffled = list(rows)
        random.Random(123).shuffle(shuffled)
        self.assertEqual(expected, select_candidates(shuffled))
        candidates, _, counts = expected
        self.assertEqual(counts["review_candidates"], {"short": 25, "medium": 30, "long": 25})
        rng = random.Random(SEED)
        groups = group_recordings(rows)
        for category, lower, upper, _, count in CATEGORIES:
            pool = [text for text in groups if lower <= len(text.split()) <= upper]
            selected = [row for row in candidates if row["category"] == category]
            self.assertEqual([row["normalized_text"] for row in selected], rng.sample(pool, count))
            self.assertEqual([row["rank_in_category"] for row in selected], list(range(1, count + 1)))
        self.assertEqual(len({row["normalized_text"] for row in candidates}), 80)
        self.assertEqual(len({row["candidate_id"] for row in candidates}), 80)
        self.assertTrue(all(row["review_status"] == "pending" and row["exclusion_reason"] == ""
                            for row in candidates))

    def test_selected_group_preserves_original_and_every_link(self) -> None:
        rows = fixture_rows()
        first = select_candidates(rows)[0][0]
        text = first["normalized_text"]
        original = "  " + text.upper().replace(" ", "   ") + "  "
        rows.append(recording(original, "!first.mp3", "synthetic-extra"))
        candidates, provenance, _ = select_candidates(rows)
        chosen = next(row for row in candidates if row["normalized_text"] == text)
        self.assertEqual(chosen["original_text"], original)
        entry = provenance[chosen["candidate_id"]]
        self.assertEqual(entry["representative_path"], "!first.mp3")
        expected = [{key: row[key] for key in ("sentence_id", "path", "client_id")}
                    for row in group_recordings(rows)[text]]
        self.assertEqual(entry["recordings"], expected)
        self.assertEqual(len(entry["recordings"]), 2)

    def test_flags_use_normalized_text(self) -> None:
        rows = fixture_rows()
        # Replace a selected short group; exactly 25 short groups ensures selection.
        rows = [r for r in rows if not r["path"].startswith("short-")][:]
        rows += [recording(f"\u00cb\u00c7 token{i}", f"flag-{i}.mp3") for i in range(25)]
        candidates, _, _ = select_candidates(rows)
        for candidate in candidates[:25]:
            self.assertTrue(candidate["has_\u00eb"])
            self.assertTrue(candidate["has_\u00e7"])
            self.assertTrue(candidate["has_digits"])

    def test_insufficient_pool_fails_instead_of_returning_fewer(self) -> None:
        with self.assertRaisesRegex(ValueError, "Insufficient eligible texts"):
            select_candidates([])

    def test_output_guard_rejects_unignored_and_raw_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside data/raw"):
            check_output_directory(Path("data/raw/synthetic-output"))
        with self.assertRaisesRegex(ValueError, "ignored and untracked"):
            check_output_directory(Path("reports/synthetic-candidates"))

    def test_output_guard_does_not_overwrite_existing_review(self) -> None:
        with patch("scripts.prepare_mms_pilot_candidates.Path.exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exist"):
                check_output_directory(Path("data/interim/synthetic-review"))


if __name__ == "__main__":
    unittest.main()
