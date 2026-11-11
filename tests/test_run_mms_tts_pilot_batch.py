"""Synthetic fixtures and mocked synthesis; never run the real pilot in tests."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.audit_common_voice import normalized_text, sha256
from scripts.run_mms_tts_pilot_batch import (
    QUOTAS, REVIEW_FIELDS, create_output, generate_batch, load_approved, seed_map,
)


def fixture_rows() -> list[dict]:
    rows = []
    for category, (lower, _, quota) in QUOTAS.items():
        for index in range(quota):
            text = "  " + "   ".join([f"SYNTHETIC{category}{index}"] + ["word"] * (lower - 1)) + "  "
            rows.append({"candidate_id": f"mms-{category}-{index + 1:03d}", "category": category,
                         "original_text": text, "normalized_text": normalized_text(text),
                         "unit_count": str(lower), "review_status": "approved"})
    return rows


def write_fixture(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fixture_rows()[0]))
        writer.writeheader()
        writer.writerows(rows)


class BatchTests(unittest.TestCase):
    def test_valid_input_preserves_original_text_and_row_order(self) -> None:
        rows = fixture_rows()
        rows = [rows[-1], *rows[:-1]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selected.csv"
            write_fixture(path, rows)
            self.assertEqual(load_approved(path), rows)

    def test_invalid_inputs_fail(self) -> None:
        variants = []
        rows = fixture_rows()
        variants.append(rows[:-1])
        for field, value in [("review_status", "pending"), ("unit_count", "1"),
                             ("original_text", "one"), ("normalized_text", "wrong"),
                             ("category", "unknown")]:
            variant = [dict(row) for row in rows]
            variant[0][field] = value
            variants.append(variant)
        duplicated_id = [dict(row) for row in rows]
        duplicated_id[1]["candidate_id"] = duplicated_id[0]["candidate_id"]
        variants.append(duplicated_id)
        duplicated_text = [dict(row) for row in rows]
        for field in ("original_text", "normalized_text"):
            duplicated_text[1][field] = duplicated_text[0][field]
        variants.append(duplicated_text)
        wrong_quotas = [dict(row) for row in rows]
        wrong_quotas[-1].update(candidate_id="mms-short-016", category="short",
                                original_text="synthetic extra", normalized_text="synthetic extra", unit_count="2")
        variants.append(wrong_quotas)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selected.csv"
            for index, variant in enumerate(variants):
                with self.subTest(variant=index):
                    write_fixture(path, variant)
                    with self.assertRaises(ValueError):
                        load_approved(path)

    def test_seed_map_uses_csv_order_not_id_sorting(self) -> None:
        rows = fixture_rows()
        rows = [rows[-1], *rows[:-1]]
        seeds = seed_map(rows)
        self.assertEqual(list(seeds), [row["candidate_id"] for row in rows])
        self.assertEqual(list(seeds.values()), list(range(1234, 1284)))
        self.assertEqual(seeds[rows[0]["candidate_id"]], 1234)

    def test_existing_output_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "batch"
            create_output(output)
            marker = output / "existing.txt"
            marker.write_text("retain", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                create_output(output)
            self.assertEqual(marker.read_text(), "retain")

    def test_failures_and_partial_outputs_preserve_other_results_and_order(self) -> None:
        rows = fixture_rows()[:4]
        calls = []

        def fake_synthesis(row, seed, directory, model, tokenizer, metadata):
            calls.append((row["candidate_id"], seed))
            metadata["inference_seconds"] = 0.01
            if len(calls) == 2:
                raise RuntimeError("Synthetic failure without WAV")
            if len(calls) == 3:
                (directory / "audio.wav").write_bytes(b"partial synthetic output")
                raise RuntimeError("Synthetic failure with partial output")
            return np.array([0.125, -0.25, 0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch("scripts.run_mms_tts_pilot_batch.synthesize_one", side_effect=fake_synthesis):
                results = generate_batch(rows, output, {"sample_rate": 16000}, None, None)
            self.assertEqual(calls, [(row["candidate_id"], 1234 + i) for i, row in enumerate(rows)])
            self.assertEqual([item["generation_status"] for item in results], ["success", "failed", "failed", "success"])
            self.assertEqual(results[1]["audio_path"], "")
            self.assertTrue(results[2]["audio_path"].endswith("audio.wav"))
            self.assertEqual((output / results[2]["audio_path"]).read_bytes(), b"partial synthetic output")
            with (output / "listening_review.csv").open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                review = list(reader)
                self.assertEqual(reader.fieldnames, REVIEW_FIELDS)
            self.assertEqual([item["candidate_id"] for item in review], [row["candidate_id"] for row in rows])
            self.assertTrue(all(item["review_status"] == "pending" for item in review))
            for item in review:
                for field in REVIEW_FIELDS[4:]:
                    self.assertEqual(item[field], "")
            for row in rows:
                metadata = json.loads((output / row["candidate_id"] / "metadata.json").read_text())
                self.assertEqual(metadata["original_text"], row["original_text"])
            snapshot = {p: sha256(p) for p in output.rglob("*") if p.is_file()}
            with self.assertRaises(FileExistsError):
                generate_batch(rows, output, {"sample_rate": 16000}, None, None)
            self.assertEqual(snapshot, {p: sha256(p) for p in output.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
