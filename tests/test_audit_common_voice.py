"""Small synthetic fixtures exercise parsing and leakage checks without raw data."""

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.audit_common_voice import audit, discover_root, normalized_text, read_tsv, sha256


class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "nested" / "sq"
        self.root.mkdir(parents=True)
        self.header = "client_id\tpath\tsentence\n"
        self.write("validated.tsv", self.header + 'a\t1.mp3\t"Opening\nb\t2.mp3\tClosing"\n')
        self.write("train.tsv", self.header + "a\t1.mp3\tËshtë mirë.\na\t2.mp3\tOther\n")
        self.write("dev.tsv", self.header + "b\t3.mp3\t  ËSHTË   MIRË. \n")
        self.write("test.tsv", self.header + "a\t1.mp3\tOther\n")

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_literal_quotes_preserve_clip_rows_and_explain_merge(self) -> None:
        result = audit(self.base)
        self.assertEqual(result["files"]["validated.tsv"]["rows"], 2)
        diagnostic = result["validated_quote_diagnostic"]
        self.assertEqual(diagnostic["rows"], 1)
        self.assertEqual(diagnostic["multiline_record_spans"],
                         [{"first_line": 2, "last_line": 3, "physical_lines": 2}])
        self.assertTrue(diagnostic["reduction_explained_by_line_spans"])
        self.assertFalse(diagnostic["reproduces_reported_6374_vs_6344"])

    def test_sentence_and_report_tables_decode_quotes(self) -> None:
        for name in ("validated_sentences.tsv", "unvalidated_sentences.tsv", "reported.tsv"):
            with self.subTest(name=name):
                path = self.root / name
                with path.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.writer(stream, delimiter="\t")
                    writer.writerow(["sentence_id", "sentence"])
                    writer.writerow(["id", 'Ai tha "po".'])
                _, rows = read_tsv(path)
                self.assertEqual(rows, [{"sentence_id": "id", "sentence": 'Ai tha "po".'}])

    def test_concentrations_and_overlap_counts(self) -> None:
        result = audit(self.base)
        train = result["files"]["train.tsv"]
        self.assertEqual(train["unique_nonempty"], {"path": 2, "client_id": 1, "sentence": 2})
        self.assertEqual(train["speaker_concentration"]["clip_counts_descending"], [2])
        self.assertEqual(train["speaker_concentration"]["largest_speaker_fraction_of_all_rows"], 1)
        train_dev, train_test, _ = result["split_overlaps"]
        self.assertEqual(train_dev["shared_unique"], {"path": 0, "client_id": 0, "sentence": 0})
        self.assertEqual(train_dev["shared_normalized_texts"], 1)
        self.assertEqual(train_test["shared_unique"], {"path": 1, "client_id": 1, "sentence": 1})

    def test_normalization_retains_punctuation_and_diacritics(self) -> None:
        self.assertEqual(normalized_text(" E\u0308SHTË\t mirë. "), "është mirë.")
        self.assertNotEqual(normalized_text("mire"), normalized_text("mirë"))
        self.assertNotEqual(normalized_text("mirë"), normalized_text("mirë."))

    def test_audit_preserves_bytes_and_records_hashes(self) -> None:
        before = {p: p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        result = audit(self.base)
        after = {p: p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        path = self.root / "validated.tsv"
        self.assertEqual(result["files"][path.name]["sha256"], hashlib.sha256(before[path]).hexdigest())
        self.assertEqual(sha256(path), hashlib.sha256(before[path]).hexdigest())
        self.assertEqual(result["dataset_root_relative_to_argument"], "nested/sq")
        self.assertEqual(result, audit(self.base))

    def test_duplicate_paths_are_not_counted_as_unique_rows(self) -> None:
        self.write("validated.tsv", self.header + "a\t1.mp3\tHello\na\t1.mp3\tHello\n")
        result = audit(self.base)
        summary = result["files"]["validated.tsv"]
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["unique_nonempty"]["path"], 1)
        self.assertEqual(summary["duplicate_row_excess"], 1)

    def test_diagnostic_does_not_claim_unobserved_merge(self) -> None:
        self.write("validated.tsv", self.header + "a\t1.mp3\tHello\n")
        diagnostic = audit(self.base)["validated_quote_diagnostic"]
        self.assertEqual(diagnostic["row_reduction"], 0)
        self.assertEqual(diagnostic["multiline_record_spans"], [])
        self.assertFalse(diagnostic["reduction_explained_by_line_spans"])
        self.assertFalse(diagnostic["reproduces_reported_6374_vs_6344"])

    def test_ambiguous_root_and_missing_split_fail(self) -> None:
        self.write("validated.tsv", self.header)
        extra = self.base / "validated.tsv"
        extra.write_text(self.header, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Expected one"):
            discover_root(self.base)
        extra.unlink()
        (self.root / "dev.tsv").unlink()
        with self.assertRaisesRegex(ValueError, "Missing required"):
            discover_root(self.base)

    def test_bad_width_unknown_file_and_missing_columns_fail(self) -> None:
        for name, text, message in [
            ("validated.tsv", self.header + "a\t1.mp3\n", "field count"),
            ("unknown.tsv", self.header, "No parser policy"),
            ("train.tsv", "sentence\nHello\n", "Missing required clip"),
        ]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    read_tsv(self.write(name, text))

    def test_blank_ids_are_excluded_and_empty_split_is_supported(self) -> None:
        self.write("train.tsv", self.header + " \t1.mp3\tHello\n")
        self.write("dev.tsv", self.header)
        result = audit(self.base)
        self.assertEqual(result["files"]["train.tsv"]["unique_nonempty"]["client_id"], 0)
        self.assertEqual(result["files"]["train.tsv"]["missing_blank"]["client_id"], 1)
        self.assertIsNone(result["files"]["dev.tsv"]["speaker_concentration"]["largest_speaker_fraction_of_all_rows"])


if __name__ == "__main__":
    unittest.main()
