"""Synthetic fixtures only; no checkpoint download or speech inference."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_mms_tts_cpu_smoke import load_candidate, observed_edits, save_and_check_wav


class SmokeHelperTests(unittest.TestCase):
    def test_candidate_is_read_verbatim_and_must_be_unique_and_approved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selected.csv"
            original = '  Synthetic "text  '
            row = {"candidate_id": "mms-short-001", "review_status": "approved", "original_text": original}
            for rows, valid in [([row], True), ([row, row], False), ([], False),
                                ([{**row, "review_status": "pending"}], False)]:
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                if valid:
                    self.assertEqual(load_candidate(path)["original_text"], original)
                else:
                    with self.assertRaises(ValueError):
                        load_candidate(path)

    def test_edit_diagnostic_reports_observed_changes(self) -> None:
        edits = observed_edits("A b!", "a b")
        self.assertEqual([item["operation"] for item in edits], ["replace", "delete"])
        self.assertEqual(edits[0]["original"], "A")
        self.assertEqual(edits[0]["prepared"], "a")
        self.assertEqual(edits[1]["original"], "!")
        self.assertEqual(observed_edits("same", "same"), [])

    def test_vits_decode_keeps_blanks_until_tokenizer_removes_them(self) -> None:
        from transformers.models.vits.tokenization_vits import VitsTokenizer

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vocab.json"
            path.write_text(json.dumps({"|": 0, "a": 1, "b": 2, " ": 3, "<unk>": 4}), encoding="utf-8")
            tokenizer = VitsTokenizer(str(path), pad_token="|", phonemize=False, is_uroman=False)
            ids = tokenizer("Ab!")["input_ids"]
            self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False), "ab")
            self.assertNotEqual(tokenizer.decode(ids, skip_special_tokens=True), "ab")

    def test_float_wav_roundtrip_and_peak_without_clipping_or_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.wav"
            samples = np.array([0, 0.25, -1, 1.25], dtype=np.float32)
            result = save_and_check_wav(path, samples, 16000)
            self.assertTrue(result["read_back_exact"])
            self.assertEqual(result["num_samples"], 4)
            self.assertEqual(result["peak_amplitude"], 1.25)
            self.assertEqual(result["samples_abs_ge_1"], 2)
            with self.assertRaises(FileExistsError):
                save_and_check_wav(path, samples, 16000)

    def test_invalid_signals_fail_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.wav"
            for samples in [np.array([], dtype=np.float32), np.zeros(4, dtype=np.float32),
                            np.array([np.nan], dtype=np.float32), np.array([np.inf], dtype=np.float32),
                            np.array([1], dtype=np.float64)]:
                with self.subTest(samples=samples):
                    with self.assertRaises(ValueError):
                        save_and_check_wav(path, samples, 16000)
                    self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
