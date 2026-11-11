"""Synthesize only mms-short-001, read from the local approved CSV.

Run: python -B -m scripts.run_mms_tts_cpu_smoke
Existing output files are never overwritten. No source text is embedded here.
"""

import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter

from scripts.audit_common_voice import sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "mms-short-001"
MODEL_ID = "facebook/mms-tts-sqi"
REVISION = "1ff16df5f391884d5ea36627298acb2e68b45be2"
SEED = 1234
SOURCE = PROJECT_ROOT / "data/interim/mms_tts_pilot_candidates/selected_texts.csv"
OUTPUT = PROJECT_ROOT / "data/interim/mms_tts_cpu_smoke_mms-short-001_run02"
CACHE = PROJECT_ROOT / "data/interim/huggingface_mms_cache"


def load_candidate(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        matches = [row for row in csv.DictReader(stream) if row["candidate_id"] == CANDIDATE_ID]
    if len(matches) != 1:
        raise ValueError("Expected exactly one mms-short-001 row.")
    row = matches[0]
    if row["review_status"] != "approved" or not row["original_text"].strip():
        raise ValueError("The target must have approved, nonempty original text.")
    return row


def observed_edits(before: str, after: str) -> list[dict]:
    """Describe observed edits, without claiming unique causal alignment."""
    edits = []
    for operation, i, j, k, end in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if operation != "equal":
            edits.append({"operation": operation, "original_range": [i, j],
                          "prepared_range": [k, end], "original": before[i:j],
                          "prepared": after[k:end]})
    return edits


def save_and_check_wav(path: Path, waveform, sample_rate: int) -> dict:
    """Write float32 directly, read back, and verify exact sample preservation."""
    import numpy as np
    from scipy.io import wavfile

    if waveform.ndim != 1 or waveform.size == 0 or waveform.dtype != np.float32:
        raise ValueError("Expected a nonempty, one-dimensional float32 waveform.")
    if not np.isfinite(waveform).all() or not np.any(waveform != 0):
        raise ValueError("Waveform must be finite and not entirely zero.")
    with path.open("xb") as stream:
        wavfile.write(stream, sample_rate, waveform)
    read_rate, restored = wavfile.read(path)
    if read_rate != sample_rate or restored.dtype != np.float32 or not np.array_equal(restored, waveform):
        raise ValueError("WAV read-back differs from the generated float32 samples.")
    return {
        "sample_rate": sample_rate, "num_samples": int(waveform.size),
        "duration_seconds": waveform.size / sample_rate, "wav_dtype": "float32",
        "peak_amplitude": float(np.max(np.abs(waveform))),
        "samples_abs_ge_1": int(np.count_nonzero(np.abs(waveform) >= 1)),
        "finite": True, "nonempty": True, "not_all_zero": True,
        "read_back_exact": True, "wav_sha256": sha256(path),
    }


def run() -> None:
    if Path(sys.executable).resolve() != (PROJECT_ROOT / ".venv313/Scripts/python.exe").resolve():
        raise RuntimeError("Run this smoke test with the project .venv313 interpreter.")
    files = [OUTPUT / name for name in ("audio.wav", "metadata.json", "tokenizer_diagnostics.json")]
    for path in [*files, CACHE / "cache-check"]:
        if subprocess.run(["git", "check-ignore", "--quiet", "--",
                           path.relative_to(PROJECT_ROOT).as_posix()], cwd=PROJECT_ROOT).returncode != 0:
            raise ValueError("Outputs and checkpoint cache must be ignored and untracked.")
    if any(path.exists() for path in files):
        raise FileExistsError("Output already exists; refusing to overwrite it.")
    source_hash = sha256(SOURCE)
    candidate = load_candidate(SOURCE)
    text = candidate["original_text"]

    import torch
    from transformers import AutoTokenizer, VitsModel, set_seed

    start = perf_counter()
    model = VitsModel.from_pretrained(
        MODEL_ID, revision=REVISION, use_safetensors=True,
        trust_remote_code=False, cache_dir=str(CACHE),
    ).to("cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=False, cache_dir=str(CACHE),
    )
    load_seconds = perf_counter() - start
    checkpoint_config = model.config.to_dict()
    synthesis_parameters = {
        name: getattr(model, name)
        for name in ("speaking_rate", "noise_scale", "noise_scale_duration")
    }
    for name, value in synthesis_parameters.items():
        if value != getattr(model.config, name):
            raise ValueError("Runtime synthesis parameters differ from the checkpoint.")

    inputs = tokenizer(text, return_tensors="pt")
    prepared_text, _ = tokenizer.prepare_for_tokenization(text)
    input_ids = inputs["input_ids"][0].tolist()
    # VitsTokenizer removes its interspersed blanks internally during decoding.
    # Keep special tokens here so they are not removed twice.
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded_text != prepared_text:
        raise ValueError("Decoded input IDs do not match the tokenizer-prepared text.")
    diagnostics = {
        "candidate_id": CANDIDATE_ID, "original_text": text,
        "prepared_text": prepared_text, "input_ids": input_ids,
        "attention_mask": inputs.get("attention_mask").tolist() if "attention_mask" in inputs else None,
        "decoded_text": decoded_text,
        "decode_options": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
        "decoded_matches_prepared": True,
        "tokenizer_class": type(tokenizer).__name__,
        "settings": {name: getattr(tokenizer, name, None)
                     for name in ("language", "normalize", "phonemize", "is_uroman", "add_blank")},
        "observed_edits_original_to_prepared": observed_edits(text, prepared_text),
        "edit_limit": "Observed string alignment, not a unique causal character trace; repeated characters can admit other alignments. Indices are Python Unicode character indices, not byte offsets.",
        "source_text_modified": False,
    }
    set_seed(SEED)
    start = perf_counter()
    with torch.inference_mode():
        output = model(**inputs)
    inference_seconds = perf_counter() - start
    waveform = output.waveform[0].cpu().numpy()
    if model.config.to_dict() != checkpoint_config:
        raise ValueError("Checkpoint configuration changed during inference.")
    if sha256(SOURCE) != source_hash:
        raise ValueError("Approved source CSV changed during the run.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    measurements = save_and_check_wav(files[0], waveform, model.config.sampling_rate)
    metadata = {
        "candidate_id": CANDIDATE_ID, "original_text": text,
        "selected_texts_file": SOURCE.relative_to(PROJECT_ROOT).as_posix(),
        "selected_texts_sha256": source_hash, "model_id": MODEL_ID, "revision": REVISION,
        "seed": SEED, "seed_method": "transformers.set_seed immediately before forward",
        "python_executable": str(Path(sys.executable).resolve()), "python_version": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "huggingface_hub", "tokenizers", "safetensors", "numpy", "scipy", "soundfile")},
        "device": "cpu", "model_dtype": str(next(model.parameters()).dtype),
        "model_training": model.training, "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "synthesis_parameters": synthesis_parameters, "checkpoint_config": checkpoint_config,
        "use_safetensors": True, "trust_remote_code": False,
        "load_seconds": load_seconds, "inference_seconds": inference_seconds,
        "inference_timing_scope": "model forward only; excludes loading, tokenization and WAV I/O",
        "postprocessing": "None: no resampling, loudness normalization or trimming",
        "manual_listening": "Not performed; intelligibility is unverified",
        "script_sha256": sha256(Path(__file__)), **measurements,
    }
    for path, data in [(files[1], metadata), (files[2], diagnostics)]:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    # Aggregate diagnostics only; original text stays in ignored local files.
    print(json.dumps({"candidate_id": CANDIDATE_ID, "inference_seconds": inference_seconds,
                      "load_seconds": load_seconds, **measurements}, indent=2))


if __name__ == "__main__":
    run()
