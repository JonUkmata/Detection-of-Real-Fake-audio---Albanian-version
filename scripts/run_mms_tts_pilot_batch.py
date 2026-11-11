"""Generate the 50 approved pilot texts once, sequentially, using the local cache.

Run: python -B -m scripts.run_mms_tts_pilot_batch
Never resumes, overwrites, retries, or selects a preferred synthesis result.
"""

import csv
import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter

from scripts.audit_common_voice import normalized_text, sha256
from scripts.run_mms_tts_cpu_smoke import (
    CACHE, MODEL_ID, PROJECT_ROOT, REVISION, SOURCE, observed_edits, save_and_check_wav,
)


OUTPUT = PROJECT_ROOT / "data/interim/mms_tts_pilot_batch_001"
QUOTAS = {"short": (2, 6, 15), "medium": (7, 11, 20), "long": (12, 14, 15)}
RESULT_FIELDS = ["candidate_id", "category", "seed", "generation_status", "audio_path",
                 "sample_rate", "num_samples", "duration_seconds", "inference_seconds",
                 "peak_amplitude", "samples_abs_ge_1", "wav_sha256", "error"]
REVIEW_FIELDS = ["candidate_id", "audio_path", "generation_status", "review_status",
                 "intelligibility_0_2", "pronunciation_notes", "missing_or_repeated_words",
                 "naturalness_notes"]


def load_approved(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 50 or any(row["review_status"] != "approved" for row in rows):
        raise ValueError("Expected exactly 50 approved rows.")
    ids = set()
    texts = set()
    for row in rows:
        category = row["category"]
        text = normalized_text(row["original_text"])
        candidate_id = row["candidate_id"]
        if category not in QUOTAS or not re.fullmatch(rf"mms-{category}-[0-9]{{3}}", candidate_id):
            raise ValueError("Invalid category or candidate ID.")
        lower, upper, _ = QUOTAS[category]
        length = len(text.split())
        if text != row["normalized_text"] or not lower <= length <= upper or length != int(row["unit_count"]):
            raise ValueError("Normalized text or unit-count validation failed.")
        if candidate_id in ids or text in texts:
            raise ValueError("Candidate IDs and normalized texts must be unique.")
        ids.add(candidate_id)
        texts.add(text)
    if Counter(row["category"] for row in rows) != {name: quota for name, (_, _, quota) in QUOTAS.items()}:
        raise ValueError("Expected short/medium/long quotas 15/20/15.")
    return rows


def seed_map(rows: list[dict[str, str]]) -> dict[str, int]:
    return {row["candidate_id"]: 1234 + index for index, row in enumerate(rows)}


def write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def create_output(path: Path) -> None:
    # Requiring a completely new directory also protects partial prior batches.
    path.mkdir(parents=True, exist_ok=False)


def synthesize_one(row: dict, seed: int, directory: Path, model, tokenizer, metadata: dict):
    import torch
    from transformers import set_seed

    text = row["original_text"]
    inputs = tokenizer(text, return_tensors="pt")
    prepared, _ = tokenizer.prepare_for_tokenization(text)
    ids = inputs["input_ids"][0].tolist()
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    write_json(directory / "tokenizer_diagnostics.json", {
        "candidate_id": row["candidate_id"], "original_text": text,
        "prepared_text": prepared, "decoded_text": decoded, "input_ids": ids,
        "decoded_matches_prepared": decoded == prepared,
        "decode_options": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
        "settings": {name: getattr(tokenizer, name, None)
                     for name in ("language", "normalize", "phonemize", "is_uroman", "add_blank")},
        "observed_edits_original_to_prepared": observed_edits(text, prepared),
        "edit_limit": "Observed alignment; repeated characters may admit other alignments. Source text is unchanged.",
    })
    if decoded != prepared:
        raise ValueError("Decoded IDs differ from tokenizer-prepared text.")
    for name, value in metadata["synthesis_parameters"].items():
        if getattr(model, name) != value:
            raise ValueError("Effective synthesis parameters changed.")
    set_seed(seed)
    start = perf_counter()
    try:
        with torch.inference_mode():
            output = model(**inputs)
    finally:
        metadata["inference_seconds"] = perf_counter() - start
    if model.config.to_dict() != metadata["checkpoint_config"]:
        raise ValueError("Checkpoint configuration changed.")
    return output.waveform[0].cpu().numpy()


def generate_batch(rows: list[dict], directory: Path, common: dict, model, tokenizer) -> list[dict]:
    """One attempt per row; persist each terminal status and retain partial files."""
    import soundfile as sf

    results = []
    seeds = seed_map(rows)
    with (directory / "results.csv").open("x", encoding="utf-8", newline="") as result_file, \
         (directory / "listening_review.csv").open("x", encoding="utf-8", newline="") as review_file:
        result_writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDS)
        review_writer = csv.DictWriter(review_file, fieldnames=REVIEW_FIELDS)
        result_writer.writeheader()
        review_writer.writeheader()
        for index, row in enumerate(rows):
            candidate_id = row["candidate_id"]
            candidate_dir = directory / candidate_id
            candidate_dir.mkdir()
            wav = candidate_dir / "audio.wav"
            metadata = {**common, "candidate_id": candidate_id, "category": row["category"],
                        "original_text": row["original_text"], "seed": seeds[candidate_id],
                        "generation_status": "failed", "error": None,
                        "inference_seconds": None, "audio_path": ""}
            try:
                waveform = synthesize_one(row, metadata["seed"], candidate_dir, model, tokenizer, metadata)
                measurements = save_and_check_wav(wav, waveform, common["sample_rate"])
                metadata.update(measurements)
                info = sf.info(wav)
                if (info.format != "WAV" or info.subtype != "FLOAT" or info.channels != 1
                        or info.frames != measurements["num_samples"] or info.samplerate != common["sample_rate"]):
                    raise ValueError("WAV format, channels, sample count or rate failed verification.")
                metadata.update(wav_format=info.format, wav_subtype=info.subtype, channels=info.channels,
                                generation_status="success")
            except Exception as exc:
                metadata["error"] = {"type": type(exc).__name__, "message": str(exc),
                                     "traceback": traceback.format_exc()}
            if wav.exists():
                metadata["audio_path"] = wav.relative_to(directory).as_posix()
                metadata["wav_sha256"] = sha256(wav)
            write_json(candidate_dir / "metadata.json", metadata)
            result = {field: metadata.get(field, "") for field in RESULT_FIELDS}
            result["error"] = json.dumps(metadata["error"], ensure_ascii=False) if metadata["error"] else ""
            result_writer.writerow(result)
            review = {field: "" for field in REVIEW_FIELDS}
            review.update(candidate_id=candidate_id, audio_path=metadata["audio_path"],
                          generation_status=metadata["generation_status"], review_status="pending")
            review_writer.writerow(review)
            result_file.flush()
            review_file.flush()
            results.append(metadata)
            print(f"{index + 1}/{len(rows)} {candidate_id}: {metadata['generation_status']}", flush=True)
    return results


def protected_hashes() -> dict[str, str]:
    paths = [PROJECT_ROOT / name for name in (
        "requirements.txt", "notebooks/01_dataset_exploration.ipynb", "scripts/audit_common_voice.py",
        "scripts/run_mms_tts_cpu_smoke.py", "reports/common_voice_27_sq_metadata_audit.json")]
    for name in ("mms_tts_pilot_candidates", "mms_tts_cpu_smoke_mms-short-001",
                 "mms_tts_cpu_smoke_mms-short-001_run02"):
        paths.extend(p for p in (PROJECT_ROOT / "data/interim" / name).rglob("*") if p.is_file())
    paths.extend((PROJECT_ROOT / "data/raw/common_voice_27_sq").rglob("*.tsv"))
    return {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def run() -> None:
    if Path(sys.executable).resolve() != (PROJECT_ROOT / ".venv313/Scripts/python.exe").resolve():
        raise RuntimeError("Use the project .venv313 interpreter.")
    for path in (OUTPUT / "config.json", CACHE / "cache-check"):
        if subprocess.run(["git", "check-ignore", "--quiet", "--", path.relative_to(PROJECT_ROOT).as_posix()],
                          cwd=PROJECT_ROOT).returncode != 0:
            raise ValueError("Batch outputs and checkpoint cache must be ignored and untracked.")
    if OUTPUT.exists():
        raise FileExistsError("Batch directory already exists; do not regenerate or overwrite it.")
    before = protected_hashes()
    source_hash = sha256(SOURCE)
    rows = load_approved(SOURCE)
    if sha256(SOURCE) != source_hash:
        raise ValueError("Source CSV changed during validation.")

    import torch
    from transformers import AutoTokenizer, VitsModel

    start = perf_counter()
    model = VitsModel.from_pretrained(MODEL_ID, revision=REVISION, use_safetensors=True,
                                     trust_remote_code=False, cache_dir=str(CACHE), local_files_only=True).to("cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False,
                                              cache_dir=str(CACHE), local_files_only=True)
    load_seconds = perf_counter() - start
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("Expected checkpoint float32 parameters.")
    parameters = {name: getattr(model, name) for name in ("speaking_rate", "noise_scale", "noise_scale_duration")}
    if any(value != getattr(model.config, name) for name, value in parameters.items()):
        raise ValueError("Synthesis parameters differ from checkpoint defaults.")
    common = {
        "model_id": MODEL_ID, "revision": REVISION, "device": "cpu", "dtype": "float32",
        "python_executable": str(Path(sys.executable).resolve()), "python_version": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "huggingface_hub", "tokenizers", "safetensors", "numpy", "scipy", "soundfile")},
        "selected_texts_file": SOURCE.relative_to(PROJECT_ROOT).as_posix(), "selected_texts_sha256": source_hash,
        "sample_rate": model.config.sampling_rate, "synthesis_parameters": parameters,
        "checkpoint_config": model.config.to_dict(), "model_training": model.training,
        "use_safetensors": True, "trust_remote_code": False, "local_files_only": True,
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "inference_timing_scope": "Model forward only; includes failed forward time when measurable",
        "postprocessing": "None: no trimming, resampling or loudness normalization",
        "manual_listening": "pending; technical checks do not establish intelligibility or naturalness",
        "batch_script_sha256": sha256(Path(__file__)),
        "smoke_script_sha256": sha256(PROJECT_ROOT / "scripts/run_mms_tts_cpu_smoke.py"),
    }
    create_output(OUTPUT)
    write_json(OUTPUT / "config.json", {**common, "seed_rule": "1234 + zero-based CSV row index; set immediately before each forward",
                                      "candidate_seeds": seed_map(rows), "selection_seed_separate": 42,
                                      "audio_path_base": "Directory containing results.csv and listening_review.csv",
                                      "load_seconds": load_seconds, "protected_inputs_sha256": before,
                                      "retry_policy": "One attempt per row; retain failures and partial outputs; no regeneration"})
    (OUTPUT / "LISTENING_GUIDE.md").write_text(
        "# D\u00ebgjimi manual i batch-it\n\n"
        "T\u00eb 50 rreshtat fillojn\u00eb pending; numri q\u00eb do t\u00eb d\u00ebgjohet ende nuk \u00ebsht\u00eb vendosur.\n\n"
        "- 2 = e qart\u00eb, kuptohet pa u mb\u00ebshtetur te teksti.\n"
        "- 1 = pjes\u00ebrisht e qart\u00eb, disa fjal\u00eb k\u00ebrkojn\u00eb tekstin.\n"
        "- 0 = e pakuptueshme ose mungon t\u00eb folurit.\n\n"
        "Shqiptimi, fjal\u00ebt e munguara/p\u00ebrs\u00ebritura dhe natyrshm\u00ebria sh\u00ebnohen ve\u00e7mas.\n"
        "audio_path \u00ebsht\u00eb relativ ndaj k\u00ebtij folderi. Kontrollet teknike nuk jan\u00eb vler\u00ebsim d\u00ebgjimor.\n"
        "Vler\u00ebsimi i p\u00ebrdoruesit p\u00ebr smoke test-in ishte: \u201cshum\u00eb mir\u00eb d\u00ebgjohet dhe kuptohet\u201d.\n"
        "Ky vler\u00ebsim nuk transferohet te audioja e batch-it.\n", encoding="utf-8")
    results = generate_batch(rows, OUTPUT, common, model, tokenizer)
    durations = [item["duration_seconds"] for item in results if item["generation_status"] == "success"]
    summary = {
        "total_inputs": len(rows), "statuses": dict(Counter(item["generation_status"] for item in results)),
        "total_inference_seconds": sum(item["inference_seconds"] or 0 for item in results),
        "successful_audio_duration_seconds": {"total": sum(durations), "min": min(durations),
            "mean": statistics.mean(durations), "median": statistics.median(durations), "max": max(durations)} if durations else None,
        "manual_review_status": "50 pending; no ratings entered; listening count not decided",
        "protected_inputs_unchanged": protected_hashes() == before,
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["protected_inputs_unchanged"]:
        raise RuntimeError("Protected inputs changed; preserve batch and investigate without regenerating.")


if __name__ == "__main__":
    run()
