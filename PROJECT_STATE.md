
# Project State — Albanian Real/Fake Audio Detection

Last Updated: 2026-11-02

## Project Overview

Bachelor's thesis project focused on detecting AI-generated
versus real human speech in the Albanian language using
Deep Learning.

## Current Phase

Phase 1 — Dataset Collection & Analysis

Status: In progress. Metadata audit and exploration milestone verified;
awaiting review before further dataset selection or implementation.

## Completed

- Created and cloned the GitHub repository.
- Established AGENTS.md and the documentation workflow.
- Created a separate personal thesis development journal.
- Configured Python 3.13.7 using .venv313.
- Configured .gitignore and requirements.txt.
- Installed and verified Librosa 1.0.0.
- Installed and verified PyTorch 2.14.1+cpu.
- Verified dependencies using pip check.
- Completed and pushed the initial setup commits.
- Implemented the read-only, standard-library metadata audit in
  `scripts/audit_common_voice.py`, with 10 synthetic-fixture unit tests in
  `tests/test_audit_common_voice.py`.
- Saved the reference audit in `reports/common_voice_27_sq_metadata_audit.json`.
  It records input TSV SHA-256 hashes, Python version, parser settings,
  counts, speaker concentration and split overlaps.
- Created `notebooks/01_dataset_exploration.ipynb`, reusing the audit module
  and reference JSON, with explanations, aggregate tables and four Matplotlib
  plots: supplied split sizes, validated top-10 speaker contributions,
  training speaker contributions and validated duration distribution.
- Executed all 9 notebook code cells using the actual `.venv313` interpreter
  (Python 3.13.7, Matplotlib 3.10.7). The saved runtime output identifies it.
- Added selected plotting/notebook dependency pins to the existing
  `requirements.txt`; no dependencies were installed during this milestone
  finalization. Installed IPykernel: 7.4.0; IPython: 9.18.1; nbformat: 5.11.1;
  jupyter-client: 8.10.0. Notebook execution used jupyter-client; nbclient is
  not installed and was not needed.



## Technology Stack (Planned)

- Python 3.13
- PyTorch
- Librosa / Torchaudio
- Pandas / NumPy
- Scikit-learn
- FastAPI
- React
- Git / GitHub

## Key Decisions

- Python 3.13.7 selected as the primary development environment.
- Python 3.11 retained as a fallback.
- PyTorch 2.14.1+cpu verified for basic CPU operations.
- Local NVIDIA GPU support has not been validated.
- Google Colab GPU is the proposed training environment
  for computationally expensive experiments.
- Maintain shared model and preprocessing code between
  local development and Google Colab.
- Save trained model checkpoints for local inference.
- Use Python modules for reusable audit/parsing logic and notebooks for
  educational exploration and presentation. Retain necessary scientific checks.
- Preserve raw data and the reference audit. No pilot sentences, new splits,
  synthetic recordings or classifier experiments have been created.

## Verified Metadata Findings

These observations apply to the current local Common Voice 27.0 Albanian
snapshot (`cv-corpus-27.0-2026-09-11`), discovered beneath
`data/raw/common_voice_27_sq/`.

- `validated.tsv`: 6,374 rows, 6,374 unique paths, 147 nonempty speaker IDs,
  and 6,364 exact unique texts.
- Supplied train/dev/test sizes: 2,658 / 1,795 / 1,911 recordings, with
  2 / 7 / 138 speaker IDs. The largest training speaker contributes
  2,009 recordings (75.6%); the other contributes 649.
- Pairwise exact path, speaker-ID and text overlaps are zero. After Unicode
  NFC, case folding and whitespace collapsing, shared unique text counts are
  4 (train/dev), 2 (train/test), and 1 (dev/test). Punctuation and diacritics
  remain significant. These checks do not establish comprehensive leakage safety.
- Ten validated recordings are outside the supplied split union.
- Duration metadata covers 6,553 recordings (9.263 hours); the validated
  subset totals 8.988 hours. These are metadata durations, not decoded audio.
- The 6,374 versus 6,344 discrepancy is reproduced on the current bytes:
  default quote-aware CSV parsing merges physical lines 4,205-4,235 into one
  record, losing 30 rows from the count. Clip manifests require literal quotes
  (`csv.QUOTE_NONE`); sentence-corpus/report tables use `csv.QUOTE_MINIMAL`.
  An earlier file hash is unavailable, so historical byte identity is unproven.

## Verification and Limitations

- All 10 existing tests passed under `.venv313` using:
  `.venv313/Scripts/python.exe -B -m unittest discover -s tests -v`.
- All notebook code cells executed successfully using the registered
  `albanian-voice-venv313` kernel. Audit comparisons, input-hash guards,
  duration coverage and histogram checks passed.
- Notebook schema validation with nbformat passed. Saved outputs contain four
  compact plots and no cell errors, audio or speaker identifiers.
- Before/after SHA-256 checks confirmed all 6,564 raw files (including audio)
  unchanged, along with the existing audit script, tests and reference JSON.
- `git diff --check` passed; no commit or push was made.
- `pip check` reported no broken requirements. Dependency pins describe
  the relevant installed packages; a clean environment rebuild has not been tested.
- Metadata analysis does not verify audio quality, decoding, sample rate,
  waveform duplication or human origin. Speaker IDs are not verified identities.
- Demographic metadata is incomplete/self-declared. Speaker and source/domain
  imbalance limit representativeness; the notebook does not recompute the earlier
  source join. No classifier performance or generalization has been measured.
- Personal thesis notes remain outside the repository; documentation handoff
  is supplied for review rather than creating repository copies.



## Current Task

Review the verified metadata audit and exploration milestone before proceeding
to dataset selection or additional implementation. Changes remain uncommitted.

## Next Steps

1. Review the notebook, audit evidence and personal documentation handoff.
2. Agree on audio-quality checks and later selection criteria, including speaker,
   source/domain and duration coverage.
3. Decide the real/synthetic collection strategy and evaluation protocol.
4. Design speaker-independent evaluation with normalized-text grouping before
   generating speech or creating thesis-specific splits.

## Open Questions

- Which real and synthetic audio sources will be included?
- Which Deep Learning models will be compared?
- What evaluation protocol will be used?
- What application architecture will be finalized?
