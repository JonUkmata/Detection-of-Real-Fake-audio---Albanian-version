
# Project State — Albanian Real/Fake Audio Detection

Last Updated: 2026-11-11

## Project Overview

Bachelor's thesis project focused on detecting AI-generated
versus real human speech in the Albanian language using
Deep Learning.

## Current Phase

Phase 1 — Dataset Collection & Analysis

Status: MMS-TTS pilot implementation, technical verification and the author's
manual listening review are complete: 50 reviewed, 47 ratings of 2 and 3 of 1.
Phase 1 remains in progress. Real-audio validation remains pending. Review the
pilot results with the user and mentor before deciding the next scientific
step. Do not start new experiments automatically.

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
- Initially executed all 9 notebook code cells using the actual `.venv313` interpreter
  (Python 3.13.7, Matplotlib 3.10.7). The saved runtime output identifies it.
- Added selected plotting/notebook dependency pins to the existing
  `requirements.txt`; no dependencies were installed during this milestone
  finalization. Installed IPykernel: 7.4.0; IPython: 9.18.1; nbformat: 5.11.1;
  jupyter-client: 8.10.0. Notebook execution used jupyter-client; nbclient is
  not installed and was not needed.
- Extended the notebook with normalized unique-text length analysis: 6,355
  nonempty unique texts, no empty texts; min/mean/median/max 1/8.456/9/14
  (mean rounded), P25/P75 6/11. Length is whitespace-unit count; quartiles use
  inclusive linear interpolation. The notebook now has five scientific plots.



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
- Preserve raw data and the reference audit. Fifty pilot texts are now approved;
  50 synthetic recordings have been generated in a separate pilot batch.
  No new splits or classifier experiments have been created.

## MMS-TTS Pilot Text Approval — 2026-11-05

- Recorded the user's explicit text-review decisions in the ignored local
  `data/interim/mms_tts_pilot_candidates/candidates.csv`: short ranks 1-15,
  medium ranks 1-20, and long ranks 1-15 are approved. The remaining 30 are
  pending, not rejected and not individually reviewed. Exclusion reasons remain blank.
- Saved the exact original CSV as
  `candidates.before_text_approval.2026-11-05.csv` before updating it.
- Created local `selected_texts.csv` with only the 50 approved rows, preserving
  source columns and order. Local `review_decisions.md` records review scope,
  limitations and file hashes. All these local files remain outside Git.
- Approved quotas: 15 short (2-6 whitespace units), 20 medium (7-11), and
  15 long (12-14). The existing seed-42 candidate ranking was followed;
  no reserve replacement or new random selection was needed.
- This is text approval for the synthesis pilot, not verification of real audio.
  Understandable sentences, titles, phrases and names are accepted.
- Texts remain uncorrected, including grammatical inaccuracies, missing
  diacritics and unmatched quotation marks. This limits interpretation and
  attribution of later TTS errors.
- Pilot texts are planned to be excluded from the thesis final test; no new
  split has been created and final-test isolation has not yet been established.
- Original texts, normalized texts, IDs, order and provenance links are preserved.
  The original generation config and provenance JSON remain unchanged.
- Verification under `.venv313` confirmed exactly 50 approved and 30 pending,
  quotas 15/20/15, 50 unique normalized selected texts, and exact agreement
  between selected rows and approved source rows. The selected groups link to
  51 original recordings. Backup bytes, provenance links, input hashes and
  ignored/untracked status of the local review files were checked.

## MMS-TTS CPU Pilot Batch — 2026-11-05

This section records the generation-time state; the completed listening review
is recorded separately below.

- Implemented `scripts/run_mms_tts_pilot_batch.py`, reusing WAV checks and
  observed-text-edit diagnostics from `scripts/run_mms_tts_cpu_smoke.py`.
  Added synthetic-fixture tests in `tests/test_run_mms_tts_pilot_batch.py`.
- Validated all 50 approved source rows, unique IDs and normalized texts,
  quotas 15/20/15, length bounds and the SHA-256 of `selected_texts.csv`.
  Original texts and CSV order were preserved.
- Executed once with `.venv313`, Python 3.13.7, PyTorch 2.14.1+cpu and
  Transformers 4.57.6. Loaded `facebook/mms-tts-sqi` and its tokenizer once
  from the local cache, both at revision
  `1ff16df5f391884d5ea36627298acb2e68b45be2`.
- Used CPU/float32, safetensors, `trust_remote_code=False`, `model.eval()`
  and `torch.inference_mode()`. Checkpoint parameters remained unchanged:
  speaking rate 1.0, noise scale 0.667, duration noise scale 0.8.
- Seeds are `1234 + zero-based selected CSV row index` (1234-1283), fixed
  immediately before each inference and recorded by candidate ID in the local
  config. This is separate from seed 42 used for candidate selection.
- Results: **50 successful generations, 0 failures**. Total model-forward
  inference time: **118.186 seconds**, excluding loading, tokenization and I/O.
- Outputs are under ignored local `data/interim/mms_tts_pilot_batch_001/`:
  per-candidate WAV, metadata and tokenizer diagnostics; shared `config.json`,
  `results.csv`, `summary.json`, `verification.json`, `listening_review.csv`
  and `LISTENING_GUIDE.md`. The previous smoke outputs remain separate.
- All 50 WAVs were reread and verified as mono float32 at 16,000 Hz, with
  expected sample counts, finite nonempty/nonzero signals and matching hashes.
  No trimming, resampling, loudness normalization or other signal processing
  was applied. Maximum peak amplitude: 0.280033. The sum of `samples_abs_ge_1`
  in the existing `results.csv` is **0**, meaning zero samples with `abs(x) >= 1`.
- Audio durations: total 218.384 seconds; minimum 1.184, mean 4.36768,
  median 4.784, maximum 7.728 seconds. These are technical observations,
  not evidence of intelligibility or naturalness.
- At generation time all 50 listening rows were `pending` in source order,
  with manual ratings and notes blank. Listening was initially deferred.
  Scale: 2 = clear without relying
  on the text; 1 = partly clear, some words require the text; 0 = unintelligible
  or speech absent. Pronunciation, missing/repeated words and naturalness are
  documented separately from intelligibility.
- Separately, the user's assessment of the earlier smoke test was:
  “shumë mirë dëgjohet dhe kuptohet”. This is a user-reported smoke assessment;
  it is not transferred to any batch recording or entered as a batch rating.
- All 29 tests passed, including input validation, seed mapping/order,
  overwrite protection, and preserving successful/partial outputs on failure.
  Independent checks confirmed 50 terminal generation statuses and 50 pending
  listening rows. Input CSVs, provenance, raw TSV hashes, requirements and both
  smoke runs were unchanged. Outputs and checkpoint cache remain outside Git.
- The batch was not rerun for testing. No split creation, classifier training,
  requirements change, commit or push was performed.

## Pilot Milestone Closure — 2026-11-05

Historical implementation/technical closure, before the manual batch review.

- Reviewed the notebook, candidate preparation, smoke/batch scripts, tests and
  existing verification artifacts without regenerating speech or editing data.
  Summary: `reports/mms_tts_pilot_milestone.md`.
- Added seven already-installed MMS-TTS pins to the existing requirements:
  Transformers 4.57.6, huggingface_hub 0.36.0, tokenizers 0.22.2,
  safetensors 0.7.0, PyYAML 6.0.3, regex 2025.11.3 and tqdm 4.67.1.
  Existing pins were preserved; no package was installed or upgraded.
- Historical inventories show 75 -> 82 packages with no existing package
  changed or removed. Current `pip check` passes. All 29 tests pass under
  `.venv313`, and notebook schema validation passes. A clean rebuild remains untested.
- Raw environment inventories and resolver reports contain absolute machine
  paths and are now excluded by `reports/environment/` in `.gitignore`.
  The milestone report contains only aggregate evidence and relative paths.
  The notebook's saved interpreter-path line was made project-relative; no
  scientific code/results were changed and no notebook cells were rerun.
- At that closure, the 50 selected text groups linked to 51 original recording rows, without
  verification of those real recordings. Text inaccuracies remain unchanged.
  This is one model/one declared synthetic voice, with no listening assessment
  of the batch, no classifier experiment and no final dataset split.
- Closure SHA-256 tree comparisons confirmed all 6,564 raw files and all files
  in the candidate, both smoke, batch and checkpoint-cache folders unchanged.
  `git diff --check` passed. The proposed commit files contain no absolute
  machine paths, individual pilot texts or actual speaker identifiers.
- Personal-note entries are supplied in the review response only; no repository
  copies of development-journal.md, technical-decisions.md, experiments.md or
  references.md were created. No Git staging, commit or push was performed.

## Completed Manual Listening Review — verified 2026-11-11

- Read the current local `listening_review.csv`: 50 unique candidate IDs,
  all 50 `reviewed`; 47 ratings of 2 (94%), 3 ratings of 1 (6%), no 0 or blank
  ratings. All three note fields are populated for every recording.
- The evaluator is the project author. The 94% is the proportion assigned
  intelligibility 2 in this pilot, not model/classifier accuracy. Some recordings
  rated 2 still have local pronunciation issues. This is neither MOS nor a
  multi-listener assessment; actual listening/text-reveal order is not proven
  by the CSV. These are subjective listening observations, not objective error rates.
- Confirmed the author's corrections: short-009 has no missing word;
  long-002 records unusual stress; short-008 records an overly short vowel as
  a pronunciation problem, without a missing word. Individual source texts and
  detailed notes remain local and are not reproduced here.
- Verified the eight columns, unique IDs and source order against
  `selected_texts.csv`, and audio paths/generation statuses against `results.csv`.
  Selected original texts agree with per-recording metadata. The reviewed CSV
  SHA-256 is `e83164665011db994ba9ed5caa87c85158e201c26191f66b28fe649c6fbf2516`.
- No audio was regenerated to improve the listening results. Generation-time
  `summary.json`, `verification.json`, metadata and listening guide retain their historical pending
  state; the current review CSV is the source for completed listening results.
  The local HTML page also embeds the initial pending state; browser progress
  and CSV exports are separate from the source CSV, not synchronized to it.
- Current checks: all 29 unit tests pass under `.venv313`; `pip check` and
  nbformat schema validation pass. No notebook cells or model inference were
  rerun. Clean-environment rebuilding and additional independent listening
  remain untested. The next scientific step remains for discussion.

## Authorized Local Commit Preparation — 2026-11-11

- Verified the existing `albanian-voice-venv313` kernel.json and launched its
  interpreter: both point to the project's `.venv313`, Python 3.13.7. Restored
  only the notebook kernelspec name/display name; no absolute interpreter path
  was embedded. Before/after comparison confirmed identical cells and outputs.
- nbformat validation and `git diff --check` pass. The prior 29/29 unit tests
  and passing `pip check` were not repeated for metadata/documentation changes.
- Known batch I/O limitation remains unfixed: directory creation and metadata
  writing are outside the per-recording error handler, so filesystem errors
  there can interrupt the batch. Review before its next use; no refactor was made.
- The user authorized three explicit local commit groups: notebook; pilot
  scripts/tests, pins and ignore rule; project documentation. No push or new
  scientific experiment is authorized. Data and local review artifacts remain
  excluded from these commits.

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

- At the original audit milestone, all 10 audit tests passed under `.venv313` using:
  `.venv313/Scripts/python.exe -B -m unittest discover -s tests -v`.
- All notebook code cells executed successfully using the registered
  `albanian-voice-venv313` kernel. Audit comparisons, input-hash guards,
  duration coverage and histogram checks passed.
- Notebook schema validation with nbformat passed. Current saved outputs contain five
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

Listening is complete. The user authorized the three reviewed local commits.
After local commit verification, wait for discussion with the user and mentor.
Do not push or start experiments.

## Return Point

Përdoruesi e ka përfunduar dëgjimin e 50 audios dhe listening_review.csv.
Pas shqyrtimit të rezultateve, ndryshimeve dhe përditësimit të shënimeve,
diskutohet hapi tjetër me mentorin. Mos nis automatikisht eksperimente të reja.

## Next Steps

1. Review the completed listening results and proposed commit files with the mentor.
2. Update personal thesis notes outside the repository; discuss the next scientific step.
3. Decide the real/synthetic collection strategy and evaluation protocol.
4. Design speaker-independent evaluation with normalized-text grouping before
   generating speech or creating thesis-specific splits.

## Open Questions

- Which real and synthetic audio sources will be included?
- Which Deep Learning models will be compared?
- What evaluation protocol will be used?
- What application architecture will be finalized?
