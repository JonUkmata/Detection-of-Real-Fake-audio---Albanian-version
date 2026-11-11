# MMS-TTS pilot: milestone review

Implementation/technical closure was reviewed on 2026-11-05, when listening was
deferred. Updated on 2026-11-11 after verifying the author's completed manual
review of all 50 recordings. No audio was regenerated to improve these results.

## Evidence and text selection

- The [exploration notebook](../notebooks/01_dataset_exploration.ipynb) records
  6,355 unique normalized validated texts, with no empty texts. Length is
  `len(text.split())` after NFC, casefold and whitespace collapse; punctuation
  and Albanian diacritics are retained. Each normalized text counts once.
  Lengths: minimum 1, mean 8.456 (rounded), median 9, maximum 14, P25 6 and
  P75 11. Quartiles use `statistics.quantiles(..., method="inclusive")`, with
  linear interpolation at zero-based position `(n - 1) * p`.
- The candidate script samples sorted normalized groups with one
  `random.Random(42)`, in short/medium/long order. It prepared 25/30/25 review
  candidates; the user approved 15/20/15, with bounds 2-6/7-11/12-14 whitespace
  units. No reserve replacement was needed. The other 30 remain pending.
- `selected_texts.csv` contains 50 approved, unique normalized texts linked by
  the unchanged provenance file to 51 real recording rows. This is a metadata
  link, not a claim that the real recordings were listened to or validated.
- Original texts remain uncorrected, including grammatical inaccuracies,
  missing diacritics and unmatched quotation marks. Understandable sentences,
  titles, phrases and names were accepted for the pilot.

## Environment and generation

Python 3.13.7 in `.venv313`, PyTorch 2.14.1+cpu, CPU/float32. The seven installed
MMS-TTS packages are now pinned in the existing [requirements.txt](../requirements.txt);
previous pins were retained. Historical installation reports show 75 -> 82
packages, with no existing version changed or removed. Current `pip check`
passes. A clean-environment rebuild has **not** been tested.

Model: `facebook/mms-tts-sqi`; model and tokenizer revision:
`1ff16df5f391884d5ea36627298acb2e68b45be2`. The batch loads them once from the
local cache, uses safetensors, `trust_remote_code=False`, `model.eval()` and
`torch.inference_mode()`. Inference seeds are 1234-1283 (`1234 + CSV row index`),
separate from selection seed 42. Checkpoint speaking rate 1.0, noise scale
0.667 and duration noise scale 0.8 are unchanged. There is no trimming,
resampling or loudness normalization. The checkpoint declares one speaker.

## Recorded batch results

The local `results.csv` has 50 terminal statuses: 50 successes and 0 failures.
Its totals agree with `summary.json` and `verification.json`.

| Technical observation | Value |
|---|---:|
| Total model-forward inference time, seconds | 118.1862453 |
| Total audio duration, seconds | 218.384 |
| Duration minimum / mean / median / maximum, seconds | 1.184 / 4.36768 / 4.784 / 7.728 |
| Maximum peak amplitude | 0.280032933 |
| Sum of samples satisfying `abs(x) >= 1`, from `results.csv` | 0 |
| WAV files technically verified | 50 |

The prior verification records mono float32 WAVs at 16,000 Hz, matching sample
counts and SHA-256 values, finite nonempty/nonzero signals, and decoded tokenizer
text matching the prepared text. Smoke decoding was corrected to retain special
blank tokens until VitsTokenizer removes them internally; the original smoke
outputs were preserved. These checks do not measure intelligibility or naturalness.

## Listening, limitations and return point

At generation and the 2026-11-05 technical closure, all 50 listening rows were
pending and ratings/notes were blank. The current local `listening_review.csv`
now records the completed assessment by one evaluator: the project author.

| Current manual observation | Count | Percentage |
|---|---:|---:|
| Reviewed recordings | 50 | 100% |
| Intelligibility 2 | 47 | 94% |
| Intelligibility 1 | 3 | 6% |
| Intelligibility 0 | 0 | 0% |
| Blank ratings / pending reviews | 0 | 0% |

The scale is 2 = clear without relying on text; 1 = partly clear, some words
require text; 0 = unintelligible or speech absent. All three separate note fields
(pronunciation, missing/repeated words and naturalness) are populated for all 50.
The 94% is the proportion assigned intelligibility 2, **not accuracy**. Some
recordings assigned 2 still have local pronunciation problems. This is not MOS
or a multi-listener evaluation. The CSV does not prove the actual order of
listening and revealing the source text, and note contents have not been
systematically coded into error frequencies.

The author's corrections were checked in the CSV: short-009 records no missing
word; long-002 records unusual stress; short-008 records an overly short vowel
as a pronunciation issue, with no missing word. No individual source sentences
or detailed listening notes are published in this report.

Verification confirmed 50 unique IDs, the eight expected columns and the same
order as `selected_texts.csv`. Audio paths and generation statuses match
`results.csv`; original selected texts match recording metadata. Current review
CSV SHA-256: `e83164665011db994ba9ed5caa87c85158e201c26191f66b28fe649c6fbf2516`.
The CSV and all generation artifacts were read only during this review.

The user's positive smoke-test assessment was “shumë mirë dëgjohet dhe kuptohet”.
It applies only to the smoke test and is not transferred to batch recordings.
The pilot covers one model/one declared synthetic voice and a small text set
with retained inaccuracies. Listening results are subjective observations from
one author/evaluator; no classifier result or final train/dev/test split exists.
Pilot texts are planned outside the final
test; that isolation has not been implemented or verified.

## Local evidence and closure checks

The tree comparisons and generation checks below refer to the 2026-11-05
closure. The review CSV was subsequently updated by the author. Historical
`summary.json`, `verification.json`, metadata and the guide keep their generation-time pending state;
use the current CSV for listening results. The ignored HTML page embeds the
initial review data; browser storage/export and the source CSV do not synchronize.

- Selection: `data/interim/mms_tts_pilot_candidates/` contains the generation
  config, reviewed candidates, selection, backup, review record and provenance.
- Batch: `data/interim/mms_tts_pilot_batch_001/` contains per-text WAVs/metadata/
  tokenizer diagnostics, shared config, technical results, summary, verification,
  listening CSV and guide. Both smoke runs remain separate under `data/interim/`.
- Environment evidence remains local under `reports/environment/`, excluded
  from Git because raw inventories/resolver reports contain machine paths and
  extensive third-party metadata. This report intentionally contains no individual
  source texts, speaker identifiers or absolute machine paths.
- Closure checks: all 29 synthetic-fixture tests pass; notebook schema is valid;
  existing dependency pins are unchanged and all pins match installed metadata.
  The saved notebook interpreter path was made project-relative for publication;
  scientific cells, plots and results were preserved without rerunning them.
- Data/audio/text/provenance/cache remain ignored and untracked. No package
  installation, speech generation, data editing, Git staging, commit or push
  is part of this closure. Personal note files were not created or updated here.
- Complete before/after SHA-256 tree comparisons matched for all 6,564 raw
  files and for the candidate, both smoke, batch and checkpoint-cache folders.
  `git diff --check` passed; the proposed 11 files were checked for absolute
  machine paths, actual pilot texts and speaker identifiers, with none found.

Current checks (2026-11-11): 29 tests passed under `.venv313`, `pip check` passed
and nbformat schema validation passed. Notebook cells and model inference were
not rerun; no packages were installed. The notebook's generic `python3`
kernelspec was identified during review and subsequently restored to the verified
`albanian-voice-venv313` kernel. Its kernel.json and launched Python 3.13.7
interpreter both match the project's `.venv313`. Only kernelspec metadata changed;
all notebook cells, outputs and other metadata remained identical. nbformat and
`git diff --check` passed again; the prior unit tests and pip check were not
repeated for these metadata/documentation changes. Clean rebuilding remains untested.

Known I/O limitation, not corrected: directory creation and metadata writing in
the batch are outside its per-recording error handler. Filesystem errors there
can interrupt the batch. Review this before the next use; no refactor or new
generation was performed. The user authorized three reviewed local commits,
with data/private artifacts excluded and no push.

Përdoruesi e ka përfunduar dëgjimin dhe plotësimin e 50 rreshtave. Pas shqyrtimit
të rezultateve, propozimit të commit-it dhe përditësimit të shënimeve, diskutohet
hapi tjetër me mentorin.
Mos nis automatikisht eksperimente të reja.
