
# Project State — Albanian Real/Fake Audio Detection

Last Updated: 2026-10-28

## Project Overview

Bachelor's thesis project focused on detecting AI-generated
versus real human speech in the Albanian language using
Deep Learning.

## Current Phase

Phase 1 — Dataset Collection & Analysis

Status: Ready to begin.

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



## Current Task

Design the initial dataset workflow and establish the
source code structure for audio data exploration.

## Next Steps

1. Define the dataset sources and initial collection strategy.
2. Create the initial project folder structure.
3. Implement audio loading and metadata validation.
4. Perform exploratory data analysis.
5. Design speaker-independent, leakage-safe dataset splits.

## Open Questions

- Which real and synthetic audio sources will be included?
- Which Deep Learning models will be compared?
- What evaluation protocol will be used?
- What application architecture will be finalized?
