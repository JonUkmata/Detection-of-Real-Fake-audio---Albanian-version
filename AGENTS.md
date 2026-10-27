
# AGENTS.md — Albanian Real/Fake Audio Detection

## Project Overview

This repository contains a Bachelor's thesis project focused on developing a Deep Learning system to distinguish real human speech from AI-generated speech in the Albanian language.

The project has two main goals:
1. Develop a functional and reliable audio classification system.
2. Conduct scientifically meaningful experiments that can support an academic thesis.

## Development Philosophy

This is an educational and research-oriented project.

The student must understand the code, technical decisions, machine learning methods, and experimental results.

Prioritize learning, clarity, reproducibility, and scientific correctness over implementation speed.

## Working Rules

1. Work on one clearly defined task at a time.
2. Before making significant changes, explain the proposed approach.
3. Keep implementations simple, modular, and understandable.
4. Explain important functions, algorithms, and design decisions.
5. Do not introduce unnecessary dependencies or architectural complexity.
6. Do not modify unrelated files.
7. Add appropriate tests for new functionality.
8. Do not create Git commits or push changes unless explicitly requested.
9. Never delete existing datasets, experiments, or results without permission.
10. Ask before making important scientific or architectural decisions.

## Machine Learning and Research Rules

- Do not invent experimental results, metrics, or dataset statistics.
- Avoid data leakage between training, validation, and test sets.
- Consider speaker independence when splitting audio datasets.
- Track dataset origins, licenses, and preprocessing operations.
- Ensure experiments are reproducible whenever practical.
- Record random seeds, configurations, model versions, and evaluation metrics.
- Clearly distinguish proposed methods from experimentally validated results.
- Do not automatically select models based solely on accuracy.
- Explain the reasoning behind preprocessing, feature extraction, and model architecture choices.

## Documentation Rules

- Keep technical documentation concise and accurate.
- Document significant implementation decisions and experiment outcomes.
- Update PROJECT_STATE.md when explicitly requested or when completing a milestone as agreed with the student.
- Do not write fictional experimental observations.
- Personal thesis notes are supporting material, not a finalized academic thesis.
- Preserve references and sources used for research claims.

## Code Quality

- Use Python 3.11 unless the project configuration changes.
- Prefer readable code with meaningful variable and function names.
- Use type hints where appropriate.
- Avoid unnecessary abstractions.
- Use clear error handling.
- Keep data processing, model training, and evaluation logically separated.

## Before Completing a Task

Provide:
1. A summary of changes.
2. An explanation of important implementation details.
3. Tests performed and their results.
4. Any limitations or unresolved questions.
5. Suggested next steps.

The final implementation must remain understandable and maintainable by the student.
