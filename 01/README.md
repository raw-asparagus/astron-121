# Lab 1: SDR and Mixers

This directory contains the report notebook and an installable Python package for Lab 1.

## Quick Start

```bash
cd labs/01
python -m pip install -e .
python -m pip install -r requirements-dev.txt
pytest
```

## Structure

- `01.ipynb`: polished lab report notebook
- `src/ugradio_lab1`: installable package code
- `scripts/`: command-line scripts for simulation/acquisition/analysis/plotting
- `data/`: raw/interim/processed data and manifests
- `tests/`: unit, integration, and hardware tests
- `docs/requirements/allmixers_traceability.md`: mapping to manual requirements
