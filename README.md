# SphereEmbedding: Spherical Coordinate Embedding Model with Distillation

## Overview
This project creates a lightweight embedding model that maps text into spherical coordinates `(r, theta, phi)`, distilled from `sentence-transformers/all-MiniLM-L6-v2` for efficient Retrieval-Augmented Generation (RAG) on CPU and edge devices.

## Installation
Install dependencies using uv:
```bash
uv sync
```

## Project Structure
```text
E:\SphereEmbedding
|-- src/
|   |-- spherical_embedding.py  # Spherical coordinate embedding model
|   |-- distillation.py         # Knowledge distillation trainer
|   |-- real_data_loader.py     # Synthetic corpus generation
|   `-- rag.py                  # RAG system with FAISS
|-- tests/
|-- data/
|-- models/
|-- train.py
|-- train_small.py
|-- pyproject.toml
`-- README.md
```

## Quick Start
Run the automated tests:
```bash
uv run pytest
```

Run a small CPU training smoke test:
```bash
uv run python train_small.py
```

Run full training:
```bash
uv run python train.py --threads 7 --pair-workers -1
```

Compare teacher retrieval against the spherical model manually:
```bash
uv run python compare_rag_retrieval.py
```

Run the chunked dual-FAISS benchmark:
```bash
uv run python benchmark_chunked_rag.py --sentences-per-topic 24 --top-k 5
```

Useful training options:
```bash
uv run python train.py --sentences 10000 --epochs 10 --batch-size 64 --pairs-per-epoch 10000 --skip-rag-test
```

Train the 3D model with labeled topic separation:
```bash
uv run python train.py --device auto --sentences 128 --epochs 14 --batch-size 32 --pairs-per-epoch 512 --skip-rag-test --supervised-topics
```

Longer supervised run with checkpoints:
```bash
uv run python train.py --device auto --sentences 512 --epochs 40 --batch-size 64 --pairs-per-epoch 1024 --threads 7 --pair-workers -1 --skip-rag-test --supervised-topics --checkpoint-dir models/checkpoints --checkpoint-every 10
```

Train 3D as a retrieval ranking model with teacher top-k neighborhoods and mined hard negatives:
```bash
uv run python train.py --device auto --sentences 256 --epochs 30 --batch-size 32 --pairs-per-epoch 512 --threads 7 --skip-rag-test --supervised-topics --retrieval-distillation --teacher-top-k 8 --mined-hard-negatives
```

`--prototype-loss` is available for experiments, but the best observed compact run did not use it.

## Notes
- Training now mines positives and hard negatives with TF-IDF nearest neighbors instead of using random adjacent sentences.
- The frozen MiniLM base model is reused as the teacher signal during each student forward pass, avoiding a second teacher encode per batch.
- PyTorch CPU thread count is configurable with `--threads`; semantic pair mining uses scikit-learn workers via `--pair-workers`.

## Key Components
- Teacher model: `sentence-transformers/all-MiniLM-L6-v2`
- Student model: custom spherical coordinate encoder and decoder
- RAG system: FAISS search over spherical coordinates converted to 3D Cartesian vectors

## Documentation
For detailed information about the distillation process, see [DISTILLATION.md](DISTILLATION.md).
