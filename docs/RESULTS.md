# 3D Retrieval Distillation Results

Best observed compact run:

```bash
uv run python -u train.py --sentences 160 --epochs 12 --batch-size 16 --pairs-per-epoch 256 --threads 7 --skip-rag-test --supervised-topics --retrieval-distillation --teacher-top-k 8 --mined-hard-negatives
```

Chunked dual-FAISS benchmark:

| Method | P@5 | MRR |
| --- | ---: | ---: |
| 384D MiniLM teacher | 1.000 | 1.000 |
| Initial 3D smoke checkpoint | 0.467 | 0.569 |
| Pairwise + supervised topics | 0.717 | 0.781 |
| Retrieval distillation + mined negatives | 0.817 | 0.883 |
| Residual separate-head encoder + retrieval distillation | 0.967 | 1.000 |
| Longer retrieval run, 256 sentences | 0.725 | 0.743 |
| Retrieval distillation + prototype loss | 0.692 | 0.719 |

Takeaway: teacher top-k neighborhoods, listwise KL ranking, mined hard negatives, and a stronger residual encoder with separate radius/angle heads gave the best 3D retrieval quality. Prototype loss is implemented but was not beneficial in the compact test.
