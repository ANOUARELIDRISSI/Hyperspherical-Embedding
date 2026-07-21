import os

from src.real_data_loader import load_real_corpus
from src.distillation import DistillationTrainer


def main():
    print("=== SMALL TRAINING TEST ===")
    print("=" * 40)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    training_texts = load_real_corpus(500)  # Small corpus
    print(f"Loaded {len(training_texts)} training texts")
    threads = max(1, (os.cpu_count() or 2) - 1)
    trainer = DistillationTrainer(num_threads=threads)
    trainer.train(
        training_texts,
        num_epochs=2,
        batch_size=16,
        learning_rate=1e-3,
        pair_workers=-1,
        pairs_per_epoch=500,
    )
    print("\n=== TEST FINISHED ===")


if __name__ == "__main__":
    main()
