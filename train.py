import argparse
import os

from src.real_data_loader import load_real_corpus
from src.distillation import DistillationTrainer
from src.spherical_embedding import SphericalEmbeddingModel
from src.rag import RAGSystem


def main():
    parser = argparse.ArgumentParser(description="Train the spherical embedding model on CPU.")
    parser.add_argument("--sentences", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--pair-workers", type=int, default=-1)
    parser.add_argument("--pairs-per-epoch", type=int, default=None)
    parser.add_argument("--skip-rag-test", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    print("=== Spherical Embedding RAG Training ===")
    print("=" * 50)
    print(f"CPU threads: {args.threads}")
    
    # Step 1: Load real-world corpus
    print("\n1. Loading training corpus...")
    training_texts = load_real_corpus(total_sentences=args.sentences)
    print(f"Loaded {len(training_texts)} training sentences")
    
    # Step 2: Initialize trainer
    print("\n2. Initializing distillation trainer...")
    trainer = DistillationTrainer(num_threads=args.threads)
    
    # Step 3: Train model
    print("\n3. Starting training with contrastive pairwise objective...")
    trainer.train(
        training_texts,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pair_workers=args.pair_workers,
        pairs_per_epoch=args.pairs_per_epoch,
    )
    
    # Step 4: Save model
    print("\n4. Saving trained model...")
    model_path = "models/spherical_embedding_model.pt"
    trainer.save_student_model(model_path)

    if args.skip_rag_test:
        print("\n=== Training Complete ===")
        return
    
    # Step 5: Quick RAG test
    print("\n5. Testing RAG retrieval...")
    trained_model = SphericalEmbeddingModel.from_pretrained(model_path)
    
    # Create test corpus
    test_texts = load_real_corpus(total_sentences=5000)
    print(f"\nLoaded {len(test_texts)} test documents")
    
    # Test both modes
    rag_spherical = RAGSystem(model=trained_model, mode="spherical")
    rag_spherical.add_documents(test_texts)
    
    # Test queries
    queries = [
        "What is artificial intelligence?",
        "Tell me about sports",
        "What is climate change?",
        "Explain quantum computing",
        "What is music?",
        "Tell me about biology"
    ]
    
    for query in queries:
        print(f"\n=== Query: {query} ===")
        results = rag_spherical.retrieve(query, k=5)
        for i, res in enumerate(results):
            print(f"  {i+1}. {res['document'][:100]}... (distance: {res['distance']:.4f})")
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
