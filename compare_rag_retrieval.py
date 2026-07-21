import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.spherical_embedding import SphericalEmbeddingModel


def load_large_corpus():
    """Load the large corpus from file"""
    with open("data/large_corpus.txt", "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]
    return corpus


def main():
    print("=== Loading Large Corpus ===")
    corpus = load_large_corpus()
    print(f"Loaded {len(corpus)} documents")
    
    print("\n=== Loading all-MiniLM-L6-v2 Teacher Model ===")
    teacher_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("\n=== Loading Trained Spherical Embedding Model ===")
    spherical_model = SphericalEmbeddingModel.from_pretrained("models/spherical_embedding_model.pt")
    spherical_model.eval()
    
    print("\n=== Generating Embeddings ===")
    # Generate 384D embeddings using teacher model
    embeddings_384d = teacher_model.encode(corpus, convert_to_numpy=True).astype(np.float32)
    print(f"Generated 384D embeddings: shape {embeddings_384d.shape}")
    
    # Generate spherical coordinates and convert to 3D Cartesian
    with torch.no_grad():
        r, theta, phi, _, _ = spherical_model(corpus)
        cartesian_3d = spherical_model.spherical_to_cartesian(r, theta, phi)
        cartesian_3d = torch.nn.functional.normalize(cartesian_3d, p=2, dim=1).numpy().astype(np.float32)
    print(f"Generated spherical 3D Cartesian embeddings: shape {cartesian_3d.shape}")
    
    print("\n=== Building FAISS Indexes ===")
    # Build FAISS index for 384D
    index_384d = faiss.IndexFlatL2(384)
    index_384d.add(embeddings_384d)
    
    # Build FAISS index for 3D
    index_3d = faiss.IndexFlatL2(3)
    index_3d.add(cartesian_3d)
    print("Indexes built successfully!")
    
    print("\n=== Testing RAG Retrieval (TOP 20) ===")
    test_queries = [
        "Tell me about machine learning and artificial intelligence",
        "What is the best way to stay healthy and fit?",
        "How do I start a business or startup?",
        "What are popular travel destinations?",
        "Tell me about space and astronomy"
    ]
    
    top_k = 20  # Top 20 results
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # Get query embeddings
        query_embedding_384d = teacher_model.encode([query], convert_to_numpy=True).astype(np.float32)
        
        with torch.no_grad():
            q_r, q_theta, q_phi, _, _ = spherical_model([query])
            query_embedding_3d = spherical_model.spherical_to_cartesian(q_r, q_theta, q_phi)
            query_embedding_3d = torch.nn.functional.normalize(query_embedding_3d, p=2, dim=1).numpy().astype(np.float32)
        
        # Retrieve from 384D index
        distances_384d, indices_384d = index_384d.search(query_embedding_384d, k=top_k)
        
        # Retrieve from 3D index
        distances_3d, indices_3d = index_3d.search(query_embedding_3d, k=top_k)
        
        print(f"\n--- all-MiniLM-L6-v2 (384D) Top {top_k} Results ---")
        for i, idx in enumerate(indices_384d[0]):
            print(f"  {i+1:2d}. Score: {distances_384d[0][i]:.4f} - {corpus[idx]}")
        
        print(f"\n--- Spherical Coordinates (3D) Top {top_k} Results ---")
        for i, idx in enumerate(indices_3d[0]):
            print(f"  {i+1:2d}. Score: {distances_3d[0][i]:.4f} - {corpus[idx]}")


if __name__ == "__main__":
    main()
