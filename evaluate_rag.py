import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.spherical_embedding import SphericalEmbeddingModel
from datasets import load_dataset


def load_real_world_dataset():
    try:
        dataset = load_dataset("ag_news", trust_remote_code=True)
        texts = dataset["train"]["text"]
        print(f"Loaded {len(texts)} real-world training samples!")
        return texts
    except Exception as e:
        print(f"Error loading AG News: {e}")
        print("Falling back to local synthetic corpus...")
        with open("data/large_corpus.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def compute_recall_at_k(relevant_indices, retrieved_indices, k):
    """Compute Recall@k"""
    retrieved_topk = retrieved_indices[:k]
    hits = len(set(relevant_indices) & set(retrieved_topk))
    return hits / len(relevant_indices)


def compute_mrr(relevant_indices, retrieved_indices):
    """Compute Mean Reciprocal Rank"""
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0


def compute_dcg(relevance_scores, k):
    """Discounted Cumulative Gain at k"""
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        dcg += relevance_scores[i] / np.log2(i + 2)
    return dcg


def compute_ndcg(relevant_indices, retrieved_indices, k):
    """Normalized Discounted Cumulative Gain at k"""
    relevance_scores = [1.0 if idx in relevant_indices else 0.0 for idx in retrieved_indices]
    dcg = compute_dcg(relevance_scores, k)
    ideal_relevance = [1.0] * len(relevant_indices) + [0.0] * (len(retrieved_indices) - len(relevant_indices))
    idcg = compute_dcg(ideal_relevance, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(corpus, query_set, model_name="teacher"):
    """Evaluate retrieval performance for a given embedding model"""
    print(f"\nEvaluating {model_name} model...")
    
    if model_name == "teacher":
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        def embed(texts):
            return embedder.encode(texts, convert_to_numpy=True).astype(np.float32)
    else:
        spherical_model = SphericalEmbeddingModel.from_pretrained("models/spherical_embedding_model.pt")
        spherical_model.eval()
        if model_name == "spherical":
            def embed(texts):
                with torch.no_grad():
                    r, theta, phi, _, _ = spherical_model(texts)
                    return spherical_model.spherical_to_cartesian(r, theta, phi).numpy().astype(np.float32)
        else:  # reconstructed
            def embed(texts):
                with torch.no_grad():
                    _, _, _, _, recon_emb = spherical_model(texts)
                    return recon_emb.numpy().astype(np.float32)
    
    # Build index
    print("Building FAISS index...")
    corpus_embs = embed(corpus)
    index = faiss.IndexFlatL2(corpus_embs.shape[1])
    index.add(corpus_embs)
    
    # Evaluate
    k_list = [1, 5, 10, 20]
    total_recall = {k: 0.0 for k in k_list}
    total_mrr = 0.0
    total_ndcg = {k: 0.0 for k in k_list}
    num_queries = len(query_set)
    
    print("Evaluating queries...")
    for query, relevant_idx in query_set:
        query_emb = embed([query])
        _, indices = index.search(query_emb, max(k_list))
        retrieved = indices[0]
        
        for k in k_list:
            total_recall[k] += compute_recall_at_k([relevant_idx], retrieved, k)
            total_ndcg[k] += compute_ndcg([relevant_idx], retrieved, k)
        total_mrr += compute_mrr([relevant_idx], retrieved)
    
    # Average
    for k in k_list:
        total_recall[k] /= num_queries
        total_ndcg[k] /= num_queries
    total_mrr /= num_queries
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"MRR: {total_mrr:.4f}")
    for k in k_list:
        print(f"Recall@{k}: {total_recall[k]:.4f}")
        print(f"nDCG@{k}: {total_ndcg[k]:.4f}")
    
    return {
        "mrr": total_mrr,
        "recall": total_recall,
        "ndcg": total_ndcg
    }


def main():
    print("=== Spherical Embedding RAG Evaluation ===\n")
    
    # Load corpus
    corpus = load_real_world_dataset()[:1000]  # Smaller corpus for faster evaluation
    print(f"Using {len(corpus)} documents in corpus")
    
    # Create query set - select random docs as queries, use their own index as relevant
    print("\nCreating evaluation query set...")
    np.random.seed(42)
    query_indices = np.random.choice(len(corpus), size=50, replace=False)
    query_set = [(corpus[i], i) for i in query_indices]
    print(f"Created {len(query_set)} evaluation queries\n")
    
    # Evaluate all models
    teacher_results = evaluate_retrieval(corpus, query_set, "teacher")
    spherical_results = evaluate_retrieval(corpus, query_set, "spherical")
    recon_results = evaluate_retrieval(corpus, query_set, "reconstructed")
    
    # Print comparison
    print("\n=== Final Comparison ===")
    print(f"\n{'Metric':<12} {'Teacher':<12} {'Spherical':<12} {'Reconstructed':<12}")
    print("-" * 52)
    print(f"{'MRR':<12} {teacher_results['mrr']:<12.4f} {spherical_results['mrr']:<12.4f} {recon_results['mrr']:<12.4f}")
    for k in [1,5,10,20]:
        print(f"\n{'Recall@'+str(k):<12} {teacher_results['recall'][k]:<12.4f} {spherical_results['recall'][k]:<12.4f} {recon_results['recall'][k]:<12.4f}")
        print(f"{'nDCG@'+str(k):<12} {teacher_results['ndcg'][k]:<12.4f} {spherical_results['ndcg'][k]:<12.4f} {recon_results['ndcg'][k]:<12.4f}")


if __name__ == "__main__":
    main()
