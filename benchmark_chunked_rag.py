import argparse
from collections import defaultdict

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.real_data_loader import TOPICS, build_topic_documents, chunk_text
from src.spherical_embedding import SphericalEmbeddingModel


def build_chunks(sentences_per_topic=80, chunk_sentences=4, overlap=1):
    chunks = []
    for document in build_topic_documents(sentences_per_topic=sentences_per_topic):
        for chunk_id, chunk in enumerate(chunk_text(document["text"], chunk_sentences, overlap)):
            chunks.append({
                "id": f"{document['id']}-{chunk_id}",
                "label": document["label"],
                "text": chunk,
            })
    return chunks


def normalize_numpy(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def embed_spherical(model, texts):
    with torch.no_grad():
        r, theta, phi, _, _ = model(texts)
        embeddings = model.spherical_to_cartesian(r, theta, phi)
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy().astype(np.float32)


def build_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index


def score_results(chunks, retrieved_indices, expected_label):
    labels = [chunks[idx]["label"] for idx in retrieved_indices]
    hits = [label == expected_label for label in labels]
    precision = sum(hits) / len(hits)
    reciprocal_rank = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / rank
            break
    return precision, reciprocal_rank, labels


def main():
    parser = argparse.ArgumentParser(description="Compare teacher and spherical RAG over chunked documents.")
    parser.add_argument("--sentences-per-topic", type=int, default=80)
    parser.add_argument("--chunk-sentences", type=int, default=4)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-path", default="models/spherical_embedding_model.pt")
    args = parser.parse_args()

    chunks = build_chunks(args.sentences_per_topic, args.chunk_sentences, args.overlap)
    texts = [chunk["text"] for chunk in chunks]
    print(f"Built {len(chunks)} chunks from {len(TOPICS)} topic documents")

    teacher = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    spherical = SphericalEmbeddingModel.from_pretrained(args.model_path)
    spherical.eval()

    teacher_embeddings = teacher.encode(texts, convert_to_numpy=True).astype(np.float32)
    teacher_embeddings = normalize_numpy(teacher_embeddings)
    spherical_embeddings = embed_spherical(spherical, texts)

    teacher_index = build_index(teacher_embeddings)
    spherical_index = build_index(spherical_embeddings)

    totals = {
        "teacher": defaultdict(float),
        "spherical": defaultdict(float),
    }
    query_count = 0

    for label, topic in TOPICS.items():
        for query in topic["queries"]:
            query_count += 1
            teacher_query = normalize_numpy(teacher.encode([query], convert_to_numpy=True).astype(np.float32))
            spherical_query = embed_spherical(spherical, [query])

            _, teacher_indices = teacher_index.search(teacher_query, args.top_k)
            _, spherical_indices = spherical_index.search(spherical_query, args.top_k)

            teacher_precision, teacher_rr, teacher_labels = score_results(chunks, teacher_indices[0], label)
            spherical_precision, spherical_rr, spherical_labels = score_results(chunks, spherical_indices[0], label)

            totals["teacher"]["precision"] += teacher_precision
            totals["teacher"]["mrr"] += teacher_rr
            totals["spherical"]["precision"] += spherical_precision
            totals["spherical"]["mrr"] += spherical_rr

            print(f"\nQuery [{label}]: {query}")
            print(f"Teacher labels:   {teacher_labels} P@{args.top_k}={teacher_precision:.2f} RR={teacher_rr:.2f}")
            print(f"Spherical labels: {spherical_labels} P@{args.top_k}={spherical_precision:.2f} RR={spherical_rr:.2f}")
            print(f"Teacher top:   {chunks[teacher_indices[0][0]]['text'][:140]}...")
            print(f"Spherical top: {chunks[spherical_indices[0][0]]['text'][:140]}...")

    print("\n=== Chunked RAG Summary ===")
    for name in ("teacher", "spherical"):
        precision = totals[name]["precision"] / query_count
        mrr = totals[name]["mrr"] / query_count
        print(f"{name:<10} P@{args.top_k}: {precision:.3f}  MRR: {mrr:.3f}")


if __name__ == "__main__":
    main()
