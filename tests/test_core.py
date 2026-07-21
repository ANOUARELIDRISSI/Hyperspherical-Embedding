import torch

from src.real_data_loader import load_real_corpus
from src.rag import RAGSystem
from src.spherical_embedding import SphericalEmbeddingModel


def test_data_loader_returns_requested_number_of_texts():
    texts = load_real_corpus(25)

    assert len(texts) == 25
    assert all(isinstance(text, str) and text for text in texts)


def test_spherical_model_forward_shapes():
    model = SphericalEmbeddingModel()
    texts = ["Hello world", "Machine learning is useful"]

    with torch.no_grad():
        r, theta, phi, base_embeddings, reconstructed_embeddings = model(texts)

    assert r.shape == (2,)
    assert theta.shape == (2,)
    assert phi.shape == (2,)
    assert base_embeddings.shape == (2, 384)
    assert reconstructed_embeddings.shape == (2, 384)


def test_rag_retrieve_returns_top_k_documents():
    model = SphericalEmbeddingModel()
    documents = [
        "Python programming is useful for data work.",
        "Football teams train for championship matches.",
        "Astronomy studies planets and stars.",
    ]
    rag = RAGSystem(model=model, mode="spherical")

    rag.add_documents(documents)
    results = rag.retrieve("Tell me about Python", k=2)

    assert len(results) == 2
    assert all("document" in result and "distance" in result for result in results)
