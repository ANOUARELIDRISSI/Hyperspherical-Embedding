import numpy as np
import faiss
from .spherical_embedding import SphericalEmbeddingModel


class RAGSystem:
    """
    RAG system using spherical coordinate embeddings and FAISS
    """
    def __init__(self, model=None, mode="spherical"):
        """
        mode: "spherical" (3D Cartesian from spherical) or "reconstructed" (384D from decoder)
        """
        if model is None:
            self.model = SphericalEmbeddingModel()
        else:
            self.model = model
            
        self.mode = mode
        if mode == "spherical":
            index_dim = 3
        elif mode == "reconstructed":
            index_dim = 384
        else:
            raise ValueError("mode must be 'spherical' or 'reconstructed'")
        
        # FAISS index
        self.index = faiss.IndexFlatL2(index_dim)
        self.documents = []
        
    def add_documents(self, documents):
        """Add documents to the RAG system"""
        self.documents.extend(documents)
        
        # Get embeddings for documents
        r, theta, phi, base_embeddings, reconstructed_embeddings = self.model(documents)
        
        if self.mode == "spherical":
            embeddings = self.model.spherical_to_cartesian(r, theta, phi).detach().numpy()
        else:
            embeddings = reconstructed_embeddings.detach().numpy()
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
    def retrieve(self, query, k=3):
        """Retrieve top-k relevant documents for a query"""
        r, theta, phi, base_embeddings, reconstructed_embeddings = self.model([query])
        
        if self.mode == "spherical":
            query_embedding = self.model.spherical_to_cartesian(r, theta, phi).detach().numpy()
        else:
            query_embedding = reconstructed_embeddings.detach().numpy()
        
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'distance': distances[0][i]
                })
        
        return results
