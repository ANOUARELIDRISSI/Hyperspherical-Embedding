import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class SphericalEmbeddingModel(nn.Module):
    """
    A model that maps text to spherical coordinates (r, theta, phi) with decoder for distillation
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=384, hidden_dim=256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        
        # Freeze base model initially for distillation
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Encoder: embedding -> spherical coordinates (r, theta, phi)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # r, theta, phi
        )
        
        # Decoder: spherical coordinates -> embedding space (for distillation)
        self.decoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)
    
    def forward(self, texts):
        base_embeddings = self.encode_base_embeddings(texts)
        return self.forward_embeddings(base_embeddings)

    def encode_base_embeddings(self, texts):
        device = next(self.parameters()).device
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        with torch.no_grad():
            model_output = self.base_model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input["attention_mask"])

    def forward_embeddings(self, base_embeddings):
        # Encode to spherical coordinates
        spherical_coords = self.encoder(base_embeddings.detach())
        
        # Normalize coordinates
        r = torch.abs(spherical_coords[:, 0])  # r >= 0
        theta = torch.sigmoid(spherical_coords[:, 1]) * torch.pi  # theta in [0, pi]
        phi = torch.sigmoid(spherical_coords[:, 2]) * 2 * torch.pi  # phi in [0, 2pi]
        
        # Decode spherical coordinates back to embedding space
        spherical_cartesian = self.spherical_to_cartesian(r, theta, phi)
        reconstructed_embeddings = self.decoder(spherical_cartesian)
        
        return r, theta, phi, base_embeddings, reconstructed_embeddings
    
    def spherical_to_cartesian(self, r, theta, phi):
        """Convert spherical coordinates to Cartesian coordinates"""
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        return torch.stack([x, y, z], dim=1)
    
    def save_pretrained(self, path):
        """Save model weights"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'base_model_name': self.base_model.name_or_path,
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path, model_name=None):
        """Load model from saved weights"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if model_name is None:
            model_name = checkpoint['base_model_name']
        model = cls(model_name=model_name)
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        model.eval()
        print(f"Model loaded from {path}")
        return model
