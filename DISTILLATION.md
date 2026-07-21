# Spherical Embedding Distillation

## Overview

This document explains the knowledge distillation process used to train the spherical coordinate embedding model using the `sentence-transformers/all-MiniLM-L6-v2` teacher model, focusing on semantic preservation for RAG applications.

## Architecture

### Teacher Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Purpose**: High-quality, pre-trained sentence embeddings (384-dimensional)
- **Use**: Provides "ground truth" similarity scores for pairwise training

### Student Model

#### Base Component
- Uses the same `all-MiniLM-L6-v2` base as teacher for tokenization and mean pooling
- **Frozen**: Base model weights are kept fixed during distillation

#### Encoder
Converts 384D base embeddings → spherical coordinates (r, θ, φ):
- 2 hidden layers (256 units each, ReLU activation)
- Final layer outputs 3 values (unconstrained)
- Post-processing:
  - r = absolute value of first output (non-negative radius)
  - θ = sigmoid(second output) * π (range [0, π] - polar angle)
  - φ = sigmoid(third output) * 2π (range [0, 2π] - azimuthal angle)

#### Decoder
Converts spherical coordinates → reconstructed 384D embeddings:
- First converts spherical (r, θ, φ) to Cartesian (x, y, z)
- 2 hidden layers (256 units each, ReLU activation)
- Final layer outputs 384-dimensional embedding

## Training Process

### Objective
Optimize student to match the teacher's pairwise similarity scores to preserve semantic neighborhoods for RAG.

### Pair Generation
For each training sample, we generate 4-element tuples:
1. **Anchor**: Random sentence
2. **Positive**: Neighboring sentence (from same document/article)
3. **Hard Negative**: Random sentence from the corpus
4. **Random Negative**: Another random sentence from the corpus

### Loss Function
We use a **contrastive pairwise similarity loss** to teach the geometric structure:

1. Compute cosine similarities between (anchor, positive), (anchor, hard negative), and (anchor, random negative) for both teacher and student
2. Minimize the MSE between teacher and student similarities

```python
# Loss for reconstructed 384D embeddings
recon_loss = F.mse_loss(student_similarities, teacher_similarities)

# Loss for spherical 3D embeddings
spherical_loss = F.mse_loss(student_spherical_similarities, teacher_similarities)

total_loss = recon_loss + 0.5 * spherical_loss
```

### Training Loop
1. **Pair Generation**: Create (anchor, positive, hard negative, random negative) tuples
2. **Batch Processing**: Process pairs in batches of 32
3. **Teacher/Student Forward Pass**: Compute embeddings for all 4 elements of each pair
4. **Similarity Calculation**: Compute pairwise cosine similarities
5. **Loss Calculation**: Compute MSE between student and teacher similarities
6. **Backward Pass/Optimization**: Update encoder/decoder weights

### Training Parameters
- **Number of epochs**: 50
- **Batch size**: 32
- **Learning rate**: 1e-3
- **Optimizer**: Adam
- **Training Pairs/epoch**: ~20,000

## Data Pipeline
We use a real-world data pipeline:
1. Load Wikipedia and AG News datasets
2. Split into 8–40 word sentences
3. Shuffle to mix domains
4. Generate pairwise training data

## Key Concepts

### Spherical Coordinate System
- **r (radius)**: Distance from origin
- **θ (theta)**: Polar angle from positive z-axis (0 ≤ θ ≤ π)
- **φ (phi)**: Azimuthal angle in xy-plane from positive x-axis (0 ≤ φ ≤ 2π)

### Spherical to Cartesian Conversion
```python
x = r * sin(theta) * cos(phi)
y = r * sin(theta) * sin(phi)
z = r * cos(theta)
```

## Distillation Benefits
1. **Compact Representation**: 3D spherical coordinates vs. 384D teacher embeddings
2. **Semantic Preservation**: Pairwise contrastive loss maintains nearest neighbor relationships
3. **Edge-Friendly**: Smaller embeddings suitable for resource-constrained devices
4. **Reconstruction**: Decoder can recover full-dimensional embeddings when needed

## Usage

### Training
```bash
uv run python train.py
```

### Evaluation
```bash
uv run python evaluate_rag.py
```

### RAG Retrieval
```bash
uv run python compare_rag_retrieval.py
```
