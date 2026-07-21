import torch
import torch.nn as nn
import torch.nn.functional as F
from .spherical_embedding import SphericalEmbeddingModel
import os
import random
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class DistillationTrainer:
    """
    Trainer for distillation from Sentence-BERT to spherical embedding model with contrastive pairwise training
    """
    def __init__(self, teacher_model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu", num_threads=None):
        if num_threads:
            torch.set_num_threads(num_threads)
            try:
                torch.set_num_interop_threads(max(1, min(4, num_threads)))
            except RuntimeError:
                pass
        self.device = torch.device(device)
        self.student_model = SphericalEmbeddingModel(model_name=teacher_model_name).to(self.device)
        self.eval_texts = None
        self.topic_classifier = None
        self.topic_prototypes = None
        
    def set_eval_texts(self, texts):
        """Set texts to use for evaluation during training"""
        self.eval_texts = texts
        
    def compute_retrieval_metrics(self, query_set, corpus):
        """Compute Recall@10 and MRR"""
        self.student_model.eval()
        
        with torch.no_grad():
            # Encode corpus with student spherical embeddings
            r, theta, phi, _, _ = self.student_model(corpus)
            corpus_emb = self.student_model.spherical_to_cartesian(r, theta, phi)
            corpus_emb = F.normalize(corpus_emb, p=2, dim=1).detach().cpu().numpy().astype(np.float32)
            
            # Build FAISS index
            index = faiss.IndexFlatL2(3)
            index.add(corpus_emb)
        
            recall_at_10 = 0.0
            mrr = 0.0
        
            for query, relevant_idx in query_set:
                r_q, theta_q, phi_q, _, _ = self.student_model([query])
                q_emb = self.student_model.spherical_to_cartesian(r_q, theta_q, phi_q)
                q_emb = F.normalize(q_emb, p=2, dim=1).detach().cpu().numpy().astype(np.float32)
                
                _, indices = index.search(q_emb, 10)
                retrieved_indices = indices[0]
                
                # Recall@10
                if relevant_idx in retrieved_indices:
                    recall_at_10 += 1.0
                
                # MRR
                for rank, idx in enumerate(retrieved_indices, start=1):
                    if idx == relevant_idx:
                        mrr += 1.0 / rank
                        break
        
        num_queries = len(query_set)
        recall_at_10 /= num_queries
        mrr /= num_queries
        
        return recall_at_10, mrr
        
    def build_pair_index(self, texts, n_jobs=-1):
        """Precompute TF-IDF nearest neighbors used for semantic pair mining."""
        if len(texts) < 4:
            raise ValueError("At least four texts are required to generate contrastive pairs")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
        )
        text_features = vectorizer.fit_transform(texts)
        neighbor_count = min(16, len(texts))
        neighbors = NearestNeighbors(
            n_neighbors=neighbor_count,
            metric="cosine",
            algorithm="brute",
            n_jobs=n_jobs,
        )
        neighbors.fit(text_features)
        distances, indices = neighbors.kneighbors(text_features, return_distance=True)
        return distances, indices

    def generate_pairs(self, texts, pair_index, num_pairs_per_epoch=10000, labels=None):
        """Generate semantic training pairs from a precomputed nearest-neighbor graph."""
        distances, indices = pair_index
        label_to_indices = None
        if labels is not None:
            label_to_indices = {}
            for idx, label in enumerate(labels):
                label_to_indices.setdefault(label, []).append(idx)

        pairs = []
        num_texts = len(texts)
        
        for _ in range(num_pairs_per_epoch):
            anchor_idx = random.randint(0, num_texts - 1)
            anchor = texts[anchor_idx]

            nearest = [idx for idx in indices[anchor_idx] if idx != anchor_idx]
            if label_to_indices is not None and random.random() < 0.7:
                same_label = [idx for idx in label_to_indices[labels[anchor_idx]] if idx != anchor_idx]
                positive_idx = random.choice(same_label or nearest[: max(1, min(5, len(nearest)))])
            else:
                positive_idx = random.choice(nearest[: max(1, min(5, len(nearest)))])
            positive = texts[positive_idx]

            far_neighbors = [
                idx for dist, idx in zip(distances[anchor_idx], indices[anchor_idx])
                if idx != anchor_idx and idx != positive_idx and dist > 0.35
            ]
            if label_to_indices is not None and random.random() < 0.7:
                other_labels = [label for label in label_to_indices if label != labels[anchor_idx]]
                neg_label = random.choice(other_labels)
                hard_neg_idx = random.choice(label_to_indices[neg_label])
            else:
                hard_neg_idx = random.choice(far_neighbors or nearest[-max(1, len(nearest) // 3):])
            hard_neg = texts[hard_neg_idx]

            rand_neg_idx = random.randint(0, num_texts - 1)
            while rand_neg_idx in {anchor_idx, positive_idx, hard_neg_idx}:
                rand_neg_idx = random.randint(0, num_texts - 1)
            rand_neg = texts[rand_neg_idx]

            pairs.append((anchor_idx, positive_idx, hard_neg_idx, rand_neg_idx))
        
        return pairs

    def precompute_base_embeddings(self, texts, batch_size=128):
        self.student_model.eval()
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                embeddings.append(self.student_model.encode_base_embeddings(batch).cpu())
        return torch.cat(embeddings, dim=0).to(self.device)

    def build_teacher_neighbor_index(self, base_embedding_cache, top_k=8, far_k=32):
        """Precompute teacher top-k neighbors and far negatives from MiniLM geometry."""
        with torch.no_grad():
            teacher_unit = F.normalize(base_embedding_cache, p=2, dim=1)
            similarity = teacher_unit @ teacher_unit.T
            similarity.fill_diagonal_(-1.0)
            top_count = min(top_k, similarity.shape[1] - 1)
            far_count = min(far_k, similarity.shape[1] - 1)
            top_indices = torch.topk(similarity, k=top_count, dim=1).indices.cpu().tolist()
            far_indices = torch.topk(-similarity, k=far_count, dim=1).indices.cpu().tolist()
        return {"top": top_indices, "far": far_indices}

    def mine_student_hard_negatives(self, base_embedding_cache, labels=None, top_k=16):
        """Find examples the current 3D student places close despite teacher/topic mismatch."""
        self.student_model.eval()
        with torch.no_grad():
            r, theta, phi, _, _ = self.student_model.forward_embeddings(base_embedding_cache)
            student_vectors = self.student_model.spherical_to_cartesian(r, theta, phi)
            student_unit = F.normalize(student_vectors, p=2, dim=1)
            similarity = student_unit @ student_unit.T
            similarity.fill_diagonal_(-1.0)
            near_indices = torch.topk(similarity, k=min(top_k * 3, similarity.shape[1] - 1), dim=1).indices.cpu().tolist()

        mined = []
        for anchor_idx, candidates in enumerate(near_indices):
            if labels is None:
                mined.append(candidates[:top_k])
                continue
            mismatched = [idx for idx in candidates if labels[idx] != labels[anchor_idx]]
            mined.append((mismatched or candidates)[:top_k])
        return mined

    def generate_retrieval_groups(self, teacher_neighbors, num_groups, positives=4, negatives=12, mined_negatives=None):
        """Build listwise retrieval groups: anchor plus teacher positives, far negatives, and mined hard negatives."""
        groups = []
        num_texts = len(teacher_neighbors["top"])
        for _ in range(num_groups):
            anchor_idx = random.randint(0, num_texts - 1)
            positive_pool = teacher_neighbors["top"][anchor_idx]
            far_pool = teacher_neighbors["far"][anchor_idx]
            mined_pool = mined_negatives[anchor_idx] if mined_negatives else []
            pos = random.sample(positive_pool, k=min(positives, len(positive_pool)))
            mixed_neg_pool = list(dict.fromkeys(mined_pool + far_pool))
            neg = random.sample(mixed_neg_pool, k=min(negatives, len(mixed_neg_pool)))
            candidates = list(dict.fromkeys(pos + neg))
            if candidates:
                groups.append((anchor_idx, candidates))
        return groups
        
    def train_step(self, batch_pairs, base_embedding_cache, optimizer, label_cache=None):
        self.student_model.train()
        optimizer.zero_grad()

        batch_indices = torch.tensor(
            [idx for pair in batch_pairs for idx in pair],
            dtype=torch.long,
            device=self.device,
        )
        base_embeddings = base_embedding_cache.index_select(0, batch_indices)
        r, theta, phi, base_embeddings, reconstructed_embeddings = self.student_model.forward_embeddings(base_embeddings)
        teacher_embeddings = base_embeddings.detach()
        spherical_cartesian = self.student_model.spherical_to_cartesian(r, theta, phi)
        spherical_unit = F.normalize(spherical_cartesian, p=2, dim=1)
        
        # Reshape to (batch_size, 4, dim)
        batch_size = len(batch_pairs)
        teacher_embeddings = teacher_embeddings.reshape(batch_size, 4, -1)
        spherical_cartesian = spherical_cartesian.reshape(batch_size, 4, -1)
        spherical_unit = spherical_unit.reshape(batch_size, 4, -1)
        reconstructed_embeddings = reconstructed_embeddings.reshape(batch_size, 4, -1)
        
        # Compute pairwise similarity scores
        def compute_similarity_pairs(embeddings):
            anchors = embeddings[:, 0, :]
            positives = embeddings[:, 1, :]
            hard_negs = embeddings[:, 2, :]
            rand_negs = embeddings[:, 3, :]
            
            pos_sim = F.cosine_similarity(anchors, positives, dim=1)
            hard_neg_sim = F.cosine_similarity(anchors, hard_negs, dim=1)
            rand_neg_sim = F.cosine_similarity(anchors, rand_negs, dim=1)
            
            return pos_sim, hard_neg_sim, rand_neg_sim
        
        teacher_pos_sim, teacher_hard_neg_sim, teacher_rand_neg_sim = compute_similarity_pairs(teacher_embeddings)
        recon_pos_sim, recon_hard_neg_sim, recon_rand_neg_sim = compute_similarity_pairs(reconstructed_embeddings)
        spherical_pos_sim, spherical_hard_neg_sim, spherical_rand_neg_sim = compute_similarity_pairs(spherical_unit)
        
        # Compute losses
        recon_loss = F.mse_loss(torch.stack([recon_pos_sim, recon_hard_neg_sim, recon_rand_neg_sim], dim=1),
                               torch.stack([teacher_pos_sim, teacher_hard_neg_sim, teacher_rand_neg_sim], dim=1))
        spherical_loss = F.mse_loss(torch.stack([spherical_pos_sim, spherical_hard_neg_sim, spherical_rand_neg_sim], dim=1),
                                   torch.stack([teacher_pos_sim, teacher_hard_neg_sim, teacher_rand_neg_sim], dim=1))
        flat_teacher = F.normalize(teacher_embeddings.reshape(batch_size * 4, -1), p=2, dim=1)
        flat_recon = F.normalize(reconstructed_embeddings.reshape(batch_size * 4, -1), p=2, dim=1)
        flat_spherical = spherical_unit.reshape(batch_size * 4, -1)
        teacher_similarity = flat_teacher @ flat_teacher.T
        recon_similarity = flat_recon @ flat_recon.T
        spherical_similarity = flat_spherical @ flat_spherical.T
        matrix_loss = F.mse_loss(spherical_similarity, teacher_similarity)
        recon_matrix_loss = F.mse_loss(recon_similarity, teacher_similarity)
        embedding_loss = F.mse_loss(reconstructed_embeddings, teacher_embeddings)
        margin = 0.15
        rank_loss = (
            F.relu(margin - spherical_pos_sim + spherical_hard_neg_sim).mean()
            + F.relu(margin - spherical_pos_sim + spherical_rand_neg_sim).mean()
        )
        spread_loss = F.relu(0.25 - flat_spherical.std(dim=0)).mean()
        supervised_loss = torch.tensor(0.0, device=self.device)
        if label_cache is not None and self.topic_classifier is not None:
            label_targets = label_cache.index_select(0, batch_indices)
            supervised_loss = F.cross_entropy(self.topic_classifier(flat_spherical), label_targets)
        total_loss = (
            0.5 * recon_loss
            + 0.5 * recon_matrix_loss
            + 0.25 * embedding_loss
            + 2.0 * spherical_loss
            + 4.0 * matrix_loss
            + rank_loss
            + 0.1 * spread_loss
            + 0.25 * supervised_loss
        )
        
        # Compute average teacher cosine similarity
        avg_teacher_cosine = (torch.mean(teacher_pos_sim) + torch.mean(teacher_hard_neg_sim) + torch.mean(teacher_rand_neg_sim)) / 3
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), avg_teacher_cosine.item()

    def train_retrieval_step(
        self,
        groups,
        base_embedding_cache,
        optimizer,
        label_cache=None,
        temperature=0.08,
        prototype_weight=0.1,
    ):
        self.student_model.train()
        optimizer.zero_grad()

        losses = []
        supervised_losses = []
        prototype_losses = []
        for anchor_idx, candidate_indices in groups:
            indices = torch.tensor([anchor_idx] + candidate_indices, dtype=torch.long, device=self.device)
            base_embeddings = base_embedding_cache.index_select(0, indices)
            r, theta, phi, base_embeddings, reconstructed_embeddings = self.student_model.forward_embeddings(base_embeddings)
            teacher_unit = F.normalize(base_embeddings.detach(), p=2, dim=1)
            spherical = self.student_model.spherical_to_cartesian(r, theta, phi)
            spherical_unit = F.normalize(spherical, p=2, dim=1)
            recon_unit = F.normalize(reconstructed_embeddings, p=2, dim=1)

            teacher_scores = (teacher_unit[0:1] @ teacher_unit[1:].T).squeeze(0)
            student_scores = (spherical_unit[0:1] @ spherical_unit[1:].T).squeeze(0)
            recon_scores = (recon_unit[0:1] @ recon_unit[1:].T).squeeze(0)

            teacher_distribution = F.softmax(teacher_scores / temperature, dim=0)
            student_log_distribution = F.log_softmax(student_scores / temperature, dim=0)
            recon_log_distribution = F.log_softmax(recon_scores / temperature, dim=0)
            listwise_loss = F.kl_div(student_log_distribution, teacher_distribution, reduction="batchmean")
            recon_listwise_loss = F.kl_div(recon_log_distribution, teacher_distribution, reduction="batchmean")
            embedding_loss = F.mse_loss(reconstructed_embeddings, base_embeddings.detach())
            losses.append(listwise_loss + 0.5 * recon_listwise_loss + 0.1 * embedding_loss)

            if label_cache is not None and self.topic_classifier is not None:
                label_targets = label_cache.index_select(0, indices)
                supervised_losses.append(F.cross_entropy(self.topic_classifier(spherical_unit), label_targets))

            if label_cache is not None and self.topic_prototypes is not None:
                label_targets = label_cache.index_select(0, indices)
                prototypes = F.normalize(self.topic_prototypes, p=2, dim=1)
                prototype_scores = spherical_unit @ prototypes.T
                prototype_losses.append(F.cross_entropy(prototype_scores / 0.12, label_targets))

        if not losses:
            return 0.0, 0.0

        total_loss = torch.stack(losses).mean()
        if supervised_losses:
            total_loss = total_loss + 0.15 * torch.stack(supervised_losses).mean()
        if prototype_losses:
            total_loss = total_loss + prototype_weight * torch.stack(prototype_losses).mean()

        total_loss.backward()
        optimizer.step()
        return total_loss.item(), 0.0
    
    def train(
        self,
        texts,
        num_epochs=50,
        learning_rate=1e-3,
        batch_size=32,
        pair_workers=-1,
        pairs_per_epoch=None,
        labels=None,
        checkpoint_dir=None,
        checkpoint_every=0,
        retrieval_distillation=False,
        teacher_top_k=8,
        mined_hard_negatives=False,
        prototype_loss=False,
    ):
        label_cache = None
        optimizer_params = (
            list(self.student_model.input_projection.parameters()) +
            list(self.student_model.encoder.parameters()) +
            list(self.student_model.radius_head.parameters()) +
            list(self.student_model.angle_head.parameters()) +
            list(self.student_model.decoder.parameters())
        )
        if labels is not None:
            label_names = sorted(set(labels))
            label_to_id = {label: idx for idx, label in enumerate(label_names)}
            label_cache = torch.tensor([label_to_id[label] for label in labels], dtype=torch.long, device=self.device)
            self.topic_classifier = nn.Linear(3, len(label_names)).to(self.device)
            optimizer_params += list(self.topic_classifier.parameters())
            print(f"Using supervised topic separation with {len(label_names)} labels")
            if prototype_loss:
                self.topic_prototypes = nn.Parameter(torch.randn(len(label_names), 3, device=self.device))
                optimizer_params.append(self.topic_prototypes)
                print("Using topic prototype loss")

        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=learning_rate
        )
        
        # Evaluation setup
        eval_texts = texts[:1000]
        self.set_eval_texts(eval_texts)
        eval_query_set = [(text, idx) for idx, text in enumerate(eval_texts[:50])]
        print("Precomputing frozen MiniLM embeddings...")
        base_embedding_cache = self.precompute_base_embeddings(texts)
        history = []
        teacher_neighbors = None
        mined_negatives = None
        if retrieval_distillation:
            print("Building teacher top-k retrieval neighborhoods...")
            teacher_neighbors = self.build_teacher_neighbor_index(base_embedding_cache, top_k=teacher_top_k)
        
        for epoch in range(num_epochs):
            print(f"\nGenerating training pairs for epoch {epoch+1}...")
            num_pairs = pairs_per_epoch or min(20000, len(texts) * 2)
            if retrieval_distillation:
                if mined_hard_negatives:
                    print("Mining student hard negatives...")
                    mined_negatives = self.mine_student_hard_negatives(base_embedding_cache, labels=labels)
                training_groups = self.generate_retrieval_groups(
                    teacher_neighbors,
                    num_groups=num_pairs,
                    positives=min(teacher_top_k, 4),
                    negatives=12,
                    mined_negatives=mined_negatives,
                )
            elif epoch == 0:
                print("Building semantic pair index...")
                pair_index = self.build_pair_index(texts, n_jobs=pair_workers)
            if not retrieval_distillation:
                training_pairs = self.generate_pairs(texts, pair_index, num_pairs_per_epoch=num_pairs, labels=labels)
            
            total_loss = 0.0
            total_teacher_cosine = 0.0
            num_batches = 0
            
            training_items = training_groups if retrieval_distillation else training_pairs
            for i in range(0, len(training_items), batch_size):
                batch_items = training_items[i:i+batch_size]
                if retrieval_distillation:
                    loss, teacher_cosine = self.train_retrieval_step(
                        batch_items,
                        base_embedding_cache,
                        optimizer,
                        label_cache=label_cache,
                        prototype_weight=0.1 if prototype_loss else 0.0,
                    )
                else:
                    loss, teacher_cosine = self.train_step(batch_items, base_embedding_cache, optimizer, label_cache=label_cache)
                total_loss += loss
                total_teacher_cosine += teacher_cosine
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_teacher_cosine = total_teacher_cosine / num_batches
            recall_at_10, mrr = self.compute_retrieval_metrics(eval_query_set, eval_texts)
            
            # Print epoch summary
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            print(f"Loss ............. {avg_loss:.3f}")
            print(f"Teacher cosine ... {avg_teacher_cosine:.3f}")
            print(f"Recall@10 ........ {recall_at_10*100:.1f}%")
            print(f"MRR .............. {mrr:.3f}")
            print("-" * 40)
            history.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "teacher_cosine": avg_teacher_cosine,
                "recall_at_10": recall_at_10,
                "mrr": mrr,
            })
            if checkpoint_dir and checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"spherical_embedding_epoch_{epoch + 1}.pt")
                self.save_student_model(checkpoint_path)

        return history
    
    def save_student_model(self, path="models/spherical_embedding_model.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.student_model.save_pretrained(path)
