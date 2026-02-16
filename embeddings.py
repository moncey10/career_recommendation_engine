# src/embeddings.py
import os
import pickle
from typing import Dict, List
import numpy as np


class SkillEmbeddingIndex:
    """
    - build mode: load_only=False -> loads SentenceTransformer (slow, but only used in offline build step)
    - load mode : load_only=True  -> does NOT load SentenceTransformer (fast runtime)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "cache", load_only: bool = False):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.model = None
        if not load_only:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)

        self.skill_to_vec: Dict[str, np.ndarray] = {}
        self.role_to_vec: Dict[str, np.ndarray] = {}

    def _cache_path(self, name: str) -> str:
        safe = name.replace("/", "_")
        # IMPORTANT: we always append ".pkl" once here
        return os.path.join(self.cache_dir, f"{safe}.pkl")

    def save(self, filename: str = "embeddings"):
        """
        Save embeddings to <cache_dir>/<filename>.pkl
        Pass filename WITHOUT extension to avoid ".pkl.pkl".
        """
        with open(self._cache_path(filename), "wb") as f:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "skill_to_vec": self.skill_to_vec,
                    "role_to_vec": self.role_to_vec,
                },
                f,
            )

    def load(self, filename: str = "embeddings") -> bool:
        """
        Load embeddings from <cache_dir>/<filename>.pkl
        Pass filename WITHOUT extension to avoid ".pkl.pkl".
        """
        path = self._cache_path(filename)
        if not os.path.exists(path):
            return False

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if obj.get("model_name") != self.model_name:
            return False

        self.skill_to_vec = obj.get("skill_to_vec", {})
        self.role_to_vec = obj.get("role_to_vec", {})
        return True

    def build_skill_embeddings(self, skills: List[str], batch_size: int = 64):
        if self.model is None:
            raise RuntimeError("SkillEmbeddingIndex is in load_only mode; cannot build embeddings.")

        missing = [s for s in skills if s not in self.skill_to_vec]
        if not missing:
            return

        vecs = self.model.encode(
            missing,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        for s, v in zip(missing, vecs):
            self.skill_to_vec[s] = v.astype(np.float32)

    def build_role_embeddings(self, role_to_skillweights: Dict[str, Dict[str, float]]):
        """
        role_to_skillweights example:
        {
          "Data Scientist": {"python": 0.8, "sql": 0.6, ...},
          ...
        }

        We build role vectors by weighted average of skill vectors.
        """
        if self.model is None:
            raise RuntimeError("SkillEmbeddingIndex is in load_only mode; cannot build embeddings.")

        # collect all skills we need embeddings for
        all_skills = set()
        for sw in role_to_skillweights.values():
            for s in sw.keys():
                if s:
                    all_skills.add(str(s).strip().lower())

        self.build_skill_embeddings(sorted(all_skills))

        for role, sw in role_to_skillweights.items():
            items = [(str(s).strip().lower(), float(w)) for s, w in sw.items() if s and float(w) > 0]
            if not items:
                continue

            vecs = []
            weights = []
            for s, w in items:
                v = self.skill_to_vec.get(s)
                if v is None:
                    continue
                vecs.append(v)
                weights.append(w)

            if not vecs:
                continue

            W = np.array(weights, dtype=np.float32)
            W = W / (W.sum() + 1e-9)

            V = np.vstack(vecs).astype(np.float32)
            role_vec = (V * W[:, None]).sum(axis=0)

            # normalize to unit vector (safe for cosine similarity)
            norm = float(np.linalg.norm(role_vec))
            if norm > 0:
                role_vec = role_vec / norm

            self.role_to_vec[role] = role_vec.astype(np.float32)
