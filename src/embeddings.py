import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


class SkillEmbeddingIndex:
    """
    Builds embeddings for skills and role embeddings (weighted avg of skill embeddings).
    Caches to disk so you don't recompute every time.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "cache",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # small + fast model
        self.model = SentenceTransformer(model_name)

        self.skill_to_vec: Dict[str, np.ndarray] = {}
        self.role_to_vec: Dict[str, np.ndarray] = {}

    def _cache_path(self, name: str) -> str:
        safe = name.replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe}.pkl")

    def save(self, filename: str = "embeddings_phase1.pkl"):
        path = self._cache_path(filename)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "skill_to_vec": self.skill_to_vec,
                    "role_to_vec": self.role_to_vec,
                },
                f,
            )

    def load(self, filename: str = "embeddings_phase1.pkl") -> bool:
        path = self._cache_path(filename)
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            obj = pickle.load(f)

        # load only if same model
        if obj.get("model_name") != self.model_name:
            return False

        self.skill_to_vec = obj.get("skill_to_vec", {})
        self.role_to_vec = obj.get("role_to_vec", {})
        return True

    def build_skill_embeddings(self, skills: List[str], batch_size: int = 64):
        """
        skills: list of unique skill titles (already normalized lower-case recommended)
        """
        missing = [s for s in skills if s not in self.skill_to_vec]
        if not missing:
            return

        vecs = self.model.encode(
            missing,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # already L2 normalized
        )

        for s, v in zip(missing, vecs):
            self.skill_to_vec[s] = v.astype(np.float32)

    def build_role_embeddings(self, role_to_skillweights: Dict[str, Dict[str, float]]):
        """
        role_to_skillweights: {role: {skill: weight}}
        role embedding = weighted average of skill embeddings
        """
        # build missing skill embeddings
        all_skills = set()
        for r, sw in role_to_skillweights.items():
            for sk in sw.keys():
                all_skills.add(sk)
        self.build_skill_embeddings(sorted(all_skills))

        for role, sw in role_to_skillweights.items():
            if not sw:
                continue

            vec_sum = None
            w_sum = 0.0

            for skill, w in sw.items():
                v = self.skill_to_vec.get(skill)
                if v is None:
                    continue
                if vec_sum is None:
                    vec_sum = (w * v).astype(np.float32)
                else:
                    vec_sum += (w * v).astype(np.float32)
                w_sum += float(w)

            if vec_sum is None or w_sum == 0.0:
                continue

            role_vec = vec_sum / w_sum
            role_vec = l2_normalize(role_vec)
            self.role_to_vec[role] = role_vec.astype(np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        # both should be normalized already
        return float(np.dot(a, b))
