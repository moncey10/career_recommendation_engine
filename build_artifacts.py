# build_artifacts.py
import os, json
import numpy as np
from scipy import sparse

from src.preprocess import load_clean
from src.embeddings import SkillEmbeddingIndex

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

def main(xlsx_path="data/JobRoleToSkillRecommendation.xlsx"):
    df = load_clean(xlsx_path)

    role_col = "Job Role"
    skill_col = "Skill Title"
    level_col = "Proficiency Level"

    # pivot -> dense -> sparse
    mat_df = df.pivot_table(index=role_col, columns=skill_col, values=level_col, aggfunc="max", fill_value=0)
    roles = mat_df.index.tolist()
    skills = mat_df.columns.tolist()

    role_to_idx = {r: i for i, r in enumerate(roles)}
    skill_to_idx = {s: i for i, s in enumerate(skills)}

    M = mat_df.values.astype(np.float32)
    M_sp = sparse.csr_matrix(M)

    sparse.save_npz(os.path.join(ART_DIR, "matrix.npz"), M_sp)

    with open(os.path.join(ART_DIR, "roles.json"), "w", encoding="utf-8") as f:
        json.dump(roles, f, ensure_ascii=False)

    with open(os.path.join(ART_DIR, "skills.json"), "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False)

    with open(os.path.join(ART_DIR, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump({"role_to_idx": role_to_idx, "skill_to_idx": skill_to_idx}, f)

    # embeddings: build once, save in artifacts
    # (loads model ONLY during build step)
    embed = SkillEmbeddingIndex(model_name="all-MiniLM-L6-v2", cache_dir="artifacts")

    # role -> {skill: weight}
    role_to_skillweights = {}
    for r in roles:
        row = mat_df.loc[r]
        sw = {s.lower(): float(v) for s, v in row[row > 0].items()}
        role_to_skillweights[r] = sw

    embed.build_role_embeddings(role_to_skillweights)
    embed.save("embeddings")

    print("✅ Artifacts created in ./artifacts")

if __name__ == "__main__":
    main()
