import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

FILE_PATH = "/data/JobRoleToSkillRecommendation .xlsx"

# -----------------------------
# 1) Load + build role→skills text
# -----------------------------
df = pd.read_excel(FILE_PATH)

# keep only needed cols
df = df[["Job Role", "Skill Title", "Proficiency Level"]].dropna()
df["Job Role"] = df["Job Role"].astype(str).str.strip()
df["Skill Title"] = df["Skill Title"].astype(str).str.strip()

# create weighted skill tokens (simple + fast)
# example: "python" repeated by proficiency (or use scaling)
def skill_token(skill: str, prof: int) -> str:
    prof = int(prof) if str(prof).isdigit() else 3
    return (" " + skill.lower().replace(" ", "_")) * max(1, min(prof, 5))

role_to_text = (
    df.groupby("Job Role")
      .apply(lambda g: "".join(skill_token(s, p) for s, p in zip(g["Skill Title"], g["Proficiency Level"])))
      .to_dict()
)

roles = sorted(role_to_text.keys())
role_texts = [role_to_text[r] for r in roles]

# TF-IDF role vectors (fast + good baseline)
vectorizer = TfidfVectorizer(min_df=2, max_df=0.9)
X = vectorizer.fit_transform(role_texts)
X = normalize(X)

role_to_idx = {r:i for i, r in enumerate(roles)}
idx_to_role = {i:r for r,i in role_to_idx.items()}

# precompute cosine similarity helper
def top_similar(role_idx: int, k: int = 50):
    sims = (X[role_idx] @ X.T).toarray().ravel()  # cosine because normalized
    sims[role_idx] = -1
    top = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]

# -----------------------------
# 2) Metadata: track + level from title
# -----------------------------
def extract_track(title: str) -> str:
    t = title.lower()
    # IMPORTANT: your dataset uses "Back End Developer" with a space
    if "back end" in t or "backend" in t:
        return "backend"
    if "front end" in t or "frontend" in t:
        return "frontend"
    if "full stack" in t or "fullstack" in t:
        return "fullstack"
    return "other"

def extract_level(title: str) -> int:
    t = title.lower()
    # tune this for your naming
    if "intern" in t or "trainee" in t:
        return 0
    if "junior" in t or "jr" in t:
        return 1
    if "senior" in t or "sr" in t:
        return 3
    if "lead" in t or "principal" in t:
        return 4
    if "manager" in t or "head" in t:
        return 5
    return 2  # default mid

meta = {}
for r in roles:
    meta[r] = {"track": extract_track(r), "level": extract_level(r)}

# -----------------------------
# 3) Transition constraints + scoring
# -----------------------------
def allowed(curr_role: str, next_role: str, mode: str = "stay_in_track") -> bool:
    c = meta[curr_role]
    n = meta[next_role]

    # no going backwards in level
    if n["level"] < c["level"]:
        return False

    # avoid huge jumps (allow only +0 or +1 by default)
    if n["level"] - c["level"] > 1:
        return False

    if mode == "stay_in_track":
        # strict: must be same track, OR allow backend<->fullstack as adjacent
        if c["track"] == n["track"]:
            return True
        adjacent = {("backend", "fullstack"), ("fullstack", "backend"),
                    ("frontend", "fullstack"), ("fullstack", "frontend")}
        return (c["track"], n["track"]) in adjacent

    # explore mode: allow any track but it will be penalized
    return True

def score(curr_role: str, next_role: str, sim: float, mode: str = "stay_in_track") -> float:
    c = meta[curr_role]
    n = meta[next_role]
    s = sim

    # bonus for level up
    if n["level"] == c["level"] + 1:
        s += 0.15

    # penalty for lateral same-level moves (often noisy)
    if n["level"] == c["level"]:
        s -= 0.10

    # track penalties
    if c["track"] != n["track"]:
        if (c["track"], n["track"]) in {("backend", "fullstack"), ("fullstack", "backend"),
                                        ("frontend", "fullstack"), ("fullstack", "frontend")}:
            s -= 0.12
        else:
            s -= 0.35  # hard penalty for big switches like backend->frontend

    # if mode is strict, we already filtered most switches; this is extra safety
    if mode == "stay_in_track" and c["track"] != n["track"]:
        s -= 0.10

    return s

# -----------------------------
# 4) Beam search path generation
# -----------------------------
def recommend_paths(start_role: str, steps: int = 4, beam: int = 3, mode: str = "stay_in_track"):
    if start_role not in role_to_idx:
        raise ValueError(f"Unknown role: {start_role}")

    # state: (total_score, path_roles)
    states = [(0.0, [start_role])]

    for _ in range(steps):
        new_states = []
        for total, path in states:
            curr = path[-1]
            curr_idx = role_to_idx[curr]

            # candidates from similarity
            candidates = top_similar(curr_idx, k=80)

            for idx, sim in candidates:
                nxt = idx_to_role[idx]
                if not allowed(curr, nxt, mode=mode):
                    continue
                sc = score(curr, nxt, sim, mode=mode)
                new_states.append((total + sc, path + [nxt]))

        if not new_states:
            # no valid next step; stop early
            break

        # keep best beam
        new_states.sort(key=lambda x: x[0], reverse=True)
        states = new_states[:beam]

    return states

# -----------------------------
# Example: your exact roles exist in dataset
# -----------------------------
for total, path in recommend_paths("Back End Developer", steps=4, beam=3, mode="stay_in_track"):
    print(round(total, 3), " -> ".join(path))
