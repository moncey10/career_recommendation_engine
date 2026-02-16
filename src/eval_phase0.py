# import argparse
# import numpy as np

# # Adjust if your path differs:
# from src.recommender import CareerRecommender

# def rank_skills_from_neighbors(matrix, neighbor_idxs, neighbor_sims, top_k=100):
#     """
#     Weighted skill recovery:
#     predicted_skill_score = sum(similarity * neighbor_role_skill_value)
#     """
#     weighted = np.zeros(matrix.shape[1], dtype=np.float64)
#     for idx, sim in zip(neighbor_idxs, neighbor_sims):
#         weighted += sim * matrix[idx]

#     order = np.argsort(-weighted)
#     return order[:top_k]


# def metrics_at_k(pred_skills, true_skills, k):
#     pred_k = pred_skills[:k]
#     hit = set(pred_k) & set(true_skills)

#     precision = len(hit) / float(k) if k else 0.0
#     recall = len(hit) / float(len(true_skills)) if true_skills else 0.0
#     hitrate = 1.0 if len(hit) > 0 else 0.0

#     # MRR
#     mrr = 0.0
#     true_set = set(true_skills)
#     for rank, s in enumerate(pred_k, start=1):
#         if s in true_set:
#             mrr = 1.0 / rank
#             break

#     return precision, recall, hitrate, mrr


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data", default="data/JobRoleToSkillRecommendation.xlsx")
#     ap.add_argument("--hide_ratio", type=float, default=0.2)
#     ap.add_argument("--neighbors", type=int, default=10)
#     ap.add_argument("--k", type=int, default=20)
#     ap.add_argument("--max_roles", type=int, default=0)  # 0 = all roles
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()

#     np.random.seed(args.seed)

#     rec = CareerRecommender(path=args.data)

#     # Role x Skill matrix from your recommender
#     df_mat = rec.matrix.copy()
#     mat = df_mat.values.astype(np.float64)
#     n_roles, n_skills = mat.shape

#     # cosine similarity on full matrix (baseline)
#     norms = np.linalg.norm(mat, axis=1, keepdims=True)
#     norms[norms == 0] = 1.0
#     mat_norm = mat / norms
#     sims = mat_norm @ mat_norm.T

#     indices = np.arange(n_roles)
#     if args.max_roles and args.max_roles > 0:
#         indices = indices[: min(args.max_roles, n_roles)]

#     P, R, H, M = [], [], [], []

#     for i in indices:
#         vec = mat[i].copy()
#         present = np.where(vec > 0)[0]
#         if len(present) < 10:
#             continue

#         hide_n = max(1, int(len(present) * args.hide_ratio))
#         hidden = np.random.choice(present, size=hide_n, replace=False)

#         observed_vec = vec.copy()
#         observed_vec[hidden] = 0.0

#         obs_norm = np.linalg.norm(observed_vec)
#         if obs_norm == 0:
#             continue

#         obs_unit = observed_vec / obs_norm
#         sim_row = mat_norm @ obs_unit
#         sim_row[i] = -1.0

#         neighbor_order = np.argsort(-sim_row)[: args.neighbors]
#         neighbor_sims = sim_row[neighbor_order]

#         ranked_skill_idxs = rank_skills_from_neighbors(
#             mat, neighbor_order, neighbor_sims, top_k=max(args.k, 100)
#         )

#         # Don't predict skills already observed
#         observed_set = set(np.where(observed_vec > 0)[0])
#         ranked_skill_idxs = [s for s in ranked_skill_idxs if s not in observed_set]

#         p, r, h, mrr = metrics_at_k(ranked_skill_idxs, hidden.tolist(), args.k)
#         P.append(p); R.append(r); H.append(h); M.append(mrr)

#     def avg(x): 
#         return float(np.mean(x)) if x else 0.0

#     print("\n=== Phase 0 Offline Evaluation (Proxy Skill Recovery) ===")
#     print(f"Roles evaluated: {len(P)}")
#     print(f"hide_ratio={args.hide_ratio}, neighbors={args.neighbors}, K={args.k}\n")
#     print(f"Precision@{args.k}: {avg(P):.4f}")
#     print(f"Recall@{args.k}:    {avg(R):.4f}")
#     print(f"HitRate@{args.k}:   {avg(H):.4f}")
#     print(f"MRR@{args.k}:       {avg(M):.4f}\n")


# if __name__ == "__main__":
#     main()
