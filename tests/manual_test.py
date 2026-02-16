from src.recommender import CareerRecommender

rec = CareerRecommender()

roles = rec.list_roles()
print("Total roles:", len(roles))
print("First 10 roles:", roles[:10])

current_role = roles[0]
print("\nCurrent role:", current_role)

top = rec.recommend_roles(current_role, top_k=5, same_industry=True, same_department=True)
print("\nTop recommended roles:")
for role, score in top:
    print(f"- {role} | score={score:.3f}")

if top:
    target_role = top[0][0]
    gaps = rec.skill_gap(current_role, target_role, top_n=10)

    print(f"\nSkill gaps: {current_role} -> {target_role}")
    for skill, diff in gaps:
        print(f"- {skill} (+{diff})")

print("\n2-step career path:")
roadmap = rec.career_path(current_role, steps=2, top_k_each=3)
for step in roadmap:
    print(step["from_role"], "->", step["to_role"], "|", round(step["match_score"], 3))
