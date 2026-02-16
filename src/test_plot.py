from src.recommender import CareerRecommender
from src.plot_graph import plot_career_graph

def main():
    rec = CareerRecommender()

    while True:
        role = input("\nEnter job role (or 'exit'): ").strip()
        if role.lower() == "exit":
            break

        try:
            resolved, suggestions = rec.resolve_role(role, cutoff=0.75)
            print("Resolved:", resolved)
            print("Suggestions:", suggestions)

            steps = input("Steps (default 3): ").strip()
            steps = int(steps) if steps else 3

            path = rec.graph_career_path(resolved, steps=steps)
            if not path:
                print("No path found for this role.")
                continue

            plot_career_graph(path)

        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
