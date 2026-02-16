import io
import matplotlib
matplotlib.use("Agg")  # IMPORTANT: server-safe backend (no popup)
import matplotlib.pyplot as plt
import networkx as nx

def plot_career_graph_png(graph_path) -> bytes:
    """
    Generates a PNG image (bytes) from graph_path list.
    Works inside FastAPI/uvicorn.
    """
    G = nx.DiGraph()

    for step in graph_path:
        from_role = step["from_role"]
        to_role = step["to_role"]
        weight = round(step.get("edge_weight", step.get("match_score", 0)), 2)
        G.add_edge(from_role, to_role, weight=weight)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, font_size=10, arrows=True)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Career Path Graph")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.getvalue()
