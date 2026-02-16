import io
import matplotlib
matplotlib.use("Agg")  # IMPORTANT: server-safe backend (no popup)
import matplotlib.pyplot as plt
import networkx as nx

def plot_career_graph_png(graph_path, start_role=None) -> bytes:
    """
    Generates a PNG image (bytes) from graph_path list.
    Works inside FastAPI/uvicorn.
    
    Args:
        graph_path: List of path steps with from_role, to_role, edge_weight
        start_role: Starting role (used when path is empty to show message)
    
    Returns:
        PNG image bytes
    """
    # Handle empty path - show informative message
    if not graph_path:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        
        message = f"No Career Path Found"
        if start_role:
            message += f"\n\nStarting Role: {start_role}\n\n"
            message += "Possible reasons:\n"
            message += "• No suitable next roles in the same industry/department\n"
            message += "• All potential paths have been exhausted\n"
            message += "• Role requirements don't match progression criteria\n\n"
            message += "Try:\n"
            message += "• Removing industry/department filters\n"
            message += "• Choosing a different starting role\n"
            message += "• Reducing the number of steps"
        else:
            message += "\n\nNo recommendations available for this role."
        
        ax.text(0.5, 0.5, message, 
                ha='center', va='center', 
                fontsize=14, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                transform=ax.transAxes)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return buf.getvalue()
    
    # Build graph from path
    G = nx.DiGraph()
    
    for step in graph_path:
        from_role = step["from_role"]
        to_role = step["to_role"]
        weight = round(step.get("edge_weight", step.get("match_score", 0)), 2)
        G.add_edge(from_role, to_role, weight=weight)
    
    # Calculate layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=3000, font_size=10, arrows=True)
    
    # Add edge labels (scores)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Title with path length info
    plt.title(f"Career Path Graph ({len(graph_path)} step{'s' if len(graph_path) != 1 else ''})")
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.getvalue()
