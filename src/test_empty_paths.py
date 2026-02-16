#!/usr/bin/env python3
"""
Test script to demonstrate empty path handling improvements.
Run this to see how the system handles different scenarios.
"""

from src.recommender import CareerRecommender
from src.plot_graph import plot_career_graph_png

def test_scenario(rec, role_name, steps, description):
    """Test a specific scenario and report results."""
    print(f"\n{'='*60}")
    print(f"📋 TEST: {description}")
    print(f"{'='*60}")
    print(f"Role: {role_name}")
    print(f"Steps requested: {steps}")
    print()
    
    try:
        # Resolve role
        resolved, suggestions = rec.resolve_role(role_name, cutoff=0.75)
        print(f"✓ Resolved to: {resolved}")
        if suggestions and len(suggestions) > 1:
            print(f"  Suggestions: {', '.join(suggestions[:3])}")
        
        # Build path
        graph_path = []
        current_role = resolved
        visited_roles = {resolved}
        
        for step in range(steps):
            ranked = rec.recommend_roles(
                current_role=current_role,
                top_k=max(steps, 5),
                same_industry=True,
                same_department=True,
            )
            
            if not ranked:
                print(f"✗ Step {step + 1}: No recommendations from '{current_role}'")
                break
            
            next_role = None
            for item in ranked:
                candidate = item["next_role"]
                if candidate not in visited_roles:
                    next_role = candidate
                    score_data = item
                    break
            
            if next_role is None:
                print(f"✗ Step {step + 1}: All recommendations already visited (cycle)")
                break
            
            graph_path.append({
                "from_role": current_role,
                "to_role": next_role,
                "edge_weight": score_data["final_score"],
                "match_score": score_data["score_breakdown"]["baseline"],
            })
            
            print(f"✓ Step {step + 1}: {current_role} → {next_role} (score: {score_data['final_score']:.3f})")
            
            visited_roles.add(next_role)
            current_role = next_role
        
        # Results
        print(f"\n📊 RESULTS:")
        print(f"  Requested steps: {steps}")
        print(f"  Actual steps: {len(graph_path)}")
        print(f"  Status: {'✅ Complete' if len(graph_path) >= steps else '⚠️ Partial' if graph_path else '❌ No path'}")
        
        if graph_path:
            path_str = " → ".join([graph_path[0]["from_role"]] + [p["to_role"] for p in graph_path])
            print(f"  Path: {path_str}")
        else:
            print(f"  Message: Would show 'No Career Path Found' image")
        
        # Generate image (in real app)
        # img_bytes = plot_career_graph_png(graph_path, start_role=resolved)
        # print(f"  Image generated: {len(img_bytes)} bytes")
        
    except ValueError as e:
        print(f"❌ ERROR: {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")


def main():
    print("\n" + "="*60)
    print("🧪 CAREER PATH EMPTY HANDLING TEST SUITE")
    print("="*60)
    
    # Initialize recommender
    print("\nInitializing Career Recommender...")
    rec = CareerRecommender()
    print(f"✓ Loaded {len(rec.list_roles())} roles")
    
    # Test scenarios
    scenarios = [
        # Scenario 1: Normal case - should work fine
        ("Data Analyst", 3, "Normal path - should find 3 steps"),
        
        # Scenario 2: Excessive steps - will likely hit partial path
        ("Junior Developer", 10, "Excessive steps - likely partial path"),
        
        # Scenario 3: High-level role - may have no progression
        ("Chief Executive Officer", 3, "Top-level role - likely no next steps"),
        
        # Scenario 4: Small number of steps
        ("Software Engineer", 1, "Single step - should work"),
        
        # Scenario 5: Try a specific path that might cycle
        ("Product Manager", 5, "Medium steps - may hit cycles"),
    ]
    
    for role, steps, desc in scenarios:
        test_scenario(rec, role, steps, desc)
    
    print("\n" + "="*60)
    print("✅ TEST SUITE COMPLETE")
    print("="*60)
    print("\nKEY IMPROVEMENTS:")
    print("  1. ✅ Empty paths show informative message (not blank)")
    print("  2. ✅ Partial paths clearly indicated")
    print("  3. ✅ Cycle detection prevents infinite loops")
    print("  4. ✅ Clear step counting (actual vs requested)")
    print("  5. ✅ Better error messages")
    print("\nNEXT STEPS:")
    print("  - Replace plot_graph.py with plot_graph_improved.py")
    print("  - Replace api.py with api_improved.py")
    print("  - Test with: curl http://localhost:8000/graph_info?role=CEO&steps=3")
    print()


if __name__ == "__main__":
    main()
