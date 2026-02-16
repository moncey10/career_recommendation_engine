from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Set, Tuple, Optional, List, Dict, Any
from functools import lru_cache

from src.recommender import CareerRecommender
from src.plot_graph_improved import plot_career_graph_png


# ============================================================
# Recommender loader (FAST: artifacts) with safe fallback
# ============================================================
@lru_cache(maxsize=1)
def get_rec() -> CareerRecommender:
    """
    Loads the recommender once and caches it.
    Prefer artifacts-based load for speed.
    Falls back to slow init if artifacts method doesn't exist.
    """
    # FAST path (your Step C)
    if hasattr(CareerRecommender, "from_artifacts"):
        try:
            return CareerRecommender.from_artifacts("artifacts")
        except Exception:
            # If artifacts missing or corrupted, fallback to slow init
            return CareerRecommender()

    # Slow fallback (if you didn't implement Step C yet)
    return CareerRecommender()


# ============================================================
# App setup
# ============================================================
app = FastAPI(title="Career Recommendation API (Recommend / Roadmap / Explain)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_role(role: str) -> str:
    rec = get_rec()
    resolved, _ = rec.resolve_role(role, cutoff=0.75)
    return resolved


# ============================================================
# Schemas
# ============================================================
class RecommendRequest(BaseModel):
    current_role: str
    top_k: int = 5
    top_n_skills: int = 5
    same_industry: bool = True
    same_department: bool = True


class RoadmapRequest(BaseModel):
    from_role: str
    to_role: str
    top_n_skills: int = 10


class ExplainRequest(BaseModel):
    from_role: str
    to_role: str
    top_k_overlap: int = 10


# ============================================================
# Health
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# Helpers for graph path selection
# ============================================================
def _norm(s: str) -> str:
    return (s or "").strip().lower()


def get_level(role: str) -> int:
    r = _norm(role)
    if "intern" in r:
        return 0
    if "junior" in r or "jr" in r:
        return 1
    if "senior" in r or "sr" in r:
        return 3
    if "lead" in r or "principal" in r or "staff" in r:
        return 4
    return 2


def get_track(role: str) -> str:
    r = _norm(role)
    if "full stack" in r or "fullstack" in r:
        return "fullstack"
    if "front" in r or "frontend" in r or "ui" in r:
        return "frontend"
    if "back" in r or "backend" in r or "api" in r:
        return "backend"
    return "other"


def count_future_paths(rec: CareerRecommender, role: str, visited: Set[str], max_depth: int = 2) -> int:
    """
    WARNING: expensive. Keep depth low.
    """
    if max_depth <= 0:
        return 0

    ranked = rec.recommend_roles(
        current_role=role,
        top_k=20,
        same_industry=False,
        same_department=False,
    )

    count = 0
    for item in ranked:
        next_role = item.get("next_role")
        if next_role and next_role not in visited:
            count += 1
            if max_depth > 1:
                count += count_future_paths(rec, next_role, visited | {next_role}, max_depth - 1)

    return count


def pick_next_role_optimized(
    ranked: list,
    current_role: str,
    visited: set,
    remaining_steps: int,
    rec: CareerRecommender
) -> Tuple[Optional[str], Optional[dict], str]:
    cur_level = get_level(current_role)
    cur_track = get_track(current_role)

    candidates = [it for it in ranked if it.get("next_role") and it["next_role"] not in visited]
    if not candidates:
        return None, None, "No unvisited candidates"

    scored_candidates = []

    for candidate in candidates:
        next_role = candidate["next_role"]
        next_level = get_level(next_role)
        next_track = get_track(next_role)
        base_score = float(candidate.get("final_score", 0))

        # Level bonus
        if next_level > cur_level:
            level_bonus = 0.30
        elif next_level == cur_level:
            level_bonus = 0.10
        else:
            level_bonus = -0.20

        # Track bonus
        if cur_track == "fullstack":
            track_bonus = 0.35 if next_track == "fullstack" else -0.25
        else:
            if next_track == cur_track:
                track_bonus = 0.20
            elif next_track == "fullstack":
                track_bonus = 0.10
            elif (cur_track in ("backend", "frontend")) and (next_track in ("backend", "frontend")):
                track_bonus = -0.30
            else:
                track_bonus = -0.10

        # Future bonus (keep small depth)
        future_bonus = 0.0
        if remaining_steps > 1:
            future_count = count_future_paths(rec, next_role, visited | {next_role}, max_depth=2)
            future_bonus = min(future_count * 0.05, 0.40)

        total_score = base_score + level_bonus + track_bonus + future_bonus

        scored_candidates.append({
            "candidate": candidate,
            "next_role": next_role,
            "total_score": total_score,
            "breakdown": {"base": base_score, "level": level_bonus, "track": track_bonus, "future": future_bonus},
        })

    best = max(scored_candidates, key=lambda x: x["total_score"])
    reason = (
        f"Selected total={best['total_score']:.3f} "
        f"(base={best['breakdown']['base']:.2f}, level={best['breakdown']['level']:.2f}, "
        f"track={best['breakdown']['track']:.2f}, future={best['breakdown']['future']:.2f})"
    )
    return best["next_role"], best["candidate"], reason


# ============================================================
# Role Gap / Skill Gap endpoints (between two roles)
# ============================================================
@app.get("/role_gap")
@app.get("/skill_gap")  # alias
def role_gap(
    current_role: str = Query(...),
    target_role: str = Query(...),
    top_n: int = Query(10, ge=1, le=50),
):
    rec = get_rec()
    cur, cur_sug = rec.resolve_role(current_role, cutoff=0.75)
    tgt, tgt_sug = rec.resolve_role(target_role, cutoff=0.75)

    gaps = rec.skill_gap(cur, tgt, top_n=top_n)
    return {
        "current_role": current_role,
        "resolved_current_role": cur,
        "current_suggestions": cur_sug,
        "target_role": target_role,
        "resolved_target_role": tgt,
        "target_suggestions": tgt_sug,
        "missing_skills": [{"skill": s, "gap": float(g)} for s, g in gaps],
    }


# ============================================================
# Recommend
# ============================================================
@app.post("/recommend")
def recommend(req: RecommendRequest):
    rec = get_rec()
    try:
        resolved, suggestions = rec.resolve_role(req.current_role, cutoff=0.75)

        ranked = rec.recommend_roles(
            current_role=resolved,
            top_k=req.top_k,
            same_industry=req.same_industry,
            same_department=req.same_department,
        )

        def _track(role: str) -> str:
            r = (role or "").strip().lower()
            if "full stack" in r or "fullstack" in r:
                return "fullstack"
            if "front end" in r or "frontend" in r or "ui" in r:
                return "frontend"
            if "back end" in r or "backend" in r or "api" in r:
                return "backend"
            return "other"

        cur_track = _track(resolved)
        if cur_track == "backend":
            ranked = [it for it in ranked if _track(it["next_role"]) in ("backend", "fullstack")]
        elif cur_track == "frontend":
            ranked = [it for it in ranked if _track(it["next_role"]) in ("frontend", "fullstack")]
        elif cur_track == "fullstack":
            ranked = [it for it in ranked if _track(it["next_role"]) == "fullstack"]

        out = []
        for item in ranked:
            role = item["next_role"]
            sb = item["score_breakdown"]

            gaps = rec.skill_gap(resolved, role, top_n=req.top_n_skills)
            top_missing = [{"skill": s, "gap": float(g)} for s, g in gaps]

            rm = rec.learning_roadmap(
                resolved,
                role,
                max_items=req.top_n_skills,
                baseline=sb["baseline"],
                semantic=sb["semantic"],
                gap_penalty=sb["gap_penalty"],
            )

            out.append({
                "next_role": role,
                "final_score": float(item["final_score"]),
                "confidence": rm.get("confidence"),
                "top_missing_skills": top_missing,
            })

        return {
            "input_role": req.current_role,
            "resolved_role": resolved,
            "suggestions": suggestions,
            "recommendations": out,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


# ============================================================
# Roadmap
# ============================================================
@app.post("/roadmap")
def roadmap(req: RoadmapRequest):
    rec = get_rec()
    try:
        from_resolved, from_sug = rec.resolve_role(req.from_role, cutoff=0.75)
        to_resolved, to_sug = rec.resolve_role(req.to_role, cutoff=0.75)

        base = rec._weighted_cosine(rec._get_vec(from_resolved), rec._get_vec(to_resolved))
        sem = rec._semantic_score(from_resolved, to_resolved)
        gap_pen = rec._gap_penalty(from_resolved, to_resolved)

        rm = rec.learning_roadmap(
            from_resolved,
            to_resolved,
            max_items=req.top_n_skills,
            baseline=base,
            semantic=sem,
            gap_penalty=gap_pen,
        )

        return {
            "from_role": req.from_role,
            "resolved_from": from_resolved,
            "from_suggestions": from_sug,
            "to_role": req.to_role,
            "resolved_to": to_resolved,
            "to_suggestions": to_sug,
            "roadmap": rm,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


# ============================================================
# Explain
# ============================================================
@app.post("/explain")
def explain(req: ExplainRequest):
    rec = get_rec()
    try:
        from_resolved, from_sug = rec.resolve_role(req.from_role, cutoff=0.75)
        to_resolved, to_sug = rec.resolve_role(req.to_role, cutoff=0.75)

        base = rec._weighted_cosine(rec._get_vec(from_resolved), rec._get_vec(to_resolved))
        sem = rec._semantic_score(from_resolved, to_resolved)
        prog = rec._progression_score(from_resolved, to_resolved)
        gap_pen = rec._gap_penalty(from_resolved, to_resolved)

        return {
            "from_role": req.from_role,
            "resolved_from": from_resolved,
            "from_suggestions": from_sug,
            "to_role": req.to_role,
            "resolved_to": to_resolved,
            "to_suggestions": to_sug,
            "score_breakdown": {
                "semantic": float(sem),
                "baseline": float(base),
                "progression": float(prog),
                "gap_penalty": float(gap_pen),
            },
            "top_overlap_skills": rec.top_contributing_skills(from_resolved, to_resolved, top_k=req.top_k_overlap),
            "top_missing_skills": [
                {"skill": s, "gap": float(g)}
                for s, g in rec.skill_gap(from_resolved, to_resolved, top_n=req.top_k_overlap)
            ],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


# ============================================================
# Graph Image
# ============================================================
@app.get("/graph_image")
def graph_image(role: str = Query(...), steps: int = Query(3, ge=1, le=20), _t: int = Query(None)):
    rec = get_rec()

    resolved = normalize_role(role)

    graph_path = [resolved]
    visited_roles = {resolved}
    current_role = resolved
    no_path_reason = ""

    for step_idx in range(steps):
        remaining_steps = steps - step_idx - 1

        ranked = rec.recommend_roles(
            current_role=current_role,
            top_k=20,
            same_industry=True,
            same_department=True,
        )

        next_role, score_data, reason = pick_next_role_optimized(
            ranked, current_role, visited_roles, remaining_steps, rec
        )

        if next_role is None:
            ranked2 = rec.recommend_roles(
                current_role=current_role,
                top_k=20,
                same_industry=False,
                same_department=False,
            )
            next_role, score_data, reason = pick_next_role_optimized(
                ranked2, current_role, visited_roles, remaining_steps, rec
            )

        if next_role is None:
            no_path_reason = f"Stopped at '{current_role}': {reason}"
            break

        graph_path.append(next_role)
        visited_roles.add(next_role)
        current_role = next_role

    steps_data = [{"from_role": graph_path[i], "to_role": graph_path[i + 1]} for i in range(len(graph_path) - 1)]

    img_bytes = plot_career_graph_png(steps_data, start_role=resolved)

    headers = {
        "X-Requested-Steps": str(steps),
        "X-Actual-Steps": str(max(0, len(graph_path) - 1)),
        "X-Resolved-Role": resolved,
        "X-No-Path-Reason": no_path_reason or "OK",
    }

    return Response(content=img_bytes, media_type="image/png", headers=headers)


# ============================================================
# Graph Info
# ============================================================
@app.get("/graph_info")
def graph_info(role: str = Query(...), steps: int = Query(3, ge=1, le=20)):
    rec = get_rec()

    resolved, suggestions = rec.resolve_role(role, cutoff=0.75)

    graph_path = [resolved]
    visited_roles = {resolved}
    current_role = resolved

    no_path_reason = ""
    debug_info = []

    for step_idx in range(steps):
        remaining_steps = steps - step_idx - 1

        ranked = rec.recommend_roles(
            current_role=current_role,
            top_k=20,
            same_industry=True,
            same_department=True,
        )

        next_role, score_data, reason = pick_next_role_optimized(
            ranked, current_role, visited_roles, remaining_steps, rec
        )

        if next_role is None:
            ranked2 = rec.recommend_roles(
                current_role=current_role,
                top_k=20,
                same_industry=False,
                same_department=False,
            )
            next_role, score_data, reason = pick_next_role_optimized(
                ranked2, current_role, visited_roles, remaining_steps, rec
            )

        step_debug = {
            "step": step_idx + 1,
            "from": current_role,
            "to": next_role,
            "reason": reason,
            "candidates_count": len(ranked),
        }

        if next_role is None:
            no_path_reason = f"Stopped at '{current_role}': {reason}"
            debug_info.append(step_debug)
            break

        debug_info.append(step_debug)
        graph_path.append(next_role)
        visited_roles.add(next_role)
        current_role = next_role

    actual_steps = max(0, len(graph_path) - 1)
    path_exists = actual_steps > 0

    return {
        "input_role": role,
        "resolved_role": resolved,
        "suggestions": suggestions,
        "requested_steps": steps,
        "actual_steps": actual_steps,
        "path_exists": path_exists,
        "path": graph_path,
        "reason": no_path_reason or ("OK" if path_exists else "No path found"),
        "debug": debug_info,
    }
