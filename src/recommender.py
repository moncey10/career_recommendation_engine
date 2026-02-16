import re
from difflib import get_close_matches
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from src.preprocess import (
    load_clean,
    ROLE_COL,
    SKILL_COL,
    LEVEL_COL,
    INDUSTRY_COL,
    DEPT_COL,
)

from src.embeddings import SkillEmbeddingIndex


# Career level hierarchy (lower = entry level, higher = senior level)
LEVEL_HIERARCHY = {
    "intern": 0, "trainee": 0, "apprentice": 0,
    "junior": 1, "jr": 1, "associate": 1, "entry": 1,
    "mid": 2, "mid-level": 2, "intermediate": 2,
    "senior": 3, "sr": 3, " iii": 3, "3": 3,
    "staff": 4, "principal": 4, " iv": 4, "4": 4,
    "lead": 5, "senior lead": 5, "team lead": 5,
    "manager": 6, "senior manager": 6,
    "director": 7, "senior director": 7,
    "head": 8, "vp": 9, "vice president": 9,
    "executive": 10, "svp": 10,
    "cto": 11, "chief": 11, "ceo": 11,
}

CATEGORY_WEIGHT = {
    "Technical": 1.40,
    "Domain": 1.25,
    "Functional": 1.15,
    "Behavioral": 1.00,
    "Soft": 1.00,
}
DEFAULT_CAT_W = 1.0


class CareerRecommender:
    """
    Improved Career Recommender with logical progression paths.
    
    Key improvements:
    1. Strict vertical progression within same specialization
    2. Horizontal moves only when no vertical path exists
    3. Prevents illogical jumps (Backend → Frontend → Full Stack)
    4. Clear role family detection
    """
    def from_artifacts(cls, artifacts_dir: str = "artifacts"):
        self = cls.__new__(cls)   # bypass __init__

        # load sparse matrix
        self.matrix_sp = sparse.load_npz(os.path.join(artifacts_dir, "matrix.npz"))

        # load roles + skills
        with open(os.path.join(artifacts_dir, "roles.json"), "r", encoding="utf-8") as f:
            self.roles = json.load(f)

        with open(os.path.join(artifacts_dir, "skills.json"), "r", encoding="utf-8") as f:
            self.skills = json.load(f)

        with open(os.path.join(artifacts_dir, "mappings.json"), "r", encoding="utf-8") as f:
            maps = json.load(f)

        self.role_to_idx = maps["role_to_idx"]
        self.skill_to_idx = maps["skill_to_idx"]

        # load embeddings without loading model
        self.embed_index = SkillEmbeddingIndex(
            model_name="all-MiniLM-L6-v2",
            cache_dir=artifacts_dir,
            load_only=True
        )

        ok = self.embed_index.load("embeddings.pkl")
        if not ok:
            raise RuntimeError("embeddings.pkl not found. Run build_artifacts.py")

        return self

    def __init__(self, path: str = "data/JobRoleToSkillRecommendation.xlsx"):
        self.df = load_clean(path)

        self.role_col = ROLE_COL
        self.skill_col = SKILL_COL
        self.level_col = LEVEL_COL
        self.industry_col = INDUSTRY_COL
        self.dept_col = DEPT_COL

        # Skill category column (safe fallback)
        self.skill_cat_col = "Skill Category"
        if self.skill_cat_col not in self.df.columns:
            for cand in ["skill_category", "SkillCategory", "Skill_Category", "Category"]:
                if cand in self.df.columns:
                    self.skill_cat_col = cand
                    break

        self.matrix = self.df.pivot_table(
            index=self.role_col,
            columns=self.skill_col,
            values=self.level_col,
            aggfunc="max",
            fill_value=0,
        )

        self._vec_cache: Dict[str, Dict[str, float]] = {}

        # Embeddings
        self.embed_index = SkillEmbeddingIndex(
            model_name="all-MiniLM-L6-v2",
            cache_dir="cache",
        )
        loaded = self.embed_index.load("embeddings_phase1.pkl")
        if not loaded:
            role_to_skillweights = {r: self._role_skill_vector(r) for r in self.list_roles()}
            self.embed_index.build_role_embeddings(role_to_skillweights)
            self.embed_index.save("embeddings_phase1.pkl")

    # ----------------- basics -----------------
    def list_roles(self) -> List[str]:
        return sorted(self.matrix.index.tolist())

    def resolve_role(self, role_text: str, cutoff: float = 0.6) -> Tuple[str, List[str]]:
        roles = self.list_roles()
        matches = get_close_matches(role_text, roles, n=5, cutoff=cutoff)
        if not matches:
            raise ValueError("No close role match found. Please check spelling.")
        return matches[0], matches

    def _role_mode_value(self, role: str, col: str):
        rows = self.df[self.df[self.role_col] == role]
        m = rows[col].mode()
        return m.iloc[0] if not m.empty else None

    def _allowed_roles(self, current_role: str, same_industry: bool, same_department: bool) -> set:
        cur_ind = self._role_mode_value(current_role, self.industry_col)
        cur_dep = self._role_mode_value(current_role, self.dept_col)

        allowed = set(self.matrix.index)

        if same_industry and cur_ind is not None:
            allowed &= set(self.df[self.df[self.industry_col] == cur_ind][self.role_col].unique())

        if same_department and cur_dep is not None:
            allowed &= set(self.df[self.df[self.dept_col] == cur_dep][self.role_col].unique())

        allowed.discard(current_role)
        return allowed

    # ----------------- IMPROVED career hierarchy extraction -----------------
    def _extract_role_level(self, role: str) -> int:
        """
        Extract career level from role name.
        Returns a numeric level based on LEVEL_HIERARCHY.
        """
        role_lower = role.lower()
        
        # Check for multi-word level indicators first (more specific)
        for level_name, level_value in sorted(LEVEL_HIERARCHY.items(), key=lambda x: -len(x[0])):
            if level_name in role_lower:
                return level_value
        
        # Default: base level (no level indicator found)
        return 1
    
    def _extract_base_role(self, role: str) -> str:
        """
        Extract the base role name without level indicators.
        
        IMPROVED: More accurate extraction that preserves specialization.
        
        Examples:
            "Senior Back End Developer" -> "Back End Developer"
            "Back End Developer" -> "Back End Developer"
            "Lead Front End Developer" -> "Front End Developer"
            "Full Stack Developer" -> "Full Stack Developer"
            "Data Analyst" -> "Data Analyst"
        """
        # Create pattern that matches level prefixes
        level_prefixes = [
            "chief", "cto", "ceo", "svp", "vp", "vice president",
            "senior director", "director", "senior manager", "manager",
            "head", "lead", "team lead", "senior lead",
            "principal", "staff", "senior", "sr.",
            "junior", "jr.", "intern", "trainee", "apprentice",
            "associate", "entry", "mid-level", "mid", "intermediate"
        ]
        
        # Sort by length (longest first) to match "senior director" before "senior"
        level_prefixes.sort(key=len, reverse=True)
        
        base_role = role.strip()
        
        # Remove level prefixes from the start
        for prefix in level_prefixes:
            # Case-insensitive match at word boundary
            pattern = r'^' + re.escape(prefix) + r'\s+'
            base_role = re.sub(pattern, '', base_role, flags=re.IGNORECASE)
        
        # Also handle roman numerals and numbers at the end
        base_role = re.sub(r'\s+(III|II|I|IV|V|VI)$', '', base_role, flags=re.IGNORECASE)
        base_role = re.sub(r'\s+\d+$', '', base_role)
        
        return base_role.strip()
    
    def _get_role_specialization(self, role: str) -> str:
        """
        Get the core specialization of a role.
        
        This helps distinguish between:
        - Back End Developer
        - Front End Developer  
        - Full Stack Developer
        - Mobile Developer
        etc.
        
        Returns a normalized specialization key.
        """
        base = self._extract_base_role(role).lower()
        
        # Define specialization keywords (order matters - more specific first)
        specializations = {
            'backend': ['back end', 'backend', 'back-end'],
            'frontend': ['front end', 'frontend', 'front-end'],
            'fullstack': ['full stack', 'fullstack', 'full-stack'],
            'mobile': ['mobile', 'ios', 'android'],
            'devops': ['devops', 'dev ops', 'site reliability'],
            'data': ['data engineer', 'data scientist', 'data analyst'],
            'machine_learning': ['machine learning', 'ml engineer', 'ai engineer'],
            'qa': ['qa', 'quality assurance', 'test engineer', 'sdet'],
            'security': ['security', 'cybersecurity', 'infosec'],
            'cloud': ['cloud engineer', 'cloud architect'],
            'database': ['database', 'dba'],
            'software': ['software engineer', 'software developer'],
            'web': ['web developer'],
        }
        
        # Check for specialization keywords
        for spec_key, keywords in specializations.items():
            for keyword in keywords:
                if keyword in base:
                    return spec_key
        
        # If no specific specialization found, return the base itself
        return base
    
    def _has_prerequisite_experience(self, current_role: str, target_role: str, visited_roles: set = None) -> bool:
        """
        Check if user has prerequisite experience for target role.
        
        SIMPLIFIED LOGIC:
        - Frontend/Backend → Full Stack: BLOCKED (need both specializations)
        - All other moves: ALLOWED
        
        The career priority scoring handles the rest.
        """
        cur_spec = self._get_role_specialization(current_role)
        target_spec = self._get_role_specialization(target_role)
        
        # Full Stack requires BOTH Frontend AND Backend experience
        if target_spec == 'fullstack':
            # Already Full Stack? Can progress within Full Stack
            if cur_spec == 'fullstack':
                return True  # Senior Full Stack, Lead Full Stack, etc.
            
            # Coming from single specialization? BLOCK Full Stack
            # (They need to learn the other specialization first)
            if cur_spec in ['frontend', 'backend']:
                # Check if they have experience in the other specialization
                if visited_roles:
                    visited_specs = {self._get_role_specialization(r) for r in visited_roles}
                    has_frontend = 'frontend' in visited_specs or cur_spec == 'frontend'
                    has_backend = 'backend' in visited_specs or cur_spec == 'backend'
                    
                    if has_frontend and has_backend:
                        return True  # ✅ Has both → Full Stack allowed!
                
                # Single specialization → Block Full Stack
                return False  # ❌ Need both Frontend + Backend first
        
        # All other moves are allowed
        return True
    
    def _is_same_role_family(self, role1: str, role2: str) -> bool:
        """
        Check if two roles belong to the same role family.
        
        IMPROVED: Uses specialization detection for accurate matching.
        
        Examples:
            ("Back End Developer", "Senior Back End Developer") -> True
            ("Back End Developer", "Front End Developer") -> False
            ("Back End Developer", "Full Stack Developer") -> False (different specializations)
        """
        base1 = self._extract_base_role(role1)
        base2 = self._extract_base_role(role2)
        
        # Normalize
        base1_norm = " ".join(base1.lower().split())
        base2_norm = " ".join(base2.lower().split())
        
        # Exact match
        if base1_norm == base2_norm:
            return True
        
        # Check specialization
        spec1 = self._get_role_specialization(role1)
        spec2 = self._get_role_specialization(role2)
        
        # Same specialization = same family
        return spec1 == spec2

    # ----------------- role vector & similarity -----------------
    def _role_skill_vector(self, role: str) -> Dict[str, float]:
        if role in self._vec_cache:
            return self._vec_cache[role]

        row = self.matrix.loc[role]
        skills = row[row > 0].to_dict()

        cat_map = {}
        if self.skill_cat_col in self.df.columns:
            sub = self.df[self.df[self.role_col] == role]
            cat_map = (
                sub.groupby(self.skill_col)[self.skill_cat_col]
                .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
                .to_dict()
            )

        weighted_skills = {}
        for sk, val in skills.items():
            cat = cat_map.get(sk)
            w = CATEGORY_WEIGHT.get(cat, DEFAULT_CAT_W) if cat else DEFAULT_CAT_W
            weighted_skills[sk] = val * w

        self._vec_cache[role] = weighted_skills
        return weighted_skills

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        keys = set(vec1.keys()) | set(vec2.keys())
        if not keys:
            return 0.0

        a = np.array([vec1.get(k, 0.0) for k in keys], dtype=np.float32)
        b = np.array([vec2.get(k, 0.0) for k in keys], dtype=np.float32)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))
    
    # ----------------- API compatibility methods -----------------
    def _get_vec(self, role: str) -> Dict[str, float]:
        """Alias for _role_skill_vector for API compatibility."""
        return self._role_skill_vector(role)
    
    def _weighted_cosine(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        """Weighted cosine similarity - alias for _cosine_similarity for API compatibility."""
        return self._cosine_similarity(v1, v2)
    
    def _semantic_score(self, a_role: str, b_role: str) -> float:
        """Semantic similarity using embeddings."""
        va = self.embed_index.role_to_vec.get(a_role)
        vb = self.embed_index.role_to_vec.get(b_role)
        if va is None or vb is None:
            return 0.0
        return float(np.dot(va, vb))  # normalized cosine
    
    def avg_role_level(self, role: str) -> float:
        """Average proficiency level across all skills for a role."""
        vec = self.matrix.loc[role]
        nz = vec[vec > 0]
        return float(nz.mean()) if len(nz) else 0.0
    
    def _progression_score(self, from_role: str, to_role: str) -> float:
        """
        Calculate progression score based on average skill level difference.
        Expects a modest upward progression.
        """
        a = self.avg_role_level(from_role)
        b = self.avg_role_level(to_role)
        delta = b - a

        if delta < -0.15:
            return 0.0

        score = 1.0 - abs(delta - 0.35) / 0.9
        return float(np.clip(score, 0.0, 1.0))
    
    def _gap_penalty(self, from_role: str, to_role: str, top_n: int = 20) -> float:
        """
        Calculate penalty based on skill gap size.
        Larger gaps = higher penalty.
        """
        gaps = self.skill_gap(from_role, to_role, top_n=top_n)
        total_gap = float(sum(g for _, g in gaps))
        pen = (total_gap - 6.0) / (30.0 - 6.0)
        return float(np.clip(pen, 0.0, 1.0))
    
    def top_contributing_skills(self, from_role: str, to_role: str, top_k: int = 10) -> List[Dict[str, float]]:
        """
        Find skills that contribute most to similarity between two roles.
        """
        from_vec = self._get_vec(from_role)
        to_vec = self._get_vec(to_role)

        contrib = []
        for sk in set(from_vec.keys()) & set(to_vec.keys()):
            c = float(from_vec.get(sk, 0.0) * to_vec.get(sk, 0.0))
            if c > 0:
                contrib.append((sk, c))

        contrib.sort(key=lambda x: x[1], reverse=True)
        return [{"skill": s, "contribution": c} for s, c in contrib[:top_k]]
    
    # ----------------- roadmap helpers -----------------
    def _skill_category(self, skill_name: str) -> str:
        """Get the category of a skill."""
        s = skill_name.strip().lower()
        if self.skill_cat_col in self.df.columns:
            rows = self.df[self.df[self.skill_col].astype(str).str.strip().str.lower() == s]
            if not rows.empty:
                val = rows[self.skill_cat_col].mode()
                if not val.empty and pd.notna(val.iloc[0]):
                    return str(val.iloc[0]).strip().title()
        return "General"

    def _clean_category(self, category: str) -> str:
        """Clean and normalize category names."""
        c = category.strip().title()
        mapping = {
            "Data Analytics And Information Technology Management": "Data & Tech",
            "Analytical Thinking": "Analytics",
            "Business Management": "Business",
            "Business Strategy": "Strategy",
        }
        return mapping.get(c, c)

    def _priority_from_gap(self, gap: float, category_raw: str) -> str:
        """Assign priority (High/Medium/Low) based on gap size and category."""
        cat = category_raw.lower()
        core = any(k in cat for k in ["technical", "domain", "data", "analytics", "engineering", "it"])

        if gap >= 4:
            return "High"
        if gap >= 2.5:
            return "High" if core else "Medium"
        if gap >= 1.5:
            return "Medium"
        return "Low"

    def _weeks_from_gap(self, gap: float, category_raw: str) -> int:
        """Estimate learning time in weeks based on gap and category."""
        if gap >= 5:
            base = 6
        elif gap >= 4:
            base = 4
        elif gap >= 3:
            base = 3
        elif gap >= 2:
            base = 2
        else:
            base = 1

        cat = category_raw.lower()
        if gap >= 3 and any(k in cat for k in ["technical", "domain", "data", "engineering", "it"]):
            base += 1

        return max(1, int(base))

    def _phase_from_item(self, item: Dict[str, Any]) -> str:
        """Assign learning phase based on priority."""
        if item["priority"] == "High":
            return "Foundation"
        if item["priority"] == "Medium":
            return "Core"
        return "Advanced"

    def _confidence(self, baseline: float, semantic: float, gap_penalty: float) -> str:
        """Calculate confidence level for a recommendation."""
        score = (0.45 * semantic) + (0.35 * baseline) + (0.20 * (1.0 - gap_penalty))
        if score >= 0.78:
            return "High"
        if score >= 0.62:
            return "Medium"
        return "Low"

    def learning_roadmap(
        self,
        from_role: str,
        to_role: str,
        max_items: int = 10,
        baseline: float = None,
        semantic: float = None,
        gap_penalty: float = None,
    ) -> Dict[str, Any]:
        """
        Generate a detailed learning roadmap from one role to another.
        Includes phases, week plan, and grouped categories.
        """
        gaps = self.skill_gap(from_role, to_role, top_n=max_items)

        items: List[Dict[str, Any]] = []
        for skill, gap in gaps:
            cat_raw = self._skill_category(skill)
            cat = self._clean_category(cat_raw)
            pr = self._priority_from_gap(gap, cat_raw)
            weeks = self._weeks_from_gap(gap, cat_raw)

            items.append(
                {
                    "skill": skill,
                    "gap": float(gap),
                    "category": cat,
                    "priority": pr,
                    "estimated_weeks": int(weeks),
                    "phase": self._phase_from_item({"priority": pr}),
                }
            )

        pr_rank = {"High": 0, "Medium": 1, "Low": 2}
        items.sort(key=lambda x: (pr_rank.get(x["priority"], 9), -x["gap"]))

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            grouped.setdefault(it["category"], []).append(it)

        # week plan: High alone, medium/low bundled 2 per block
        week_plan: List[Dict[str, Any]] = []
        week_cursor = 1
        bucket: List[Dict[str, Any]] = []

        def add_block(block_items: List[Dict[str, Any]], duration_weeks: int):
            nonlocal week_cursor, week_plan
            duration_weeks = max(1, int(duration_weeks))
            start = week_cursor
            end = week_cursor + duration_weeks - 1
            week_plan.append(
                {
                    "weeks": f"Week {start}" if start == end else f"Week {start}-{end}",
                    "focus_skills": block_items,
                }
            )
            week_cursor = end + 1

        for it in items:
            if it["priority"] == "High":
                if bucket:
                    dur = max(x["estimated_weeks"] for x in bucket)
                    add_block(bucket, dur)
                    bucket = []
                add_block([it], it["estimated_weeks"])
            else:
                bucket.append(it)
                if len(bucket) >= 2:
                    dur = max(x["estimated_weeks"] for x in bucket)
                    add_block(bucket, dur)
                    bucket = []

        if bucket:
            dur = max(x["estimated_weeks"] for x in bucket)
            add_block(bucket, dur)

        total_weeks = week_cursor - 1

        # phases
        phases_order = ["Foundation", "Core", "Advanced"]
        phases: List[Dict[str, Any]] = []
        for ph in phases_order:
            ph_items = [x for x in items if x["phase"] == ph]
            if not ph_items:
                continue
            phases.append(
                {
                    "name": ph,
                    "skills": ph_items,
                }
            )

        conf = None
        if baseline is not None and semantic is not None and gap_penalty is not None:
            conf = self._confidence(float(baseline), float(semantic), float(gap_penalty))

        return {
            "to_role": to_role,
            "confidence": conf,
            "total_estimated_weeks": int(total_weeks),
            "items": items,
            "phases": phases,
            "week_plan": week_plan,
        }

    # ----------------- skill gap calculation -----------------
    def skill_gap(self, current_role: str, target_role: str, top_n: int = 10) -> List[Tuple[str, float]]:
        if current_role not in self.matrix.index or target_role not in self.matrix.index:
            return []

        cur_vec = self._role_skill_vector(current_role)
        tgt_vec = self._role_skill_vector(target_role)

        gaps = {}
        for skill, tgt_val in tgt_vec.items():
            cur_val = cur_vec.get(skill, 0.0)
            if tgt_val > cur_val:
                gaps[skill] = tgt_val - cur_val

        sorted_gaps = sorted(gaps.items(), key=lambda x: -x[1])
        return sorted_gaps[:top_n]

    # ----------------- IMPROVED role recommendation -----------------
    def recommend_roles(
        self,
        current_role: str,
        top_k: int = 5,
        same_industry: bool = True,
        same_department: bool = True,
        prioritize_career_ladder: bool = True,
        visited_roles: set = None,  # NEW: Track career path history
    ) -> List[Dict[str, Any]]:
        """
        Recommend next career roles with IMPROVED logic.
        
        Key improvements:
        1. Prioritizes vertical progression within same specialization
        2. Only allows horizontal moves when no vertical path exists
        3. Prevents illogical jumps between specializations
        4. Checks prerequisites (e.g., can't be Full Stack without Backend experience)
        
        Args:
            visited_roles: Set of roles already visited in career path (for prerequisite checking)
        """
        if current_role not in self.matrix.index:
            raise ValueError(f"Role not found: {current_role}")

        allowed = self._allowed_roles(current_role, same_industry, same_department)
        if not allowed:
            return []

        cur_vec = self._role_skill_vector(current_role)
        cur_level = self._extract_role_level(current_role)
        cur_spec = self._get_role_specialization(current_role)
        
        # Initialize visited_roles if not provided
        if visited_roles is None:
            visited_roles = {current_role}

        # Calculate scores for all candidates
        candidates = []
        
        for target_role in allowed:
            target_vec = self._role_skill_vector(target_role)
            target_level = self._extract_role_level(target_role)
            target_spec = self._get_role_specialization(target_role)
            
            # Skip down-level moves (demotion)
            if target_level < cur_level:
                continue
            
            # **NEW: Check prerequisites**
            if not self._has_prerequisite_experience(current_role, target_role, visited_roles):
                continue  # Skip roles that require prerequisite experience
            
            # Calculate baseline similarity
            baseline_sim = self._cosine_similarity(cur_vec, target_vec)
            
            # Calculate skill gap penalty
            gaps = self.skill_gap(current_role, target_role, top_n=100)
            total_gap = sum(g[1] for g in gaps)
            gap_penalty = min(1.0, total_gap / 10.0)
            
            # Calculate embedding similarity
            emb_sim = 0.0
            if current_role in self.embed_index.role_to_vec and target_role in self.embed_index.role_to_vec:
                cur_emb = self.embed_index.role_to_vec[current_role]
                tgt_emb = self.embed_index.role_to_vec[target_role]
                emb_sim = self.embed_index.cosine(cur_emb, tgt_emb)
            
            # Calculate progression score (for API compatibility)
            progression = self._progression_score(current_role, target_role)
            
            # **CRITICAL: Career Ladder Priority**
            # This is the key to preventing illogical jumps
            career_priority = self._calculate_career_priority(
                current_role, cur_level, cur_spec,
                target_role, target_level, target_spec
            )
            
            # Combine scores with heavy weight on career priority
            if prioritize_career_ladder:
                final_score = (
                    0.15 * baseline_sim +
                    0.10 * emb_sim +
                    0.60 * career_priority +  # HIGHEST WEIGHT
                    0.15 * (1.0 - gap_penalty)
                )
            else:
                final_score = (
                    0.40 * baseline_sim +
                    0.30 * emb_sim +
                    0.30 * (1.0 - gap_penalty)
                )
            
            candidates.append({
                "next_role": target_role,
                "final_score": final_score,
                "score_breakdown": {
                    "baseline": baseline_sim,
                    "semantic": emb_sim,  # API expects 'semantic' not 'embedding'
                    "embedding": emb_sim,  # Keep both for compatibility
                    "progression": progression,  # API may need this
                    "gap_penalty": gap_penalty,
                    "career_priority": career_priority,
                },
                "metadata": {
                    "current_level": cur_level,
                    "target_level": target_level,
                    "level_jump": target_level - cur_level,
                    "same_specialization": cur_spec == target_spec,
                    "specialization": target_spec,
                },
            })
        
        # Sort by final score
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        return candidates[:top_k]
    
    def _calculate_career_priority(
        self,
        current_role: str,
        cur_level: int,
        cur_spec: str,
        target_role: str,
        target_level: int,
        target_spec: str,
    ) -> float:
        """
        Calculate career progression priority.
        
        UPDATED LOGIC - Skill Expansion First (T-Shaped Development):
        
        For Junior/Mid developers (Level 1-2):
        1. **Skill Expansion FIRST** (Frontend ↔ Backend) = 1.00
        2. Vertical Progression (Senior) = 0.85
        3. Management = 0.60
        
        For Senior developers (Level 3+):
        1. Vertical Progression = 1.00
        2. Management = 0.80
        3. Skill Expansion = 0.70
        
        This ensures:
        - Frontend → Backend (#1 priority at junior level)
        - Backend → Frontend (#1 priority at junior level)
        - Senior Frontend → Lead Frontend (#1 priority at senior level)
        """
        
        # **TIER 0: Block Full Stack without prerequisites**
        # This is handled in _has_prerequisite_experience, but we can also de-prioritize here
        if target_spec == 'fullstack' and cur_spec != 'fullstack':
            # Full Stack from single specialization = very low priority
            # (Will be blocked by prerequisite check anyway)
            return 0.10
        
        # **Determine career stage**
        is_junior_mid = cur_level <= 2  # Level 1-2 = Junior/Mid
        is_senior_plus = cur_level >= 3  # Level 3+ = Senior/Lead/Principal
        
        # **TIER 1: Skill Expansion (For Junior/Mid developers)**
        # Frontend ↔ Backend at SAME or +1 level
        if is_junior_mid:
            if (cur_spec == 'frontend' and target_spec == 'backend') or \
               (cur_spec == 'backend' and target_spec == 'frontend'):
                # Same level or +1 level lateral expansion
                if target_level <= cur_level + 1:
                    return 1.00  # 🥇 HIGHEST - Learn complementary skills first!
        
        # **TIER 2: Vertical Progression in Same Specialization**
        if cur_spec == target_spec:
            if target_level == cur_level + 1:
                # Next level in same spec
                if is_senior_plus:
                    return 1.00  # Seniors go vertical
                else:
                    return 0.85  # Juniors: vertical is secondary to expansion
            elif target_level == cur_level + 2:
                # Skip one level
                return 0.75
            elif target_level > cur_level:
                # Higher level
                return 0.70
            elif target_level == cur_level:
                # Lateral move within same spec
                return 0.50
        
        # **TIER 3: Combination Role (Full Stack) - After Prerequisites Met**
        if target_spec == 'fullstack':
            if cur_spec in ['frontend', 'backend']:
                # This will only be reached if prerequisites are met
                # (otherwise blocked by prerequisite check)
                if target_level <= cur_level + 1:
                    return 0.90  # High - natural progression to generalist
        
        # **TIER 4: Management Track**
        target_lower = target_role.lower()
        if any(keyword in target_lower for keyword in ['manager', 'director', 'head', 'lead', 'vp', 'chief']):
            if target_level >= cur_level:
                if is_senior_plus:
                    return 0.80  # Seniors → Management is natural
                else:
                    return 0.60  # Juniors → Management is less common
        
        # **TIER 5: Skill Expansion (For Senior developers)**
        if is_senior_plus:
            if (cur_spec == 'frontend' and target_spec == 'backend') or \
               (cur_spec == 'backend' and target_spec == 'frontend'):
                if target_level <= cur_level + 1:
                    return 0.70  # Seniors can still expand, but lower priority
        
        # **TIER 6: Related Specialization**
        if self._is_related_specialization(cur_spec, target_spec):
            if target_level > cur_level:
                return 0.45
            elif target_level == cur_level:
                return 0.30
        
        # **TIER 7: Different Specialization, Higher Level**
        if target_level > cur_level:
            return 0.25
        
        # **TIER 8: Career Change (Different spec, same level)**
        if target_level == cur_level:
            return 0.20
        
        # Should not reach here (downlevel moves filtered out earlier)
        return 0.0
    
    def _is_related_specialization(self, spec1: str, spec2: str) -> bool:
        """
        Check if two specializations are related.
        
        Examples of related specializations:
        - Backend + Frontend = related (both development)
        - Backend + Full Stack = related (full stack includes backend)
        - Data Engineer + Data Scientist = related (both data roles)
        - Backend + QA = not closely related
        """
        # Define specialization relationships
        related_groups = [
            {'backend', 'frontend', 'fullstack', 'web', 'software'},
            {'mobile', 'ios', 'android'},
            {'data', 'machine_learning'},
            {'devops', 'cloud', 'security'},
            {'qa', 'sdet'},
        ]
        
        for group in related_groups:
            if spec1 in group and spec2 in group:
                return True
        
        return False
    
    # ----------------- IMPROVED career path building -----------------
    def build_career_path(
        self,
        start_role: str,
        steps: int = 5,
        same_industry: bool = True,
        same_department: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Build a logical career path with STRICT progression rules.
        
        Algorithm:
        1. Start with current role
        2. For each step:
           a. Get recommendations (prioritized by career ladder)
           b. Select first unvisited role that follows logical progression
           c. Add to path and continue
        3. Stop when no valid next role exists
        """
        if start_role not in self.matrix.index:
            raise ValueError(f"Role not found: {start_role}")
        
        resolved, _ = self.resolve_role(start_role, cutoff=0.75)
        
        path = []
        current_role = resolved
        visited_roles = {resolved}
        
        current_level = self._extract_role_level(current_role)
        current_spec = self._get_role_specialization(current_role)
        
        for step in range(steps):
            # Get recommendations with career ladder prioritization
            recommendations = self.recommend_roles(
                current_role=current_role,
                top_k=10,  # Get more candidates to find best progression
                same_industry=same_industry,
                same_department=same_department,
                prioritize_career_ladder=True,  # CRITICAL
                visited_roles=visited_roles,  # NEW: Pass career history for prerequisite checking
            )
            
            if not recommendations:
                # No more roles available
                break
            
            # Find the best next role
            next_role = None
            score_data = None
            
            for item in recommendations:
                candidate = item["next_role"]
                
                # Skip if already visited
                if candidate in visited_roles:
                    continue
                
                # This is the best available next role
                next_role = candidate
                score_data = item
                break
            
            # Stop if no valid next role found
            if next_role is None:
                break
            
            # Add to path
            path.append({
                "from_role": current_role,
                "to_role": next_role,
                "edge_weight": round(score_data["final_score"], 2),
                "match_score": round(score_data["score_breakdown"]["baseline"], 2),
                "step": step + 1,
                "progression_type": self._get_progression_type(
                    current_role, next_role,
                    current_level, score_data["metadata"]["target_level"],
                    current_spec, score_data["metadata"]["specialization"]
                ),
            })
            
            # Update state
            visited_roles.add(next_role)
            current_role = next_role
            current_level = score_data["metadata"]["target_level"]
            current_spec = score_data["metadata"]["specialization"]
        
        return path
    
    def _get_progression_type(
        self,
        from_role: str,
        to_role: str,
        from_level: int,
        to_level: int,
        from_spec: str,
        to_spec: str,
    ) -> str:
        """
        Classify the type of career progression.
        """
        if from_spec == to_spec:
            if to_level > from_level:
                return "vertical"  # Same specialization, promoted
            else:
                return "lateral_same_spec"  # Same specialization, same level
        else:
            if self._is_related_specialization(from_spec, to_spec):
                if to_level > from_level:
                    return "diagonal_related"  # Related field, promoted
                else:
                    return "lateral_related"  # Related field, same level
            else:
                if to_level > from_level:
                    return "diagonal_unrelated"  # Different field, promoted
                else:
                    return "career_change"  # Different field, same level