import pandas as pd

# ---- Dataset Column Names ----
# Try to auto-detect these from the Excel file
ROLE_COL = "Job Role"
SKILL_COL = "Skill Title"
LEVEL_COL = "Proficiency Level"
INDUSTRY_COL = "Industry"
DEPT_COL = "Department"
SKILL_CAT_COL = "Skill Category"
SKILL_TYPE_COL = "Skill Type"

def load_clean(path: str = "data/JobRoleToSkillRecommendation.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)
    
    # Print available columns for debugging
    print(f"Available columns in {path}: {list(df.columns)}")
    
    # Detect columns automatically
    global ROLE_COL, SKILL_COL, LEVEL_COL, INDUSTRY_COL, DEPT_COL, SKILL_CAT_COL, SKILL_TYPE_COL
    
    col_mapping = {
        'role': ['job role', 'role', 'jobrole', 'position', 'title'],
        'skill': ['skill title', 'skill', 'skillname', 'skill_name', 'technology'],
        'level': ['proficiency level', 'level', 'proficiency', 'skill level', 'proficiencylevel'],
        'industry': ['industry', 'sector', 'field'],
        'dept': ['department', 'dept', 'team', 'unit'],
        'skill_cat': ['skill category', 'skillcategory', 'skill_category', 'category', 'skilltype'],
        'skill_type': ['skill type', 'skilltype', 'skill_type', 'type'],
    }
    
    def detect_column(df, possible_names):
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        return None
    
    # Detect each column
    ROLE_COL = detect_column(df, col_mapping['role']) or ROLE_COL
    SKILL_COL = detect_column(df, col_mapping['skill']) or SKILL_COL
    LEVEL_COL = detect_column(df, col_mapping['level']) or LEVEL_COL
    INDUSTRY_COL = detect_column(df, col_mapping['industry']) or INDUSTRY_COL
    DEPT_COL = detect_column(df, col_mapping['dept']) or DEPT_COL
    SKILL_CAT_COL = detect_column(df, col_mapping['skill_cat']) or SKILL_CAT_COL
    SKILL_TYPE_COL = detect_column(df, col_mapping['skill_type']) or SKILL_TYPE_COL
    
    print(f"Detected columns:")
    print(f"  ROLE_COL: {ROLE_COL}")
    print(f"  SKILL_COL: {SKILL_COL}")
    print(f"  LEVEL_COL: {LEVEL_COL}")
    print(f"  INDUSTRY_COL: {INDUSTRY_COL}")
    print(f"  DEPT_COL: {DEPT_COL}")
    print(f"  SKILL_CAT_COL: {SKILL_CAT_COL}")
    print(f"  SKILL_TYPE_COL: {SKILL_TYPE_COL}")
    
    # Build KEEP_COLS with only columns that exist
    KEEP_COLS = []
    if ROLE_COL in df.columns:
        KEEP_COLS.append(ROLE_COL)
    if SKILL_COL in df.columns:
        KEEP_COLS.append(SKILL_COL)
    if LEVEL_COL in df.columns:
        KEEP_COLS.append(LEVEL_COL)
    if INDUSTRY_COL in df.columns:
        KEEP_COLS.append(INDUSTRY_COL)
    if DEPT_COL in df.columns:
        KEEP_COLS.append(DEPT_COL)
    if SKILL_CAT_COL in df.columns:
        KEEP_COLS.append(SKILL_CAT_COL)
    if SKILL_TYPE_COL in df.columns:
        KEEP_COLS.append(SKILL_TYPE_COL)
    
    if not KEEP_COLS:
        raise ValueError(f"Could not detect any required columns in {path}")
    
    # Keep only required columns
    df = df[KEEP_COLS].copy()
    
    # Drop rows missing core fields
    df = df.dropna(subset=[ROLE_COL, SKILL_COL, LEVEL_COL])
    
    # Normalize text fields
    df[ROLE_COL] = df[ROLE_COL].astype(str).str.strip()
    df[SKILL_COL] = df[SKILL_COL].astype(str).str.strip().str.lower()
    if INDUSTRY_COL in df.columns:
        df[INDUSTRY_COL] = df[INDUSTRY_COL].astype(str).str.strip()
    if DEPT_COL in df.columns:
        df[DEPT_COL] = df[DEPT_COL].astype(str).str.strip()
    
    # Ensure numeric proficiency
    df[LEVEL_COL] = pd.to_numeric(df[LEVEL_COL], errors="coerce")
    df = df.dropna(subset=[LEVEL_COL])
    
    # Keep non-negative proficiency
    df = df[df[LEVEL_COL] >= 0]
    
    return df

if __name__ == "__main__":
    df = load_clean()
    print("✅ Cleaned shape:", df.shape)
    print("✅ Unique roles:", df[ROLE_COL].nunique())
    print("✅ Unique skills:", df[SKILL_COL].nunique())
    print("\nSample rows:\n", df.head(5))
