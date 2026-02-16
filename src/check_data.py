import pandas as pd

df = pd.read_excel(
    r"C:\Users\moncey\career_path_recommendetion\data\JobRoleToSkillRecommendation.xlsx"
)

print("SHAPE:")
print(df.shape)

print("\nCOLUMNS:")
print(list(df.columns))

print("\nFIRST 3 ROWS:")
print(df.head(3))

