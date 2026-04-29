# 🎯 Career Recommendation Engine — AI-Powered Career Path & Skill Gap Analyzer

> Analyzes a user's current skills, suggests best next roles, identifies skill gaps, and estimates learning time — powered by embedding-based semantic matching.


## 🚀 What It Does
- 🧠 Converts user skills into semantic embedding vectors
- 🎯 Matches user profile to job roles using cosine similarity
- 📉 Identifies exactly which skills are missing for target roles
- ⏱️ Estimates realistic learning time to bridge each gap
- 🔄 Personalized per user — not generic roadmaps

## 🛠️ Tech Stack
| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI + Python |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Similarity | Cosine Similarity (scikit-learn) |
| Data Processing | Pandas + NumPy |

## ⚙️ Setup
```bash
git clone https://github.com/moncey10/career_recommendation_engine.git
cd career_recommendation_engine
pip install -r requirements.txt
uvicorn main:app --reload
```

## 📡 Sample API Response
```json
{
  "recommended_roles": [
    {
      "role": "ML Engineer",
      "match_score": 0.89,
      "missing_skills": ["Kubernetes", "MLflow"],
      "estimated_learning_time": "3-4 weeks"
    }
  ]
}
```

## 👤 Author
**Moncey Patel** — AI/ML Engineer | [LinkedIn](https://linkedin.com/in/your-linkedin) | [GitHub](https://github.com/moncey10)
