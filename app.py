import streamlit as st
import pdfplumber
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="HireLens AI | ATS Intelligence",
    page_icon="💼",
    layout="wide"
)

# -----------------------------
# UI STYLE
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f8f9fc;
    font-family: 'Segoe UI', sans-serif;
}

h1 {
    text-align: center;
    color: #111827;
    font-weight: 800;
}

[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.title("💼 HireLens AI – ATS Intelligence Platform")
st.write("Resume vs Job Description Analysis")

# -----------------------------
# INPUTS
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=250)
run = st.button("Run ATS Analysis")

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text(pdf):
    text = ""
    with pdfplumber.open(pdf) as pdf_file:
        for page in pdf_file.pages:
            text += page.extract_text() or ""
    return text.lower()

# -----------------------------
# CLEAN TOKENIZATION
# -----------------------------
def tokenize(text):
    return re.findall(r'\b[a-zA-Z][a-zA-Z\-]{2,}\b', text.lower())

# -----------------------------
# SKILLS DATABASE
# -----------------------------
SKILLS_DB = [
    "python","java","c++","javascript","typescript","sql",
    "pandas","numpy","matplotlib","seaborn","scipy",
    "machine learning","deep learning","neural networks",
    "tensorflow","keras","pytorch","scikit-learn",
    "nlp","transformers","langchain","rag","llm","faiss",
    "aws","azure","gcp",
    "docker","kubernetes","flask","fastapi","streamlit",
    "mysql","postgresql","mongodb",
    "mlflow","airflow"
]

# -----------------------------
# SKILL EXTRACTION (FIXED)
# -----------------------------
def extract_skills(text):
    text = text.lower()
    found = set()

    for skill in SKILLS_DB:
        if skill in text:
            found.add(skill)

    return found

# -----------------------------
# JOB KEYWORDS EXTRACTION (IMPORTANT FIX)
# -----------------------------
def extract_job_keywords(text):
    words = tokenize(text)

    stop_words = {
        "and","the","for","with","this","that","you","are",
        "was","were","will","from","into","using","about","your"
    }

    return set([w for w in words if w not in stop_words])

# -----------------------------
# RECRUITER FEEDBACK
# -----------------------------
def recruiter_feedback(score, missing):
    missing = list(missing)[:5]

    if score >= 0.80:
        return "Strong match. Candidate is interview-ready."
    elif score >= 0.60:
        return "Good match. Improve: " + ", ".join(missing)
    else:
        return "Weak match. Critical gaps: " + ", ".join(missing)

# -----------------------------
# MAIN
# -----------------------------
if run and uploaded_file and job_description:

    resume_text = extract_text(uploaded_file)

    # embeddings
    resume_emb = model.encode(resume_text)
    job_emb = model.encode(job_description)

    semantic_score = cosine_similarity(
        [resume_emb], [job_emb]
    )[0][0]

    # -----------------------------
    # SKILLS LOGIC (FIXED CORE)
    # -----------------------------
    resume_skills = extract_skills(resume_text)
    job_skills_db = extract_skills(job_description)
    job_keywords = extract_job_keywords(job_description)

    job_skills = job_skills_db.union(job_keywords)

    matched_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills - resume_skills

    skill_score = (
        len(matched_skills) / len(job_skills)
        if len(job_skills) > 0 else 0
    )

    final_score = (semantic_score * 0.7) + (skill_score * 0.3)

    # -----------------------------
    # UI OUTPUT
    # -----------------------------
    st.markdown("## 📊 ATS Report")

    col1, col2, col3 = st.columns(3)

    col1.metric("ATS Score", f"{final_score*100:.1f}%")
    col2.metric("Semantic Match", f"{semantic_score*100:.1f}%")
    col3.metric("Skill Match", f"{skill_score*100:.1f}%")

    st.progress(float(final_score))

    # -----------------------------
    # RESULT
    # -----------------------------
    st.subheader("AI Recommendation")
    st.info(recruiter_feedback(final_score, missing_skills))

    # -----------------------------
    # SKILLS
    # -----------------------------
    st.subheader("Matched Skills")
    if matched_skills:
        st.write(", ".join(sorted(matched_skills)))
    else:
        st.warning("No matched skills found")

    st.subheader("Missing Skills")
    if missing_skills:
        st.write(", ".join(list(missing_skills)[:20]))
    else:
        st.success("No major gaps detected")

else:
    st.info("Upload resume and paste job description to start analysis")
