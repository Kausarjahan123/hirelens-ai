import streamlit as st
import pdfplumber
import re
import numpy as np
import matplotlib.pyplot as plt
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
# PREMIUM UI
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

.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
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
st.title("💼 HireLens AI – Premium ATS Intelligence Platform")
st.write("Executive-level Resume vs Job Description Analysis")

# -----------------------------
# INPUTS
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=250
)

run = st.button("Run ATS Analysis")

# -----------------------------
# PDF EXTRACTION
# -----------------------------
def extract_text(pdf):

    text = ""

    with pdfplumber.open(pdf) as pdf_file:

        for page in pdf_file.pages:

            text += page.extract_text() or ""

    return text.lower()

# -----------------------------
# SKILLS DATABASE
# -----------------------------
SKILLS_DB = [

    # Programming
    "python",
    "java",
    "c++",
    "javascript",
    "typescript",
    "sql",

    # Data
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scipy",

    # ML
    "machine learning",
    "deep learning",
    "neural networks",
    "tensorflow",
    "keras",
    "pytorch",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",

    # NLP / LLM
    "nlp",
    "transformers",
    "hugging face",
    "langchain",
    "rag",
    "llm",
    "large language models",
    "prompt engineering",
    "faiss",

    # Cloud
    "aws",
    "azure",
    "gcp",

    # Deployment
    "docker",
    "kubernetes",
    "streamlit",
    "flask",
    "fastapi",

    # Databases
    "mysql",
    "postgresql",
    "mongodb",
    "oracle",

    # MLOps
    "mlflow",
    "airflow",
    "ci/cd",

    # Advanced AI
    "distributed training",
    "gpu optimization",
    "quantization",
    "model compression",
    "inference optimization",
    "kv cache",
    "batching",
    "scheduling",
    "observability",
    "performance profiling",
    "data pipelines",
    "multimodal llm"
]

# -----------------------------
# EXTRACT SKILLS
# -----------------------------
def extract_skills(text):

    found = []

    text = text.lower()

    for skill in SKILLS_DB:

        if skill in text:

            found.append(skill)

    return sorted(list(set(found)))

# -----------------------------
# SKILL CATEGORIES
# -----------------------------
def categorize_skills(skills):

    categories = {

        "Programming": [
            "python","java","c++","javascript","typescript","sql"
        ],

        "AI/ML": [
            "machine learning",
            "deep learning",
            "tensorflow",
            "keras",
            "pytorch",
            "scikit-learn",
            "llm",
            "rag",
            "transformers",
            "langchain",
            "faiss"
        ],

        "Cloud": [
            "aws","azure","gcp"
        ],

        "MLOps": [
            "docker",
            "kubernetes",
            "mlflow",
            "airflow"
        ],

        "Database": [
            "mysql",
            "postgresql",
            "mongodb",
            "oracle"
        ]
    }

    result = Counter()

    for skill in skills:

        for category, values in categories.items():

            if skill in values:

                result[category] += 1

    return result

# -----------------------------
# RECRUITER FEEDBACK
# -----------------------------
def recruiter_feedback(score, missing_skills):

    if score >= 0.80:
        return "Strong alignment with the role. Candidate is suitable for immediate interview consideration."

    elif score >= 0.60:
        return (
            "Good alignment. Consider strengthening experience in: "
            + ", ".join(list(missing_skills)[:5])
        )

    else:
        return (
            "Significant skill gaps detected. Focus on acquiring: "
            + ", ".join(list(missing_skills)[:5])
        )

# -----------------------------
# CHART
# -----------------------------
def skill_chart(matched, missing):

    fig, ax = plt.subplots()

    ax.bar(
        ["Matched Skills", "Missing Skills"],
        [len(matched), len(missing)]
    )

    ax.set_title("Skill Coverage")

    return fig

# -----------------------------
# MAIN
# -----------------------------
if run and uploaded_file and job_description:

    resume_text = extract_text(uploaded_file)

    # -----------------------------
    # SEMANTIC MATCH
    # -----------------------------
    resume_embedding = model.encode(resume_text)

    job_embedding = model.encode(job_description)

    semantic_score = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    # -----------------------------
    # SKILLS MATCH
    # -----------------------------
    resume_skills = extract_skills(resume_text)

    job_skills = extract_skills(job_description)

    matched_skills = set(resume_skills) & set(job_skills)

    missing_skills = set(job_skills) - set(resume_skills)

    if len(job_skills) > 0:

        skill_score = len(matched_skills) / len(job_skills)

    else:

        skill_score = 0

    # -----------------------------
    # FINAL SCORE
    # -----------------------------
    final_score = (
        semantic_score * 0.70 +
        skill_score * 0.30
    )

    # -----------------------------
    # CATEGORIES
    # -----------------------------
    skill_categories = categorize_skills(resume_skills)

    # -----------------------------
    # REPORT
    # -----------------------------
    st.markdown("## 📊 Executive ATS Report")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "ATS Fit Score",
        f"{final_score*100:.1f}%"
    )

    c2.metric(
        "Semantic Match",
        f"{semantic_score*100:.1f}%"
    )

    c3.metric(
        "Skill Match",
        f"{skill_score*100:.1f}%"
    )

    st.progress(float(final_score))

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3 = st.tabs([
        "📄 Overview",
        "📌 Skills",
        "🧠 Insights"
    ])

    # -----------------------------
    # OVERVIEW
    # -----------------------------
    with tab1:

        st.subheader("AI Recommendation")

        st.info(
            recruiter_feedback(
                final_score,
                missing_skills
            )
        )

        st.subheader("Skill Coverage")

        st.pyplot(
            skill_chart(
                matched_skills,
                missing_skills
            )
        )

    # -----------------------------
    # SKILLS
    # -----------------------------
    with tab2:

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Matched Skills")

            if matched_skills:

                for skill in sorted(matched_skills):

                    st.success(skill.title())

            else:

                st.warning("No matching skills detected")

        with col2:

            st.subheader("Missing Skills")

            if missing_skills:

                for skill in sorted(missing_skills):

                    st.error(skill.title())

            else:

                st.success("No major skill gaps detected")

        st.subheader("Skill Categories")

        st.write(dict(skill_categories))

    # -----------------------------
    # INSIGHTS
    # -----------------------------
    with tab3:

        st.subheader("Hiring Insight")

        st.write(
            f"""
            The resume achieved an ATS Fit Score of {final_score*100:.1f}%.

            Semantic relevance to the job description is
            {semantic_score*100:.1f}% while technical
            skill coverage is {skill_score*100:.1f}%.

            The strongest improvement opportunity is closing
            the missing skill gaps identified above.
            """
        )

else:

    st.info(
        "Upload a resume and paste a job description to generate the ATS report."
    )
