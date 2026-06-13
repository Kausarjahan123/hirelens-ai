import streamlit as st
import pdfplumber
import re
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
st.write("Resume vs Job Description Deep Matching Engine")

# -----------------------------
# INPUTS
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=250)

run = st.button("Run Analysis")

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.lower()

# -----------------------------
# SKILLS DB
# -----------------------------
SKILLS_DB = [
    "python","java","sql","javascript",
    "machine learning","deep learning",
    "pytorch","tensorflow","scikit-learn",
    "nlp","llm","transformers",
    "aws","docker","kubernetes",
    "mlflow","flask","fastapi"
]

# -----------------------------
# SKILL EXTRACTION
# -----------------------------
def extract_skills(text):
    found = []
    for skill in SKILLS_DB:
        if skill in text:
            found.append(skill)
    return list(set(found))

# -----------------------------
# FEEDBACK ENGINE
# -----------------------------
def feedback(score, missing):
    if score > 0.8:
        return "Strong match. Candidate is highly aligned."
    elif score > 0.6:
        return "Moderate match. Improve: " + ", ".join(list(missing)[:5])
    else:
        return "Weak match. Critical gaps: " + ", ".join(list(missing)[:5])

# -----------------------------
# MAIN
# -----------------------------
if run and uploaded_file and job_description:

    resume_text = extract_text(uploaded_file)

    # embeddings
    resume_vec = model.encode(resume_text)
    job_vec = model.encode(job_description)

    semantic_score = cosine_similarity([resume_vec], [job_vec])[0][0]

    # skills
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    matched = set(resume_skills) & set(job_skills)
    missing = set(job_skills) - set(resume_skills)

    skill_score = len(matched) / (len(job_skills) + 1)

    final_score = (0.7 * semantic_score) + (0.3 * skill_score)

    # -----------------------------
    # OUTPUT DASHBOARD
    # -----------------------------
    st.markdown("## 📊 ATS Report")

    c1, c2, c3 = st.columns(3)

    c1.metric("ATS Score", f"{final_score*100:.1f}%")
    c2.metric("Semantic Match", f"{semantic_score*100:.1f}%")
    c3.metric("Skill Match", f"{skill_score*100:.1f}%")

    st.progress(float(final_score))

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["Overview", "Skills", "Insights"])

    # -----------------------------
    # TAB 1
    # -----------------------------
    with tab1:
        st.subheader("AI Feedback")
        st.info(feedback(final_score, missing))

        st.subheader("Coverage Summary")
        st.write(f"Matched: {len(matched)}")
        st.write(f"Missing: {len(missing)}")
        st.write(f"Job Skills Found: {len(job_skills)}")

        st.progress(float(skill_score))

    # -----------------------------
    # TAB 2
    # -----------------------------
    with tab2:
        st.subheader("Matched Skills")
        st.write(list(matched) if matched else "None")

        st.subheader("Missing Skills")
        st.write(list(missing) if missing else "None")

    # -----------------------------
    # TAB 3
    # -----------------------------
    with tab3:
        st.subheader("Insight")
        st.write(
            f"""
            ATS Score: {final_score*100:.1f}%

            Semantic Match: {semantic_score*100:.1f}%

            Skill Coverage: {skill_score*100:.1f}%

            System uses embeddings + keyword matching.
            """
        )

else:
    st.info("Upload resume and paste job description to start analysis.")
