import streamlit as st
import pdfplumber
import re
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# -----------------------------
# PAGE CONFIG (PREMIUM SaaS)
# -----------------------------
st.set_page_config(
    page_title="HireLens AI | ATS Intelligence",
    page_icon="💼",
    layout="wide"
)

# -----------------------------
# PREMIUM CLEAN UI THEME
# -----------------------------
st.markdown("""
<style>

/* background */
.stApp {
    background-color: #f6f7fb;
    font-family: 'Segoe UI', sans-serif;
}

/* title */
h1 {
    text-align: center;
    color: #111827;
    font-weight: 800;
}

/* cards */
.block {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.06);
}

/* metric boxes */
[data-testid="metric-container"] {
    background-color: white;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0px 1px 6px rgba(0,0,0,0.08);
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
# INPUT
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

run = st.button("Run ATS Analysis")

# -----------------------------
# FUNCTIONS
# -----------------------------
def extract_text(pdf):
    text = ""
    with pdfplumber.open(pdf) as f:
        for page in f.pages:
            text += page.extract_text() or ""
    return text

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# -----------------------------
# SKILL CLASSIFICATION
# -----------------------------
def categorize(words):
    tech = {"python","java","sql","aws","docker","git","linux","api"}
    ml = {"machine","learning","deep","neural","nlp","pandas","numpy","tensorflow","pytorch","sklearn"}
    soft = {"communication","leadership","team","management"}

    counts = Counter()

    for w in words:
        if w in tech:
            counts["Technical"] += 1
        elif w in ml:
            counts["AI/ML"] += 1
        elif w in soft:
            counts["Soft Skills"] += 1

    return counts

# -----------------------------
# AI FEEDBACK (RECRUITER STYLE)
# -----------------------------
def recruiter_feedback(score, missing):
    if score > 0.8:
        return "Strong candidate. Recommend immediate interview shortlist."
    elif score > 0.6:
        return f"Good candidate. Improve missing areas: {', '.join(list(missing)[:5])}"
    else:
        return f"Candidate not aligned. Critical gaps: {', '.join(list(missing)[:5])}"

# -----------------------------
# CHARTS
# -----------------------------
def bar_chart(overlap, missing):
    fig, ax = plt.subplots()
    ax.bar(["Matched", "Missing"], [len(overlap), len(missing)])
    ax.set_title("Skill Coverage")
    return fig

# -----------------------------
# MAIN
# -----------------------------
if run and uploaded_file and job_description:

    resume = extract_text(uploaded_file)

    resume_emb = model.encode(resume)
    job_emb = model.encode(job_description)

    semantic_score = cosine_similarity([resume_emb], [job_emb])[0][0]

    resume_words = set(tokenize(resume))
    job_words = set(tokenize(job_description))

    overlap = resume_words & job_words
    missing = job_words - resume_words

    skill_score = len(overlap) / (len(job_words) + 1)

    # -----------------------------
    # FINAL REALISTIC ATS SCORE
    # -----------------------------
    final_score = (0.7 * semantic_score) + (0.3 * skill_score)

    skill_categories = categorize(tokenize(resume))

    # -----------------------------
    # DASHBOARD HEADER
    # -----------------------------
    st.markdown("## 📊 Executive ATS Report")

    col1, col2, col3 = st.columns(3)

    col1.metric("ATS Fit Score", f"{final_score*100:.1f}%")
    col2.metric("Semantic Match", f"{semantic_score*100:.1f}%")
    col3.metric("Skill Match", f"{skill_score*100:.1f}%")

    st.progress(float(final_score))

    # -----------------------------
    # TABS (VERY IMPORTANT FOR PREMIUM FEEL)
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["📄 Overview", "📌 Skills", "🧠 Insights"])

    # -----------------------------
    # TAB 1 - OVERVIEW
    # -----------------------------
    with tab1:
        st.markdown("### AI Recommendation")
        st.info(recruiter_feedback(final_score, missing))

        st.markdown("### Skill Coverage")
        st.pyplot(bar_chart(overlap, missing))

    # -----------------------------
    # TAB 2 - SKILLS
    # -----------------------------
    with tab2:
        st.markdown("### Skill Categories Detected")
        st.write(skill_categories)

        st.markdown("### Missing Skills")
        st.write(list(missing)[:25])

    # -----------------------------
    # TAB 3 - INSIGHTS
    # -----------------------------
    with tab3:
        st.markdown("### Hiring Insight")
        st.write(
            "This analysis is based on semantic similarity + ATS keyword matching. "
            "It simulates modern resume screening systems used in recruitment pipelines."
        )

else:
    st.info("Upload resume and job description to generate premium ATS report 🚀")
