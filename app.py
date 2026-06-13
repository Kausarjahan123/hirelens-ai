import streamlit as st
import pdfplumber
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="HireLens AI",
    page_icon="💼",
    layout="wide"
)

# -----------------------------
# SIMPLE SAAS UI STYLE
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ffe6f0, #f8bbd0, #f3e5f5);
}

h1 {
    text-align: center;
    color: #6a1b9a;
    font-weight: 800;
}

.stButton>button {
    background-color: #ec407a;
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------
# TITLE
# -----------------------------
st.title("💼 HireLens AI - Resume Analyzer")

st.write("Upload your resume and compare it with a job description")

# -----------------------------
# INPUTS
# -----------------------------
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("🧾 Paste Job Description")

run = st.button("🚀 Analyze")

# -----------------------------
# FUNCTIONS
# -----------------------------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def clean_words(text):
    return re.findall(r'\b\w+\b', text.lower())

# -----------------------------
# MAIN LOGIC
# -----------------------------
if run and uploaded_file and job_description:

    # Resume text
    resume_text = extract_text(uploaded_file)

    # AI embeddings
    resume_emb = model.encode(resume_text)
    job_emb = model.encode(job_description)

    # similarity
    semantic_score = cosine_similarity([resume_emb], [job_emb])[0][0]

    # keyword logic (ATS style)
    resume_words = set(clean_words(resume_text))
    job_words = set(clean_words(job_description))

    overlap = resume_words & job_words
    missing = job_words - resume_words

    skill_score = len(overlap) / (len(job_words) + 1)

    final_score = (0.65 * semantic_score) + (0.35 * skill_score)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("📊 Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Match", f"{final_score*100:.2f}%")
    col2.metric("Semantic", f"{semantic_score*100:.2f}%")
    col3.metric("Skill Match", f"{skill_score*100:.2f}%")

    st.progress(float(final_score))

    # decision
    if final_score > 0.7:
        st.success("💚 Strong Match")
    elif final_score > 0.5:
        st.warning("💛 Medium Match")
    else:
        st.error("💔 Weak Match")

    # missing skills
    st.subheader("📌 Missing Skills")
    st.write(list(missing)[:20])

else:
    st.info("Upload resume + paste job description, then click Analyze 🚀")
