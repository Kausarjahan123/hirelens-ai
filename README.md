# HireLens AI - Resume Intelligence System

HireLens AI is an AI-powered resume evaluation system that analyzes the match between a candidate’s resume and a job description using NLP techniques and transformer-based embeddings. It generates an ATS-style report with skill gap analysis and semantic matching.

---

## Live Applications

HireLens AI (Premium Version)
https://hirelens-ai-ba45tgkrgbprvscxvbk8hd.streamlit.app/

AI Resume Matcher (Base Version)
https://ai-resume-matcher-ms57xeu8lvcn3f227u4vnh.streamlit.app/

---

## Project Overview

This project focuses on automating resume screening by combining:

- Semantic similarity using transformer embeddings
- Keyword-based ATS scoring logic
- Skill extraction and gap analysis
- Structured evaluation reports

The system helps simulate how modern Applicant Tracking Systems evaluate candidates.

---

## Key Features

### Resume Analysis
- PDF resume text extraction
- Structured parsing of skills and content
- Noise-free preprocessing

### AI Matching Engine
- SentenceTransformer embeddings
- Cosine similarity scoring
- Weighted hybrid scoring system

### ATS Evaluation
- Skill matching percentage
- Missing skill identification
- Final candidate fit score

### Reporting System
- Clean dashboard-style output
- Structured evaluation summary
- Recruiter-style recommendation logic

---

## Tech Stack

- Python
- Streamlit
- SentenceTransformers
- Scikit-learn
- NumPy
- PDFplumber
- Matplotlib

---

## System Architecture

Resume PDF
→ Text Extraction
→ NLP Preprocessing
→ Transformer Embedding Model
→ Job Description Embedding
→ Cosine Similarity Calculation
→ ATS Scoring Engine
→ Streamlit Dashboard Output

---

## Project Structure
hirelens-ai/
│
├── app.py
├── requirements.txt
├── README.md
│
└── notebooks/
└── AI_Resume_Matcher.ipynb

---

## How It Works

1. Upload resume in PDF format
2. Paste job description
3. System extracts and processes text
4. Embeddings are generated for both inputs
5. Similarity score is calculated
6. ATS scoring engine evaluates match quality
7. Final report is displayed

---

## Core Concepts Used

- Natural Language Processing
- Sentence Embeddings
- Cosine Similarity
- Feature Engineering
- Rule-based scoring system
- Hybrid ML evaluation approach

---

## Future Improvements

- GPT-based feedback system
- Multi-resume ranking
- User authentication system
- Database integration
- Resume PDF report generation
- Advanced ATS simulation engine

---

## Author

Kausar Jahan  
M.Tech Computer Science  
AI/ML Engineering Portfolio Project
