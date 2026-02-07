import streamlit as st
import nltk
import re
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

nltk.download('stopwords')

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text

# ---------------- KEYWORD EXTRACTION ----------------
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    keywords = [word for word in words if word not in stop_words]
    return set(keywords)

# ---------------- PDF READER ----------------
def read_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# ---------------- WORD READER ----------------
def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + " "
    return text

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Resume Gap Analyzer", layout="wide")

st.title("🧠 Intelligent Resume Gap Analyzer (NLP)")
st.write("Upload your resume (PDF/DOCX) and compare it with a job description.")

resume_file = st.file_uploader(
    "📄 Upload Resume (PDF or DOCX)",
    type=["pdf", "docx"]
)

job_description = st.text_area(
    "🧾 Paste Job Description",
    height=250
)

if st.button("🔍 Analyze Resume"):
    if resume_file and job_description:

        # Read resume file
        if resume_file.type == "application/pdf":
            resume_text = read_pdf(resume_file)
        else:
            resume_text = read_docx(resume_file)

        # Clean text
        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_description)

        # TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])

        # Cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        match_percentage = round(similarity * 100, 2)

        # Skill gap analysis
        resume_keywords = extract_keywords(resume_clean)
        job_keywords = extract_keywords(job_clean)
        missing_skills = job_keywords - resume_keywords

        # ---------------- RESULTS ----------------
        st.subheader("📊 Match Score")
        st.success(f"Resume matches the job by **{match_percentage}%**")

        st.subheader("❌ Missing Skills / Keywords")
        if missing_skills:
            st.write(", ".join(sorted(missing_skills)))
        else:
            st.success("No major skill gaps found 🎯")

    else:
        st.warning("Please upload a resume and paste the job description.")

