from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import fitz
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")

def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join(page.get_text() for page in doc)
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return " ".join(p.text for p in doc.paragraphs)
    return ""

st.title("ATS Resume Tracker")
st.caption("Powered by sentence-transformers + Groq")

jd = st.text_area("Paste Job Description here")
resumes = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Rank Resumes") and jd and resumes:
    jd_embedding = model.encode([jd])
    results = []
    resume_texts = {}

    for resume in resumes:
        text = extract_text(resume)
        resume_texts[resume.name] = text
        embedding = model.encode([text])
        score = cosine_similarity(jd_embedding, embedding)[0][0]
        results.append((resume.name, round(float(score) * 100, 2)))

    results.sort(key=lambda x: x[1], reverse=True)

    st.session_state["results"] = results
    st.session_state["resume_texts"] = resume_texts
    st.session_state["jd"] = jd
    st.session_state["output"] = ""

if "results" in st.session_state:
    results = st.session_state["results"]
    resume_texts = st.session_state["resume_texts"]
    jd = st.session_state["jd"]

    st.subheader("Rankings")
    for i, (name, score) in enumerate(results, 1):
        st.write(f"{i}. {name} — {score}% match")

    st.divider()

    selected = st.selectbox("Select a candidate", [name for name, _ in results], key="candidate_select")
    resume_text = resume_texts[selected]
    score = dict(results)[selected]

    col1, col2, col3 = st.columns(3)

    with col1:
        btn1 = st.button("Tell Me About the Resume", key="btn1")
    with col2:
        btn2 = st.button("How Can I Improvise my Skills", key="btn2")
    with col3:
        btn3 = st.button("Percentage Match", key="btn3")

    if btn1:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Summarize this resume in 4 bullet points:\n{resume_text[:1500]}"}]
            )
            st.session_state["output"] = response.choices[0].message.content

    if btn2:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Suggest 4 skills to improve based on this JD and resume:\n\nJD: {jd[:800]}\n\nResume: {resume_text[:800]}"}]
            )
            st.session_state["output"] = response.choices[0].message.content

    if btn3:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Explain in 3 bullet points why this resume matches {score}% with the JD:\n\nJD: {jd[:800]}\n\nResume: {resume_text[:800]}"}]
            )
            st.session_state["output"] = f"Match Score: {score}%\n\n" + response.choices[0].message.content

    if "output" in st.session_state and st.session_state["output"]:
        st.divider()
        st.write(st.session_state["output"])
