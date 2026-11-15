import streamlit as st
import os
from dotenv import load_dotenv
from markitdown import MarkItDown
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import json
import tempfile

load_dotenv()

st.set_page_config(page_title="AI Job Application", layout="wide")
st.title("AI Job Application Prototype")

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
MODEL = "openai/gpt-6-luna"

class SkillSuggestion(BaseModel):
    skill: str
    question: str
    reason: str

class AnalysisResult(BaseModel):
    match_score: int
    summary: str
    missing_skills: List[SkillSuggestion]
    suggestions: List[str]

def pdf_to_markdown(pdf_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        path = tmp.name
    md = MarkItDown()
    result = md.convert(path)
    os.unlink(path)
    return result.text_content

def analyze(cv_md: str, job: str) -> AnalysisResult:
    prompt = f"""
CV:
{cv_md}

Job:
{job}

Berikan skor 0-100, skill yang kurang (dengan pertanyaan), dan saran. Output JSON.
"""
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return AnalysisResult(**json.loads(res.choices[0].message.content))

def generate(cv_md: str, job: str, skills: List[str]) -> str:
    skills_text = ", ".join(skills) if skills else "tidak ada"
    prompt = f"""
Buat CV Markdown yang disesuaikan.

CV Asli:
{cv_md}

Job:
{job}

Skill yang dikonfirmasi: {skills_text}

Hanya tambahkan skill yang dikonfirmasi. Output hanya Markdown.
"""
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# UI
uploaded = st.file_uploader("Upload CV (PDF)", type=["pdf"])
job_desc = st.text_area("Job Description", height=150)

if st.button("Analisis", disabled=not (uploaded and job_desc)):
    with st.spinner("Memproses..."):
        cv_md = pdf_to_markdown(uploaded)
        st.session_state["cv"] = cv_md
        st.session_state["job"] = job_desc
        st.session_state["analysis"] = analyze(cv_md, job_desc)

if "analysis" in st.session_state:
    a = st.session_state["analysis"]
    st.metric("Skor", f"{a.match_score}/100")
    st.write(a.summary)
    
    st.subheader("Skill yang Kurang")
    confirmed = []
    for i, s in enumerate(a.missing_skills):
        if st.checkbox(f"{s.skill} — {s.question}", key=f"s{i}"):
            confirmed.append(s.skill)
    st.session_state["confirmed"] = confirmed
    
    if st.button("Generate CV"):
        with st.spinner("Membuat CV..."):
            result = generate(
                st.session_state["cv"],
                st.session_state["job"],
                st.session_state.get("confirmed", [])
            )
        st.code(result, language="markdown")
        st.download_button("Download", result, "tailored_cv.md")