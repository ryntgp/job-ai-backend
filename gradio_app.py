import os
import gradio as gr
from dotenv import load_dotenv
from markitdown import MarkItDown
from openai import OpenAI
import json
import markdown
from xhtml2pdf import pisa

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
MODEL = "openai/gpt-6-luna"

def pdf_to_markdown(pdf_path: str) -> str:
    md = MarkItDown()
    result = md.convert(pdf_path)
    return result.text_content

def markdown_to_pdf(md_text: str) -> str:
    html_body = markdown.markdown(md_text)
    html_full = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 40px; 
                line-height: 1.5;
                color: #333;
            }}
            h1 {{ font-size: 22px; margin-bottom: 8px; }}
            h2 {{ font-size: 16px; color: #444; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 20px; }}
            ul {{ margin: 8px 0; }}
            li {{ margin: 4px 0; }}
            p {{ margin: 8px 0; }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """
    
    pdf_path = "tailored_cv.pdf"
    with open(pdf_path, "wb") as f:
        pisa_status = pisa.CreatePDF(html_full, dest=f)
    
    if pisa_status.err:
        raise Exception("Gagal membuat PDF")
    
    return pdf_path

def analyze_and_generate(pdf_file, job_desc, confirmed_skills_text):
    if not pdf_file or not job_desc:
        return "Upload CV dan isi Job Description dulu.", "", None
    
    try:
        # Convert PDF ke Markdown
        cv_md = pdf_to_markdown(pdf_file)
        
        # Analisis
        prompt = f"""
Analisis CV ini terhadap job description.

CV:
{cv_md}

Job Description:
{job_desc}

Output JSON dengan format PERSIS seperti ini:
{{
  "match_score": 70,
  "summary": "ringkasan singkat",
  "missing_skills": ["skill1", "skill2"],
  "suggestions": ["saran1", "saran2"]
}}
"""
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        
        data = json.loads(res.choices[0].message.content)
        
        # Format hasil analisis
        score = data.get("match_score", "N/A")
        summary = data.get("summary", "Tidak ada ringkasan")
        missing = data.get("missing_skills", [])
        suggestions = data.get("suggestions", [])
        
        analysis_text = f"## Skor: {score}/100\n\n"
        analysis_text += f"**Ringkasan:** {summary}\n\n"
        
        if missing:
            analysis_text += "**Skill yang kurang:**\n"
            for s in missing:
                analysis_text += f"- {s}\n"
            analysis_text += "\n"
        
        if suggestions:
            analysis_text += "**Saran:**\n"
            for s in suggestions:
                analysis_text += f"- {s}\n"
        
        # Generate tailored CV
        confirmed = [s.strip() for s in confirmed_skills_text.split(",") if s.strip()] if confirmed_skills_text else []
        
        gen_prompt = f"""
Buat CV dalam format Markdown yang disesuaikan dengan job description.

CV Asli:
{cv_md}

Job Description:
{job_desc}

Skill yang dikonfirmasi user: {', '.join(confirmed) if confirmed else 'tidak ada'}

Aturan:
- Hanya tambahkan skill yang dikonfirmasi
- Jangan mengarang pengalaman atau skill baru
- Perbaiki wording agar lebih kuat dan ATS-friendly
- Output HANYA Markdown CV, tanpa penjelasan tambahan
"""
        gen_res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": gen_prompt}]
        )
        tailored = gen_res.choices[0].message.content
        
        # Convert ke PDF
        pdf_path = markdown_to_pdf(tailored)
        
        return analysis_text, tailored, pdf_path
        
    except Exception as e:
        return f"**Error:** {str(e)}", "", None

# UI
with gr.Blocks(title="AI Job Application") as demo:
    gr.Markdown("# AI Job Application Prototype")
    gr.Markdown("Upload CV PDF → Analisis → Generate CV baru (Markdown + PDF)")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload CV (PDF)", file_types=[".pdf"])
        job_input = gr.Textbox(label="Job Description", lines=6, placeholder="Tempel job description di sini...")
    
    skills_input = gr.Textbox(
        label="Skill yang kamu konfirmasi (pisahkan dengan koma)",
        placeholder="Contoh: Excel, PowerPoint, Python"
    )
    
    btn = gr.Button("Analisis & Generate CV", variant="primary")
    
    analysis_output = gr.Markdown(label="Hasil Analisis")
    cv_output = gr.Code(label="CV Baru (Markdown)", language="markdown")
    pdf_output = gr.File(label="Download PDF")
    
    btn.click(
        analyze_and_generate,
        inputs=[pdf_input, job_input, skills_input],
        outputs=[analysis_output, cv_output, pdf_output]
    )

if __name__ == "__main__":
    demo.launch()