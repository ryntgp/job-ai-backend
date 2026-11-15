from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from markitdown import MarkItDown
from openai import OpenAI
import os
import re
import json
import markdown
from xhtml2pdf import pisa
from dotenv import load_dotenv
import uuid
from io import BytesIO
from datetime import datetime, timezone
from typing import Optional
import httpx

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
MODEL = "openai/gpt-6-luna"

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

NUMERIC_FIELD_RULES = (
    ("gpa", ("gpa", "cgpa", "grade point", "grade-point")),
    ("ipk", ("ipk", "indeks prestasi")),
    ("toefl", ("toefl",)),
    ("ielts", ("ielts",)),
    ("gre", ("gre",)),
    ("gmat", ("gmat",)),
)
FIELD_LABELS = {
    "gpa": "GPA",
    "ipk": "IPK",
    "toefl": "TOEFL",
    "ielts": "IELTS",
    "gre": "GRE",
    "gmat": "GMAT",
}
FIELD_HINTS = {
    "gpa": "Contoh: 3.75",
    "ipk": "Contoh: 3.75",
    "toefl": "Contoh: 100",
    "ielts": "Contoh: 7.5",
    "gre": "Contoh: 320",
    "gmat": "Contoh: 700",
}
FILLIN_RE = re.compile(
    r"\[(?:"
    r"(?:add|insert|your|isi|masukkan|todo|tbd|placeholder)[^\]]*"
    r"|(?:gpa|ipk|cgpa|toefl|ielts|gre|gmat)(?:\s*/\s*(?:gpa|ipk|cgpa))?"
    r")\](?!\()",
    re.IGNORECASE,
)
ACADEMIC_PLACEHOLDER_RE = re.compile(
    r"\[\s*(?:(?:add|insert|your|isi|masukkan)\s+)?"
    r"(?:gpa|ipk|cgpa)(?:\s*/\s*(?:gpa|ipk|cgpa))?\s*\](?!\()",
    re.IGNORECASE,
)
EMPTY_SCORE_LINE_RE = re.compile(
    r"^\s*[-*]?\s*\**\s*"
    r"(?:GPA(?:\s*/\s*IPK)?|IPK(?:\s*/\s*GPA)?|CGPA|TOEFL|IELTS|GRE|GMAT)"
    r"\s*\**\s*[:\-–]?\s*\**\s*$",
    re.IGNORECASE,
)


def field_keys_in_text(text: str) -> list:
    lowered = text.lower()
    found = []
    for key, terms in NUMERIC_FIELD_RULES:
        if any(re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", lowered) for term in terms):
            found.append(key)
    return found


def number_field(key: str, label: str, hint: str) -> dict:
    return {"key": key, "label": label, "input_type": "number", "hint": hint}


def build_fields(keys: list) -> list:
    unique = []
    for key in keys:
        if key not in unique:
            unique.append(key)
    fields = []
    academic = [key for key in unique if key in ("gpa", "ipk")]
    if academic:
        if academic == ["ipk"]:
            fields.append(number_field("ipk", "IPK", FIELD_HINTS["ipk"]))
        elif "ipk" in academic:
            fields.append(number_field("gpa", "GPA / IPK", FIELD_HINTS["gpa"]))
        else:
            fields.append(number_field("gpa", "GPA", FIELD_HINTS["gpa"]))
    for key in ("toefl", "ielts", "gre", "gmat"):
        if key in unique:
            fields.append(number_field(key, FIELD_LABELS[key], FIELD_HINTS[key]))
    return fields


def normalize_analysis(raw: dict) -> dict:
    found_keys = []
    skills = []
    seen = set()

    def absorb(text: str) -> bool:
        keys = field_keys_in_text(text)
        if not keys:
            return False
        found_keys.extend(keys)
        return True

    for item in raw.get("missing_skills") or []:
        text = str(item).strip()
        if not text or absorb(text):
            continue
        marker = text.lower()
        if marker not in seen:
            seen.add(marker)
            skills.append(text)

    for item in raw.get("missing_fields") or []:
        if isinstance(item, dict):
            text = str(item.get("label") or item.get("key") or "")
        else:
            text = str(item)
        absorb(text)

    suggestions = raw.get("suggestions") if isinstance(raw.get("suggestions"), list) else []
    return {
        "match_score": raw.get("match_score"),
        "summary": raw.get("summary") or "",
        "missing_skills": skills,
        "missing_fields": build_fields(found_keys),
        "suggestions": suggestions,
    }


def normalize_number(value: str):
    cleaned = value.strip().replace(",", ".")
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3})?", cleaned):
        return cleaned
    return None


def parse_field_values(raw: str) -> list:
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    values = []
    for item in data:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip().lower()
        if key not in FIELD_LABELS:
            continue
        number = normalize_number(str(item.get("value") or ""))
        if not number:
            continue
        label = str(item.get("label") or "").strip() or FIELD_LABELS[key]
        values.append({"key": key, "label": label, "value": number})
    return values


def clean_tailored_cv(text: str, field_values: list) -> str:
    academic_value = next((item["value"] for item in field_values if item["key"] in ("gpa", "ipk")), None)
    if academic_value:
        text = ACADEMIC_PLACEHOLDER_RE.sub(academic_value, text)
    for item in field_values:
        if item["key"] in ("gpa", "ipk"):
            continue
        pattern = re.compile(
            rf"\[\s*(?:(?:add|insert|your|isi|masukkan)\s+)?{re.escape(item['key'])}\s*\](?!\()",
            re.IGNORECASE,
        )
        text = pattern.sub(item["value"], text)

    text = FILLIN_RE.sub("", text)
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if EMPTY_SCORE_LINE_RE.match(line) or (stripped and re.fullmatch(r"[-*•\s.,:;]+", stripped)):
            continue
        kept.append(re.sub(r"[ \t]{2,}", " ", line).rstrip())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return ensure_score_lines(text, field_values)


SCORE_ALIASES = {
    "gpa": ["GPA / IPK", "IPK / GPA", "CGPA", "GPA", "IPK"],
    "ipk": ["GPA / IPK", "IPK / GPA", "CGPA", "IPK", "GPA"],
    "toefl": ["TOEFL"],
    "ielts": ["IELTS"],
    "gre": ["GRE"],
    "gmat": ["GMAT"],
}


def score_line_present(text: str, item: dict) -> bool:
    value = re.escape(item["value"])
    aliases = SCORE_ALIASES.get(item["key"], [item["label"]])
    for alias in aliases:
        label = re.escape(alias)
        if re.search(rf"{label}[^\n]{{0,40}}(?<!\d){value}(?!\d)", text, re.IGNORECASE):
            return True
    return False


def ensure_score_lines(text: str, field_values: list) -> str:
    lines = text.splitlines()
    for item in field_values:
        if score_line_present("\n".join(lines), item):
            continue
        score_line = f"{item['label']}: {item['value']}"
        insert_at = None
        for index, line in enumerate(lines):
            if re.search(r"^\s{0,3}#{1,3}\s+.*(pendidikan|education)\b", line, re.IGNORECASE):
                insert_at = index + 1
                break
        if insert_at is None:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(score_line)
        else:
            lines.insert(insert_at, score_line)
    return "\n".join(lines).strip()



def require_config():
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL atau SUPABASE_ANON_KEY belum diisi di .env",
        )


def bearer_token(authorization: Optional[str]) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Login diperlukan")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Login diperlukan")
    return token


async def current_user_id(token: str) -> str:
    require_config()
    async with httpx.AsyncClient(timeout=20) as http:
        response = await http.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {token}",
            },
        )
    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Sesi login tidak valid. Silakan login ulang.")
    user_id = response.json().get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Sesi login tidak valid.")
    return user_id


def supabase_headers(token: str, prefer: Optional[str] = None) -> dict:
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def raise_for_supabase(response: httpx.Response, fallback: str):
    if response.status_code < 400:
        return
    message = fallback
    try:
        payload = response.json()
        if payload.get("code") == "PGRST205":
            message = (
                "Tabel cvs belum ada. Jalankan supabase/schema.sql "
                "di SQL Editor Supabase, lalu coba lagi."
            )
        elif payload.get("message"):
            message = payload["message"]
    except Exception:
        pass
    raise HTTPException(status_code=502, detail=message)


async def save_cv(token: str, user_id: str, filename: str, markdown_text: str):
    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.post(
            f"{SUPABASE_URL}/rest/v1/cvs",
            params={"on_conflict": "user_id"},
            headers=supabase_headers(token, "resolution=merge-duplicates,return=minimal"),
            json={
                "user_id": user_id,
                "filename": filename,
                "markdown": markdown_text,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    raise_for_supabase(response, "CV gagal disimpan")


async def load_cv(token: str, user_id: str) -> str:
    async with httpx.AsyncClient(timeout=20) as http:
        response = await http.get(
            f"{SUPABASE_URL}/rest/v1/cvs",
            params={"user_id": f"eq.{user_id}", "select": "markdown"},
            headers=supabase_headers(token),
        )
    raise_for_supabase(response, "CV gagal dibaca")
    rows = response.json()
    if not rows or not rows[0].get("markdown"):
        raise HTTPException(status_code=400, detail="CV belum diupload untuk akun ini")
    return rows[0]["markdown"]


def pdf_to_markdown(pdf_bytes: bytes) -> str:
    md = MarkItDown()
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)
    try:
        result = md.convert(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return result.text_content


def markdown_to_pdf(md_text: str) -> bytes:
    html_body = markdown.markdown(md_text)
    html_full = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, Helvetica, sans-serif; margin: 40px 50px; line-height: 1.45; color: #222; font-size: 11.5px; }}
            h1 {{ font-size: 22px; margin-bottom: 4px; color: #1a1a1a; }}
            h2 {{ font-size: 14px; color: #333; border-bottom: 1.5px solid #444; padding-bottom: 3px; margin-top: 18px; margin-bottom: 8px; }}
            h3 {{ font-size: 12.5px; margin: 10px 0 4px 0; }}
            ul {{ margin: 6px 0; padding-left: 18px; }}
            li {{ margin-bottom: 3px; }}
            p {{ margin: 4px 0; }}
            strong {{ color: #111; }}
        </style>
    </head>
    <body>{html_body}</body>
    </html>
    """
    pdf_buffer = BytesIO()
    pisa.CreatePDF(html_full, dest=pdf_buffer)
    return pdf_buffer.getvalue()


@app.post("/upload-cv")
async def upload_cv(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    token = bearer_token(authorization)
    user_id = await current_user_id(token)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File CV kosong")
    cv_md = pdf_to_markdown(content)
    filename = file.filename or "cv.pdf"
    await save_cv(token, user_id, filename, cv_md)
    return {"user_id": user_id, "filename": filename, "message": "CV uploaded"}


@app.post("/analyze")
async def analyze(
    job_description: str = Form(...),
    authorization: Optional[str] = Header(None),
):
    token = bearer_token(authorization)
    user_id = await current_user_id(token)
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description kosong")
    cv_md = await load_cv(token, user_id)

    prompt = f"""Analisis CV terhadap Job Description ini.

CV:
{cv_md}

Job Description:
{job_description}

Output JSON PERSIS format ini (tanpa tambahan teks):
{{
  "match_score": 72,
  "summary": "ringkasan singkat kecocokan dalam 2-3 kalimat",
  "missing_skills": ["skill yang benar-benar berupa kemampuan, misalnya Python"],
  "missing_fields": [
    {{"key": "gpa", "label": "GPA / IPK", "input_type": "number"}}
  ],
  "suggestions": ["saran konkret 1", "saran 2"]
}}

Aturan pemisahan:
- missing_skills hanya berisi kemampuan yang dijawab Ya atau Tidak. Contoh: Python, SQL, Excel, komunikasi.
- GPA, IPK, CGPA, TOEFL, IELTS, GRE, dan GMAT bukan skill. Masukkan ke missing_fields dengan input_type "number".
- Jangan taruh GPA atau IPK di missing_skills.
- missing_fields kosong [] jika CV sudah memuat angka itu atau job tidak memintanya.
- Jangan membuat placeholder seperti [Add GPA].
"""
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    analysis = normalize_analysis(json.loads(res.choices[0].message.content))
    return {"analysis": analysis}


@app.post("/generate-cv")
async def generate_cv(
    job_description: str = Form(...),
    confirmed_skills: str = Form(""),
    field_values: str = Form(""),
    authorization: Optional[str] = Header(None),
):
    token = bearer_token(authorization)
    user_id = await current_user_id(token)
    cv_md = await load_cv(token, user_id)

    confirmed = [s.strip() for s in confirmed_skills.split(",") if s.strip()]
    numbers = parse_field_values(field_values)
    if numbers:
        number_lines = "\n".join(f"- {item['label']}: {item['value']}" for item in numbers)
    else:
        number_lines = "tidak ada"

    prompt = f"""Buat CV dalam format Markdown yang sangat ATS-friendly dan menarik.

CV Asli:
{cv_md}

Job Description:
{job_description}

Skill yang dikonfirmasi user (boleh ditambahkan jika relevan): {', '.join(confirmed) if confirmed else 'tidak ada'}

Data angka yang diisi user:
{number_lines}

Aturan ketat:
- Hanya tambahkan skill yang dikonfirmasi user. Jangan mengarang skill atau pengalaman baru.
- Data angka yang diisi user wajib muncul sebagai baris sendiri di bagian Pendidikan, dengan label dan angka persis. Contoh: GPA: 3.1
- Jangan dibulatkan dan jangan diubah. Jangan menaruh angka itu hanya di kalimat lain.
- Jika sebuah data angka tidak diisi, jangan tulis baris, label, atau kalimat tentang data itu.
- Dilarang menulis placeholder, termasuk [Add GPA], [GPA], [IPK], [Your GPA], [Insert GPA], [Masukkan IPK], TBD, TODO, dan N/A.
- Jangan mengarang GPA, IPK, atau angka lain.
- Perbaiki wording agar lebih kuat, action-oriented, dan quantifiable jika memungkinkan.
- Struktur jelas: Contact, Summary/Profile, Experience, Education, Skills.
- Gunakan bullet points yang kuat.
- Optimalkan untuk ATS (keyword dari job description yang relevan, formatting sederhana).
- Buat summary/profile yang lebih menarik dan targeted.
- Output HANYA Markdown CV, tanpa penjelasan tambahan.

Mulai langsung dengan nama atau heading pertama.
"""
    gen_res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    tailored = clean_tailored_cv(gen_res.choices[0].message.content.strip(), numbers)
    return {"tailored_cv": tailored}


def strip_placeholders(text: str) -> str:
    text = FILLIN_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


WEAK_OPENING_RE = re.compile(
    r"\b("
    r"i am writing to apply|i am writing to express|i am pleased to apply|"
    r"i am excited to apply|i would like to apply|please accept this|"
    r"i am applying for|this letter is to apply|i wish to apply|"
    r"saya menulis untuk melamar|saya menulis surat ini untuk|"
    r"dengan ini saya mengajukan|dengan hormat saya mengajukan|"
    r"saya bermaksud melamar|surat ini saya tulis untuk|"
    r"saya ingin melamar|melalui surat ini saya"
    r")\b",
    re.IGNORECASE,
)
CTA_RE = re.compile(
    r"\b(interview|conversation|speak|call|discuss|meeting|"
    r"wawancara|berdiskusi|diskusi|ngobrol|hubungi|jadwal|bicara)\b",
    re.IGNORECASE,
)


def weak_cover_opening(text: str) -> bool:
    opening = " ".join(text.strip().splitlines()[:4])
    return bool(WEAK_OPENING_RE.search(opening))


def has_call_to_action(text: str) -> bool:
    return bool(CTA_RE.search(text[-600:]))


def cover_letter_needs_revision(text: str) -> bool:
    return weak_cover_opening(text) or not has_call_to_action(text)


def cover_letter_prompt(tailored_cv: str, job_description: str) -> str:
    return f"""Tulis surat lamaran yang terasa seperti ditulis orangnya, bukan template.

CV yang akan dikirim:
{tailored_cv}

Lowongan:
{job_description}

Bahasa surat sama dengan bahasa CV.

Struktur, dalam urutan ini:
1. Hook. Setelah sapaan singkat, kalimat pertama harus spesifik: satu pengamatan tentang masalah, produk, atau taruhan peran ini, yang diambil dari lowongan, lalu kaitkan dengan satu fakta dari CV. Jangan mulai dengan "I am writing to apply", "I am excited to apply", "Saya menulis untuk melamar", atau "Dengan ini saya mengajukan lamaran".
2. Pemahaman peran. Sebut nama peran dan perusahaan jika ada di lowongan. Tunjukkan bahwa penulis mengerti apa yang sedang dicari peran itu, dengan menyebut 1-2 kebutuhan yang benar-benar tertulis di lowongan.
3. Bukti. Hubungkan dua pengalaman dari CV ke kebutuhan itu. Sebut jabatan atau konteks aslinya. Jelaskan apa yang dikerjakan dan mengapa itu relevan untuk peran ini, bukan hanya "saya punya pengalaman di bidang terkait".
4. Ajakan. Paragraf terakhir meminta satu langkah yang jelas, misalnya percakapan atau wawancara, dan menyebut kapan atau cara menindaklanjutinya. Jangan menutup hanya dengan "terima kasih" atau "saya menunggu kabar".

Nada:
- Profesional, hangat, dan langsung. Bukan kaku dan bukan terlalu akrab.
- Kalimat pendek boleh bercampur dengan kalimat yang sedikit lebih panjang.
- Jangan pakai Furthermore, Moreover, Dear Sir or Madam, "valuable asset", "perfect fit", atau "It is with great enthusiasm".

Fakta:
- Hanya gunakan pengalaman, angka, dan nama yang ada di CV atau lowongan.
- Jangan mengarang nama manajer, email, atau detail perusahaan yang tidak tertulis.
- Jangan menulis placeholder seperti [Add GPA], [Your Name], [Perusahaan], TBD, atau N/A.
- Jika sebuah data tidak ada, lewati.

Panjang 220-340 kata. Tanpa judul, tanpa bullet, tanpa penjelasan di luar surat.
"""


def revise_cover_letter_prompt(draft: str) -> str:
    return f"""Perbaiki surat ini. Pertahankan bahasa, fakta, dan panjangnya.

Yang wajib berubah:
- Jika pembukanya generik, ganti dengan hook yang spesifik. Dilarang "I am writing to apply" dan "Saya menulis untuk melamar".
- Pastikan ada pemahaman konkret tentang peran atau perusahaan dari lowongan yang sudah ada di surat.
- Pastikan dua pengalaman terhubung ke kebutuhan posisi, bukan daftar keahlian.
- Paragraf terakhir harus mengajak wawancara atau percakapan, dengan langkah yang jelas.
- Nada tetap profesional dan tidak kaku.

Surat:
{draft}

Output HANYA surat yang sudah diperbaiki.
"""


@app.post("/generate-cover-letter")
async def generate_cover_letter(
    job_description: str = Form(...),
    tailored_cv: str = Form(...),
    authorization: Optional[str] = Header(None),
):
    token = bearer_token(authorization)
    await current_user_id(token)
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description kosong")
    if not tailored_cv.strip():
        raise HTTPException(status_code=400, detail="CV hasil generate masih kosong")

    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": cover_letter_prompt(tailored_cv, job_description)}],
    )
    letter = strip_placeholders(result.choices[0].message.content.strip())
    if cover_letter_needs_revision(letter):
        revised = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": revise_cover_letter_prompt(letter)}],
        )
        letter = strip_placeholders(revised.choices[0].message.content.strip())
    return {"cover_letter": letter}


@app.post("/download-pdf")
async def download_pdf(markdown_cv: str = Form(...)):
    pdf_bytes = markdown_to_pdf(markdown_cv)
    return Response(content=pdf_bytes, media_type="application/pdf")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
