import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import datetime
import tempfile
import base64
import pandas as pd
from PyPDF2 import PdfReader
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from werkzeug.utils import secure_filename
import requests
from bs4 import BeautifulSoup
import http.client
import json
from fpdf import FPDF
import plotly.express as px
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
import io


# Configure Gemini API
genai.configure(api_key="AIzaSyBwrxR_jXcVCczEKom-nYUCPqvx9EReHuo")  # Replace with your real key
model = genai.GenerativeModel("gemini-1.5-flash")
RAPI_KEY = "1783318fc1msh4e42d25e7b9ffe8p1ef51cjsn3fb8eb99b350"  # Replace with your real RapidAPI key


# ─── Streamlit Setup ──────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Toolbox Pro", layout="wide")
st.sidebar.title("🧠 AI Toolbox Pro")
actions = [
    "Job Based Resume Creator",
    "Extract Text from File",
    "Describe Uploaded Image",
    "Translate Text",
    "Generate from Prompt",
    "Summarize Text",
    "Keyword Extractor",
    "Resume Analyzer",
    "Email Draft Generator",
    "Voice to Text",
    "Text to Voice",
    "Text to AI Image",
    "Data Analyzer",
    "Fetch from URL",
    "Chat with AI",
    "Conversion Tools",
    "Code Assistant 💻"
]
action = st.sidebar.selectbox("Choose Task", actions)
output_format = st.sidebar.selectbox("Choose Output Format", ["txt","csv","xlsx","pdf","png","json","zip"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("✨ AI Toolbox Pro Dashboard")
st.caption(f"🕒 {datetime.datetime.now().strftime('%A, %d %B %Y, %I:%M %p')}")

UPLOAD_DIR = "uploads"
TEMP_DIR = tempfile.mkdtemp()
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Common Inputs ────────────────────────────────────────────────────────────
user_prompt = st.text_area("💬 Enter your prompt or instructions:")
uploaded_file = st.file_uploader("📁 Upload file", type=["txt","pdf","csv","xlsx","png","jpg","jpeg","mp3","json"])
filename = ""
if uploaded_file:
    filename = os.path.join(UPLOAD_DIR, secure_filename(uploaded_file.name))
    with open(filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

# ─── Action-Specific UI ───────────────────────────────────────────────────────
if action == "Translate Text":
    target_lang = st.text_input("Target Language (e.g., French, Spanish):")

elif action == "Email Draft Generator":
    email_to = st.text_input("Recipient Name/Role:")
    email_subject = st.text_input("Subject:")
    email_purpose = st.text_area("Purpose / Key Points:")
    email_tone = st.selectbox("Tone:", ["Formal","Informal","Friendly","Persuasive"])

elif action == "Chat with AI":
    chat_input = st.text_input("You:")

elif action == "Data Analyzer":
    data_analysis_option = st.selectbox("Data Action:", ["Generate Insights","Find Hidden Trends","Show Graphs & Plots","Map Visualization"])

elif action == "Resume Analyzer":
    resume_options = [
        "Introduction of Candidate",
        "Explain Past Roles and Experience",
        "Project to Explain in Interview",
        "Interview Questions with Answers",
        "Why are you looking for a new role?",
        "Resume Score and JD Matching",
        "LinkedIn Summary Generator",
        "Export Full PDF Resume Report"
    ]
    resume_query = st.selectbox("Resume Action:", resume_options)

elif action == "Fetch from URL":
    url_input = st.text_input("Enter URL:")

elif action == "Conversion Tools":
    conversion_tool = st.selectbox("Choose Conversion:", [
        "Text to PDF", "PDF to Text", "PDF to CSV", "CSV to PDF",
        "JSON to PDF", "PDF to JSON", "Image to PDF",
        "Excel to PDF", "PDF to Excel"
    ])

# ─── Helpers: Extraction + Saving ─────────────────────────────────────────────
def extract_text(path: str) -> str:
    if path.endswith(".txt"):
        return open(path, "r", encoding="utf-8").read()
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    if path.endswith(".csv"):
        return pd.read_csv(path).to_string()
    if path.endswith(".xlsx"):
        return pd.read_excel(path).to_string()
    if path.lower().endswith(("png","jpg","jpeg")):
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    if path.endswith(".json"):
        return json.dumps(json.load(open(path)), indent=2)
    return ""

def save_file(text: str, ext: str) -> str:
    out = os.path.join(TEMP_DIR, f"output.{ext}")
    if ext == "txt":
        open(out,"w",encoding="utf-8").write(text)
    elif ext in ("csv","xlsx"):
        df = pd.DataFrame(text.split("\n"), columns=["Response"])
        df.to_csv(out if ext=="csv" else out, index=False)
    elif ext == "pdf":
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf.output(out)
    elif ext == "png":
        img = Image.new("RGB", (1000,1000), (255,255,255))
        d = ImageDraw.Draw(img)
        d.multiline_text((10,10), text, font=ImageFont.load_default())
        img.save(out)
    elif ext == "json":
        json.dump({"response": text}, open(out,"w"), indent=2)
    return out

# ─── Conversion Implementations ─────────────────────────────────────────────
def text_to_pdf(text, path):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(path)

def csv_to_pdf(path_in, path_out):
    df = pd.read_csv(path_in)
    pdf = FPDF(orientation='L'); pdf.add_page(); pdf.set_font("Arial", size=10)
    colw = pdf.epw / len(df.columns)
    for col in df.columns:
        pdf.cell(colw, 10, col, border=1)
    pdf.ln()
    for _, row in df.iterrows():
        for cell in row:
            pdf.cell(colw, 10, str(cell), border=1)
        pdf.ln()
    pdf.output(path_out)

def json_to_pdf(path_in, path_out):
    data = json.load(open(path_in))
    text = json.dumps(data, indent=2)
    text_to_pdf(text, path_out)

def image_to_pdf(path_in, path_out):
    img = Image.open(path_in).convert("RGB")
    img.save(path_out)

def excel_to_pdf(path_in, path_out):
    df = pd.read_excel(path_in)
    csv_tmp = path_in.replace('.xlsx','.csv')
    df.to_csv(csv_tmp, index=False)
    csv_to_pdf(csv_tmp, path_out)

# ─── Main “Run” Handler ──────────────────────────────────────────────────────
if st.button("🚀 Run"):
    response_text = ""
    try:
        # ── Conversion Tools ────────────────────────────────────────────────
        if action == "Conversion Tools" and filename:
            out = os.path.join(TEMP_DIR, "converted")
            if conversion_tool == "Text to PDF":
                txt = user_prompt or extract_text(filename)
                pdf_out = out + ".pdf"
                text_to_pdf(txt, pdf_out)
                st.download_button("Download PDF", open(pdf_out, "rb"), file_name="text.pdf", mime="application/pdf")

            elif conversion_tool == "PDF to Text":
                txt = extract_text(filename)
                st.text_area("Extracted Text", txt, height=300)
                response_text = txt

            elif conversion_tool == "PDF to CSV":
                txt = extract_text(filename)
                rows = [r.split() for r in txt.split("\n") if r.strip()]
                df = pd.DataFrame(rows)
                csv_out = out + ".csv"
                df.to_csv(csv_out, index=False)
                st.download_button("Download CSV", open(csv_out, "rb"), file_name="data.csv")

            elif conversion_tool == "CSV to PDF":
                pdf_out = out + ".pdf"
                csv_to_pdf(filename, pdf_out)
                st.download_button("Download PDF", open(pdf_out, "rb"), file_name="data.pdf", mime="application/pdf")

            elif conversion_tool == "JSON to PDF":
                pdf_out = out + ".pdf"
                json_to_pdf(filename, pdf_out)
                st.download_button("Download PDF", open(pdf_out, "rb"), file_name="data.pdf", mime="application/pdf")

            elif conversion_tool == "PDF to JSON":
                txt = extract_text(filename)
                js = {"text": txt}
                json_out = out + ".json"
                json.dump(js, open(json_out, "w"), indent=2)
                st.download_button("Download JSON", open(json_out, "rb"), file_name="data.json", mime="application/json")

            elif conversion_tool == "Image to PDF":
                pdf_out = out + ".pdf"
                image_to_pdf(filename, pdf_out)
                st.download_button("Download PDF", open(pdf_out, "rb"), file_name="image.pdf", mime="application/pdf")

            elif conversion_tool == "Excel to PDF":
                pdf_out = out + ".pdf"
                excel_to_pdf(filename, pdf_out)
                st.download_button("Download PDF", open(pdf_out, "rb"), file_name="sheet.pdf", mime="application/pdf")

            elif conversion_tool == "PDF to Excel":
                txt = extract_text(filename)
                rows = [r.split() for r in txt.split("\n") if r.strip()]
                df = pd.DataFrame(rows)
                xlsx_out = out + ".xlsx"
                df.to_excel(xlsx_out, index=False)
                st.download_button("Download Excel", open(xlsx_out, "rb"), file_name="data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ── Fetch from URL ────────────────────────────────────────────────
        elif action == "Fetch from URL" and url_input:
            r = requests.get(url_input)
            soup = BeautifulSoup(r.text, "html.parser")
            txt = soup.get_text(separator=" ", strip=True)
            st.subheader("Fetched Page Text")
            st.text_area("", txt[:2000], height=200)
            response_text = model.generate_content(f"Summarize this content:\n{txt[:2000]}").text

        # ── Simple Model Calls ─────────────────────────────────────────────
        base = user_prompt or (extract_text(filename) if filename else "")
        if action == "Generate from Prompt" and user_prompt:
            response_text = model.generate_content(user_prompt).text

        elif action == "Summarize Text" and base:
            response_text = model.generate_content(f"Summarize:\n{base}").text

        elif action == "Keyword Extractor" and base:
            response_text = model.generate_content(f"Extract keywords from:\n{base}").text

        elif action == "Translate Text" and base and target_lang:
            response_text = model.generate_content(f"Translate to {target_lang}:\n{base}").text

        elif action == "Email Draft Generator" and email_purpose:
            prompt = f"Draft a {email_tone} email to {email_to} with subject '{email_subject}' covering: {email_purpose}"
            response_text = model.generate_content(prompt).text

        elif action == "Voice to Text" and filename:
            wav = filename
            if filename.lower().endswith(".mp3"):
                wav = filename.replace(".mp3", ".wav")
                AudioSegment.from_mp3(filename).export(wav, format="wav")
            r = sr.Recognizer()
            with sr.AudioFile(wav) as src:
                audio = r.record(src)
            response_text = r.recognize_google(audio)

        elif action == "Text to Voice" and user_prompt:
            tts = gTTS(user_prompt)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            st.audio(buf.getvalue(), format="audio/mp3")
            response_text = "🔊 Audio generated"

        elif action == "Chat with AI" and chat_input:
            response_text = model.generate_content(chat_input).text
            st.session_state.chat_history.append(("Chat", chat_input, response_text))

        elif action == "Data Analyzer" and filename:
            df = pd.read_csv(filename) if filename.endswith(".csv") else pd.read_excel(filename)
            if data_analysis_option == "Generate Insights":
                response_text = model.generate_content(f"Insights:\n{df.head().to_string()}").text
            elif data_analysis_option == "Find Hidden Trends":
                response_text = model.generate_content(f"Trends:\n{df.head().to_string()}").text
            elif data_analysis_option == "Show Graphs & Plots":
                st.subheader("Plots")
                for c in df.select_dtypes("number").columns:
                    st.plotly_chart(px.histogram(df, x=c, title=c))
            elif data_analysis_option == "Map Visualization" and {"latitude","longitude"}.issubset(df.columns):
                st.map(df)

        # ── Display & Download ────────────────────────────────────────────
        if response_text:
            st.subheader("💡 AI Response")
            st.write(response_text)
            path = save_file(response_text, output_format)
            data = open(path, "rb").read()
            b64 = base64.b64encode(data).decode()
            st.markdown(
                f'<a href="data:application/octet-stream;base64,{b64}" '
                f'download="{os.path.basename(path)}">Download</a>',
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Error: {e}")

# ─── Sidebar: Chat History ────────────────────────────────────────────────────
if st.sidebar.checkbox("📜 Show History"):
    for i, (act, pr, res) in enumerate(reversed(st.session_state.chat_history), 1):
        with st.sidebar.expander(f"{i}. {act}"):
            st.write(f"**Prompt:** {pr}")
            st.write(f"**Response:** {res}")

# ─── Code Assistant ──────────────────────────────────────────────────────────
if action == "Code Assistant 💻":
    st.subheader("🧠 Code Assistant")
    coding_question = st.text_area("Ask a coding question:")
    if st.button("Generate Code"):
        try:
            code = model.generate_content(f"Answer with code: {coding_question}").text
            st.code(code, language="python")
        except Exception as e:
            st.error(f"Failed: {e}")
