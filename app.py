import os
import json
import re
from dotenv import load_dotenv
import streamlit as st
import pdfplumber
from PyPDF2 import PdfMerger
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Load environment
load_dotenv()
GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model=GENAI_MODEL, google_api_key=GOOGLE_API_KEY)

# Define tool functions
def extract_title_and_type(input_text):
    return llm.invoke(f"Extract the title and type of this document in JSON.\nText:\n{input_text}")

def summarize_content(input_text):
    return llm.invoke(f"Summarize this legal document:\n{input_text}")

# Register tools
tools = [
    Tool(name="TitleExtractor", func=extract_title_and_type, description="Extract title and type from document."),
    # Tool(name="Summarizer", func=summarize_content, description="Summarize the document content."),
]

# Main orchestrator agent
orchestrator_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Parse response utility
def parse_gemini_response(response):
    if hasattr(response, "content"):
        response = response.content
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())

# Generate cover
def create_cover_page(exhibits):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, 750, "Client Exhibit List")
    c.setFont("Helvetica", 12)
    y = 700
    for ex in exhibits:
        c.drawString(100, y, ex['label'])
        y -= 30  # add more space between lines since no summary now
    c.save()
    buffer.seek(0)
    return buffer


def split_text_to_lines(text, max_chars):
    words = text.split()
    lines = []
    current = []
    for word in words:
        current.append(word)
        if len(" ".join(current)) > max_chars:
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))
    return lines

# --- Streamlit UI ---
st.title("\U0001F4D1 Legal Document Exhibit Bundler (Agentic Reasoning Mode)")

# uploaded_files = st.file_uploader("Upload legal PDF documents", type=["pdf"], accept_multiple_files=True)
uploaded_files_raw = st.file_uploader("Upload legal PDF documents", type=["pdf"], accept_multiple_files=True)
uploaded_files = []

if uploaded_files_raw:
    st.markdown("### 📄 Arrange Document Order")
    indices = st.multiselect("Select document order", options=list(range(len(uploaded_files_raw))), default=list(range(len(uploaded_files_raw))), format_func=lambda x: uploaded_files_raw[x].name)
    if len(indices) == len(uploaded_files_raw):
        uploaded_files = [uploaded_files_raw[i] for i in indices]
    else:
        st.warning("Please select all documents to arrange them in order.")
        
if uploaded_files:
    if st.button("\U0001F680 Generate Exhibit Bundle"):
        exhibits = []
        page_counter = 0

        # Temporary cover page
        temp_cover = create_cover_page([{"label": "...temporary..."}])
        temp_cover.seek(0)
        final_merger = PdfMerger()
        final_merger.append(temp_cover)
        page_counter += 1

        for i, file in enumerate(uploaded_files):
            with pdfplumber.open(file) as pdf:
                first_page_text = pdf.pages[0].extract_text() or "No text found."

            st.write(f"\U0001F50D Processing Document {i+1}: {file.name}")
            print(f"\n\U0001F50D Processing Document {i+1}: {file.name}")

            query = f"""
            You are a legal assistant. Analyze the following text from the first page of a document and extract:

            - Full name of the client (if available)
            - Form number (if available)

            If not found, return null.
            Dont return any text or anything else at all. I just need this json. If anything is missing, return null.
            Text:
            {first_page_text}

            Return in JSON:
            {{
            "client_name": "...",
            "form_number": "..."
            }}
            """
            try:
                response = orchestrator_agent.run(query)
                print("Raw agent response:\n", response)
                st.code(response, language="json")

                result = parse_gemini_response(response)
                # title = result.get("title", "Untitled")
                # summary = result.get("summary", "")
                client_name = result.get("client_name", "Unknown Client")
                form_number = result.get("form_number", "Unknown Form")
            except Exception as e:
                st.error(f"❌ Failed to parse response for Document {i+1}: {e}")
                print(f"❌ Failed to parse response for Document {i+1}: {e}")
                title = "Untitled"
                summary = ""
                client_name = "Unknown Client"
                form_number = "Unknown Form"

            # st.text_area(f"📝 Summary for Document {i+1}", summary)
            # print(f"📝 Summary for Document {i+1}:\n{summary}")

            label = f"Exhibit {i+1}: Client: {client_name} — Form: {form_number} — Starts on Page {page_counter + 1}"
            exhibits.append({"label": label})

            file.seek(0)
            final_merger.append(file)
            with pdfplumber.open(file) as pdf:
                page_counter += len(pdf.pages)

        # Generate correct cover
        final_output = BytesIO()
        corrected_cover = create_cover_page(exhibits)
        corrected_cover.seek(0)

        merged = PdfMerger()
        merged.append(corrected_cover)
        for file in uploaded_files:
            file.seek(0)
            merged.append(file)

        merged.write(final_output)
        final_output.seek(0)

        st.success("\U0001F389 Exhibit bundle created successfully!")
        st.download_button("\U0001F4E5 Download Final Exhibit PDF", final_output, file_name="exhibit_bundle.pdf")
