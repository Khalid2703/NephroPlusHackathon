#!/usr/bin/env python3
"""
Unified Medical Report OCR Parser (Combined from main3.py and main4.py)
- Smooth, robust processing
- Advanced dashboard, ChromaDB, error/debug handling
- Modular and user-friendly
"""

import streamlit as st
import os
import cv2
import numpy as np
import json
import requests
from pathlib import Path
import glob
from datetime import datetime, timedelta
import pytesseract
import easyocr
import traceback
import time
import tempfile
import pandas as pd
import plotly.express as px
from PIL import Image
import fitz  # PyMuPDF for PDFs
from io import BytesIO
import re
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ====== CONFIGURATION ======
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"
BASE_TIMEOUT = 7 * 60  # 7 minutes base timeout
DEBUG_MODE = True
CHROMA_PATH = './chroma_db'
# ===========================

# Initialize ChromaDB client and embedding model (global, so only loaded once)
chroma_client = chromadb.Client(Settings(persist_directory=CHROMA_PATH))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    return embedding_model.encode(texts).tolist()

def build_chroma_collection(jsons):
    collection = chroma_client.get_or_create_collection('medical_reports')
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
    records, metadatas, ids = [], [], []
    for idx, report in enumerate(jsons):
        if 'error' in report or 'patient_info' not in report:
            continue
        patient = report.get('patient_info', {})
        tests = report.get('test_results', [])
        for t_idx, test in enumerate(tests):
            text = f"Patient: {patient.get('name', 'Unknown')}, Age: {patient.get('age', '')}, Gender: {patient.get('gender', '')}, Test: {test.get('test_name', '')}, Result: {test.get('result_value', '')}, Status: {test.get('status', '')}, File: {report.get('_metadata', {}).get('source_file', '')}"
            records.append(text)
            metadatas.append({
                'patient_name': patient.get('name', 'Unknown'),
                'test_name': test.get('test_name', ''),
                'file': report.get('_metadata', {}).get('source_file', ''),
                'test_idx': t_idx
            })
            ids.append(f"{idx}_{t_idx}")
    if records:
        embeddings = embed_texts(records)
        collection.add(documents=records, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return collection

def query_chroma(query, top_n=10):
    collection = chroma_client.get_or_create_collection('medical_reports')
    query_emb = embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_emb], n_results=top_n)
    return results['documents'][0], results['metadatas'][0]

class MultiOCRProcessor:
    def __init__(self):
        self.tesseract_available = True
        self.easyocr_available = True
        try:
            self.easyocr_reader = easyocr.Reader(['en'])
            st.success("✅ EasyOCR initialized successfully")
        except Exception as e:
            st.warning(f"⚠️ EasyOCR not available: {str(e)}")
            self.easyocr_available = False
        try:
            pytesseract.get_tesseract_version()
            st.success("✅ Tesseract OCR available")
        except Exception as e:
            st.warning(f"⚠️ Tesseract not available: {str(e)}")
            self.tesseract_available = False
    def preprocess_image(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = np.array(image)
        if img is None:
            raise ValueError("Could not process image")
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        denoised = cv2.fastNlMeansDenoising(gray)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)
        return enhanced
    def extract_with_tesseract(self, image):
        if not self.tesseract_available:
            return "", []
        try:
            processed_img = self.preprocess_image(image)
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            extracted_texts = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:
                    text = data['text'][i].strip()
                    if text:
                        extracted_texts.append({
                            'text': text,
                            'confidence': int(data['conf'][i]),
                            'method': 'tesseract'
                        })
            full_text = ' '.join([item['text'] for item in extracted_texts])
            return full_text, extracted_texts
        except Exception as e:
            st.error(f"Tesseract OCR error: {str(e)}")
            return "", []
    def extract_with_easyocr(self, image):
        if not self.easyocr_available:
            return "", []
        try:
            processed_img = self.preprocess_image(image)
            results = self.easyocr_reader.readtext(processed_img)
            extracted_texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:
                    extracted_texts.append({
                        'text': text.strip(),
                        'confidence': int(confidence * 100),
                        'method': 'easyocr',
                        'bbox': bbox
                    })
            full_text = ' '.join([item['text'] for item in extracted_texts])
            return full_text, extracted_texts
        except Exception as e:
            st.error(f"EasyOCR error: {str(e)}")
            return "", []
    def combine_ocr_results(self, tesseract_text, tesseract_details, easyocr_text, easyocr_details):
        all_texts = tesseract_details + easyocr_details
        unique_texts = {}
        for item in all_texts:
            text = item['text'].lower().strip()
            if text and (text not in unique_texts or item['confidence'] > unique_texts[text]['confidence']):
                unique_texts[text] = item
        combined_text = tesseract_text
        if len(easyocr_text) > len(tesseract_text):
            combined_text = easyocr_text
        elif len(easyocr_text) > 0 and len(tesseract_text) > 0:
            combined_text = f"{tesseract_text}\n\n--- EasyOCR Additional Content ---\n{easyocr_text}"
        return combined_text, list(unique_texts.values())

class EnhancedMedicalReportOCR:
    def __init__(self, ollama_url=OLLAMA_BASE_URL, model_name=OLLAMA_MODEL):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.model_warmed_up = False
        self.ocr_processor = MultiOCRProcessor()
        self.processing_stats = []
        self._test_ollama_connection()
    def _test_ollama_connection(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                if self.model_name in models:
                    st.success(f"✅ Connected to Ollama - Model {self.model_name} available")
                    return True
                else:
                    st.error(f"❌ Model {self.model_name} not found. Available: {models}")
                    return False
            else:
                st.error(f"❌ Failed to connect to Ollama at {self.ollama_url}")
                return False
        except Exception as e:
            st.error(f"❌ Ollama connection error: {e}")
            return False
    def calculate_dynamic_timeout(self, num_files):
        if num_files <= 1:
            return BASE_TIMEOUT
        elif num_files <= 3:
            return BASE_TIMEOUT + (num_files - 1) * 3 * 60
        else:
            return BASE_TIMEOUT + 6 * 60 + (num_files - 3) * 2 * 60
    def warm_up_model(self):
        if self.model_warmed_up:
            return True
        with st.spinner("🔥 Warming up AI model... This may take a few minutes..."):
            try:
                warm_up_data = {
                    "model": self.model_name,
                    "prompt": "Hello, this is a test. Respond with 'OK'.",
                    "stream": False,
                    "options": {"temperature": 0.1, "max_tokens": 10}
                }
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=warm_up_data,
                    timeout=60
                )
                if response.status_code == 200:
                    st.success("✅ AI model warmed up successfully!")
                    self.model_warmed_up = True
                    return True
                else:
                    st.error(f"❌ Model warm-up failed: HTTP {response.status_code}")
                    return False
            except Exception as e:
                st.error(f"❌ Model warm-up error: {e}")
                return False
    def extract_text_from_image(self, image):
        tesseract_text, tesseract_details = self.ocr_processor.extract_with_tesseract(image)
        easyocr_text, easyocr_details = self.ocr_processor.extract_with_easyocr(image)
        combined_text, combined_details = self.ocr_processor.combine_ocr_results(
            tesseract_text, tesseract_details, easyocr_text, easyocr_details
        )
        return combined_text, combined_details, {
            'tesseract': {'text': tesseract_text, 'blocks': len(tesseract_details)},
            'easyocr': {'text': easyocr_text, 'blocks': len(easyocr_details)}
        }
    def extract_text_from_pdf(self, pdf_bytes):
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            doc.close()
            return full_text, [{'text': full_text, 'confidence': 100, 'method': 'pdf_direct'}]
        except Exception as e:
            st.error(f"PDF extraction error: {e}")
            return "", []
    def clean_json_string(self, json_text):
        json_text = re.sub(r',([ \t\r\n]*[}\]])', r'\1', json_text)
        json_match = re.search(r'\{[\s\S]*\}', json_text)
        if json_match:
            return json_match.group(0)
        return json_text.strip()
    def generate_structured_json(self, extracted_text, filename, timeout):
        if not self.model_warmed_up:
            if not self.warm_up_model():
                return {'success': False, 'error': 'Failed to warm up model'}
        max_length = 3000
        if len(extracted_text) > max_length:
            extracted_text = extracted_text[:max_length] + "\n[TEXT TRUNCATED]"
        prompt = f"""You are an expert medical report parser. Convert this OCR-extracted text into a structured JSON format.\n\nExtracted text from medical report:\n{extracted_text}\n\nCreate a comprehensive JSON with this structure:\n... (same as main4.py) ...\nExtract ALL available information. Use null for missing fields. Return ONLY the JSON structure."""
        try:
            with st.spinner(f"🤖 Processing with AI model... (timeout: {timeout//60} minutes)"):
                start_time = time.time()
                request_data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_tokens": 2048
                    }
                }
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=request_data,
                    timeout=timeout
                )
                processing_time = time.time() - start_time
                if response.status_code == 200:
                    result = response.json()
                    json_text = result.get('response', '').strip()
                    if json_text.startswith('```json'):
                        json_text = json_text[7:]
                    elif json_text.startswith('```'):
                        json_text = json_text[3:]
                    if json_text.endswith('```'):
                        json_text = json_text[:-3]
                    json_text = json_text.strip()
                    st.write("OLLAMA RAW RESPONSE:", json_text)
                    json_str = self.clean_json_string(json_text)
                    try:
                        parsed_json = json.loads(json_str)
                    except Exception as e:
                        st.error(f"JSON parsing error: {e}")
                        st.write("Raw JSON string:", json_str)
                        debug_dir = Path('output/debug')
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        debug_path = debug_dir / f"{filename}_json_debug.txt"
                        with open(debug_path, 'w', encoding='utf-8') as f:
                            f.write(f"Error: {e}\n\nRaw JSON string (first 1000 chars):\n{json_str[:1000]}\n\nFull JSON text (first 1000 chars):\n{json_text[:1000]}")
                        return {
                            'success': False,
                            'error': f'JSON parsing error: {e}',
                            'raw_response': json_text,
                            'debug_file': str(debug_path),
                            'error_context': {
                                'error_message': str(e),
                                'first_500_chars': json_str[:500],
                                'timestamp': datetime.now().isoformat()
                            }
                        }
                    parsed_json['_metadata'] = {
                        'source_file': filename,
                        'processing_timestamp': datetime.now().isoformat(),
                        'processing_time_seconds': processing_time,
                        'model_used': self.model_name,
                        'extraction_method': 'multi_ocr_ollama'
                    }
                    self.processing_stats.append({
                        'filename': filename,
                        'processing_time': processing_time,
                        'timestamp': datetime.now()
                    })
                    output_dir = Path('output/json')
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{filename}_extracted.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed_json, f, indent=2)
                    return {
                        'success': True,
                        'json_data': parsed_json,
                        'processing_time': processing_time
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Ollama API error: HTTP {response.status_code}',
                        'response': response.text
                    }
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': f'Processing timed out after {timeout//60} minutes'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Processing error: {str(e)}'
            }
    def process_file(self, file_content, filename, file_type, timeout):
        start_time = time.time()
        try:
            if file_type == 'pdf':
                extracted_text, extraction_details = self.extract_text_from_pdf(file_content)
                ocr_stats = {'pdf_direct': {'text': extracted_text, 'blocks': 1}}
            else:
                image = Image.open(BytesIO(file_content))
                extracted_text, extraction_details, ocr_stats = self.extract_text_from_image(image)
            if not extracted_text.strip():
                return {
                    'success': False,
                    'filename': filename,
                    'error': 'No text extracted from file',
                    'processing_time': time.time() - start_time
                }
            json_result = self.generate_structured_json(extracted_text, filename, timeout)
            if json_result['success']:
                return {
                    'success': True,
                    'filename': filename,
                    'extracted_text': extracted_text,
                    'extraction_details': extraction_details,
                    'ocr_stats': ocr_stats,
                    'structured_json': json_result['json_data'],
                    'processing_time': time.time() - start_time
                }
            else:
                return {
                    'success': False,
                    'filename': filename,
                    'error': json_result['error'],
                    'extracted_text': extracted_text,
                    'processing_time': time.time() - start_time,
                    'debug_file': json_result.get('debug_file'),
                    'error_context': json_result.get('error_context')
                }
        except Exception as e:
            return {
                'success': False,
                'filename': filename,
                'error': f'File processing error: {str(e)}',
                'processing_time': time.time() - start_time
            }

# --- Helper: Load all JSONs with error handling ---
def load_all_jsons(json_dir="output/json"):
    all_data = []
    json_path = Path(json_dir)
    if not json_path.exists():
        st.warning(f"JSON directory {json_dir} does not exist")
        return all_data
    json_files = list(json_path.glob("*.json"))
    if not json_files:
        st.info(f"No JSON files found in {json_dir}")
        return all_data
    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data['_filename'] = file.name
                all_data.append(data)
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
            all_data.append({
                'error': f'Failed to load {file.name}: {str(e)}',
                '_filename': file.name
            })
    return all_data

# --- Helper: Flatten report data for dashboard/analysis ---
def flatten_report_data(reports):
    flat_data = []
    for report in reports:
        if "error" in report:
            continue
        base_info = {
            "patient_name": str(report.get("patient_info", {}).get("name", "Unknown")),
            "patient_age": str(report.get("patient_info", {}).get("age", "")),
            "patient_gender": str(report.get("patient_info", {}).get("gender", "")),
            "report_date": str(report.get("report_info", {}).get("report_date", "")),
            "source_file": str(report.get("_metadata", {}).get("source_file", ""))
        }
        for test in report.get("test_results", []):
            record = base_info.copy()
            record.update({
                "test_name": str(test.get("test_name", "")),
                "result_value": str(test.get("result_value", "")),
                "unit": str(test.get("unit", "")),
                "reference_range": str(test.get("reference_range", "")),
                "status": str(test.get("status", "unknown"))
            })
            flat_data.append(record)
    return flat_data

# --- Helper: Create dashboard visualizations ---
def create_patient_dashboard(results):
    if not results:
        st.warning("No data available for dashboard")
        return
    patients_data = []
    for result in results:
        if 'error' in result or 'patient_info' not in result:
            continue
        patient_info = result.get('patient_info', {})
        test_results = result.get('test_results', [])
        patients_data.append({
            'name': str(patient_info.get('name', 'Unknown')),
            'age': str(patient_info.get('age', 'N/A')),
            'gender': str(patient_info.get('gender', 'N/A')),
            'num_tests': len(test_results),
            'hospital': str(result.get('hospital_info', {}).get('hospital_name', 'Unknown')),
            'report_date': str(result.get('report_info', {}).get('report_date', 'N/A')),
            'filename': str(result.get('filename', result.get('_metadata', {}).get('source_file', '')))
        })
    if not patients_data:
        st.warning("No valid patient data found")
        return
    df = pd.DataFrame(patients_data)
    # Normalize gender values
    def normalize_gender(gender):
        if not gender or str(gender).strip().lower() in ['none', 'n/a', '', 'nan']:
            return 'Unknown'
        g = str(gender).strip().lower()
        if g in ['male', 'm']:
            return 'Male'
        if g in ['female', 'f']:
            return 'Female'
        return gender  # fallback to original if not matched
    df['gender'] = df['gender'].apply(normalize_gender)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Total Tests", df['num_tests'].sum())
    with col3:
        st.metric("Avg Tests per Patient", f"{df['num_tests'].mean():.1f}")
    with col4:
        unique_hospitals = df['hospital'].nunique()
        st.metric("Hospitals", unique_hospitals)
    col1, col2 = st.columns(2)
    with col1:
        try:
            def extract_age(age_str):
                if pd.isna(age_str) or age_str == 'N/A':
                    return None
                match = re.search(r'(\d+)', str(age_str))
                return int(match.group(1)) if match else None
            df['age_numeric'] = df['age'].apply(extract_age)
            valid_ages = df.dropna(subset=['age_numeric'])
            if not valid_ages.empty:
                fig_age = px.histogram(valid_ages, x='age_numeric', title='Age Distribution of Patients', nbins=20)
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("Age distribution not available (no valid numeric ages)")
        except Exception as e:
            st.info(f"Age distribution not available: {str(e)}")
    with col2:
        fig_tests = px.bar(df, x='name', y='num_tests', title='Number of Tests per Patient', color='num_tests', color_continuous_scale='viridis')
        fig_tests.update_xaxes(tickangle=45)
        st.plotly_chart(fig_tests, use_container_width=True)
    if df['gender'].notna().any():
        gender_counts = df['gender'].value_counts()
        if not gender_counts.empty:
            fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index, title='Gender Distribution')
            st.plotly_chart(fig_gender, use_container_width=True)
    hospital_counts = df['hospital'].value_counts()
    if not hospital_counts.empty:
        fig_hospital = px.bar(x=hospital_counts.index, y=hospital_counts.values, title='Patients by Hospital')
        st.plotly_chart(fig_hospital, use_container_width=True)
    st.subheader("📋 Patient Details")
    st.dataframe(df, use_container_width=True)

# --- Helper: Severity leaderboard ---
def create_severity_leaderboard(results):
    if not results:
        return
    st.subheader("🏆 Patient Severity Analysis")
    st.info("This analysis is based on the number of abnormal test results and should not replace medical judgment.")
    severity_data = []
    for result in results:
        if 'error' in result or 'patient_info' not in result:
            continue
        patient_info = result.get('patient_info', {})
        test_results = result.get('test_results', [])
        abnormal_count = 0
        total_tests = len(test_results)
        for test in test_results:
            status = str(test.get('status') or '').lower()
            if status in ['abnormal', 'high', 'low', 'critical']:
                abnormal_count += 1
        severity_score = (abnormal_count / total_tests * 100) if total_tests > 0 else 0
        severity_data.append({
            'patient_name': str(patient_info.get('name', 'Unknown')),
            'age': str(patient_info.get('age', 'N/A')),
            'total_tests': total_tests,
            'abnormal_tests': abnormal_count,
            'severity_score': severity_score,
            'hospital': str(result.get('hospital_info', {}).get('hospital_name', 'Unknown'))
        })
    if not severity_data:
        st.warning("No severity data available")
        return
    severity_df = pd.DataFrame(severity_data).sort_values('severity_score', ascending=False)
    def get_severity_color(score):
        if score >= 70:
            return "🔴 High"
        elif score >= 40:
            return "🟡 Medium"
        else:
            return "🟢 Low"
    severity_df['severity_level'] = severity_df['severity_score'].apply(get_severity_color)
    st.dataframe(
        severity_df[['patient_name', 'age', 'total_tests', 'abnormal_tests', 'severity_score', 'severity_level', 'hospital']],
        use_container_width=True
    )
    severity_counts = severity_df['severity_level'].value_counts()
    if not severity_counts.empty:
        fig = px.pie(values=severity_counts.values, names=severity_counts.index, title='Severity Distribution', color_discrete_map={'🔴 High': 'red', '🟡 Medium': 'yellow', '🟢 Low': 'green'})
        st.plotly_chart(fig, use_container_width=True)

# --- Helper: Create AI query prompt ---
def create_query_prompt(query, data):
    return f"""
You are a medical data analyst. Given the following medical report data in JSON format,
answer the user's question in a clear, concise manner with relevant statistics and findings.
Rules:
1. Always provide specific numbers when available
2. Highlight abnormal results
3. Group similar findings together
4. If the query requires calculations, show your work
5. Be precise with units and reference ranges
6. When listing patients, include their names if available
7. Clearly indicate when data is missing or unknown
User Question: {query}
Medical Data:
{json.dumps(data, indent=2)}
Provide your analysis below:
"""

def ask_ollama(prompt, timeout=600):
    request_data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "max_tokens": 2048
        }
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=request_data,
            timeout=timeout
        )
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'No answer returned.')
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error querying Ollama: {e}"

# --- Main Streamlit App ---
def main():
    st.set_page_config(
        page_title="SwasthStack",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🏥 SwasthStack")
    st.markdown("### AI-Powered Medical Report Extraction with Multi-OCR, Dashboard, and AI Query")
    if 'processor' not in st.session_state:
        st.session_state.processor = EnhancedMedicalReportOCR()
    if 'results' not in st.session_state:
        st.session_state.results = []
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("🤖 AI Model Settings")
        model_name = st.selectbox(
            "Select Ollama Model",
            ["llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "deepseek-coder:6.7b"],
            index=0
        )
        if model_name != st.session_state.processor.model_name:
            st.session_state.processor.model_name = model_name
            st.session_state.processor.model_warmed_up = False
        st.subheader("⏱️ Processing Settings")
        base_timeout = st.slider("Base Timeout (minutes)", 5, 15, 7)
        st.subheader("👁️ OCR Settings")
        st.info("Multi-OCR engines automatically combine results for better accuracy")
        st.subheader("📊 System Status")
        if st.session_state.processor.ocr_processor.tesseract_available:
            st.success("✅ Tesseract OCR")
        else:
            st.error("❌ Tesseract OCR")
        if st.session_state.processor.ocr_processor.easyocr_available:
            st.success("✅ EasyOCR")
        else:
            st.error("❌ EasyOCR")
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload & Process", "📊 Dashboard", "🏆 Severity Analysis", "📈 Processing Stats", "🤖 AI Query"])
    # --- Tab 1: Upload & Process ---
    with tab1:
        st.header("📤 Upload Medical Reports")
        uploaded_files = st.file_uploader(
            "Choose medical report files (Images: JPG, PNG, etc. | Documents: PDF)",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'pdf'],
            accept_multiple_files=True,
            help="You can upload multiple files at once. Supported formats: Images and PDFs"
        )
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} file(s) uploaded successfully!")
            for i, file in enumerate(uploaded_files):
                st.write(f"**{i+1}.** {file.name} ({file.size/1024:.1f} KB)")
            timeout = st.session_state.processor.calculate_dynamic_timeout(len(uploaded_files))
            st.info(f"⏱️ Estimated processing timeout: {timeout//60} minutes (based on {len(uploaded_files)} files)")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Start Processing", type="primary"):
                    process_files(uploaded_files, timeout)
            with col2:
                if st.button("🔥 Warm Up AI Model"):
                    st.session_state.processor.warm_up_model()
            if st.session_state.results:
                st.subheader("📋 Recent Processing Results")
                for i, result in enumerate(st.session_state.results[-3:]):
                    with st.expander(f"📄 {result['filename']} - {'✅ Success' if result['success'] else '❌ Failed'}"):
                        if result['success']:
                            patient_name = result['structured_json'].get('patient_info', {}).get('name', 'Unknown')
                            num_tests = len(result['structured_json'].get('test_results', []))
                            st.write(f"**Patient:** {patient_name}")
                            st.write(f"**Tests Found:** {num_tests}")
                            st.write(f"**Processing Time:** {result['processing_time']:.1f} seconds")
                            if st.button(f"📥 Download JSON", key=f"download_{i}"):
                                json_str = json.dumps(result['structured_json'], indent=2)
                                st.download_button(
                                    label="Download JSON",
                                    data=json_str,
                                    file_name=f"{result['filename']}_extracted.json",
                                    mime="application/json"
                                )
                        else:
                            st.error(f"**Error:** {result['error']}")
                            st.write(f"**Processing Time:** {result['processing_time']:.1f} seconds")
                            if result.get('debug_file'):
                                st.info(f"Debug info saved: {result['debug_file']}")
    # --- Tab 2: Dashboard ---
    with tab2:
        st.header("📊 Patient Dashboard")
        all_jsons = load_all_jsons("output/json")
        flat_data = flatten_report_data(all_jsons)
        if all_jsons:
            create_patient_dashboard(all_jsons)
        else:
            st.info("No extracted data available for analysis.")
    # --- Tab 3: Severity Analysis ---
    with tab3:
        st.header("🏆 Severity Analysis")
        all_jsons = load_all_jsons("output/json")
        if all_jsons:
            create_severity_leaderboard(all_jsons)
        else:
            st.info("No extracted data available for severity analysis.")
    # --- Tab 4: Processing Stats ---
    with tab4:
        st.header("📈 Processing Statistics")
        if st.session_state.results:
            successful_results = [r for r in st.session_state.results if r['success']]
            if successful_results:
                processing_times = [r['processing_time'] for r in successful_results]
                avg_time = sum(processing_times) / len(successful_results)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files Processed", len(st.session_state.results))
                with col2:
                    st.metric("Successful Extractions", len(successful_results))
                with col3:
                    st.metric("Average Processing Time", f"{avg_time:.1f}s")
                fig_time = px.line(
                    x=range(1, len(processing_times) + 1),
                    y=processing_times,
                    title="Processing Time per File",
                    labels={'x': 'File Number', 'y': 'Processing Time (seconds)'}
                )
                st.plotly_chart(fig_time, use_container_width=True)
                success_rate = (len(successful_results) / len(st.session_state.results)) * 100
                fig_success = px.pie(
                    values=[len(successful_results), len(st.session_state.results) - len(successful_results)],
                    names=['Successful', 'Failed'],
                    title=f"Success Rate: {success_rate:.1f}%",
                    color_discrete_map={'Successful': 'green', 'Failed': 'red'}
                )
                st.plotly_chart(fig_success, use_container_width=True)
                st.subheader("📋 Processing Log")
                log_data = []
                for i, result in enumerate(st.session_state.results):
                    log_data.append({
                        'File': result['filename'],
                        'Status': '✅ Success' if result['success'] else '❌ Failed',
                        'Processing Time (s)': f"{result['processing_time']:.1f}",
                        'Error': result.get('error', 'N/A')
                    })
                log_df = pd.DataFrame(log_data)
                st.dataframe(log_df, use_container_width=True)
            else:
                st.warning("No successful extractions to analyze")
        else:
            st.info("No processing results available.")
    # --- Tab 5: AI Query ---
    with tab5:
        st.header("🤖 Ask Llama About All Medical Reports")
        user_query = st.text_input("Enter your question about the reports", key="ai_query_all_jsons", placeholder="e.g. How many patients have abnormal hemoglobin?")
        if user_query:
            top_docs, top_metas = query_chroma(user_query, top_n=10)
            st.write("#### Top relevant records used for answer:")
            for doc, meta in zip(top_docs, top_metas):
                st.write(f"- {doc}")
            prompt = create_query_prompt(user_query, top_docs)
            if st.button("Ask Llama", key="ask_llama_all_jsons"):
                with st.spinner("Querying Llama..."):
                    answer = ask_ollama(prompt)
                st.write("### Llama's Answer:")
                st.markdown(answer)

def process_files(uploaded_files, timeout):
    st.write("DEBUG: process_files called")
    if not uploaded_files:
        st.error("No files uploaded")
        return
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    total_files = len(uploaded_files)
    st.write(f"DEBUG: {total_files} files to process")
    for i, uploaded_file in enumerate(uploaded_files):
        st.write(f"DEBUG: Processing file {uploaded_file.name}")
        try:
            progress = i / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{total_files})")
            file_content = uploaded_file.read()
            st.write(f"DEBUG: Read {len(file_content)} bytes from {uploaded_file.name}")
            file_type = 'pdf' if uploaded_file.name.lower().endswith('.pdf') else 'image'
            result = st.session_state.processor.process_file(
                file_content, 
                uploaded_file.name, 
                file_type, 
                timeout
            )
            st.write(f"DEBUG: Result for {uploaded_file.name}: {result}")
            results.append(result)
            if not result['success']:
                output_dir = Path('output/json')
                output_dir.mkdir(parents=True, exist_ok=True)
                error_path = output_dir / f"{uploaded_file.name}_error.json"
                with open(error_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
            if result['success']:
                st.success(f"✅ {uploaded_file.name} processed successfully!")
            else:
                st.error(f"❌ {uploaded_file.name} failed: {result['error']}")
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            st.write(f"DEBUG: Exception for {uploaded_file.name}: {str(e)}")
            error_result = {
                'success': False,
                'filename': uploaded_file.name,
                'error': str(e),
                'processing_time': 0
            }
            results.append(error_result)
            output_dir = Path('output/json')
            output_dir.mkdir(parents=True, exist_ok=True)
            error_path = output_dir / f"{uploaded_file.name}_error.json"
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2)
    progress_bar.progress(100)
    status_text.text("Processing complete!")
    st.session_state.results.extend(results)
    # After processing, update ChromaDB
    all_jsons = load_all_jsons("output/json")
    build_chroma_collection(all_jsons)

if __name__ == "__main__":
    main()