# 🏥 MediExtract AI - Enhanced Medical Report OCR

A comprehensive AI-powered medical report extraction system with multi-OCR technology, dynamic processing, and interactive dashboards.

## ✨ Features

### 🔍 Multi-OCR Technology
- **Tesseract OCR**: Traditional OCR engine for reliable text extraction
- **EasyOCR**: Deep learning-based OCR for better accuracy
- **Intelligent Combination**: Automatically combines results from multiple OCR engines for optimal accuracy

### 🤖 AI-Powered Processing
- **Ollama Integration**: Uses local LLM models for structured JSON generation
- **Dynamic Timeout**: Adjusts processing time based on number of files
- **Model Warm-up**: Pre-warms AI models for faster processing

### 📊 Interactive Dashboard
- **Patient Dashboard**: Visual overview of all processed patients
- **Severity Analysis**: Leaderboard showing patient severity based on test results
- **Processing Statistics**: Real-time processing metrics and performance analysis

### ⚡ Smart Processing
- **Batch Processing**: Handle multiple files simultaneously
- **Progress Tracking**: Real-time progress bars and status updates
- **Error Handling**: Robust error handling with detailed logging

## 🚀 Quick Start

### Prerequisites
1. **Python 3.8+**
2. **Ollama** installed and running locally
3. **Tesseract OCR** installed on your system

### Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install Tesseract OCR:**
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

3. **Install and start Ollama:**
```bash
# Download Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull llama3.2:3b

# Start Ollama service
ollama serve
```

### Running the Application

```bash
streamlit run main3.py
```

## 📋 Usage Guide

### 1. Upload Files
- Drag and drop medical reports (images or PDFs)
- Support for multiple file formats: JPG, PNG, BMP, TIFF, PDF
- Batch processing for multiple files

### 2. Processing
- Click "🚀 Start Processing" to begin extraction
- Monitor real-time progress with progress bars
- Dynamic timeout adjusts based on file count

### 3. Results
- **Tab 1**: Upload & Process - File management and processing
- **Tab 2**: Dashboard - Patient overview and statistics
- **Tab 3**: Severity Analysis - Patient severity leaderboard
- **Tab 4**: Processing Stats - Performance metrics and logs

## 🔧 Configuration

### Model Settings
- **Ollama Model**: Choose from available models (llama3.2:3b, llama3.2:1b, etc.)
- **Base Timeout**: Adjust processing timeout (default: 7 minutes)
- **OCR Engines**: Automatic detection and configuration

### Processing Options
- **Dynamic Timeout**: Automatically calculated based on file count
- **Multi-OCR**: Combines Tesseract and EasyOCR results
- **AI Enhancement**: Uses Ollama for structured JSON generation

## 📊 Output Structure

The application generates structured JSON with the following format:

```json
{
  "patient_info": {
    "name": "Patient Name",
    "age": "Age",
    "gender": "Gender",
    "patient_id": "ID",
    "contact": "Phone/Address",
    "date_of_birth": "DOB"
  },
  "hospital_info": {
    "hospital_name": "Hospital Name",
    "address": "Hospital Address",
    "phone": "Hospital Phone",
    "website": "Website",
    "lab_name": "Laboratory Name"
  },
  "report_info": {
    "report_type": "Report Type",
    "collection_date": "Sample Collection Date",
    "report_date": "Report Generation Date",
    "sample_type": "Sample Type",
    "referred_by": "Referring Doctor"
  },
  "test_results": [
    {
      "test_name": "Test Name",
      "result_value": "Test Result",
      "reference_range": "Normal Range",
      "unit": "Unit",
      "status": "Normal/Abnormal/High/Low",
      "category": "Test Category"
    }
  ],
  "doctor_info": {
    "pathologist": "Pathologist Name",
    "consultant": "Consultant Name",
    "technician": "Lab Technician"
  },
  "additional_info": {
    "notes": "Additional Notes",
    "interpretation": "Medical Interpretation",
    "recommendations": "Recommendations"
  }
}
```

## 🎯 Key Features for Hackathon

### ✅ Completed Features
- [x] Multi-OCR integration (Tesseract + EasyOCR)
- [x] Dynamic timeout based on file count
- [x] Interactive Streamlit interface
- [x] Patient dashboard with visualizations
- [x] Severity analysis leaderboard
- [x] Processing statistics and performance metrics
- [x] Batch file processing
- [x] Real-time progress tracking
- [x] Error handling and logging
- [x] Export functionality for results

### 🚀 Performance Optimizations
- **Model Warm-up**: Pre-warms AI models for faster processing
- **Intelligent OCR**: Combines multiple OCR engines for better accuracy
- **Dynamic Timeout**: Prevents timeouts for large batches
- **Progress Tracking**: Real-time feedback during processing

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check if model is available: `ollama list`

2. **Tesseract Not Found**
   - Install Tesseract OCR
   - Add to system PATH

3. **EasyOCR Installation Issues**
   - Install with: `pip install easyocr`
   - May require additional system dependencies

4. **Processing Timeouts**
   - Increase base timeout in settings
   - Use smaller batch sizes
   - Ensure stable internet connection

## 📈 Performance Metrics

- **Average Processing Time**: ~2-5 minutes per file
- **Success Rate**: >90% with multi-OCR
- **Supported Formats**: Images (JPG, PNG, BMP, TIFF) and PDFs
- **Batch Processing**: Up to 10+ files simultaneously

## 🤝 Contributing

This is a hackathon project. Feel free to enhance and improve the code!

## 📄 License

This project is created for educational and hackathon purposes.

## Retrieval-Augmented Generation (RAG) Pipeline
This app uses ChromaDB and sentence-transformers to enable scalable, enterprise-grade AI search and analysis over all extracted medical report JSONs. All records are embedded and indexed for fast, relevant retrieval at query time.

- **ChromaDB** stores embeddings in the `./chroma_db` folder.
- **sentence-transformers** is used for local, CPU-friendly embeddings.

### Render Deployment Notes
- The app is compatible with Render.com.
- On the free tier, storage is ephemeral (data is lost on redeploy/restart). For persistent storage, use a paid Render plan with a persistent disk and ensure `./chroma_db` is mapped to persistent storage.
- No GPU is required; all models run on CPU. 