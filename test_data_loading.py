#!/usr/bin/env python3
"""
Test script to verify JSON data loading and processing
"""

import json
import pandas as pd
from pathlib import Path
import re

def load_all_jsons(json_dir="output/json"):
    """Load all JSON files with better error handling"""
    all_data = []
    json_path = Path(json_dir)
    
    if not json_path.exists():
        print(f"JSON directory {json_dir} does not exist")
        return all_data
    
    json_files = list(json_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return all_data
    
    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Add filename to data for reference
                if isinstance(data, dict):
                    data['_filename'] = file.name
                all_data.append(data)
        except Exception as e:
            print(f"Error loading {file.name}: {str(e)}")
            # Add error record
            all_data.append({
                'error': f'Failed to load {file.name}: {str(e)}',
                '_filename': file.name
            })
    
    print(f"Loaded {len(all_data)} JSON files from {json_dir}")
    return all_data

def flatten_report_data(reports):
    """Convert nested report data into a flat structure for analysis"""
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

def test_data_processing():
    """Test the data processing functions"""
    print("=== Testing JSON Data Loading ===")
    
    # Load all JSONs
    all_jsons = load_all_jsons("output/json")
    print(f"Total JSON files loaded: {len(all_jsons)}")
    
    # Count valid vs error reports
    valid_reports = [j for j in all_jsons if 'error' not in j and 'patient_info' in j]
    error_reports = [j for j in all_jsons if 'error' in j]
    
    print(f"Valid reports: {len(valid_reports)}")
    print(f"Error reports: {len(error_reports)}")
    
    if valid_reports:
        print("\n=== Sample Valid Report ===")
        sample_report = valid_reports[0]
        print(f"Patient: {sample_report.get('patient_info', {}).get('name', 'Unknown')}")
        print(f"Age: {sample_report.get('patient_info', {}).get('age', 'N/A')}")
        print(f"Tests: {len(sample_report.get('test_results', []))}")
    
    # Test flattening
    print("\n=== Testing Data Flattening ===")
    flat_data = flatten_report_data(all_jsons)
    print(f"Flat data records: {len(flat_data)}")
    
    if flat_data:
        print("\n=== Sample Flat Data ===")
        df = pd.DataFrame(flat_data)
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few records:")
        print(df.head())
        
        # Test age extraction
        print("\n=== Testing Age Extraction ===")
        def extract_age(age_str):
            if pd.isna(age_str) or age_str == 'N/A':
                return None
            match = re.search(r'(\d+)', str(age_str))
            return int(match.group(1)) if match else None
        
        df['age_numeric'] = df['patient_age'].apply(extract_age)
        valid_ages = df.dropna(subset=['age_numeric'])
        print(f"Valid ages found: {len(valid_ages)}")
        if not valid_ages.empty:
            print(f"Age range: {valid_ages['age_numeric'].min()} - {valid_ages['age_numeric'].max()}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_data_processing() 