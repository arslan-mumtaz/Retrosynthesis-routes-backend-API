#!/usr/bin/env python3
"""
Download model files for AiZynthFinder
"""
import os
import requests
import yaml
from pathlib import Path

def download_file(url, local_path):
    """Download a file from URL to local path"""
    print(f"Downloading {url} to {local_path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Ensure directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✅ Downloaded {local_path}")

def main():
    """Download all required model files"""
    
    # Configuration with download URLs
    downloads = {
        # Stock database
        "zinc_stock.hdf5": "https://github.com/arslan-mumtaz/Retrosynthesis-routes-backend-API/releases/download/zinc/zinc_stock.hdf5",
        
        # ONNX models
        "uspto_model.onnx": "https://github.com/arslan-mumtaz/Retrosynthesis-routes-backend-API/releases/download/zinc/uspto_model.onnx",
        "uspto_ringbreaker_model.onnx": "https://github.com/arslan-mumtaz/Retrosynthesis-routes-backend-API/releases/download/zinc/uspto_ringbreaker_model.onnx",
        "uspto_filter_model.onnx": "https://github.com/arslan-mumtaz/Retrosynthesis-routes-backend-API/releases/download/zinc/uspto_filter_model.onnx",
        
        # Template files
        "uspto_templates.csv.gz": "https://github.com/arslan-mumtaz/Retrosynthesis-routes-backend-API/releases/download/zinc/uspto_templates.csv.gz",
        "uspto_ringbreaker_templates.csv.gz": "https://github.com/arslan-mumtaz/Retrosynthesis-routes-backend-API/releases/download/zinc/uspto_ringbreaker_templates.csv.gz",
    }
    
    # Only download if URLs are provided
    for filename, url in downloads.items():
        if url:
            try:
                local_path = Path(filename)
                if not local_path.exists():
                    download_file(url, local_path)
                else:
                    print(f"⏭️  {filename} already exists, skipping")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
        else:
            print(f"⚠️  No URL provided for {filename}")

if __name__ == "__main__":
    main()
