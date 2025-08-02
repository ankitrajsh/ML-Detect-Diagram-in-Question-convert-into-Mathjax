#!/bin/bash
# Create project directory
mkdir ML-Detect-Diagram-in-Question-convert-into-Mathjax
cd ML-Detect-Diagram-in-Question-convert-into-Mathjax

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install transformers torch torchvision fastapi uvicorn opencv-python numpy pandas scikit-learn

# Create directory structure
mkdir -p data models src api
touch src/text_detector.py src/image_detector.py api/main.py
