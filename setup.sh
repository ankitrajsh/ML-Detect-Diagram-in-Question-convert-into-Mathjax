#!/bin/bash
# Create project directory
mkdir diagram_detector
cd diagram_detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install transformers torch torchvision fastapi uvicorn opencv-python numpy pandas scikit-learn

# Create directory structure
mkdir -p data models src api
touch src/text_detector.py src/image_detector.py api/main.py