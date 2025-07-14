#!/bin/bash

# Create necessary directories
mkdir -p data/raw/images/diagram/physics
mkdir -p data/raw/images/diagram/mathematics
mkdir -p data/raw/images/diagram/chemistry
mkdir -p data/raw/images/diagram/biology
mkdir -p data/raw/images/diagram/zoology
mkdir -p data/raw/images/diagram/botany
mkdir -p data/raw/images/no_diagram
mkdir -p data/processed/processed_images
mkdir -p models/text_detector

# Install dependencies
pip install -r requirements.txt

echo "Setup complete. Project environment is ready."