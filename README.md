# Diagram Detector

## Overview
The Diagram Detector project is designed to identify and classify various types of diagrams from images and text data. This project utilizes machine learning models for text and image detection, enabling efficient processing and analysis of educational and scientific diagrams.

## Project Structure
- **data/**: Contains raw and processed datasets.
  - **raw/**: Original datasets including text and images.
    - **text_dataset.csv**: Raw text dataset for training/testing.
    - **images/**: Directory containing images categorized into diagrams and non-diagrams.
      - **diagram/**: Subdirectories for different types of diagrams (physics, mathematics, chemistry, biology, zoology, botany).
      - **no_diagram/**: Images that do not contain any diagrams.
  - **processed/**: Processed datasets ready for model training.
    - **processed_text_dataset.csv**: Processed text dataset.
    - **processed_images/**: Directory for processed images.

- **models/**: Contains trained models and related scripts.
  - **text_detector/**: Directory for text detection models.
  - **image_detector.pt**: PyTorch model file for image detection.

- **src/**: Source code for preprocessing and model implementation.
  - **preprocess.py**: Script for data preprocessing.
  - **text_detector.py**: Implementation of the text detection model.
  - **image_detector.py**: Implementation of the image detection model.

- **api/**: Contains the API for interacting with the model.
  - **main.py**: Entry point for the API.

- **tests/**: Unit tests for the API.
  - **test_api.py**: Tests to ensure API functionality.

- **Dockerfile**: Instructions for building a Docker image for the project.

- **requirements.txt**: Lists the Python dependencies required for the project.

- **setup.sh**: Script to set up the project environment.

## Features
- **Text Detection**: Analyze text questions to detect diagram-related content using BERT
- **Image Detection**: Classify uploaded images as containing diagrams or not using ResNet
- **RESTful API**: FastAPI-based API with automatic documentation
- **Robust Error Handling**: Comprehensive error handling and validation
- **Health Monitoring**: Built-in health check endpoints
- **Configurable**: Environment-based configuration system
- **Testing Suite**: Comprehensive test coverage with pytest
- **Docker Support**: Production-ready Docker container
- **Logging**: Structured logging for debugging and monitoring

## Quick Start

### Option 1: Using the Startup Script (Recommended)
1. Clone the repository:
   ```bash
   git clone https://github.com/ankitrajsh/ML-Detect-Diagram-in-Question-convert-into-Mathjax.git
   cd ML-Detect-Diagram-in-Question-convert-into-Mathjax
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the API:
   ```bash
   python start.py
   ```

### Option 2: Using Docker
1. Build the Docker image:
   ```bash
   docker build -t diagram-detector .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 diagram-detector
   ```

### Option 3: Manual Setup
1. Clone and install dependencies (steps 1-2 from Option 1)

2. Create sample data (optional):
   ```bash
   python src/preprocess.py --create-sample
   ```

3. Start the API directly:
   ```bash
   python api/main.py
   ```

## API Usage

Once the API is running, you can:

- **Access the API documentation**: http://localhost:8000/docs
- **Check API health**: http://localhost:8000/health
- **View API info**: http://localhost:8000/

### Text Detection Endpoint
```bash
curl -X POST "http://localhost:8000/detect_text" \
     -H "Content-Type: application/json" \
     -d '{"question": "Draw the structure of benzene"}'
```

### Image Detection Endpoint
```bash
curl -X POST "http://localhost:8000/detect_image" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

## Configuration

The application can be configured using environment variables. Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

Key configuration options:
- `API_HOST`: Host to bind to (default: 0.0.0.0)
- `API_PORT`: Port to bind to (default: 8000)
- `LOG_LEVEL`: Logging level (default: info)
- `MAX_FILE_SIZE_MB`: Maximum upload file size (default: 10MB)
- `TEXT_MODEL_PATH`: Path to text detection model
- `IMAGE_MODEL_PATH`: Path to image detection model

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=api

# Run specific test file
pytest tests/test_api.py

# Run basic tests without pytest
python tests/test_api.py
```

### Data Preprocessing
```bash
# Create sample dataset
python src/preprocess.py --create-sample

# Process custom text data
python src/preprocess.py --text-input data/raw/text.csv --text-output data/processed/text.csv

# Process custom image data
python src/preprocess.py --image-input data/raw/images --image-output data/processed/images
```

### Model Training
```bash
# Train text detection model
python src/text_detector.py

# Train image detection model
python src/image_detector.py
```
   ```
   bash setup.sh
   ```

## Usage
To start the API, run:
```
python api/main.py
```

You can then send requests to the API to detect and classify diagrams from images and text.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
