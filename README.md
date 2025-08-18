# Diagram Detector

## Overview
The Diagram Detector project identifies and classifies diagrams in questions and images. It uses machine learning models for text and image detection to help detect diagram-related content in educational and scientific material.

## Project structure
- `data/` - datasets
  - `raw/` - original datasets
    - `text_dataset.csv`
    - `images/`
      - `diagram/` (subfolders: biology, botany, chemistry, mathematics, physics, zoology)
      - `no_diagram/`
  - `processed/` - processed datasets
    - `processed_text_dataset.csv`
    - `processed_images/`

- `models/` - trained models
  - `text_detector/`
  - `image_detector.pt`

- `src/` - training / preprocessing / utilities
  - `preprocess.py`
  - `text_detector.py`
  - `image_detector.py`

- `api/` - FastAPI application
  - `main.py`

- `tests/` - unit tests
  - `test_api.py`

Other files: `Dockerfile`, `requirements.txt`, `setup.sh`, `start.py`.

## Features
- Text detection using transformer-based models (BERT or similar)
- Image detection using a ResNet-based classifier
- FastAPI REST endpoints with automatic docs
- Health checks and basic monitoring endpoints
- Configurable via environment variables
- Tests with pytest and included example scripts

## Requirements
- Python 3.8+
- Recommended: create and use a virtual environment (instructions below)

## Virtual environment (recommended)
Use a virtual environment to isolate project dependencies.

1. Verify Python is available (use `python3` on many Linux/macOS systems):

   ```bash
   python3 --version
   ```

2. Create the virtual environment in the project root:

   ```bash
   python3 -m venv .venv
   ```

3. Activate the virtual environment:

   - Linux / macOS (bash/zsh):
     ```bash
     source .venv/bin/activate
     ```

   - Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```

4. Upgrade packaging tools and install requirements:

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

Notes:
- If you need CUDA-enabled PyTorch, use the selector at https://pytorch.org/get-started/locally/ to obtain the correct install command (it may use a custom `--index-url`). Example CPU-only command:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

## Quick start
1. Clone the repository:
   ```bash
   git clone https://github.com/ankitrajsh/ML-Detect-Diagram-in-Question-convert-into-Mathjax.git
   cd ML-Detect-Diagram-in-Question-convert-into-Mathjax
   ```

2. Create and activate a virtual environment (see the "Virtual environment" section above for commands).

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Copy environment example and edit values if present:
   ```bash
   cp .env.example .env  # only if .env.example exists
   ```

5. Start the API (two equivalent options):
   ```bash
   # recommended: helper script
   python start.py

   # or run the FastAPI app directly
   python api/main.py
   ```

6. Open the API docs in your browser:
   http://localhost:8000/docs

### Using Docker
1. Build the image:
   ```bash
   docker build -t diagram-detector .
   ```
2. Run the container (map the port):
   ```bash
   docker run -p 8000:8000 diagram-detector
   ```

## API endpoints
- Documentation: `GET /docs`
- Health: `GET /health`
- Info: `GET /`

Example: Text detection
```bash
curl -X POST "http://localhost:8000/detect_text" \
     -H "Content-Type: application/json" \
     -d '{"question": "Draw the structure of benzene"}'
```

Example: Image detection
```bash
curl -X POST "http://localhost:8000/detect_image" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

## Configuration
Configuration is controlled via environment variables. If a `.env.example` is provided, copy it to `.env` and update values.
Important variables (defaults shown where applicable):
- `API_HOST` (default: `0.0.0.0`)
- `API_PORT` (default: `8000`)
- `LOG_LEVEL` (default: `info`)
- `MAX_FILE_SIZE_MB` (default: `10`)
- `TEXT_MODEL_PATH` - path to a saved text model (if required)
- `IMAGE_MODEL_PATH` - path to the image model (e.g., `models/image_detector.pt`)

## Development
- Run tests:
  ```bash
  pytest
  ```

- Run a single test file:
  ```bash
  pytest tests/test_api.py
  ```

- Run basic tests without pytest (if the file is executable as a script):
  ```bash
  python tests/test_api.py
  ```

- Create sample data (if `preprocess.py` supports it):
  ```bash
  python src/preprocess.py --create-sample
  ```

- Train models (if training scripts are implemented):
  ```bash
  python src/text_detector.py
  python src/image_detector.py
  ```

## Troubleshooting
- If the API fails to start, check that dependencies from `requirements.txt` are installed and that the configured model paths exist.
- Check logs (the application respects `LOG_LEVEL`).

## Contributing
Contributions are welcome. Please open issues or submit pull requests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
