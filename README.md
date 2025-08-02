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

## Setup Instructions
1. Clone the repository:
   ```
   git clone https://github.com/ankitrajsh/ML-Detect-Diagram-in-Question-convert-into-Mathjax.git
   cd ML-Detect-Diagram-in-Question-convert-into-Mathjax
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the setup script:
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
