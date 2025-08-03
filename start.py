#!/usr/bin/env python3
"""
Startup script for the Diagram Detection API
This script handles model loading, environment setup, and API startup
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'torch', 'transformers', 
        'opencv-python', 'numpy', 'pandas', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install them using: pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are installed")
    return True

def check_models():
    """Check if required models exist"""
    text_model_exists = os.path.exists(Config.TEXT_MODEL_PATH)
    image_model_exists = os.path.exists(Config.IMAGE_MODEL_PATH)
    
    if not text_model_exists:
        logger.warning(f"Text model not found at {Config.TEXT_MODEL_PATH}")
        logger.warning("Text detection will not be available")
    
    if not image_model_exists:
        logger.warning(f"Image model not found at {Config.IMAGE_MODEL_PATH}")
        logger.warning("Image detection will not be available")
    
    if not text_model_exists and not image_model_exists:
        logger.error("No models found! Please train models first or provide pre-trained models")
        return False
    
    return True

def create_sample_data():
    """Create sample data for testing if no data exists"""
    try:
        from src.preprocess import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        sample_dir = "data/sample"
        
        if not os.path.exists(sample_dir):
            logger.info("Creating sample dataset for testing...")
            success = preprocessor.create_sample_dataset(sample_dir)
            if success:
                logger.info("Sample dataset created successfully")
            else:
                logger.warning("Failed to create sample dataset")
        
    except Exception as e:
        logger.warning(f"Could not create sample data: {str(e)}")

def start_api(host=None, port=None, reload=False):
    """Start the FastAPI application"""
    try:
        import uvicorn
        from api.main import app
        
        # Use config values if not provided
        host = host or Config.API_HOST
        port = port or Config.API_PORT
        
        logger.info(f"Starting Diagram Detection API on {host}:{port}")
        logger.info(f"API documentation will be available at http://{host}:{port}/docs")
        
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            log_level=Config.LOG_LEVEL.lower(),
            reload=reload
        )
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        sys.exit(1)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Start the Diagram Detection API')
    parser.add_argument('--host', default=None, help='Host to bind to')
    parser.add_argument('--port', type=int, default=None, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and model checks')
    parser.add_argument('--create-sample', action='store_true', help='Create sample data and exit')
    
    args = parser.parse_args()
    
    logger.info("Starting Diagram Detection API...")
    
    # Create sample data if requested
    if args.create_sample:
        create_sample_data()
        return
    
    # Perform checks unless skipped
    if not args.skip_checks:
        logger.info("Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        logger.info("Checking models...")
        if not check_models():
            logger.error("Model check failed. Use --skip-checks to start anyway.")
            sys.exit(1)
    
    # Create sample data if needed
    create_sample_data()
    
    # Start the API
    start_api(args.host, args.port, args.reload)

if __name__ == "__main__":
    main()
