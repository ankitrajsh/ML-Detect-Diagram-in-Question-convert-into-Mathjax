import pandas as pd
import numpy as np
import cv2
import os
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
import json
from PIL import Image
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing utilities for diagram detection"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
    def preprocess_text_data(self, input_file: str, output_file: str, 
                           text_column: str = 'question', 
                           label_column: str = 'has_diagram') -> bool:
        """Preprocess text data for training with improved error handling"""
        try:
            logger.info(f"Starting text preprocessing: {input_file} -> {output_file}")
            
            # Validate input file
            if not os.path.exists(input_file):
                logger.error(f"Input file not found: {input_file}")
                return False
            
            # Load data
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} rows from {input_file}")
            
            # Validate required columns
            if text_column not in df.columns:
                logger.error(f"Required column '{text_column}' not found in dataset")
                return False
            
            if label_column not in df.columns:
                logger.error(f"Required column '{label_column}' not found in dataset")
                return False
            
            # Clean text data
            original_count = len(df)
            df = df.dropna(subset=[text_column, label_column])
            logger.info(f"Removed {original_count - len(df)} rows with missing values")
            
            # Clean and normalize text
            df[text_column] = df[text_column].astype(str).str.lower().str.strip()
            
            # Remove empty questions
            df = df[df[text_column].str.len() > 0]
            logger.info(f"Removed empty questions, {len(df)} rows remaining")
            
            # Remove duplicates
            original_count = len(df)
            df = df.drop_duplicates(subset=[text_column])
            logger.info(f"Removed {original_count - len(df)} duplicate questions")
            
            # Encode labels
            le = LabelEncoder()
            df['label'] = le.fit_transform(df[label_column])
            
            # Save label encoder mapping
            label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            logger.info(f"Label mapping: {label_mapping}")
            
            # Create output directory if needed
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed data
            df[[text_column, 'label']].to_csv(output_file, index=False)
            
            # Save metadata
            metadata = {
                'original_rows': original_count,
                'processed_rows': len(df),
                'label_mapping': label_mapping,
                'text_column': text_column,
                'label_column': label_column
            }
            
            metadata_file = output_path.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Processed text data saved to {output_file}")
            logger.info(f"Metadata saved to {metadata_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing text data: {str(e)}")
            return False
    
    def preprocess_images(self, input_dir: str, output_dir: str) -> bool:
        """Preprocess images for training with improved error handling"""
        try:
            logger.info(f"Starting image preprocessing: {input_dir} -> {output_dir}")
            
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            
            if not input_path.exists():
                logger.error(f"Input directory not found: {input_dir}")
                return False
            
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            processed_count = 0
            error_count = 0
            
            # Process each category
            for category in ['diagram', 'no_diagram']:
                category_input = input_path / category
                category_output = output_path / category
                
                if not category_input.exists():
                    logger.warning(f"Category directory not found: {category_input}")
                    continue
                
                category_output.mkdir(parents=True, exist_ok=True)
                
                # Process images in category
                image_files = [f for f in category_input.iterdir() 
                             if f.suffix.lower() in self.supported_image_formats]
                
                logger.info(f"Processing {len(image_files)} images in {category}")
                
                for img_file in image_files:
                    try:
                        # Load image using PIL for better format support
                        with Image.open(img_file) as img:
                            # Convert to RGB if necessary
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Resize image
                            img_resized = img.resize(self.target_size, Image.Resampling.LANCZOS)
                            
                            # Save processed image
                            output_file = category_output / img_file.name
                            img_resized.save(output_file, 'JPEG', quality=95)
                            
                            processed_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing {img_file}: {str(e)}")
                        error_count += 1
            
            # Save processing metadata
            metadata = {
                'processed_images': processed_count,
                'error_count': error_count,
                'target_size': self.target_size,
                'supported_formats': list(self.supported_image_formats)
            }
            
            metadata_file = output_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Processed {processed_count} images with {error_count} errors")
            logger.info(f"Processed images saved to {output_dir}")
            logger.info(f"Metadata saved to {metadata_file}")
            
            return error_count == 0
            
        except Exception as e:
            logger.error(f"Error preprocessing images: {str(e)}")
            return False
    
    def create_sample_dataset(self, output_dir: str) -> bool:
        """Create sample dataset for testing purposes"""
        try:
            logger.info(f"Creating sample dataset in {output_dir}")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create sample text data
            sample_text_data = {
                'question': [
                    'Draw the structure of benzene',
                    'What is the capital of France?',
                    'Sketch the water cycle diagram',
                    'Explain photosynthesis',
                    'Draw a flowchart for the algorithm',
                    'What is machine learning?',
                    'Show the circuit diagram',
                    'Define artificial intelligence'
                ],
                'has_diagram': [True, False, True, False, True, False, True, False]
            }
            
            df = pd.DataFrame(sample_text_data)
            text_file = output_path / 'sample_text_dataset.csv'
            df.to_csv(text_file, index=False)
            
            # Create sample images (simple colored squares)
            for category, color in [('diagram', (255, 100, 100)), ('no_diagram', (100, 255, 100))]:
                category_dir = output_path / 'images' / category
                category_dir.mkdir(parents=True, exist_ok=True)
                
                for i in range(3):
                    img = Image.new('RGB', self.target_size, color)
                    img.save(category_dir / f'sample_{i}.jpg', 'JPEG')
            
            logger.info(f"Sample dataset created in {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating sample dataset: {str(e)}")
            return False

def main():
    """Main preprocessing function with command line interface"""
    parser = argparse.ArgumentParser(description='Preprocess data for diagram detection')
    parser.add_argument('--text-input', help='Input text CSV file')
    parser.add_argument('--text-output', help='Output processed text CSV file')
    parser.add_argument('--image-input', help='Input image directory')
    parser.add_argument('--image-output', help='Output processed image directory')
    parser.add_argument('--create-sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--sample-dir', default='data/sample', help='Sample dataset directory')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor()
    
    success = True
    
    # Create sample dataset if requested
    if args.create_sample:
        success &= preprocessor.create_sample_dataset(args.sample_dir)
    
    # Process text data
    if args.text_input and args.text_output:
        success &= preprocessor.preprocess_text_data(args.text_input, args.text_output)
    
    # Process images
    if args.image_input and args.image_output:
        success &= preprocessor.preprocess_images(args.image_input, args.image_output)
    
    if not any([args.text_input, args.image_input, args.create_sample]):
        # Default behavior - try to process standard paths
        logger.info("No specific arguments provided, trying default paths...")
        
        # Try to process text data
        text_input = "data/raw/text_dataset.csv"
        text_output = "data/processed/processed_text_dataset.csv"
        if os.path.exists(text_input):
            success &= preprocessor.preprocess_text_data(text_input, text_output)
        else:
            logger.warning(f"Text input file not found: {text_input}")
        
        # Try to process images
        image_input = "data/raw/images"
        image_output = "data/processed/processed_images"
        if os.path.exists(image_input):
            success &= preprocessor.preprocess_images(image_input, image_output)
        else:
            logger.warning(f"Image input directory not found: {image_input}")
        
        # Create sample dataset
        success &= preprocessor.create_sample_dataset("data/sample")
    
    if success:
        logger.info("Preprocessing completed successfully")
    else:
        logger.error("Preprocessing completed with errors")
        exit(1)

if __name__ == "__main__":
    main()