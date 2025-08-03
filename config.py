"""
Configuration module for the Diagram Detection API
"""
import os
from pathlib import Path
from typing import Set

class Config:
    """Application configuration"""
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    # Model Paths
    TEXT_MODEL_PATH: str = os.getenv("TEXT_MODEL_PATH", "models/text_detector")
    IMAGE_MODEL_PATH: str = os.getenv("IMAGE_MODEL_PATH", "models/image_detector.pt")
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024  # Convert MB to bytes
    ALLOWED_EXTENSIONS: Set[str] = set(os.getenv("ALLOWED_EXTENSIONS", ".jpg,.jpeg,.png,.bmp,.tiff").split(","))
    
    # Model Configuration
    MAX_TEXT_LENGTH: int = 1000
    BERT_MAX_LENGTH: int = 128
    IMAGE_SIZE: tuple = (224, 224)
    
    # Device Configuration
    DEVICE: str = os.getenv("DEVICE", "auto")  # auto, cuda, or cpu
    
    # Validation
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        errors = []
        
        if cls.API_PORT < 1 or cls.API_PORT > 65535:
            errors.append(f"Invalid API_PORT: {cls.API_PORT}")
        
        if cls.MAX_FILE_SIZE <= 0:
            errors.append(f"Invalid MAX_FILE_SIZE: {cls.MAX_FILE_SIZE}")
        
        if not cls.ALLOWED_EXTENSIONS:
            errors.append("ALLOWED_EXTENSIONS cannot be empty")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True

# Load and validate configuration
try:
    Config.validate()
except ValueError as e:
    print(f"Configuration error: {e}")
    raise
