from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, validator
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from fastapi.responses import JSONResponse
import logging
import os
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diagram Detection API",
    description="API for detecting diagrams in text and images",
    version="1.0.0"
)

# Configuration
TEXT_MODEL_PATH = "models/text_detector"
IMAGE_MODEL_PATH = "models/image_detector.pt"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Global variables for models
text_tokenizer = None
text_model = None
image_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    """Load models with proper error handling"""
    global text_tokenizer, text_model, image_model
    
    try:
        # Load text model
        if os.path.exists(TEXT_MODEL_PATH):
            logger.info("Loading text detection model...")
            text_tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_PATH)
            text_model = BertForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
            text_model.eval()
            text_model = text_model.to(device)
            logger.info("Text model loaded successfully")
        else:
            logger.warning(f"Text model not found at {TEXT_MODEL_PATH}")
            
        # Load image model
        if os.path.exists(IMAGE_MODEL_PATH):
            logger.info("Loading image detection model...")
            image_model = models.resnet18()
            image_model.fc = torch.nn.Linear(image_model.fc.in_features, 2)
            image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
            image_model.eval()
            image_model = image_model.to(device)
            logger.info("Image model loaded successfully")
        else:
            logger.warning(f"Image model not found at {IMAGE_MODEL_PATH}")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Load models on startup
try:
    load_models()
except Exception as e:
    logger.error(f"Failed to load models on startup: {str(e)}")

# Image transform
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Pydantic models
class TextInput(BaseModel):
    question: str
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        if len(v) > 1000:
            raise ValueError('Question too long (max 1000 characters)')
        return v.strip()

class DetectionResponse(BaseModel):
    diagram_detected: bool
    confidence: float
    message: Optional[str] = None

# Text detection endpoint
@app.post("/detect_text", response_model=DetectionResponse)
async def detect_text(input: TextInput):
    """Detect if a text question contains diagram-related content"""
    if text_model is None or text_tokenizer is None:
        raise HTTPException(status_code=503, detail="Text detection model not available")
    
    try:
        logger.info(f"Processing text detection request for question: {input.question[:50]}...")
        
        question = input.question.lower()
        encoding = text_tokenizer(
            question, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = text_model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = probs.argmax().item()
            confidence = float(probs[0][prediction])
        
        result = {
            "diagram_detected": bool(prediction),
            "confidence": confidence,
            "message": "Text analysis completed successfully"
        }
        
        logger.info(f"Text detection result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in text detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text detection failed: {str(e)}")

# Image detection endpoint
@app.post("/detect_image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    """Detect if an uploaded image contains diagrams"""
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image detection model not available")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    try:
        logger.info(f"Processing image detection request for file: {file.filename}")
        
        # Read and validate file size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Decode image
        image_array = np.frombuffer(file_content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert and preprocess
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = image_model(image)
            probs = torch.softmax(outputs, dim=1)
            prediction = probs.argmax().item()
            confidence = float(probs[0][prediction])
        
        result = {
            "diagram_detected": bool(prediction),
            "confidence": confidence,
            "message": "Image analysis completed successfully"
        }
        
        logger.info(f"Image detection result: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image detection failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "text_model_loaded": text_model is not None,
        "image_model_loaded": image_model is not None,
        "device": str(device)
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Diagram Detection API",
        "version": "1.0.0",
        "endpoints": {
            "text_detection": "/detect_text",
            "image_detection": "/detect_image",
            "health": "/health"
        }
    }

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )