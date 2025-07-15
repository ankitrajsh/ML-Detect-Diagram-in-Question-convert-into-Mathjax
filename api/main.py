from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from fastapi.responses import JSONResponse

app = FastAPI()

# Load text model
text_tokenizer = BertTokenizer.from_pretrained("models/text_detector")
text_model = BertForSequenceClassification.from_pretrained("models/text_detector")
text_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_model = text_model.to(device)

# Load image model
image_model = models.resnet18()
image_model.fc = torch.nn.Linear(image_model.fc.in_features, 2)
image_model.load_state_dict(torch.load("models/image_detector.pt"))
image_model.eval()
image_model = image_model.to(device)

# Image transform
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Pydantic model for text input
class TextInput(BaseModel):
    question: str

# Text detection endpoint
@app.post("/detect_text")
async def detect_text(input: TextInput):
    question = input.question.lower()
    encoding = text_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=128)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = text_model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = probs.argmax().item()
        confidence = float(probs[0][prediction])
    
    return {
        "diagram_detected": bool(prediction),
        "confidence": confidence
    }

# Image detection endpoint
@app.post("/detect_image")
async def detect_image(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = image_model(image)
        probs = torch.softmax(outputs, dim=1)
        prediction = probs.argmax().item()
        confidence = float(probs[0][prediction])
    
    return {
        "diagram_detected": bool(prediction),
        "confidence": confidence
    }

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)