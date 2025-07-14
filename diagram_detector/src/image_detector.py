import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

class ImageDetector:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_path):
        model = models.resnet50(pretrained=False)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def predict(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
        return output

    def detect_diagrams(self, images_folder):
        results = {}
        for filename in os.listdir(images_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_folder, filename)
                output = self.predict(image_path)
                results[filename] = output
        return results

if __name__ == "__main__":
    model_path = '../models/image_detector.pt'
    images_folder = '../data/raw/images/diagram/'
    detector = ImageDetector(model_path)
    results = detector.detect_diagrams(images_folder)
    print(results)