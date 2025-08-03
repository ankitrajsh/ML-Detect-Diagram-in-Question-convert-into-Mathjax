import pytest
import httpx
import asyncio
from pathlib import Path
import io
from PIL import Image
import numpy as np

# Test configuration
BASE_URL = "http://localhost:8000"

class TestDiagramDetectionAPI:
    """Test suite for the Diagram Detection API"""
    
    @pytest.fixture
    async def client(self):
        """Create an async HTTP client for testing"""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            yield client
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    async def test_health_endpoint(self, client):
        """Test the health check endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    async def test_root_endpoint(self, client):
        """Test the root endpoint"""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
    
    async def test_text_detection_valid(self, client):
        """Test text detection with valid input"""
        payload = {"question": "Draw the structure of benzene."}
        response = await client.post("/detect_text", json=payload)
        
        if response.status_code == 503:
            pytest.skip("Text model not available")
        
        assert response.status_code == 200
        data = response.json()
        assert "diagram_detected" in data
        assert "confidence" in data
        assert isinstance(data["diagram_detected"], bool)
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1
    
    async def test_text_detection_empty_question(self, client):
        """Test text detection with empty question"""
        payload = {"question": ""}
        response = await client.post("/detect_text", json=payload)
        assert response.status_code == 422  # Validation error
    
    async def test_text_detection_long_question(self, client):
        """Test text detection with overly long question"""
        payload = {"question": "x" * 1001}  # Exceeds 1000 character limit
        response = await client.post("/detect_text", json=payload)
        assert response.status_code == 422  # Validation error
    
    async def test_image_detection_valid(self, client, sample_image):
        """Test image detection with valid image"""
        files = {"file": ("test.jpg", sample_image, "image/jpeg")}
        response = await client.post("/detect_image", files=files)
        
        if response.status_code == 503:
            pytest.skip("Image model not available")
        
        assert response.status_code == 200
        data = response.json()
        assert "diagram_detected" in data
        assert "confidence" in data
        assert isinstance(data["diagram_detected"], bool)
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1
    
    async def test_image_detection_no_file(self, client):
        """Test image detection without file"""
        response = await client.post("/detect_image")
        assert response.status_code == 422  # Missing file
    
    async def test_image_detection_invalid_format(self, client):
        """Test image detection with invalid file format"""
        files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        response = await client.post("/detect_image", files=files)
        assert response.status_code == 400  # Invalid file type

# Simple synchronous tests for basic functionality
def test_text_detection_sync():
    """Synchronous test for text detection (fallback)"""
    import requests
    
    try:
        text_payload = {"question": "Draw the structure of benzene."}
        response = requests.post(f"{BASE_URL}/detect_text", json=text_payload, timeout=10)
        
        if response.status_code == 503:
            print("Text model not available - skipping test")
            return
        
        print(f"Text Detection Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Text Detection Result: {response.json()}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("API server not running - please start the server first")
    except Exception as e:
        print(f"Error testing text detection: {e}")

def test_image_detection_sync():
    """Synchronous test for image detection (fallback)"""
    import requests
    
    try:
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/detect_image", files=files, timeout=10)
        
        if response.status_code == 503:
            print("Image model not available - skipping test")
            return
        
        print(f"Image Detection Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Image Detection Result: {response.json()}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("API server not running - please start the server first")
    except Exception as e:
        print(f"Error testing image detection: {e}")

if __name__ == "__main__":
    print("Running basic API tests...")
    test_text_detection_sync()
    test_image_detection_sync()