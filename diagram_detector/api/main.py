from flask import Flask, request, jsonify
from src.text_detector import TextDetector
from src.image_detector import ImageDetector

app = Flask(__name__)

text_detector = TextDetector()
image_detector = ImageDetector()

@app.route('/detect_text', methods=['POST'])
def detect_text():
    data = request.json
    text = data.get('text', '')
    result = text_detector.detect(text)
    return jsonify(result)

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    result = image_detector.detect(image)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)