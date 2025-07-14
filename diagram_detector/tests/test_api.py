import unittest
from api.main import app

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_health_check(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'status': 'healthy'})

    def test_diagram_detection(self):
        response = self.client.post('/detect', json={'image': 'base64_encoded_image_string'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('detected_diagram', response.json)

    def test_no_diagram_detection(self):
        response = self.client.post('/detect', json={'image': 'base64_encoded_no_diagram_image_string'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('no_diagram', response.json)

if __name__ == '__main__':
    unittest.main()