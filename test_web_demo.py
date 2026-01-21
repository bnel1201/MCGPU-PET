
import unittest
import json
import sys
import os

# Add web_demo to path to import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'web_demo')))

from app import app

class TestPETApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'MC-GPU-PET', response.data)

    def test_simulate_api(self):
        # Light payload for testing to avoid long wait, though wrapper params are respected
        # We'll use a very short time and small resolution to make it fast if possible
        # note: mcgpu execution might take a few seconds regardless.
        payload = {
            'time_sec': 1.0,
            'image_res': 64, # faster
            'detector_height': 25.0
        }
        
        # MOCKING: To avoid actually running the heavy GPU executable during this quick test,
        # we might want to mock the wrapper.run method. 
        # But for an integration test, let's try to run it IF the executable exists.
        
        if os.path.exists('./sample_simulation/MCGPU-PET.x'):
             print("Executable found, running full integration test (might take 5-10s)...")
             response = self.app.post('/api/simulate', 
                                      data=json.dumps(payload),
                                      content_type='application/json')
             
             self.assertEqual(response.status_code, 200)
             data = json.loads(response.data)
             self.assertEqual(data['status'], 'success')
             self.assertIn('sinogram', data['results'])
             self.assertIn('image', data['results'])
        else:
            print("Executable NOT found. Skipping integration test.")

if __name__ == '__main__':
    unittest.main()
