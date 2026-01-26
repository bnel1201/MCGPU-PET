
import pytest
import json
import sys
import os

# Add web_demo to path to import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'web_demo')))

from app import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'MC-GPU-PET' in response.data

def test_simulate_api(client):
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
            response = client.post('/api/simulate', 
                                    data=json.dumps(payload),
                                    content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'success'
            assert 'sinogram' in data['results']
            assert 'image' in data['results']
    else:
        print("Executable NOT found. Skipping integration test.")
        pytest.skip("Executable not found")
