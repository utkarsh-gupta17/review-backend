import requests

# Base URL of your deployed API
BASE_URL = "https://review-classifier-api-production.up.railway.app"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ /health endpoint working")
            print(f"Response: {response.json()}")
        else:
            print(f"✗ /health failed with status {response.status_code}")
    except Exception as e:
        print(f"✗ Error testing /health: {str(e)}")

def test_predict_endpoint():
    """Test the prediction endpoint with sample data"""
    print("\nTesting /predict endpoint...")
    test_data = {
        "review_text": "This product is absolutely amazing! Worth every penny.",
        "overall": 5,
        "helpful_ratio": 0.95
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✓ /predict endpoint working")
            print(f"Prediction: {response.json()}")
        else:
            print(f"✗ /predict failed with status {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"✗ Error testing /predict: {str(e)}")

if __name__ == "__main__":
    test_health_endpoint()
    test_predict_endpoint()