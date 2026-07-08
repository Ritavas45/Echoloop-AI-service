import os
import glob
import requests
import json

def find_mock_images(num_needed=5):
    """
    Dynamically scans `./mock_data` to locate actual dummy images to use for testing.
    """
    image_paths = []
    # Search in mock data directories
    for root, dirs, files in os.walk("./mock_data"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= num_needed:
                    return image_paths
    return image_paths

def print_test_case_header(title):
    border = "=" * 60
    print("\n" + border)
    print(f" TEST CASE: {title}")
    print(border)

def run_tests():
    url = "http://127.0.0.1:8000/predict"
    
    # Locate mock images
    image_files_pool = find_mock_images(5)
    if len(image_files_pool) < 5:
        print("Error: Could not locate sufficient mock images. Make sure generate_mock_data.py has run.")
        return
        
    print(f"Located {len(image_files_pool)} images for test payloads:")
    for path in image_files_pool:
        print(f"  - {path}")
        
    # --- TEST CASE 1: Standard Multi-Modal Weighted Voting ---
    print_test_case_header("Standard Weighted Soft Voting")
    
    # Construct multipart request payload
    # 5 images
    files = [('images', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in image_files_pool]
    
    data = {
        'model_age_months': 14,
        'battery_health_pct': 84.0,
        'screen_cracked': 'false',
        'functional_issues': 'false',
        'cosmetic_scratches': 1
    }
    
    try:
        response = requests.post(url, files=files, data=data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Connection Error: Is the FastAPI server running at http://127.0.0.1:8000?")
        return
    finally:
        # Close open files
        for f in files:
            f[1][1].close()

    # --- TEST CASE 2: Tabular Model Override (Confidence >= 90%) ---
    # We pass perfect Reuse parameters to trigger tabular confidence >= 90%
    print_test_case_header("Tabular Model Override (90%+ Confidence)")
    
    files = [('images', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in image_files_pool[:4]] # 4 images
    data = {
        'model_age_months': 2,
        'battery_health_pct': 98.5,
        'screen_cracked': 'false',
        'functional_issues': 'false',
        'cosmetic_scratches': 0
    }
    
    try:
        response = requests.post(url, files=files, data=data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Error executing test case 2: {e}")
    finally:
        for f in files:
            f[1][1].close()

    # --- TEST CASE 3: Validation Rejection (<4 images) ---
    print_test_case_header("Input Rejection (< 4 Images)")
    
    files = [('images', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in image_files_pool[:2]] # Only 2 images
    data = {
        'model_age_months': 12,
        'battery_health_pct': 88.0,
        'screen_cracked': 'false',
        'functional_issues': 'false',
        'cosmetic_scratches': 1
    }
    
    try:
        response = requests.post(url, files=files, data=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response Payload:\n{json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error executing test case 3: {e}")
    finally:
        for f in files:
            f[1][1].close()

if __name__ == "__main__":
    run_tests()
