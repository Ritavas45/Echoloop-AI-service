import requests

API_URL = "http://localhost:8000"

# Open the cat image 5 times to simulate a 5-image upload
files = [('images', ('cat.jpg', open('scratch/cat.jpg', 'rb'), 'image/jpeg')) for _ in range(5)]
data = {
    'model_age_months': 6,
    'battery_health_pct': 92.0,
    'screen_cracked': 'false',
    'functional_issues': 'false',
    'cosmetic_scratches': 0
}

try:
    res = requests.post(f"{API_URL}/predict", files=files, data=data)
    for f in files:
        f[1][1].close()
        
    print(f"Status Code: {res.status_code}")
    print(f"Response Body: {res.text}")
    if res.status_code == 400 and "do not appear to be a mobile phone" in res.text:
        print("🎉 SUCCESS: Validation successfully blocked the non-phone images!")
    else:
        print("❌ FAIL: Validation did not block the non-phone images.")
except Exception as e:
    print(f"Request failed: {e}")
