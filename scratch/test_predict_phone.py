import requests

API_URL = "http://localhost:8000"

files = [('images', ('phone.jpg', open('scratch/phone.jpg', 'rb'), 'image/jpeg')) for _ in range(5)]
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
    if res.status_code == 200:
        print("🎉 SUCCESS: Phone validation successfully allowed the valid phone images!")
    else:
        print("❌ FAIL: Phone validation blocked the valid phone images.")
except Exception as e:
    print(f"Request failed: {e}")
