import requests

API_URL = "http://localhost:8000"

# Open 1 phone image and 4 cat images
files = [
    ('images', ('phone.jpg', open('scratch/phone.jpg', 'rb'), 'image/jpeg')),
    ('images', ('cat.jpg', open('scratch/cat.jpg', 'rb'), 'image/jpeg')),
    ('images', ('cat2.jpg', open('scratch/cat.jpg', 'rb'), 'image/jpeg')),
    ('images', ('cat3.jpg', open('scratch/cat.jpg', 'rb'), 'image/jpeg')),
    ('images', ('cat4.jpg', open('scratch/cat.jpg', 'rb'), 'image/jpeg')),
]

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
    if res.status_code == 400:
        print("🎉 SUCCESS: Validation successfully blocked the mixed images!")
    else:
        print("❌ FAIL: Validation did not block the mixed images.")
except Exception as e:
    print(f"Request failed: {e}")
