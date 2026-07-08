import os
import sys
import requests
import subprocess
import json

API_URL = "http://localhost:8000"

def print_header(title):
    print("\n" + "="*80)
    print(f" 🔍 VERIFICATION STEP: {title}")
    print("="*80)

def main():
    print("\n" + "╔" + "═"*78 + "╗")
    print("║                                                                              ║")
    print("║             🛡️  ECHOLOOP AI - 6-POINT SYSTEM VERIFICATION SUITE  🛡️           ║")
    print("║                                                                              ║")
    print("╚" + "═"*78 + "╝")
    
    results = {}
    
    # ---------------------------------------------------------
    # STEP 1: API Server Health
    # ---------------------------------------------------------
    print_header("1. API Health Check")
    try:
        res = requests.get(f"{API_URL}/health")
        if res.status_code == 200:
            data = res.json()
            print(f"   [PASS] API is active on port 8000.")
            print(f"          - Status: {data.get('status')}")
            print(f"          - Active device: {data.get('device')}")
            print(f"          - Model loaded: {data.get('model_version')}")
            results["1. API Health"] = "PASS"
        else:
            print(f"   [FAIL] API returned status code {res.status_code}")
            results["1. API Health"] = "FAIL"
    except Exception as e:
        print(f"   [FAIL] Could not connect to API: {e}")
        results["1. API Health"] = "FAIL"

    # ---------------------------------------------------------
    # STEP 2: Database Registry / Model Status
    # ---------------------------------------------------------
    print_header("2. Active Model Registry")
    try:
        res = requests.get(f"{API_URL}/model/status")
        if res.status_code == 200:
            data = res.json()
            print(f"   [PASS] Active model registered and tracked in SQLite.")
            print(f"          - Active model: {data.get('active_model_version')}")
            print(f"          - Accuracy on test set: {data.get('accuracy_test')*100:.2f}%")
            print(f"          - Retrained samples size: {data.get('training_samples')}")
            results["2. Model Status"] = "PASS"
        else:
            print(f"   [FAIL] Model status returned code {res.status_code}")
            results["2. Model Status"] = "FAIL"
    except Exception as e:
        print(f"   [FAIL] Model status check failed: {e}")
        results["2. Model Status"] = "FAIL"

    # ---------------------------------------------------------
    # STEP 3: Multi-Modal Prediction Flow
    # ---------------------------------------------------------
    print_header("3. Multi-Modal Prediction Inference")
    # Locate mock images
    mock_images = []
    mock_dir = "./mock_data/train/Reuse/phone_001"
    if os.path.exists(mock_dir):
        mock_images = [os.path.join(mock_dir, f) for f in os.listdir(mock_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:4]
        
    if len(mock_images) < 4:
        print("   [FAIL] Could not locate 4 mock images under ./mock_data")
        results["3. Prediction Flow"] = "FAIL"
    else:
        files = [('images', (os.path.basename(p), open(p, 'rb'), 'image/jpeg')) for p in mock_images]
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
                
            if res.status_code == 200:
                pred_data = res.json()
                print(f"   [PASS] Multi-modal prediction succeeded.")
                print(f"          - Prediction ID: {pred_data.get('prediction_id')}")
                print(f"          - Predicted: {pred_data.get('prediction')} ({pred_data.get('confidence_pct')*1.0:.2f}% confidence)")
                print(f"          - Decision path: {pred_data.get('decision_path')}")
                results["3. Prediction Flow"] = "PASS"
                results["prediction_id"] = pred_data.get('prediction_id')
            else:
                print(f"   [FAIL] Prediction endpoint returned code {res.status_code}: {res.text}")
                results["3. Prediction Flow"] = "FAIL"
        except Exception as e:
            print(f"   [FAIL] Prediction request failed: {e}")
            results["3. Prediction Flow"] = "FAIL"

    # ---------------------------------------------------------
    # STEP 4: Feedback Loop Submission
    # ---------------------------------------------------------
    print_header("4. Continuous Feedback Registration")
    pred_id = results.get("prediction_id")
    if not pred_id:
        print("   [FAIL] Skipped. Prediction ID was not generated in Step 3.")
        results["4. Feedback Registry"] = "FAIL"
    else:
        payload = {
            "prediction_id": pred_id,
            "ground_truth": "Reuse"
        }
        try:
            res = requests.post(f"{API_URL}/feedback", json=payload)
            if res.status_code == 200:
                print(f"   [PASS] User feedback recorded successfully in SQLite.")
                print(f"          - Message: {res.json().get('message')}")
                results["4. Feedback Registry"] = "PASS"
            else:
                print(f"   [FAIL] Feedback registration returned code {res.status_code}")
                results["4. Feedback Registry"] = "FAIL"
        except Exception as e:
            print(f"   [FAIL] Feedback submission failed: {e}")
            results["4. Feedback Registry"] = "FAIL"

    # ---------------------------------------------------------
    # STEP 5: Collection Statistics Monitoring
    # ---------------------------------------------------------
    print_header("5. Data Collection Statistics")
    try:
        res = requests.get(f"{API_URL}/statistics")
        if res.status_code == 200:
            print(f"   [PASS] Statistics retrieved correctly.")
            print(f"          - Class distribution (7d): {res.json().get('data_collection', {}).get('class_distribution_7d')}")
            results["5. Data Stats"] = "PASS"
        else:
            print(f"   [FAIL] Statistics returned code {res.status_code}")
            results["5. Data Stats"] = "FAIL"
    except Exception as e:
        print(f"   [FAIL] Statistics check failed: {e}")
        results["5. Data Stats"] = "FAIL"

    # ---------------------------------------------------------
    # STEP 6: CLI Utility Execution
    # ---------------------------------------------------------
    print_header("6. Pipeline Orchestrator CLI")
    try:
        process = subprocess.run(
            ["./venv/bin/python", "orchestrator.py", "report"],
            capture_output=True,
            text=True
        )
        if process.returncode == 0:
            print(f"   [PASS] Pipeline orchestrator runs report correctly.")
            report = json.loads(process.stdout[process.stdout.find("{"):])
            print(f"          - Scheduler retraining threshold: {report.get('configuration', {}).get('retraining', {}).get('threshold_samples')} samples")
            print(f"          - Auto-deployment check ACC limit: >= {report.get('configuration', {}).get('deployment', {}).get('min_test_accuracy')*100:.1f}%")
            results["6. CLI Utilities"] = "PASS"
        else:
            print(f"   [FAIL] Orchestrator report command returned code {process.returncode}: {process.stderr}")
            results["6. CLI Utilities"] = "FAIL"
    except Exception as e:
        print(f"   [FAIL] CLI execution check failed: {e}")
        results["6. CLI Utilities"] = "FAIL"

    # ---------------------------------------------------------
    # SUMMARY TABLE
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("                     📝 FINAL SYSTEM VERIFICATION SUMMARY")
    print("="*80)
    print(f"   {'Check':<40} | {'Status':<15}")
    print("   " + "-"*38 + " | " + "-"*13)
    
    all_passed = True
    for key in ["1. API Health", "2. Model Status", "3. Prediction Flow", "4. Feedback Registry", "5. Data Stats", "6. CLI Utilities"]:
        status = results.get(key, "FAIL")
        print(f"   {key:<40} | {status:<15}")
        if status != "PASS":
            all_passed = False
            
    print("="*80)
    if all_passed:
        print("   🌟 SUCCESS: All points passed! The ML Continuous Learning system is in perfect shape.")
    else:
        print("   ⚠️ WARNING: Some verification points failed. Check details in logs above.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
