import os
import glob
from client import EcholooopClient

def find_phone_images(folder_path):
    """Locate up to 5 images from a phone folder."""
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    return sorted(image_paths)[:5]

def main():
    client = EcholooopClient()
    
    print("\n" + "="*80)
    print("                🔍 MODEL PREDICTION EVALUATION RUN 🔍")
    print("="*80)
    
    # Test cases mapping phone folders (mock data) and their physical attributes
    test_cases = [
        {
            "name": "Excellent Condition Phone (Expected: Reuse/Refurbish)",
            "folder": "./mock_data/train/Reuse/phone_001",
            "age": 3,
            "battery": 97.0,
            "cracked": False,
            "issues": False,
            "scratches": 0
        },
        {
            "name": "Standard Wear & Tear Phone (Expected: Refurbish/Repair)",
            "folder": "./mock_data/train/Refurbish/phone_001",
            "age": 14,
            "battery": 83.5,
            "cracked": False,
            "issues": False,
            "scratches": 2
        },
        {
            "name": "Damaged Phone needing Repair (Expected: Repair)",
            "folder": "./mock_data/train/Repair/phone_001",
            "age": 20,
            "battery": 72.0,
            "cracked": True,
            "issues": True,
            "scratches": 5
        },
        {
            "name": "Severely Broken Phone (Expected: Recycle)",
            "folder": "./mock_data/train/Recycle/phone_001",
            "age": 36,
            "battery": 45.0,
            "cracked": True,
            "issues": True,
            "scratches": 9
        }
    ]
    
    for case in test_cases:
        print(f"\n📱 {case['name']}")
        print(f"   Attributes: Age: {case['age']} months | Battery: {case['battery']}% | Cracked Screen: {case['cracked']} | Functional Issues: {case['issues']} | Scratches: {case['scratches']}")
        
        # Locate mock images
        image_paths = find_phone_images(case['folder'])
        if len(image_paths) < 4:
            print("   ⚠️ Not enough mock images found for this test case. Skipping...")
            continue
            
        print(f"   Selected Images: {[os.path.basename(p) for p in image_paths]}")
        
        # Query prediction
        try:
            res = client.predict(
                image_paths=image_paths,
                model_age_months=case['age'],
                battery_health_pct=case['battery'],
                screen_cracked=case['cracked'],
                functional_issues=case['issues'],
                cosmetic_scratches=case['scratches']
            )
            
            print(f"   ✨ Result: {res['prediction']} ({res['confidence_pct']:.2f}% Confidence)")
            print(f"   🤖 Decision Path: {res['decision_path']}")
            print(f"   🗳️ Individual Votes:")
            print(f"      - Image Model predicted: {res['individual_votes']['image_model_prediction']} ({res['individual_votes']['image_model_confidence_pct']:.2f}%)")
            print(f"      - Tabular model predicted: {res['individual_votes']['tabular_model_prediction']} ({res['individual_votes']['tabular_model_confidence_pct']:.2f}%)")
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            
    print("\n" + "="*80)
    print("                        EVALUATION RUN COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
