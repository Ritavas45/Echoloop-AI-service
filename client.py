"""
Client Library for interacting with Echoloop AI API
Useful for frontend applications, testing, and monitoring.
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time


class EcholooopClient:
    """Python client for Echoloop AI API."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
    
    def predict(
        self,
        image_paths: List[str],
        model_age_months: int,
        battery_health_pct: float,
        screen_cracked: bool,
        functional_issues: bool,
        cosmetic_scratches: int
    ) -> Dict:
        """
        Make a prediction.
        
        Args:
            image_paths: List of 4-5 image file paths
            model_age_months: Age of phone in months
            battery_health_pct: Battery health percentage (0-100)
            screen_cracked: Whether screen is cracked
            functional_issues: Whether device has functional issues
            cosmetic_scratches: Number of cosmetic scratches (0-10)
        
        Returns:
            Prediction response with prediction_id, prediction, confidence, etc.
        """
        
        # Validate images
        if len(image_paths) < 4 or len(image_paths) > 5:
            raise ValueError(f"Expected 4-5 images, got {len(image_paths)}")
        
        # Prepare files
        files = []
        for img_path in image_paths:
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            files.append(('images', open(img_path, 'rb')))
        
        # Prepare form data
        data = {
            'model_age_months': model_age_months,
            'battery_health_pct': battery_health_pct,
            'screen_cracked': screen_cracked,
            'functional_issues': functional_issues,
            'cosmetic_scratches': cosmetic_scratches
        }
        
        try:
            response = self.session.post(
                f"{self.api_url}/predict",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
        finally:
            # Close all files
            for _, f in files:
                f.close()
    
    def submit_feedback(
        self,
        prediction_id: int,
        ground_truth: str,
        user_confidence: Optional[float] = None
    ) -> Dict:
        """
        Submit ground truth feedback for a prediction.
        
        Args:
            prediction_id: ID from prediction response
            ground_truth: Correct label (Reuse/Refurbish/Repair/Recycle)
            user_confidence: Optional confidence in the label (0-1)
        
        Returns:
            Feedback response with success status
        """
        
        payload = {
            'prediction_id': prediction_id,
            'ground_truth': ground_truth,
            'user_confidence': user_confidence
        }
        
        response = self.session.post(
            f"{self.api_url}/feedback",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_model_status(self) -> Dict:
        """Get current model performance and retraining status."""
        response = self.session.get(f"{self.api_url}/model/status")
        response.raise_for_status()
        return response.json()
    
    def get_pending_feedback(self, limit: int = 20) -> Dict:
        """Get predictions awaiting user feedback."""
        response = self.session.get(
            f"{self.api_url}/predictions/pending-feedback",
            params={'limit': limit}
        )
        response.raise_for_status()
        return response.json()
    
    def get_statistics(self) -> Dict:
        """Get data collection and class distribution statistics."""
        response = self.session.get(f"{self.api_url}/statistics")
        response.raise_for_status()
        return response.json()
    
    def trigger_retrain(self) -> Dict:
        """Manually trigger model retraining."""
        response = self.session.post(f"{self.api_url}/retrain")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict:
        """Check API health."""
        response = self.session.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()
    
    def wait_for_retraining(self, timeout: int = 3600, check_interval: int = 30) -> bool:
        """
        Wait for model retraining to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            check_interval: How often to check status in seconds
        
        Returns:
            True if retrain completed, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_model_status()
            if not status.get('retraining_required'):
                print("✓ Retrain completed!")
                return True
            
            print(f"Waiting for retrain... ({int(time.time() - start_time)}s)")
            time.sleep(check_interval)
        
        print("✗ Retraining timeout exceeded")
        return False


# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_workflow():
    """Example of complete prediction and feedback workflow."""
    
    client = EcholooopClient("http://localhost:8000")
    
    # 1. Health check
    print("Checking API health...")
    health = client.health_check()
    print(f"✓ API is healthy: {health}")
    
    # 2. Make predictions
    print("\nMaking predictions...")
    predictions = []
    
    for i in range(3):
        result = client.predict(
            image_paths=[
                f"./test_data/phone_{i}_1.jpg",
                f"./test_data/phone_{i}_2.jpg",
                f"./test_data/phone_{i}_3.jpg",
                f"./test_data/phone_{i}_4.jpg",
            ],
            model_age_months=24,
            battery_health_pct=75.0,
            screen_cracked=False,
            functional_issues=False,
            cosmetic_scratches=2
        )
        print(f"Prediction {i+1}: {result['prediction']} ({result['confidence_pct']:.1f}%)")
        predictions.append(result)
    
    # 3. Submit feedback
    print("\nSubmitting feedback...")
    for pred in predictions:
        feedback_result = client.submit_feedback(
            prediction_id=pred['prediction_id'],
            ground_truth="Refurbish",
            user_confidence=0.95
        )
        print(f"✓ Feedback submitted for prediction {pred['prediction_id']}")
    
    # 4. Check model status
    print("\nChecking model status...")
    status = client.get_model_status()
    print(f"Model: {status['active_model_version']}")
    print(f"Accuracy: {status['accuracy_test']:.2%}")
    print(f"Retraining needed: {status['retraining_required']}")
    
    # 5. Get statistics
    print("\nGetting statistics...")
    stats = client.get_statistics()
    print(f"Class distribution (7d): {stats['data_collection']['class_distribution_7d']}")
    
    # 6. Check pending feedback
    print("\nPending feedback items:")
    pending = client.get_pending_feedback(limit=10)
    print(f"Count: {pending['count']}")
    for pred in pending['predictions'][:3]:
        print(f"  - ID {pred['prediction_id']}: {pred['fused_prediction']} ({pred['confidence']*100:.1f}%)")


if __name__ == "__main__":
    # Test client connection
    client = EcholooopClient()
    try:
        health = client.health_check()
        print("✓ Connected to API successfully!")
        print(json.dumps(health, indent=2))
    except Exception as e:
        print(f"✗ Could not connect to API: {e}")
        print("Make sure the API is running: python -m uvicorn app_updated:app --host 0.0.0.0 --port 8000")
