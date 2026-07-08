import os
import glob
from pathlib import Path
try:
    from google.cloud import storage
    GCP_STORAGE_AVAILABLE = True
except ImportError:
    GCP_STORAGE_AVAILABLE = False

class GCPStorageManager:
    """Manages downloading and uploading model checkpoints to/from Google Cloud Storage."""
    
    def __init__(self):
        self.bucket_name = os.environ.get("GCS_BUCKET_NAME")
        self.enabled = GCP_STORAGE_AVAILABLE and bool(self.bucket_name)
        self.client = None
        
        if self.enabled:
            try:
                # If GOOGLE_APPLICATION_CREDENTIALS is set, it will be automatically picked up by storage.Client()
                self.client = storage.Client()
                print(f"[GCPStorageManager] Initialized for bucket: {self.bucket_name}")
            except Exception as e:
                print(f"[GCPStorageManager] Warning: Failed to initialize GCP storage client: {e}")
                self.enabled = False

    def download_file(self, blob_name: str, local_path: str) -> bool:
        """Downloads a single blob from GCS bucket to a local file."""
        if not self.enabled:
            return False
        
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            if not blob.exists():
                print(f"[GCPStorageManager] Blob {blob_name} does not exist in bucket {self.bucket_name}.")
                return False
                
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            print(f"[GCPStorageManager] Downloading {blob_name} to {local_path}...")
            blob.download_to_filename(local_path)
            print(f"[GCPStorageManager] Successfully downloaded {blob_name}.")
            return True
        except Exception as e:
            print(f"[GCPStorageManager] Error downloading {blob_name}: {e}")
            return False

    def upload_file(self, local_path: str, blob_name: str) -> bool:
        """Uploads a single local file to GCS bucket."""
        if not self.enabled:
            return False
            
        if not os.path.exists(local_path):
            print(f"[GCPStorageManager] Local file {local_path} does not exist. Cannot upload.")
            return False
            
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            print(f"[GCPStorageManager] Uploading {local_path} to {blob_name}...")
            blob.upload_from_filename(local_path)
            print(f"[GCPStorageManager] Successfully uploaded {blob_name}.")
            return True
        except Exception as e:
            print(f"[GCPStorageManager] Error uploading {local_path}: {e}")
            return False

    def sync_checkpoints_from_gcs(self):
        """Syncs all model files (pth and json) from GCS to local directories."""
        if not self.enabled:
            print("[GCPStorageManager] GCS checkpoint sync skipped (GCS not enabled/configured).")
            return
            
        print("[GCPStorageManager] Syncing checkpoints from Google Cloud Storage...")
        # Checkpoints files
        self.download_file("checkpoints/best_model.pth", "./checkpoints/best_model.pth")
        
        # XGBoost files
        try:
            bucket = self.client.bucket(self.bucket_name)
            blobs = self.client.list_blobs(self.bucket_name, prefix="models/")
            for blob in blobs:
                if blob.name.endswith(".json"):
                    local_file = os.path.join(".", blob.name)
                    self.download_file(blob.name, local_file)
        except Exception as e:
            print(f"[GCPStorageManager] Error listing models/ blobs in GCS: {e}")

    def sync_checkpoints_to_gcs(self):
        """Syncs local checkpoints and models to GCS."""
        if not self.enabled:
            return
            
        print("[GCPStorageManager] Syncing local checkpoints to Google Cloud Storage...")
        # Upload best_model.pth if it exists
        if os.path.exists("./checkpoints/best_model.pth"):
            self.upload_file("./checkpoints/best_model.pth", "checkpoints/best_model.pth")
            
        # Upload all xgboost json models in models/
        for model_path in glob.glob("./models/*.json"):
            blob_name = os.path.join("models", os.path.basename(model_path))
            self.upload_file(model_path, blob_name)
            
        # Upload root level xgboost model if it exists
        if os.path.exists("./xgboost_model.json"):
            self.upload_file("./xgboost_model.json", "models/xgboost_model.json")
