"""
Continuous Training Pipeline
Handles automated model retraining with new user feedback data.
Implements active learning and data collection strategies.
"""

import xgboost as xgb
import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import uuid

from database import ECholooopDataStore
from model import LateFusionEfficientNet
from dataset import get_transforms
from gcp_storage import GCPStorageManager


class ContinuousTrainer:
    """Manages continuous model retraining and version control."""
    
    def __init__(self, device: str = None, data_store: ECholooopDataStore = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() 
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.data_store = data_store or ECholooopDataStore()
        self.checkpoint_dir = Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)
        self.CLASSES = ['Reuse', 'Refurbish', 'Repair', 'Recycle']
    
    def generate_model_version(self) -> str:
        """Generate a unique model version identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"v1_retrain_{timestamp}_{unique_id}"
    
    def prepare_training_data(
        self,
        exclude_model_version: str = None,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare training, validation, and test data from collected feedback.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print("[ContinuousTrainer] Fetching labeled data from database...")
        X, y = self.data_store.get_training_data(model_version=exclude_model_version)
        
        if len(X) == 0:
            raise ValueError("No labeled training data available for retraining!")
        
        print(f"[ContinuousTrainer] Total labeled samples: {len(X)}")
        print(f"[ContinuousTrainer] Class distribution:\n{y.value_counts()}")
        
        # Split: 70% train+val, 30% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Split temp into train (70%) and val (30%)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )
        
        print(f"[ContinuousTrainer] Data split:")
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_tabular_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparams: Dict = None
    ) -> Tuple[xgb.XGBClassifier, Dict]:
        """
        Train or retrain the XGBoost tabular model.
        """
        print("\n[ContinuousTrainer] Training XGBoost tabular model...")
        
        # Default hyperparameters (can be tuned with Optuna)
        params = hyperparams or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        
        print(f"[ContinuousTrainer] XGBoost Training Accuracy: {train_acc:.4f}")
        print(f"[ContinuousTrainer] XGBoost Validation Accuracy: {val_acc:.4f}")
        
        metrics = {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'hyperparams': params
        }
        
        return model, metrics
    
    def save_tabular_model(
        self,
        model: xgb.XGBClassifier,
        model_version: str
    ) -> str:
        """Save trained XGBoost model."""
        model_path = str(self.models_dir / f"xgboost_{model_version}.json")
        model.save_model(model_path)
        print(f"[ContinuousTrainer] XGBoost model saved to: {model_path}")
        return model_path
    
    def retrain_pipeline(
        self,
        exclude_current_model: bool = True,
        hyperparams: Dict = None
    ) -> Dict:
        """
        Full retraining pipeline:
        1. Fetch new labeled data
        2. Prepare train/val/test splits
        3. Retrain tabular model
        4. Evaluate on test set
        5. Log metadata
        6. Return model version info
        """
        job_id = str(uuid.uuid4())
        model_version = self.generate_model_version()
        
        print(f"\n{'='*60}")
        print(f"[ContinuousTrainer] Starting Retraining Pipeline")
        print(f"[ContinuousTrainer] Job ID: {job_id}")
        print(f"[ContinuousTrainer] Model Version: {model_version}")
        print(f"{'='*60}\n")
        
        try:
            # Step 1: Log job start
            self.data_store.log_training_job(
                job_id=job_id,
                status="STARTED"
            )
            
            # Step 2: Prepare data
            active_model = self.data_store.get_active_model()
            exclude_version = active_model['model_version'] if (
                exclude_current_model and active_model
            ) else None
            
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_training_data(
                exclude_model_version=exclude_version
            )
            
            # Step 3: Train tabular model
            tabular_model, metrics = self.train_tabular_model(
                X_train, y_train, X_val, y_val,
                hyperparams=hyperparams
            )
            
            # Step 4: Evaluate on test set
            test_acc = tabular_model.score(X_test, y_test)
            print(f"[ContinuousTrainer] XGBoost Test Accuracy: {test_acc:.4f}")
            metrics['test_accuracy'] = float(test_acc)
            
            # Step 5: Save model
            model_path = self.save_tabular_model(tabular_model, model_version)
            
            # Step 6: Log to database
            self.data_store.log_model_metadata(
                model_version=model_version,
                model_type="xgboost",
                training_data_size=len(X_train),
                validation_accuracy=metrics['val_accuracy'],
                test_accuracy=test_acc,
                model_path=model_path,
                metrics=metrics
            )
            
            # Step 7: Update training job
            self.data_store.log_training_job(
                job_id=job_id,
                status="COMPLETED",
                training_samples=len(X_train),
                validation_samples=len(X_val),
                metrics=metrics
            )
            
            # Sync newly trained models to GCP Storage
            try:
                storage_mgr = GCPStorageManager()
                if storage_mgr.enabled:
                    storage_mgr.sync_checkpoints_to_gcs()
            except Exception as e:
                print(f"[ContinuousTrainer] Warning: Failed to sync new checkpoints to GCS: {e}")
            
            print(f"\n{'='*60}")
            print(f"[ContinuousTrainer] Retraining Pipeline COMPLETED")
            print(f"[ContinuousTrainer] New Model: {model_version}")
            print(f"[ContinuousTrainer] Test Accuracy: {test_acc:.4f}")
            print(f"{'='*60}\n")
            
            return {
                'success': True,
                'job_id': job_id,
                'model_version': model_version,
                'model_path': model_path,
                'metrics': metrics,
                'test_accuracy': test_acc
            }
            
        except Exception as e:
            print(f"\n[ContinuousTrainer] ERROR during retraining: {e}")
            self.data_store.log_training_job(
                job_id=job_id,
                status="FAILED",
                error_message=str(e)
            )
            return {
                'success': False,
                'job_id': job_id,
                'error': str(e)
            }
    
    def compare_models(
        self,
        model_v1: str,
        model_v2: str,
        test_data: pd.DataFrame = None
    ) -> Dict:
        """
        Compare performance of two models.
        """
        print(f"\n[ContinuousTrainer] Comparing models {model_v1} vs {model_v2}...")
        
        # If no test data provided, use database test set
        if test_data is None:
            _, _, _, _, test_data, y_test = self.prepare_training_data()
        
        # Load models and evaluate
        comparison = {
            'model_v1': model_v1,
            'model_v2': model_v2,
        }
        
        return comparison
    
    def get_training_statistics(self) -> Dict:
        """Get overall training and data collection statistics."""
        stats = {
            'class_distribution_7d': self.data_store.get_class_distribution(days_back=7),
            'class_distribution_30d': self.data_store.get_class_distribution(days_back=30),
        }
        
        return stats


def check_and_retrain_if_needed(
    threshold_samples: int = 100,
    hyperparams: Dict = None
) -> Optional[Dict]:
    """
    Convenience function to check if retraining is needed and execute pipeline.
    """
    store = ECholooopDataStore()
    
    if not store.should_retrain(threshold_samples=threshold_samples):
        print(f"[ContinuousTraining] Not enough new data for retraining. Threshold: {threshold_samples}")
        return None
    
    trainer = ContinuousTrainer(data_store=store)
    result = trainer.retrain_pipeline(hyperparams=hyperparams)
    
    return result
