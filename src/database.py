"""
Database module for storing predictions, ground truth labels, and model metadata.
Supports continuous learning by tracking prediction feedback and retraining data.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

DB_PATH = "./data/echoloop_data.db"

class ECholooopDataStore:
    """SQLite-based data store for predictions, feedback, and metadata."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_db_exists()
        self._init_tables()
    
    def _ensure_db_exists(self):
        """Ensure database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    def _init_tables(self):
        """Initialize database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table 1: Predictions Log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT,
                model_age_months INTEGER,
                battery_health_pct REAL,
                screen_cracked INTEGER,
                functional_issues INTEGER,
                cosmetic_scratches INTEGER,
                image_pred TEXT,
                image_confidence REAL,
                tabular_pred TEXT,
                tabular_confidence REAL,
                fused_prediction TEXT,
                fused_confidence REAL,
                decision_path TEXT,
                raw_probabilities TEXT,
                image_filenames TEXT,
                feedback_received INTEGER DEFAULT 0,
                ground_truth TEXT,
                feedback_timestamp DATETIME,
                confidence_correct INTEGER,
                is_training_data INTEGER DEFAULT 0
            )
        ''')
        
        # Table 2: Model Metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT UNIQUE,
                model_type TEXT,
                created_date DATETIME,
                training_data_size INTEGER,
                validation_accuracy REAL,
                test_accuracy REAL,
                is_active INTEGER DEFAULT 0,
                model_path TEXT,
                config TEXT,
                metrics TEXT
            )
        ''')
        
        # Table 3: Training Jobs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE,
                created_date DATETIME,
                started_date DATETIME,
                completed_date DATETIME,
                status TEXT,
                training_samples INTEGER,
                validation_samples INTEGER,
                new_model_version TEXT,
                metrics TEXT,
                error_message TEXT
            )
        ''')
        
        # Table 4: Data Imbalance Log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS class_distribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT,
                reuse_count INTEGER,
                refurbish_count INTEGER,
                repair_count INTEGER,
                recycle_count INTEGER,
                total_count INTEGER
            )
        ''')
        
        # Table 5: Continuous Learning Config
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS continuous_learning_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                retraining_threshold_samples INTEGER DEFAULT 100,
                auto_retrain_enabled INTEGER DEFAULT 1,
                min_feedback_confidence REAL DEFAULT 0.5,
                last_retraining_date DATETIME,
                next_scheduled_retraining DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(
        self,
        model_version: str,
        model_age_months: int,
        battery_health_pct: float,
        screen_cracked: bool,
        functional_issues: bool,
        cosmetic_scratches: int,
        image_pred: str,
        image_confidence: float,
        tabular_pred: str,
        tabular_confidence: float,
        fused_prediction: str,
        fused_confidence: float,
        decision_path: str,
        raw_probabilities: Dict,
        image_filenames: List[str] = None
    ) -> int:
        """
        Log a prediction to the database.
        Returns the prediction ID for later reference.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions_log (
                model_version, model_age_months, battery_health_pct, screen_cracked,
                functional_issues, cosmetic_scratches, image_pred, image_confidence,
                tabular_pred, tabular_confidence, fused_prediction, fused_confidence,
                decision_path, raw_probabilities, image_filenames
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_version, model_age_months, battery_health_pct, int(screen_cracked),
            int(functional_issues), cosmetic_scratches, image_pred, image_confidence,
            tabular_pred, tabular_confidence, fused_prediction, fused_confidence,
            decision_path, json.dumps(raw_probabilities),
            json.dumps(image_filenames) if image_filenames else None
        ))
        
        conn.commit()
        pred_id = cursor.lastrowid
        conn.close()
        
        return pred_id
    
    def submit_feedback(
        self,
        prediction_id: int,
        ground_truth: str,
        correct: bool
    ) -> bool:
        """
        Submit feedback (ground truth label) for a prediction.
        This is crucial for continuous learning.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions_log
            SET feedback_received = 1, ground_truth = ?, 
                feedback_timestamp = CURRENT_TIMESTAMP, confidence_correct = ?
            WHERE id = ?
        ''', (ground_truth, int(correct), prediction_id))
        
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    
    def get_unfeedback_predictions(self, limit: int = 100) -> pd.DataFrame:
        """Get predictions waiting for feedback."""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT * FROM predictions_log
            WHERE feedback_received = 0
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df
    
    def get_training_data(self, model_version: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get all labeled predictions for model training/retraining.
        Returns (features_df, labels_series)
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get labeled predictions
        query = '''
            SELECT * FROM predictions_log
            WHERE feedback_received = 1 AND ground_truth IS NOT NULL
        '''
        if model_version:
            query += f" AND model_version != '{model_version}'"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return pd.DataFrame(), pd.Series(dtype=str)
        
        # Extract features
        feature_cols = [
            'model_age_months', 'battery_health_pct', 'screen_cracked',
            'functional_issues', 'cosmetic_scratches'
        ]
        X = df[feature_cols].copy()
        y = df['ground_truth'].copy()
        
        return X, y
    
    def log_model_metadata(
        self,
        model_version: str,
        model_type: str,
        training_data_size: int,
        validation_accuracy: float,
        test_accuracy: float,
        model_path: str,
        config: Dict = None,
        metrics: Dict = None
    ) -> bool:
        """Log metadata about a trained model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO model_metadata (
                    model_version, model_type, created_date, training_data_size,
                    validation_accuracy, test_accuracy, model_path, config, metrics
                ) VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
            ''', (
                model_version, model_type, training_data_size,
                validation_accuracy, test_accuracy, model_path,
                json.dumps(config) if config else None,
                json.dumps(metrics) if metrics else None
            ))
            conn.commit()
            success = True
        except Exception as e:
            print(f"Error logging model metadata: {e}")
            success = False
        finally:
            conn.close()
        
        return success
    
    def set_active_model(self, model_version: str) -> bool:
        """Set a model as the active production model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Deactivate all models
        cursor.execute('UPDATE model_metadata SET is_active = 0')
        
        # Activate the specified model
        cursor.execute(
            'UPDATE model_metadata SET is_active = 1 WHERE model_version = ?',
            (model_version,)
        )
        
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    
    def get_active_model(self) -> Optional[Dict]:
        """Get the currently active model metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM model_metadata WHERE is_active = 1 LIMIT 1
        ''')
        result = cursor.fetchone()
        conn.close()
        
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        return None
    
    def get_class_distribution(self, days_back: int = 7) -> Dict:
        """Get class distribution of predictions for the last N days."""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT 
                SUM(CASE WHEN ground_truth = 'Reuse' THEN 1 ELSE 0 END) as reuse_count,
                SUM(CASE WHEN ground_truth = 'Refurbish' THEN 1 ELSE 0 END) as refurbish_count,
                SUM(CASE WHEN ground_truth = 'Repair' THEN 1 ELSE 0 END) as repair_count,
                SUM(CASE WHEN ground_truth = 'Recycle' THEN 1 ELSE 0 END) as recycle_count
            FROM predictions_log
            WHERE feedback_received = 1 
            AND datetime(feedback_timestamp) >= datetime('now', '-' || ? || ' days')
        '''
        df = pd.read_sql_query(query, conn, params=(days_back,))
        conn.close()
        
        distribution = df.iloc[0].to_dict() if len(df) > 0 else {}
        return distribution
    
    def log_training_job(
        self,
        job_id: str,
        status: str,
        training_samples: int = 0,
        validation_samples: int = 0,
        metrics: Dict = None,
        error_message: str = None
    ) -> bool:
        """Log a training job execution."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        try:
            # Check if job exists
            cursor.execute('SELECT id FROM training_jobs WHERE job_id = ?', (job_id,))
            if cursor.fetchone():
                # Update existing
                cursor.execute('''
                    UPDATE training_jobs
                    SET status = ?, completed_date = CURRENT_TIMESTAMP,
                        training_samples = ?, validation_samples = ?, 
                        metrics = ?, error_message = ?
                    WHERE job_id = ?
                ''', (status, training_samples, validation_samples,
                      json.dumps(metrics) if metrics else None, error_message, job_id))
            else:
                # Insert new
                cursor.execute('''
                    INSERT INTO training_jobs (
                        job_id, created_date, status, training_samples,
                        validation_samples, metrics, error_message
                    ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
                ''', (job_id, status, training_samples, validation_samples,
                      json.dumps(metrics) if metrics else None, error_message))
            
            conn.commit()
            success = True
        except Exception as e:
            print(f"Error logging training job: {e}")
            success = False
        finally:
            conn.close()
        
        return success
    
    def should_retrain(self, threshold_samples: int = 100) -> bool:
        """Check if enough new labeled data exists to trigger retraining."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM predictions_log
            WHERE feedback_received = 1 AND is_training_data = 0
        ''')
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count >= threshold_samples


# Initialize global store
store = ECholooopDataStore()
