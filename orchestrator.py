"""
Pipeline Orchestration and Scheduling
Manages scheduled retraining, monitoring, and model deployment.
"""

import schedule
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import sys

from database import ECholooopDataStore
from continuous_training import ContinuousTrainer

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Orchestrator')

# ============================================================================
# Pipeline Orchestrator
# ============================================================================

class PipelineOrchestrator:
    """
    Manages the complete ML lifecycle:
    - Scheduled retraining
    - Model evaluation
    - Deployment decisions
    - Data collection monitoring
    """
    
    def __init__(self, config_path: str = "./config/pipeline_config.json"):
        self.config = self._load_config(config_path)
        self.data_store = ECholooopDataStore()
        self.trainer = ContinuousTrainer(data_store=self.data_store)
        
        logger.info("PipelineOrchestrator initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration."""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "retraining": {
                    "enabled": True,
                    "threshold_samples": 50,
                    "schedule": "daily",
                    "time": "02:00"  # 2 AM
                },
                "monitoring": {
                    "enabled": True,
                    "check_interval_hours": 6,
                    "alert_threshold": 0.05  # Alert if accuracy drops 5%
                },
                "deployment": {
                    "auto_deploy": False,
                    "min_test_accuracy": 0.80,
                    "max_performance_drop": 0.03
                },
                "data_collection": {
                    "retention_days": 365,
                    "alert_on_imbalance": True,
                    "max_class_ratio": 3.0
                }
            }
    
    def save_config(self, config_path: str = "./config/pipeline_config.json"):
        """Save current configuration."""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    
    def check_retraining_needed(self) -> bool:
        """Check if retraining should be triggered."""
        threshold = self.config['retraining']['threshold_samples']
        should_retrain = self.data_store.should_retrain(threshold)
        
        if should_retrain:
            logger.info(f"Retraining needed: threshold {threshold} samples reached")
        
        return should_retrain
    
    def scheduled_retrain_job(self) -> Optional[Dict]:
        """Execute scheduled retraining."""
        logger.info("=" * 60)
        logger.info("Starting Scheduled Retraining Job")
        logger.info("=" * 60)
        
        try:
            result = self.trainer.retrain_pipeline()
            
            if result['success']:
                logger.info(f"✓ Retraining successful: {result['model_version']}")
                logger.info(f"  Test Accuracy: {result['metrics']['test_accuracy']:.4f}")
                
                # Check deployment criteria
                if self.config['deployment']['auto_deploy']:
                    self._check_and_deploy(result)
            else:
                logger.error(f"✗ Retraining failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in scheduled retrain job: {e}")
            return None
    
    def _check_and_deploy(self, retrain_result: Dict):
        """Check if new model meets deployment criteria."""
        min_accuracy = self.config['deployment']['min_test_accuracy']
        max_drop = self.config['deployment']['max_performance_drop']
        
        new_accuracy = retrain_result['metrics']['test_accuracy']
        
        # Get current production model
        active_model = self.data_store.get_active_model()
        current_accuracy = active_model['test_accuracy'] if active_model else 0
        
        # Check criteria
        if new_accuracy >= min_accuracy and (new_accuracy >= (current_accuracy - max_drop)):
            logger.info(f"✓ New model meets deployment criteria")
            logger.info(f"  New Accuracy: {new_accuracy:.4f}, Current: {current_accuracy:.4f}")
            
            self.data_store.set_active_model(retrain_result['model_version'])
            logger.info(f"✓ Deployed new model: {retrain_result['model_version']}")
        else:
            logger.warning(f"✗ New model does not meet deployment criteria")
            logger.warning(f"  New Accuracy: {new_accuracy:.4f} (min: {min_accuracy:.4f})")
            logger.warning(f"  Current: {current_accuracy:.4f} (max drop: {max_drop:.4f})")
    
    def monitor_data_distribution(self):
        """Monitor class distribution for imbalance."""
        logger.info("Checking data distribution...")
        
        dist_7d = self.data_store.get_class_distribution(days_back=7)
        
        if not dist_7d:
            logger.warning("No feedback data in past 7 days")
            return
        
        total = sum([v for v in dist_7d.values() if v])
        if total == 0:
            return
        
        # Calculate ratios
        counts = [v for v in dist_7d.values() if v]
        if counts:
            max_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
            
            logger.info(f"Class distribution (7d):")
            for class_name, count in dist_7d.items():
                pct = (count / total * 100) if total > 0 else 0
                logger.info(f"  {class_name}: {count} ({pct:.1f}%)")
            
            if max_ratio > self.config['data_collection']['max_class_ratio']:
                logger.warning(f"⚠ Data imbalance detected! Ratio: {max_ratio:.2f}")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive pipeline report."""
        active_model = self.data_store.get_active_model()
        stats = self.trainer.get_training_statistics()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "active_model": {
                "version": active_model['model_version'] if active_model else None,
                "accuracy_test": active_model['test_accuracy'] if active_model else None,
                "accuracy_val": active_model['validation_accuracy'] if active_model else None,
                "created_date": active_model['created_date'] if active_model else None
            },
            "data_collection": {
                "pending_feedback": len(self.data_store.get_unfeedback_predictions(limit=999999)),
                "retraining_needed": self.data_store.should_retrain(self.config['retraining']['threshold_samples']),
                "class_distribution_7d": stats.get('class_distribution_7d', {}),
                "class_distribution_30d": stats.get('class_distribution_30d', {})
            },
            "configuration": self.config
        }
        
        return report
    
    def start_scheduler(self):
        """Start the scheduler for periodic tasks."""
        logger.info("Starting Pipeline Scheduler...")
        
        # Schedule retraining job
        if self.config['retraining']['enabled']:
            schedule_time = self.config['retraining']['time']
            schedule.every().day.at(schedule_time).do(self.scheduled_retrain_job)
            logger.info(f"Scheduled daily retraining at {schedule_time}")
        
        # Schedule data distribution check
        if self.config['monitoring']['enabled']:
            schedule.every(self.config['monitoring']['check_interval_hours']).hours.do(
                self.monitor_data_distribution
            )
            logger.info(f"Scheduled data monitoring every {self.config['monitoring']['check_interval_hours']} hours")
        
        # Keep scheduler running
        logger.info("Scheduler running... (Press Ctrl+C to stop)")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped")
    
    def run_once(self):
        """Run all scheduled tasks once (useful for testing)."""
        logger.info("Running all tasks once...")
        
        self.scheduled_retrain_job()
        self.monitor_data_distribution()
        
        logger.info("Tasks completed")


# ============================================================================
# CLI Commands
# ============================================================================

def main():
    """Command-line interface for pipeline management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Echoloop ML Pipeline Orchestrator')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Retrain command
    retrain_parser = subparsers.add_parser('retrain', help='Trigger model retraining')
    retrain_parser.add_argument('--force', action='store_true', help='Force retrain even if threshold not met')
    
    # Monitor command
    subparsers.add_parser('monitor', help='Monitor data distribution')
    
    # Report command
    subparsers.add_parser('report', help='Generate pipeline report')
    
    # Scheduler command
    subparsers.add_parser('scheduler', help='Start continuous scheduler')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('--show', action='store_true', help='Show current config')
    config_parser.add_argument('--reset', action='store_true', help='Reset to default config')
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator()
    
    if args.command == 'retrain':
        logger.info("Manual retrain triggered")
        result = orchestrator.scheduled_retrain_job()
        print(json.dumps(result, indent=2))
    
    elif args.command == 'monitor':
        orchestrator.monitor_data_distribution()
    
    elif args.command == 'report':
        report = orchestrator.generate_report()
        print(json.dumps(report, indent=2))
    
    elif args.command == 'scheduler':
        orchestrator.start_scheduler()
    
    elif args.command == 'config':
        if args.show:
            print(json.dumps(orchestrator.config, indent=2))
        elif args.reset:
            # Reset to default by removing config file
            config_path = "./config/pipeline_config.json"
            if Path(config_path).exists():
                Path(config_path).unlink()
                logger.info("Configuration reset to default")
        else:
            print(json.dumps(orchestrator.config, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
