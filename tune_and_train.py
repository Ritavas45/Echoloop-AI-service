import os
import time
import joblib
import optuna
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from pipelines import build_preprocessing_pipeline

# Configure Optuna to be less verbose
optuna.logging.set_verbosity(optuna.logging.WARNING)

def print_confusion_matrix(cm, classes):
    """
    Prints a text-based confusion matrix beautifully formatted for command line viewing.
    """
    header = f"{'Actual \\ Pred':<15} | " + " | ".join([f"{cls:<10}" for cls in classes])
    border = "-" * len(header)
    print("\n" + border)
    print(header)
    print(border)
    for i, row in enumerate(cm):
        row_str = f"{classes[i]:<15} | " + " | ".join([f"{val:<10}" for val in row])
        print(row_str)
    print(border + "\n")

def main():
    # 1. Load data
    data_path = "./phone_tabular_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing tabular dataset file: {data_path}. Please run generate_tabular_data.py first.")
        
    print(f"Loading tabular dataset from: {os.path.abspath(data_path)}")
    df = pd.read_csv(data_path)
    
    # 2. Separate features X and target y
    X = df.drop(columns=['condition'])
    y_raw = df['condition']
    
    # Encode Target Labels
    classes = ['Reuse', 'Refurbish', 'Repair', 'Recycle']
    class_mapping = {cls: idx for idx, cls in enumerate(classes)}
    y = y_raw.map(class_mapping)
    
    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Dataset split completed:")
    print(f"  - Train samples: {X_train.shape[0]}")
    print(f"  - Test samples: {X_test.shape[0]}")
    
    # 4. Optuna Hyperparameter Tuning
    print("\nStarting Optuna Hyperparameter Tuning (30 trials using 5-fold Stratified Cross-Validation)...")
    start_tuning = time.time()
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 350),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        }
        
        # Build training pipeline
        model = xgb.XGBClassifier(**params)
        pipeline = Pipeline(steps=[
            ('preprocessor', build_preprocessing_pipeline()),
            ('classifier', model)
        ])
        
        # 5-fold CV to evaluate trial performance
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Using f1_macro because we want high performance across all 4 condition classes
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    tuning_time = time.time() - start_tuning
    print(f"Optuna tuning completed in {tuning_time:.1f} seconds!")
    print(f"Best Trial F1-Macro score: {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
        
    # 5. Fit Final Model on Entire Training Set
    print("\nFitting final model pipeline on entire training split...")
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['eval_metric'] = 'mlogloss'
    
    final_model = xgb.XGBClassifier(**best_params)
    final_pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessing_pipeline()),
        ('classifier', final_model)
    ])
    
    final_pipeline.fit(X_train, y_train)
    print("Final pipeline fitted successfully!")
    
    # 6. Evaluation on Hold-Out Test Set
    print("\nEvaluating on hold-out Test Set...")
    y_pred = final_pipeline.predict(X_test)
    
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1-Macro: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print_confusion_matrix(cm, classes)
    
    # 7. SHAP Feature Importance Visualization
    print("\nCalculating SHAP Feature Importance...")
    # Get preprocessed test features
    preprocessor = final_pipeline.named_steps['preprocessor']
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Get clean feature names out of the ColumnTransformer
    feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = [name.split('__')[1] if '__' in name else name for name in feature_names]
    
    # Create DataFrame for SHAP explanation
    X_test_df = pd.DataFrame(X_test_preprocessed, columns=clean_feature_names)
    
    # Build tree explainer on fitted XGBoost classifier
    classifier = final_pipeline.named_steps['classifier']
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_df)
    
    # SHAP Multiclass Plot
    print("Generating and saving SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    # Make style professional
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot stacked bar chart representing mean impact per class
    shap.summary_plot(
        shap_values, 
        X_test_df, 
        plot_type="bar", 
        class_names=classes, 
        show=False
    )
    
    plt.title("SHAP Feature Importance (Condition Prediction Model)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=12)
    plt.tight_layout()
    
    shap_plot_path = "shap_importance.png"
    plt.savefig(shap_plot_path, dpi=200)
    plt.close()
    print(f"SHAP feature importance visualization saved to: {os.path.abspath(shap_plot_path)}")
    
    # 8. Model Saving with Joblib
    model_path = "phone_condition_model.joblib"
    print(f"\nSaving final fitted pipeline to {model_path}...")
    joblib.dump(final_pipeline, model_path)
    print(f"Model saved successfully to: {os.path.abspath(model_path)}")
    
    # 9. Verify Inference with Saved Pipeline on Unseen Raw Data
    print("\nVerifying inference on a simulated new customer record...")
    loaded_pipeline = joblib.load(model_path)
    
    raw_sample = pd.DataFrame([{
        'brand': 'Apple',
        'model': 'iPhone 13',
        'age_years': 1.2,
        'battery_health_pct': 87,
        'screen_condition': 3,
        'body_condition': 2,
        'functional_issues': 'battery',
        'water_damage_history': 0,
        'repair_history': 0,
        'original_purchase_price': 899.0
    }])
    
    predicted_idx = loaded_pipeline.predict(raw_sample)[0]
    predicted_probs = loaded_pipeline.predict_proba(raw_sample)[0]
    predicted_label = classes[predicted_idx]
    
    print("\nInput Phone Data:")
    for k, v in raw_sample.iloc[0].to_dict().items():
        print(f"  - {k}: {v}")
        
    print(f"\nPredicted Condition: {predicted_label} (Class Index: {predicted_idx})")
    print("Class Probabilities:")
    for idx, cls_name in enumerate(classes):
        print(f"  - {cls_name}: {predicted_probs[idx]*100:.2f}%")
        
    print("\nPipeline execution, tuning, training, and verification completed successfully!")

if __name__ == "__main__":
    main()
