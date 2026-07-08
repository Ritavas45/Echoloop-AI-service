import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

def generate_synthetic_data(num_samples=500, random_seed=42):
    np.random.seed(random_seed)
    data = []
    
    classes = [0, 1, 2, 3] # 0: Reuse, 1: Refurbish, 2: Repair, 3: Recycle
    
    # We want to generate ~125 samples per class
    samples_per_class = num_samples // 4
    
    for cls in classes:
        for _ in range(samples_per_class):
            if cls == 0: # Reuse
                model_age_months = np.random.randint(1, 10)
                battery_health_pct = np.random.uniform(90.0, 100.0)
                screen_cracked = 0
                functional_issues = 0
                cosmetic_scratches = np.random.choice([0, 1], p=[0.9, 0.1])
                
            elif cls == 1: # Refurbish
                model_age_months = np.random.randint(8, 18)
                battery_health_pct = np.random.uniform(82.0, 92.0)
                screen_cracked = 0
                functional_issues = 0
                cosmetic_scratches = np.random.choice([0, 1, 2], p=[0.2, 0.7, 0.1])
                
            elif cls == 2: # Repair
                model_age_months = np.random.randint(6, 24)
                battery_health_pct = np.random.uniform(78.0, 90.0)
                # Repair implies screen cracked or hardware functional issues exist, but battery/age still okay
                screen_cracked = np.random.choice([0, 1], p=[0.3, 0.7])
                functional_issues = np.random.choice([0, 1], p=[0.4, 0.6])
                if screen_cracked == 0 and functional_issues == 0:
                    screen_cracked = 1 # enforce at least one repairable issue
                cosmetic_scratches = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
                
            else: # Recycle
                model_age_months = np.random.randint(20, 48)
                battery_health_pct = np.random.uniform(50.0, 79.9)
                screen_cracked = np.random.choice([0, 1], p=[0.5, 0.5])
                functional_issues = np.random.choice([0, 1], p=[0.2, 0.8])
                cosmetic_scratches = np.random.choice([0, 1, 2], p=[0.1, 0.3, 0.6])
                
            data.append({
                'model_age_months': model_age_months,
                'battery_health_pct': battery_health_pct,
                'screen_cracked': screen_cracked,
                'functional_issues': functional_issues,
                'cosmetic_scratches': cosmetic_scratches,
                'label': cls
            })
            
    df = pd.DataFrame(data)
    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df

def main():
    print("Generating synthetic phone metadata...")
    df = generate_synthetic_data(num_samples=600)
    
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Print sample distribution
    print("\nDataset Class Distribution:")
    print(y.value_counts())
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining XGBoost classifier...")
    # Initialize XGBClassifier
    # eval_metric='mlogloss' is standard for multi-class classification
    model = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    classes = ['Reuse', 'Refurbish', 'Repair', 'Recycle']
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Save the model
    model_path = "./xgboost_model.json"
    model.save_model(model_path)
    print(f"XGBoost model saved successfully to: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    main()
