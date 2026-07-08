import os
import random
import numpy as np
import pandas as pd

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_mock_phone_data(n_samples=2000):
    brands = ['Apple', 'Samsung', 'Google', 'OnePlus', 'Xiaomi']
    models_by_brand = {
        'Apple': [('iPhone 12', 799), ('iPhone 13', 899), ('iPhone 14 Pro', 1099), ('iPhone SE', 429)],
        'Samsung': [('Galaxy S20', 999), ('Galaxy S21', 799), ('Galaxy S22 Ultra', 1199), ('Galaxy A53', 449)],
        'Google': [('Pixel 5', 699), ('Pixel 6', 599), ('Pixel 7 Pro', 899), ('Pixel 6a', 449)],
        'OnePlus': [('OnePlus 9', 729), ('OnePlus 10 Pro', 899), ('OnePlus Nord 2', 399)],
        'Xiaomi': [('Mi 11', 749), ('Redmi Note 11', 299), ('Xiaomi 12 Pro', 999)]
    }
    
    possible_issues = ['battery', 'camera', 'speaker', 'charging port', 'screen', 'buttons']
    
    data = []
    
    for _ in range(n_samples):
        brand = random.choice(brands)
        model, base_price = random.choice(models_by_brand[brand])
        
        # Original purchase price is slightly varied around base price
        original_purchase_price = float(np.round(base_price + np.random.normal(0, 30), 2))
        
        # Age in years (0.1 to 6.0)
        age = float(np.round(np.random.beta(1.5, 3) * 6.0, 2))
        if age < 0.1: age = 0.1
        
        # Battery health % (highly correlated with age)
        # Typically degrades ~5% to 10% per year
        expected_battery = 100 - (age * np.random.uniform(5, 10))
        # Add some random noise
        battery_health = int(np.clip(expected_battery + np.random.normal(0, 3), 40, 100))
        
        # Screen and body condition (ordinal 0-3: 3=Excellent, 2=Good, 1=Fair, 0=Poor)
        # Also degrades with age and randomly
        screen_cond_prob = np.clip(np.random.dirichlet([2 / (age + 0.5), 1.5, 1.0, 0.5]), 0, 1)
        screen_condition = int(np.random.choice([3, 2, 1, 0], p=screen_cond_prob / screen_cond_prob.sum()))
        
        body_cond_prob = np.clip(np.random.dirichlet([2 / (age + 0.5), 1.5, 1.0, 0.5]), 0, 1)
        body_condition = int(np.random.choice([3, 2, 1, 0], p=body_cond_prob / body_cond_prob.sum()))
        
        # Water damage history (binary)
        water_damage = 1 if random.random() < 0.08 else 0
        
        # Repair history (count: 0 to 3+)
        repair_history = int(np.random.poisson(0.5 + 0.2 * age))
        repair_history = min(repair_history, 4)
        
        # Functional issues (multi-hot: comma-separated list of issues)
        num_issues = 0
        issues = []
        
        # Probability of having functional issues increases with age and bad physical conditions
        issue_chance = 0.1 + 0.1 * age + 0.15 * (3 - screen_condition) + 0.15 * (3 - body_condition)
        issue_chance = min(max(issue_chance, 0.05), 0.9)
        
        if random.random() < issue_chance:
            # Decide how many issues
            num_issues = random.choice([1, 2, 3])
            # Sample issues
            issues = random.sample(possible_issues, num_issues)
            
            # If screen condition is 0, screen is highly likely to be in issues
            if screen_condition == 0 and 'screen' not in issues:
                issues.append('screen')
            # If battery health is very low, battery is highly likely to be in issues
            if battery_health < 70 and 'battery' not in issues:
                issues.append('battery')
                
            issues = list(set(issues)) # remove duplicates if any
            
        functional_issues = ",".join(issues) if issues else "none"
        
        # Determine the Ground Truth label based on conditions
        # We define logical scoring rules so the model can learn high-fidelity decision boundaries
        if water_damage == 1 or battery_health < 55 or age > 5.5 or (screen_condition == 0 and body_condition == 0 and len(issues) >= 3):
            # Water damage, extreme battery wear, very old, or completely broken physical/functional
            label = 'Recycle'
        elif screen_condition == 0 or len(issues) >= 2 or 'screen' in issues or 'camera' in issues or body_condition == 0:
            # Physical screen cracked, body broken, or heavy functional issues require intensive repair
            label = 'Repair'
        elif age > 2.5 or battery_health < 80 or screen_condition == 1 or body_condition == 1 or len(issues) > 0:
            # Minor wear or moderate age or minor issues like speaker/buttons -> needs refurbishing
            label = 'Refurbish'
        else:
            # Young age, great battery, excellent body & screen, no issues -> direct Reuse
            label = 'Reuse'
            
        # Add tiny label noise (2% random misclassification to simulate human logging errors)
        if random.random() < 0.02:
            label = random.choice(['Reuse', 'Refurbish', 'Repair', 'Recycle'])
            
        data.append({
            'brand': brand,
            'model': model,
            'age_years': age,
            'battery_health_pct': battery_health,
            'screen_condition': screen_condition,
            'body_condition': body_condition,
            'functional_issues': functional_issues,
            'water_damage_history': water_damage,
            'repair_history': repair_history,
            'original_purchase_price': original_purchase_price,
            'condition': label
        })
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating simulated tabular dataset...")
    df = generate_mock_phone_data(2000)
    
    # Save the dataframe
    output_path = "./phone_tabular_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully with {len(df)} rows.")
    print(f"Saved to: {os.path.abspath(output_path)}")
    print("\nClass distribution:")
    print(df['condition'].value_counts(normalize=True) * 100)
    print("\nSample records:")
    print(df.head(3))
