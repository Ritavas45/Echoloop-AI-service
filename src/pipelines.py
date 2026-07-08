import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class MultiHotIssueEncoder(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to convert a column containing comma-separated 
    functional issues (e.g., "battery,camera") into binary multi-hot features.
    """
    def __init__(self, issues=None):
        self.issues = issues
        
    def fit(self, X, y=None):
        # If issues list is not specified, dynamically extract all unique issues from data
        if self.issues is None:
            unique_issues = set()
            # X can be a DataFrame or Series, convert to flat array of strings
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X_flat = X.values.flatten()
            else:
                X_flat = np.asarray(X).flatten()
            issues_col = pd.Series(X_flat)
            for val in issues_col.dropna():
                if val.lower() != 'none' and val.strip() != '':
                    for item in val.split(','):
                        unique_issues.add(item.strip().lower())
            self.issues_ = sorted(list(unique_issues))
        else:
            self.issues_ = [issue.strip().lower() for issue in self.issues]
        return self
        
    def transform(self, X):
        # Convert input to a 1D flat array to support both single and multi-sample shapes robustly
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values.flatten()
        else:
            X_arr = np.asarray(X).flatten()
            
        n_samples = len(X_arr)
        n_features = len(self.issues_)
        
        # Initialize binary matrix
        output_matrix = np.zeros((n_samples, n_features), dtype=np.int32)
        
        for idx, val in enumerate(X_arr):
            if pd.isna(val) or not isinstance(val, str) or val.lower() == 'none' or val.strip() == '':
                continue
            
            # Split and clean the issues for this phone
            current_issues = [item.strip().lower() for item in val.split(',')]
            for issue_idx, issue in enumerate(self.issues_):
                if issue in current_issues:
                    output_matrix[idx, issue_idx] = 1
                    
        return pd.DataFrame(output_matrix, columns=self.get_feature_names_out())
        
    def get_feature_names_out(self, input_features=None):
        return [f"issue_{issue.replace(' ', '_')}" for issue in self.issues_]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to perform feature engineering inside the pipeline.
    Calculates Price-to-Age Ratio and Log Price from 'original_purchase_price' and 'age_years'.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # X is expected to be a DataFrame/Array with columns [original_purchase_price, age_years]
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            # Align column names if they are shuffled or different
            price = df.iloc[:, 0]
            age = df.iloc[:, 1]
        else:
            df = pd.DataFrame(X)
            price = df.iloc[:, 0]
            age = df.iloc[:, 1]
            
        # Feature 1: Price-Age ratio (add 0.1 to age to prevent division by zero)
        price_age_ratio = price / (age + 0.1)
        
        # Feature 2: Log purchase price (log1p to handle any zero prices gracefully)
        log_price = np.log1p(price)
        
        # Combine into DataFrame
        engineered_df = pd.DataFrame({
            'price_age_ratio': price_age_ratio,
            'log_price': log_price
        })
        
        return engineered_df
        
    def get_feature_names_out(self, input_features=None):
        return ['price_age_ratio', 'log_price']


def build_preprocessing_pipeline():
    """
    Builds and returns the full scikit-learn preprocessing ColumnTransformer.
    """
    numeric_features = ['age_years', 'battery_health_pct', 'original_purchase_price', 'repair_history']
    categorical_features = ['brand', 'model']
    ordinal_features = ['screen_condition', 'body_condition']
    binary_features = ['water_damage_history']
    multi_hot_features = ['functional_issues']
    
    # We pass 'original_purchase_price' and 'age_years' for engineering
    feature_eng_features = ['original_purchase_price', 'age_years']
    
    # 1. Numerical preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Categorical preprocessor (using OneHotEncoder with handle_unknown='ignore')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Ordinal preprocessor
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 4. Binary preprocessor
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    # 5. Multi-hot functional issues preprocessor
    multi_hot_transformer = Pipeline(steps=[
        ('multihot', MultiHotIssueEncoder(issues=['battery', 'camera', 'speaker', 'charging port', 'screen', 'buttons']))
    ])
    
    # 6. Feature engineering preprocessor
    feature_eng_transformer = Pipeline(steps=[
        ('engineer', FeatureEngineer())
    ])
    
    # Assembling ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('bin', binary_transformer, binary_features),
            ('multihot', multi_hot_transformer, multi_hot_features),
            ('eng', feature_eng_transformer, feature_eng_features)
        ],
        remainder='drop'  # Drop any other unmapped columns
    )
    
    return preprocessor

if __name__ == "__main__":
    # Test block
    print("Testing custom transformers and preprocessor pipeline...")
    # Create dummy phone data
    dummy_data = pd.DataFrame({
        'brand': ['Apple', 'Samsung'],
        'model': ['iPhone 13', 'Galaxy S21'],
        'age_years': [1.5, 3.0],
        'battery_health_pct': [88, 72],
        'screen_condition': [3, 1],
        'body_condition': [2, 1],
        'functional_issues': ['battery,speaker', 'none'],
        'water_damage_history': [0, 1],
        'repair_history': [0, 2],
        'original_purchase_price': [899.0, 799.0]
    })
    
    preprocessor = build_preprocessing_pipeline()
    transformed_data = preprocessor.fit_transform(dummy_data)
    feature_names = preprocessor.get_feature_names_out()
    
    print("\nTransformed data shape:", transformed_data.shape)
    print("Feature names:", feature_names)
    print("\nPreprocessed row 1 (Apple iPhone 13):")
    print(transformed_data[0])
    print("\nAll tests completed successfully!")
