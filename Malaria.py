# Malaria Outbreak Prediction System
# SDG 3: Good Health and Well-being - Reduce Malaria Cases

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load dataset (example structure - replace with real malaria data)
def load_data():
    """
    Load malaria dataset from estimated_numbers.csv.
    The CSV should have columns: year, month, rainfall, temperature, humidity, cases
    """
    csv_path = os.path.join(os.path.dirname(__file__), "estimated_numbers.csv")
    df = pd.read_csv(csv_path)
    return df

# Feature engineering
def preprocess_data(df):
    """Create additional time-based features"""
    # Create time features
    df['time_index'] = (df['year'] - 2020) * 12 + df['month']
    
    # Lag features (previous month's cases)
    df['prev_cases'] = df['cases'].shift(1)
    df['prev_cases'].fillna(method='bfill', inplace=True)
    
    # Seasonality features
    df['rainy_season'] = df['month'].apply(lambda x: 1 if 4 <= x <= 9 else 0)
    
    return df

# Train and evaluate model
def train_model(X_train, y_train, X_test, y_test):
    """Train Random Forest model and evaluate performance"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"- MAE: {mae:.2f}")
    print(f"- R² Score: {r2:.2f}")
    
    # Feature importance
    feature_imp = pd.Series(model.feature_importances_,
                           index=X_train.columns).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.title('Feature Importance')
    plt.show()
    
    return model

# Predict future cases
def predict_future(model, last_known_data, months_to_predict):
    """Generate predictions for future months"""
    future_predictions = []
    current_data = last_known_data.copy()
    
    for month in range(1, months_to_predict + 1):
        # Update time index
        current_data['time_index'] += 1
        current_month = current_data['month'].iloc[0]
        current_year = current_data['year'].iloc[0]
        # Update month and year
        if current_month < 12:
            new_month = current_month + 1
            new_year = current_year
        else:
            new_month = 1
            new_year = current_year + 1
        current_data['month'] = new_month
        current_data['year'] = new_year
        
        # Predict cases
        pred = model.predict(current_data.drop('cases', axis=1))
        
        # Update previous cases for next prediction
        current_data['prev_cases'] = pred[0]
        current_data['rainy_season'] = 1 if 4 <= current_data['month'] <= 9 else 0
        
        future_predictions.append({
            'year': current_data['year'].values[0],
            'month': current_data['month'].values[0],
            'predicted_cases': pred[0]
        })
    
    return pd.DataFrame(future_predictions)

def main():
    # Load and prepare data
    print("Loading and preprocessing data...")
    df = load_data()
    df = preprocess_data(df)
    
    # Split features and target
    X = df.drop('cases', axis=1)
    y = df['cases']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save model for future use
    joblib.dump(model, 'malaria_prediction_model.pkl')
    
    # Predict future cases (next 6 months)
    print("\nGenerating future predictions...")
    last_known = df.iloc[-1:].copy()
    future = predict_future(model, last_known, 6)
    
    print("\nPredicted Malaria Cases for Next 6 Months:")
    print(future)
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(df['time_index'], df['cases'], label='Historical Cases')
    plt.plot(future['time_index'], future['predicted_cases'], 
             'r--', label='Predicted Cases')
    plt.title('Malaria Cases Prediction')
    plt.xlabel('Time Index')
    plt.ylabel('Number of Cases')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()