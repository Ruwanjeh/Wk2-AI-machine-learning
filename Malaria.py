"""
Malaria Outbreak Prediction System
SDG 3: Good Health and Well-being - Reduce Malaria Cases

This project addresses the United Nations Sustainable Development Goal 3 (Good Health and Well-being)
by analyzing and predicting malaria trends globally. The system uses machine learning to:
1. Analyze historical malaria data
2. Predict future trends
3. Identify patterns in malaria cases and deaths

Ethical Considerations:
1. Data Bias:
   - The data may be biased towards countries with better healthcare reporting systems
   - Some regions may have underreported cases due to limited healthcare infrastructure
   - The model should be used with caution in regions with limited historical data

2. Fairness and Sustainability:
   - The model aims to promote health equity by providing insights for resource allocation
   - Predictions can help target interventions in high-risk areas
   - The system supports sustainable healthcare planning by identifying long-term trends

3. Limitations:
   - The model doesn't account for local healthcare infrastructure
   - Environmental factors (climate change, urbanization) may affect future predictions
   - Cultural and social factors are not considered in the current implementation

4. Responsible Use:
   - Predictions should be used as one of many tools for healthcare planning
   - Local healthcare experts should validate predictions
   - Regular model updates are needed to account for changing conditions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Load dataset (example structure - replace with real malaria data)
def load_data():
    """
    Load malaria dataset from estimated_numbers.csv.
    The CSV has columns: Country, Year, No. of cases, No. of deaths, etc.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "estimated_numbers.csv")
    df = pd.read_csv(csv_path)
    
    # Clean up column names
    df.columns = [col.strip() for col in df.columns]
    
    # Convert case numbers to numeric, removing any brackets and ranges
    df['No. of cases'] = df['No. of cases'].str.extract(r'(\d+)').astype(float)
    df['No. of deaths'] = df['No. of deaths'].str.extract(r'(\d+)').astype(float)
    
    # Fill missing values with 0
    df = df.fillna(0)
    
    return df

# Feature engineering
def preprocess_data(df):
    """Create additional features for analysis"""
    # Group by year to get global totals
    yearly_data = df.groupby('Year').agg({
        'No. of cases': 'sum',
        'No. of deaths': 'sum'
    }).reset_index()
    
    # Create time features
    yearly_data['time_index'] = yearly_data['Year'] - yearly_data['Year'].min()
    
    # Calculate case fatality rate
    yearly_data['case_fatality_rate'] = (yearly_data['No. of deaths'] / 
                                        yearly_data['No. of cases'] * 100)
    
    # Calculate year-over-year changes
    yearly_data['cases_yoy_change'] = yearly_data['No. of cases'].pct_change() * 100
    yearly_data['deaths_yoy_change'] = yearly_data['No. of deaths'].pct_change() * 100
    
    # Fill any remaining NaN values with 0
    yearly_data = yearly_data.fillna(0)
    
    return yearly_data

def train_multiple_models(X_train, y_train, X_test, y_test):
    """Train and compare multiple models"""
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'XGBoost': XGBRegressor(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"\n{name} Performance:")
            print(f"- MAE: {mae:,.2f}")
            print(f"- R² Score: {r2:.2f}")
            print(f"- RMSE: {rmse:,.2f}")
            print(f"- Cross-validation score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        except Exception as e:
            print(f"\nError training {name}: {str(e)}")
            continue
    
    return results

def plot_model_comparison(results):
    """Create comparison plots for different models"""
    # Prepare data for plotting
    metrics = ['mae', 'r2', 'rmse']
    model_names = list(results.keys())
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        axes[i].bar(model_names, values)
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def analyze_regional_trends(df):
    """Analyze malaria trends by WHO region"""
    # Group by region and year
    regional_data = df.groupby(['WHO Region', 'Year']).agg({
        'No. of cases': 'sum',
        'No. of deaths': 'sum'
    }).reset_index()
    
    # Calculate case fatality rate
    regional_data['case_fatality_rate'] = (regional_data['No. of deaths'] / 
                                         regional_data['No. of cases'] * 100)
    
    # Plot trends by region
    fig = plt.figure(figsize=(12, 6))
    for region in regional_data['WHO Region'].unique():
        region_data = regional_data[regional_data['WHO Region'] == region]
        plt.plot(region_data['Year'], region_data['No. of cases'], 
                marker='o', label=region)
    
    plt.title('Malaria Cases by WHO Region Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Cases')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    return regional_data, fig

def main():
    # Load and prepare data
    print("Loading and preprocessing data...")
    df = load_data()
    yearly_data = preprocess_data(df)
    
    # Analyze regional trends
    print("\nAnalyzing regional trends...")
    regional_data, regional_fig = analyze_regional_trends(df)
    
    # Split features and target
    X = yearly_data.drop(['No. of cases', 'No. of deaths'], axis=1)
    y = yearly_data['No. of cases']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train and compare multiple models
    print("\nTraining and comparing multiple models...")
    model_results = train_multiple_models(X_train, y_train, X_test, y_test)
    
    if not model_results:
        print("No models were successfully trained. Please check the data and try again.")
        return
    
    # Plot model comparison
    comparison_fig = plot_model_comparison(model_results)
    
    # Save best model (Random Forest)
    if 'Random Forest' in model_results:
        best_model = model_results['Random Forest']['model']
        joblib.dump(best_model, 'malaria_prediction_model.pkl')
        print("\nModel saved successfully!")
    
    # Create historical data plot
    historical_fig = plt.figure(figsize=(12, 6))
    plt.plot(yearly_data['Year'], yearly_data['No. of cases'], 
             marker='o', label='Historical Cases')
    plt.title('Global Malaria Cases Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Cases')
    plt.legend()
    plt.grid(True)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total number of countries: {df['Country'].nunique()}")
    print(f"Years covered: {df['Year'].min()} to {df['Year'].max()}")
    print(f"Total cases in most recent year: {yearly_data['No. of cases'].iloc[-1]:,.0f}")
    print(f"Total deaths in most recent year: {yearly_data['No. of deaths'].iloc[-1]:,.0f}")
    print(f"Average case fatality rate: {yearly_data['case_fatality_rate'].mean():.2f}%")
    
    # Print regional analysis
    print("\nRegional Analysis:")
    latest_year = regional_data['Year'].max()
    latest_regional = regional_data[regional_data['Year'] == latest_year]
    print("\nCases by Region (Most Recent Year):")
    for _, row in latest_regional.iterrows():
        print(f"{row['WHO Region']}: {row['No. of cases']:,.0f} cases, "
              f"Fatality Rate: {row['case_fatality_rate']:.2f}%")
    
    return {
        'yearly_data': yearly_data,
        'regional_data': regional_data,
        'model_results': model_results,
        'figures': {
            'historical': historical_fig,
            'regional': regional_fig,
            'comparison': comparison_fig
        }
    }

if __name__ == "__main__":
    results = main()
    # Save figures
    for name, fig in results['figures'].items():
        fig.savefig(f'{name}_plot.png')
    plt.close('all')