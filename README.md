# Malaria Prediction System 🌍

## Project Overview
This project addresses the United Nations Sustainable Development Goal 3 (Good Health and Well-being) by developing a machine learning system to analyze and predict malaria trends globally. The system uses historical WHO data to identify patterns and make predictions that can help in healthcare planning and resource allocation.

## Features
- Global malaria trend analysis
- Regional breakdown of cases and fatalities
- Multiple machine learning models (Random Forest, Linear Regression, XGBoost)
- Interactive web interface using Streamlit
- Cross-validation and model comparison
- Ethical considerations and limitations documentation

## Technical Implementation
- **Data Source**: WHO malaria data (estimated_numbers.csv)
- **Models**: 
  - Random Forest Regressor (Best performing model)
  - Linear Regression
  - XGBoost
- **Metrics**: MAE, R² Score, RMSE
- **Visualization**: Matplotlib, Seaborn
- **Web Interface**: Streamlit

## Detailed Setup Instructions

### 1. Environment Setup

#### Option 1: Using Python Virtual Environment (Recommended)
```bash
# Create a new virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using Anaconda
```bash
# Create a new conda environment
conda create -n malaria python=3.11
conda activate malaria

# Install dependencies
conda install pandas numpy scikit-learn xgboost matplotlib seaborn
pip install streamlit joblib
```

### 2. Project Structure
```
malaria-prediction/
├── malaria.py          # Main script for data processing and model training
├── app.py             # Streamlit web interface
├── estimated_numbers.csv  # WHO malaria dataset
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

### 3. Running the Project

1. **Train the Model**:
   ```bash
   python malaria.py
   ```
   This will:
   - Load and preprocess the malaria data
   - Train multiple models
   - Generate performance metrics
   - Save the best model (Random Forest)
   - Create visualization plots

2. **Launch the Web Interface**:
   ```bash
   streamlit run app.py
   ```
   This will:
   - Start a local web server
   - Open the interactive dashboard in your browser
   - Allow you to explore data and make predictions

## How It Works

### Data Processing
1. The system loads WHO malaria data from `estimated_numbers.csv`
2. Data is preprocessed to:
   - Clean and normalize values
   - Handle missing data
   - Create time-based features
   - Calculate regional statistics

### Model Training
1. Multiple models are trained:
   - Random Forest (Best performing)
   - Linear Regression
   - XGBoost
2. Models are evaluated using:
   - Mean Absolute Error (MAE)
   - R² Score
   - Root Mean Square Error (RMSE)
   - Cross-validation scores

### Web Interface
The Streamlit app provides three main sections:
1. **Overview**:
   - Global malaria trends
   - Key statistics
   - Historical data visualization

2. **Regional Analysis**:
   - Regional breakdown of cases
   - Fatality rates by region
   - Regional trends over time

3. **Predictions**:
   - Interactive prediction interface
   - Year and region selection
   - Predicted case numbers

## Dependencies
```
pandas>=1.3.0        # Data manipulation and analysis
numpy>=1.20.0        # Numerical computing
scikit-learn>=0.24.0 # Machine learning algorithms
xgboost>=1.4.0       # Gradient boosting framework
matplotlib>=3.4.0    # Data visualization
seaborn>=0.11.0      # Statistical data visualization
streamlit>=0.84.0    # Web interface
joblib>=1.0.0        # Model persistence
```

## Troubleshooting

### Common Issues and Solutions

1. **Model Not Found Error**:
   - Ensure you've run `python malaria.py` before starting the web app
   - Check if `malaria_prediction_model.pkl` exists in the project directory

2. **Import Errors**:
   - Verify all dependencies are installed: `pip list`
   - Check if you're in the correct virtual environment
   - Ensure all files are in the correct directory

3. **Plotting Issues**:
   - Make sure matplotlib is properly installed
   - Check if you're running in an environment that supports GUI

4. **Memory Issues**:
   - Reduce the dataset size if working with limited memory
   - Use smaller batch sizes for training

## Ethical Considerations
1. **Data Bias**:
   - Data may be biased towards countries with better healthcare reporting
   - Some regions may have underreported cases
   - Model should be used with caution in regions with limited data

2. **Fairness and Sustainability**:
   - Aims to promote health equity
   - Helps target interventions in high-risk areas
   - Supports sustainable healthcare planning

3. **Limitations**:
   - Doesn't account for local healthcare infrastructure
   - Environmental factors not considered
   - Cultural and social factors not included

## Future Improvements
- Integration with real-time weather data
- Addition of more environmental factors
- Implementation of deep learning models
- Enhanced regional analysis
- Real-time data updates

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Pitch deck presentation
https://aigroup25.my.canva.site/

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- World Health Organization for providing the malaria data
- United Nations for the Sustainable Development Goals framework

  https://app.screencast.com/WDeamGn3WOaWj
