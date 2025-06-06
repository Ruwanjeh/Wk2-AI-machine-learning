import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from malaria import load_data, preprocess_data, analyze_regional_trends

st.set_page_config(page_title="Malaria Prediction System", page_icon="🦟")

st.title("Malaria Prediction System")
st.subheader("SDG 3: Good Health and Well-being")

# Load data
@st.cache_data
def load_cached_data():
    return load_data()

df = load_cached_data()
yearly_data = preprocess_data(df)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Regional Analysis", "Predictions"])

if page == "Overview":
    st.header("Global Malaria Overview")
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Countries", df['Country'].nunique())
    with col2:
        st.metric("Years Covered", f"{df['Year'].min()} - {df['Year'].max()}")
    with col3:
        st.metric("Latest Year Cases", f"{yearly_data['No. of cases'].iloc[-1]:,.0f}")
    
    # Plot global trends
    st.subheader("Global Malaria Cases Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(yearly_data['Year'], yearly_data['No. of cases'], marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Cases')
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(f"Total deaths in most recent year: {yearly_data['No. of deaths'].iloc[-1]:,.0f}")
    st.write(f"Average case fatality rate: {yearly_data['case_fatality_rate'].mean():.2f}%")

elif page == "Regional Analysis":
    st.header("Regional Analysis")
    
    # Regional trends
    regional_data, regional_fig = analyze_regional_trends(df)
    st.pyplot(regional_fig)
    plt.close(regional_fig)
    
    # Display regional metrics
    st.subheader("Cases by Region (Most Recent Year)")
    latest_year = regional_data['Year'].max()
    latest_regional = regional_data[regional_data['Year'] == latest_year]
    
    for _, row in latest_regional.iterrows():
        st.write(f"**{row['WHO Region']}**:")
        st.write(f"- Cases: {row['No. of cases']:,.0f}")
        st.write(f"- Fatality Rate: {row['case_fatality_rate']:.2f}%")
        st.write("---")

elif page == "Predictions":
    st.header("Malaria Predictions")
    
    # Load model
    try:
        model = joblib.load('malaria_prediction_model.pkl')
        st.success("Model loaded successfully!")
        
        # Input parameters
        st.subheader("Input Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Year", min_value=2020, max_value=2030, value=2023)
        
        with col2:
            region = st.selectbox("WHO Region", df['WHO Region'].unique())
        
        # Make prediction
        if st.button("Predict"):
            # Prepare input data
            input_data = pd.DataFrame({
                'Year': [year],
                'time_index': [year - df['Year'].min()],
                'WHO Region': [region]
            })
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Display prediction
            st.subheader("Prediction Results")
            st.write(f"Predicted number of cases: {prediction[0]:,.0f}")
            
    except Exception as e:
        st.error("Model not found. Please run the training script first.")

# Footer
st.markdown("---")
st.markdown("""
### About
This application is part of a project addressing SDG 3 (Good Health and Well-being).
It uses machine learning to analyze and predict malaria trends globally.

### Ethical Considerations
- Data may be biased towards countries with better healthcare reporting
- Predictions should be used as one of many tools for healthcare planning
- Regular model updates are needed to account for changing conditions
""") 