"""
Streamlit Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="🌍 Tourist Predictor", page_icon="🌍")

st.title("🌍 Tourist Arrival Prediction")
st.markdown("Forecast tourist arrivals based on historical patterns")

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('models/best_model.pkl')
    except:
        st.error("Model not found! Train first: `python src/train.py`")
        return None

model = load_model()

# Inputs
st.sidebar.header("📊 Input Parameters")

date = st.sidebar.date_input("Date", datetime(2020, 1, 1))
area = st.sidebar.selectbox("Tourist Area", list(range(1, 11)))
tourism_idx = st.sidebar.slider("Tourism Index", 0, 5000, 2000)
weather_idx = st.sidebar.slider("Weather Index", 0, 100, 50)

# Extract features
year = date.year
month = date.month
quarter = (month - 1) // 3 + 1
day_of_week = date.weekday()
is_weekend = 1 if day_of_week >= 5 else 0

# Predict button
if st.sidebar.button("🎯 Predict", type="primary"):
    if model:
        # Prepare features
        features = pd.DataFrame({
            'tourist_area': [area],
            'tourism_index': [tourism_idx],
            'weather_index': [weather_idx],
            'year': [year],
            'month': [month],
            'quarter': [quarter],
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'month_sin': [np.sin(2 * np.pi * month / 12)],
            'month_cos': [np.cos(2 * np.pi * month / 12)]
        })
        
        # Load feature columns
        try:
            feature_cols = joblib.load('models/features.pkl')
            for col in feature_cols:
                if col not in features.columns:
                    features[col] = 0
            features = features[feature_cols]
        except:
            pass
        
        try:
            prediction = model.predict(features)[0]
            
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Area", f"Area {area}")
            with col2:
                st.metric("Month", date.strftime('%B'))
            with col3:
                st.metric("Weather", weather_idx)
            
            st.markdown("---")
            st.markdown("### 👥 Predicted Tourist Arrivals")
            st.markdown(f"# {int(prediction):,} tourists")
            
            if is_weekend:
                st.info("📅 Weekend - Higher arrivals expected")
            if month in [7, 8, 12]:
                st.info("🌞 Peak season")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("### 📊 Model Information")
st.markdown("""
**Model**: Random Forest / XGBoost  
**Features**: Time, location, tourism index, weather  
**Use Case**: Tourism planning & forecasting
""")
