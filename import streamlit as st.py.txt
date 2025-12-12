import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- File Name for the Scaler ---
SCALER_FILE = 'scaler.pkl' 

## 1. Function to Load the Scaler
@st.cache_resource
def load_scaler(file_path):
    """Loads the StandardScaler object from the pickle file."""
    try:
        with open(file_path, 'rb') as f:
            scaler = pickle.load(f)
        st.success(f"✅ StandardScaler loaded successfully from {file_path}")
        return scaler
    except FileNotFoundError:
        st.error(f"🚨 Error: Scaler file not found at {file_path}")
        return None
    except Exception as e:
        st.error(f"🚨 Error loading scaler: {e}")
        return None

## 2. Main Streamlit Application Logic
def app():
    st.title('StandardScaler Demo in Streamlit')
    st.markdown('Load a pre-trained `StandardScaler` and apply it to sample data.')

    # Load the scaler
    scaler = load_scaler(SCALER_FILE)

    if scaler is None:
        st.stop() # Stop if scaler loading failed

    # Extract the feature names (based on the pickle content)
    # The feature names are extracted from the pickle file data 
    feature_names = ['Year', 'Present_Price', 'Kms_Driven', 'Owner'] 
    
    st.subheader('Enter Data to Scale')
    
    # Create input fields for the features
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # The 'Year' feature will be the first column in the scaler's input
        input_year = st.number_input('Year', min_value=1900, max_value=2025, value=2015)
    with col2:
        # The 'Present_Price' feature will be the second column
        input_price = st.number_input('Present_Price (Lakhs)', min_value=0.0, max_value=50.0, value=9.85, step=0.01)
    with col3:
        # The 'Kms_Driven' feature will be the third column
        input_kms = st.number_input('Kms_Driven', min_value=0, max_value=500000, value=30000)
    with col4:
        # The 'Owner' feature will be the fourth column
        input_owner = st.number_input('Owner Count', min_value=0, max_value=3, value=0)

    # 3. Prepare data for scaling
    # Create a DataFrame or Numpy array from the user inputs
    input_data = pd.DataFrame([[input_year, input_price, input_kms, input_owner]],
                              columns=feature_names)

    st.subheader('Unscaled Input Data')
    st.dataframe(input_data)
    
    if st.button('Scale Data'):
        try:
            # 4. Apply the loaded scaler's transform method
            scaled_array = scaler.transform(input_data)
            
            # Convert the scaled output back to a DataFrame for display
            scaled_df = pd.DataFrame(scaled_array, columns=[f"{name}_scaled" for name in feature_names])

            st.subheader('Scaled Output Data (Z-scores)')
            st.dataframe(scaled_df)

            # Optional: Display the scaler's internal values for transparency
            st.subheader('Scaler Internal Parameters')
            st.write(f"**Mean values:** {scaler.mean_}") # mean values for each feature 
            st.write(f"**Scale (Std Dev) values:** {scaler.scale_}") # scale values for each feature [cite: 2]

        except Exception as e:
            st.error(f"🚨 An error occurred during scaling: {e}")

if __name__ == '__main__':
    app()