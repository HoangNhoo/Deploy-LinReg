import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Set Streamlit theme to white
st.set_page_config(layout="centered", page_title="Sales Prediction", initial_sidebar_state="auto")

# Load the dataset
df = pd.read_csv('advertising.csv')
X = df[['TV', 'Radio']]
y = df['Sales']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Streamlit app layout
st.title('Sales Prediction')

# Input fields on one line
col1, col2 = st.columns(2)
with col1:
    tv_input = st.number_input('TV', min_value=0.0, max_value=300.0, step=0.1)
with col2:
    radio_input = st.number_input('Radio', min_value=0.0, max_value=50.0, step=0.1)

# Centered and long Predict button
if st.button('Predict', use_container_width=True):
    if tv_input == 0 and radio_input == 0:
        st.error('Please input TV or Radio ads')
    else:
        input_data = np.array([[tv_input, radio_input]])
        prediction = model.predict(input_data)
        st.success(f'Sales Prediction: {prediction[0]:.3f}')