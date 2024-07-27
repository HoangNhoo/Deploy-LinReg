import numpy as np
import streamlit as st
import pickle

st.set_page_config(layout="centered", page_title="Sales Prediction", initial_sidebar_state="auto")

st.title('Sales Prediction')

model = pickle.load(open('sales_model.pkl', 'rb'))
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