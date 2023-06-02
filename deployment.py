# A simple streamlit app that uses the models

import streamlit as st
from joblib import load
import torch
import torch.nn as nn
from model import Net


        
# load the models
rf = load('model.joblib')
nn = Net()
nn.load_state_dict(torch.load("model.pt"))

# simple interface
st.title('Revenue Prediction')

weather = st.selectbox('Weather', ['sunny', 'cloudy', 'rainy'])
temperature = st.slider('Temperature', 0, 40, 20)
humidity = st.slider('Humidity', 0, 100, 50)
wind = st.slider('Wind', 0, 40, 20)

# button to predict
if st.button('Predict'):
    
    # map weather to integers
    weather_map = {'sunny': 0, 'cloudy': 1, 'rainy': 2}

    # predict
    
    prediction_rf = rf.predict([[weather_map[weather], temperature, humidity, wind]])
    prediction_nn = nn(torch.tensor([weather_map[weather], temperature, humidity, wind], dtype=torch.float32))
    
    st.write('Random Forest Prediction: ', prediction_rf[0])
    st.write('Neural Network Prediction: ', prediction_nn.argmax().item())
    