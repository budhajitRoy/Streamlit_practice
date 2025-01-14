import streamlit as st 
import pickle
import numpy as np

# users passing inputs through UI, get these inputs
st.title('Car Price Prediction')

km_driven = st.number_input('Enter km driven', 0.65108157)
mileage = st.number_input('Mileage', -0.06884605)
engine = st.number_input('Engine', -0.13477578)
max_power = st.number_input('Max power', -0.14170849)
age = st.number_input('Age', -0.69128483)

test_data = np.array([km_driven, mileage, engine, max_power, age]).reshape(1, -1)

# features to use to predict price : ['km_driven', 'mileage', 'engine', 'max_power', 'age']
# sample test data = [[ 0.65108157, -0.06884605, -0.13477578, -0.14170849, -0.69128483]]

# test_data  = [[0.65108157, -0.06884605, -0.13477578, -0.14170849, -0.69128483]]

# pass these to the model predict function, get the predicted price and display

# Load the model
with open('car_linear_model.pkl', 'rb') as file:
    car_model = pickle.load(file)

# define function predict
def predict_price(test_data):  #('km_driven', 'mileage', 'engine', 'max_power', 'age'):
    predicted_price = car_model.predict(test_data)
    st.write("Predicted Price : " + str(np.round(predicted_price[0], 2)) + ' lakh(s)')

if st.button('Predict'):
    predict_price(test_data)