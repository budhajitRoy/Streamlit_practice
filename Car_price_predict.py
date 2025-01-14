import streamlit as st 
import pickle
import numpy as np
import joblib

# users passing inputs through UI, get these inputs
st.title('Car Price Prediction')

km_driven = st.number_input('Enter km driven', 70000.00)
mileage = st.number_input('Mileage', 18.60)
engine = st.number_input('Engine', 1197.00)
max_power = st.number_input('Max power', 81.83)
age = st.slider("Age of the car", 0, 20, 20)
# age = st.number_input('Age', 6.00)

test_data = np.array([km_driven, mileage, engine, max_power, age]).reshape(1, -1)

# features to use to predict price : ['km_driven', 'mileage', 'engine', 'max_power', 'age']
# sample test data = [[ 0.65108157, -0.06884605, -0.13477578, -0.14170849, -0.69128483]]

# original values
ori_data = [[70000.00, 18.60, 1197.00, 81.83, 6.00]]

# test_data  = [[0.65108157, -0.06884605, -0.13477578, -0.14170849, -0.69128483]]

# pass these to the model predict function, get the predicted price and display

# Load the model
with open('car_linear_model.pkl', 'rb') as file:
    car_model = pickle.load(file)

# load scalar object
scalar = joblib.load('scalar.pkl')

# define function predict
# def predict_price(test_data):  #('km_driven', 'mileage', 'engine', 'max_power', 'age'):
#     predicted_price = car_model.predict(test_data)
#     st.write("Predicted Price : " + str(np.round(predicted_price[0], 2)) + ' lakh(s)')

def predict_price(ori_data):  #using original data:
    scaled_data = scalar.transform(ori_data)
    # st.write(scaled_data)
    predicted_price = car_model.predict(scaled_data)
    st.write("Predicted Price : " + str(np.round(predicted_price[0], 2)) + ' lakh(s)')

if st.button('Predict'):
    predict_price(test_data)