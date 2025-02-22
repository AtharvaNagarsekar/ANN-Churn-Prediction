import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

def predict(data):
    data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True, errors='ignore')
    data['Gender'] = le.transform(data['Gender'])
    geo = ohe.transform(data[['Geography']])
    op = pd.DataFrame(geo.toarray(), columns=ohe.get_feature_names_out(['Geography']))
    data = pd.concat([data, op], axis=1)
    data.drop(['Geography'], axis=1, inplace=True)
    data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    ans = model.predict(data)
    return ans[0][0]

model = load_model('model2.h5')

with open('scaler2.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('labelencoder2.pkl', 'rb') as f:
    le = pickle.load(f)
with open('onehotencoder2.pkl', 'rb') as f:
    ohe = pickle.load(f)


st.title('Customer Salary Prediction')
st.write('Customer Salary prediction app.')

geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', le.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember':  [is_active_member],
    'Exited': [exited]
}
input_data = pd.DataFrame(input_data)
if st.button('Predict'):
    result = predict(input_data)
    st.write(f'Predicted Salary: {result}.')

