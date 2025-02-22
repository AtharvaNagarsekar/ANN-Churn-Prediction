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
    if ans > 0.5:
        return f'The customer is going to churn.'
    else:
        return 'The customer is going to not going to churn'

model = load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('le.pkl', 'rb') as f:
    le = pickle.load(f)
with open('ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

st.title('Customer Churn Prediction')
st.write('Customer churn prediction app.')

geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', le.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
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
    'EstimatedSalary': [estimated_salary]
}
input_data = pd.DataFrame(input_data)
if st.button('Predict'):
    result = predict(input_data)
    st.write(f'Prediction: {result}.')

