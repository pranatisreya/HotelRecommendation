import streamlit as st
import numpy as np
import pandas as pd
#import joblib
import pickle
import os
#from xgboost import XGBRegressor

# Loading the pre-trained model
#path = os.path.dirname(os.path.abspath(__file__))
#rec = joblib.load(os.path.join(path, 'HRec.pkl'))

rec = pickle.load(open('HRec.pkl', 'rb'))

# Streamlit app title
st.title("Hotel Recommendations")

# Injecting CSS styles
st.markdown("""
<style>
body {
    background-color: #f0f0f0;
    font-family: Arial, sans-serif;
}

h1 {
    text-align: center;
    background-color: #94c5f8;
    color: #fff;
    padding: 20px;
}

.container {
    margin: 20px auto;
    padding: 20px;
    background-color: #fff;
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
}

form {
    max-width: 500px;
    margin: 0 auto;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

input[type="text"],
input[type="number"],
select {
    width: 100%;
    padding: 10px;
    margin-bottom: 15px;
    border: 1px solid #ccc;
    border-radius: 5px;
}

input[type="submit"] {
    background-color: #94c5f8;
    color: #fff;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
}

.image-container {
    text-align: center;
    margin-top: 20px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
}
table, th, td {
    border: 1px solid #ccc;
}
th, td {
    padding: 10px;
    text-align: left;
}

th {
    background-color: #94c5f8;
    color: #fff;
}

tr:nth-child(even) {
    background-color: #f2f2f2;
}

@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    input[type="text"],
    input[type="number"],
    select {
        margin-bottom: 10px;
    }
}
</style>
""", unsafe_allow_html=True)

# User input section
st.header("Enter Your Preferences")

state = st.selectbox("Enter state name:", ["Banglore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])
ra = st.number_input("Enter rating required (0-5):", min_value=0.0, max_value=5.0)

description_options = ["Average", "Excellent", "Good", "Poor", "Very Good", "No value"]
d_value= st.selectbox("Enter the description of hotel expected:", description_options)
d = description_options.index(d_value)

re = st.number_input("Number of reviews expected for hotel (0-15000):", min_value=0, max_value=15000)
s = st.selectbox("The star rating expected for hotel:", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
p = st.number_input("The price expected for hotel:")
t = st.number_input("The tax expected for hotel:")


if state == 'Banglore':
    Banglore = 1.0
    Chennai = Delhi = Hyderabad = Kolkata = Mumbai = 0.0
elif state == 'Chennai':
    Chennai = 1.0
    Banglore = Delhi = Hyderabad = Kolkata = Mumbai = 0.0
elif state == 'Delhi':
    Delhi = 1.0
    Banglore = Chennai = Hyderabad = Kolkata = Mumbai = 0.0
elif state == 'Hyderabad':
    Hyderabad = 1.0
    Banglore = Chennai = Delhi = Kolkata = Mumbai = 0.0
elif state == 'Kolkata':
    Kolkata = 1.0
    Banglore = Chennai = Delhi = Hyderabad = Mumbai = 0.0
elif state == 'Mumbai':
    Mumbai = 1.0
    Banglore = Chennai = Delhi = Hyderabad = Kolkata = 0.0

# Predict button
if st.button("Predict"):
    # Predict the hotel using the pre-trained model
    ip = np.array([[Banglore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai, ra, d, re, s, p, t]],dtype=object)
    predicted_Hrec = rec.predict(ip)[0]

    df = pd.read_csv('D:\mp_project\mp_project\static\hotel_data.csv')
    predicted_Hrec1 = df.loc[int(predicted_Hrec)]
    c = ['Hotel Name', 'Rating', 'Rating Description', 'Reviews', 'Star Rating', 'Location', 'Price', 'Tax']
    data1 = [{'column': column, 'value': predicted_Hrec1[column]} for column in c]

    # Display the results
    st.header("Recommended Hotels:")
    for item in data1:
        if item['value'] is not None:
            st.write(f"{item['column']}: {item['value']}")

# Display an image
st.image("D:\mp_project\mp_project\static\image.jpg", caption="Image", use_column_width=True)
