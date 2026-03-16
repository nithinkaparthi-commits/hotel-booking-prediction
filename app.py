import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Hotel Booking Cancellation Prediction")

lead_time = st.number_input("Lead Time")
weekend_nights = st.number_input("Weekend Nights")
week_nights = st.number_input("Week Nights")
adults = st.number_input("Adults")
children = st.number_input("Children")
babies = st.number_input("Babies")
adr = st.number_input("Average Daily Rate")

if st.button("Predict"):

    features = np.array([[lead_time, weekend_nights, week_nights,
                          adults, children, babies, adr]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Booking will likely be CANCELLED")
    else:
        st.success("Booking will likely NOT be cancelled")