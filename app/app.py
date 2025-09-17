import streamlit as st
import pandas as pd
import joblib

model = joblib.load(r'models\SVM_heart.pkl')
scaler = joblib.load(r'models\scaler.pkl')
columns = joblib.load(r'models\columns.pkl')

st.title('Heart Stroke Prediction by Shaurya')
st.markdown('Provide the following details')\


# 'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'

age = st.slider('Age', 1, 100, 40)
sex = st.selectbox('sex', ['M', 'F'])
chest_pain = st.selectbox('Chest Pain Type', ['NAP', 'ATA', 'TA', 'ASY'])
RestingBP = st.number_input('Resting Blood Pressure (mmHg)', 80, 200, 120 )
Cholesterol = st.number_input('Cholesterol', 100, 600, 200)
FastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])
RestingECG = st.selectbox(' Resting ECG', ['Normal', 'ST', 'LVH'])
MaxHR = st.slider('Max Heart Rate', 60, 220, 150)
Exercise_angina = st.selectbox('Exercise induced angina', ['Y', 'N'])
Oldpeak = st.slider('Old peak (ST Depression)', 0.0, 6.0, 1.0)
ST_slope = st.selectbox('ST slope', ['Up', 'Flat', 'Down'])

if st.button('Predict'):
    raw_input = {
        'Age' : age,
        'RestingBP' : RestingBP,
        'Cholesterol' : Cholesterol,
        'FastingBS' : FastingBS,
        'MaxHR' : MaxHR,
        'OldPeak' : Oldpeak,
        'Sex' + sex : 1,
        'ChestPainType_' + chest_pain : 1,
        'RestingECG' + RestingECG : 1,
        'ExerciseAngina' + Exercise_angina : 1,
        'ST_slope' + ST_slope : 1

    }
    input_df = pd.DataFrame([raw_input])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]
    input_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.transform(input_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error('Warning, Heart Disease predicted')
    else:

        st.success('No disease')
