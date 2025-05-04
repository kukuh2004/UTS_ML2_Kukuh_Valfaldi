import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'Education': ['High School', 'Phd', 'Bachelor', 'High School', 'Phd'] * 10,
    'Job_Title': ['Manager', 'Director', 'Manager', 'Director', 'Analyst'] * 10,
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'] * 10,
    'Location': ['Urban', 'Suburban', 'Suburban', 'Rural', 'Urban'] * 10,
    'Salary': [84620.053665, 142591.255894, 97800.255404, 96834.671282, 132157.786175] * 10
}
df = pd.DataFrame(data)

le_edu = LabelEncoder()
le_job = LabelEncoder()
le_gender = LabelEncoder()
le_loc = LabelEncoder()

df['Education'] = le_edu.fit_transform(df['Education'])
df['Job_Title'] = le_job.fit_transform(df['Job_Title'])
df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Location'] = le_loc.fit_transform(df['Location'])

X = df.drop("Salary", axis=1)
y = df["Salary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

st.title("💼 Prediksi Gaji")

edu = st.selectbox("Education", le_edu.classes_)
job = st.selectbox("Job Title", le_job.classes_)
gender = st.selectbox("Gender", le_gender.classes_)
loc = st.selectbox("Location", le_loc.classes_)

if st.button("Prediksi"):
    input_data = [[
        le_edu.transform([edu])[0],
        le_job.transform([job])[0],
        le_gender.transform([gender])[0],
        le_loc.transform([loc])[0]
    ]]
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    st.success(f"Predicted Salary: ${pred:,.2f}")