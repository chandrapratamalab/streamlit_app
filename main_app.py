import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import base64

# Konversi Yes/No
@st.cache_data
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict.get(val, 0)

# Ambil nilai dari dictionary
def get_value(val, my_dict):
    return my_dict.get(val, 0)

# Sidebar halaman
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])

# ===========================
# ===== HALAMAN HOME ========
# ===========================
if app_mode == 'Home':
    st.title('LOAN PREDICTION :')
    st.image('loan_image.jpg')
    st.markdown('### Dataset:')
    data = pd.read_csv('loan_dataset.csv')
    st.write(data.head())

    st.markdown('### Applicant Income VS Loan Amount')
    st.bar_chart(data[['ApplicantIncome', 'LoanAmount']].head(20))

# ===============================
# ===== HALAMAN PREDIKSI ========
# ===============================
elif app_mode == 'Prediction':
    st.image('slider-short-3.jpg')
    st.subheader('YOU need to fill all necessary informations in order to get a reply to your loan request!')
    st.sidebar.header("Informations about the client:")

    # Kamus input
    gender_dict = {"Male": 1, "Female": 2}
    feature_dict = {"No": 1, "Yes": 2}
    edu = {'Graduate': 1, 'Not Graduate': 2}
    prop = {'Rural': 1, 'Urban': 2, 'Semiurban': 3}

    # Input dari sidebar
    ApplicantIncome = st.sidebar.slider('ApplicantIncome', 0, 10000, 0)
    CoapplicantIncome = st.sidebar.slider('CoapplicantIncome', 0, 10000, 0)
    LoanAmount = st.sidebar.slider('LoanAmount in K$', 9.0, 700.0, 200.0)
    Loan_Amount_Term = st.sidebar.selectbox('Loan_Amount_Term', (12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0))
    Credit_History = st.sidebar.radio('Credit_History', (0.0, 1.0))
    Gender = st.sidebar.radio('Gender', tuple(gender_dict.keys()))
    Married = st.sidebar.radio('Married', tuple(feature_dict.keys()))
    Self_Employed = st.sidebar.radio('Self Employed', tuple(feature_dict.keys()))
    Dependents = st.sidebar.radio('Dependents', options=['0', '1', '2', '3+'])
    Education = st.sidebar.radio('Education', tuple(edu.keys()))
    Property_Area = st.sidebar.radio('Property_Area', tuple(prop.keys()))

    # One-hot untuk Dependents
    class_0 = 1 if Dependents == '0' else 0
    class_1 = 1 if Dependents == '1' else 0
    class_2 = 1 if Dependents == '2' else 0
    class_3 = 1 if Dependents == '3+' else 0

    # One-hot untuk Property_Area
    Rural = 1 if Property_Area == 'Rural' else 0
    Urban = 1 if Property_Area == 'Urban' else 0
    Semiurban = 1 if Property_Area == 'Semiurban' else 0

    # Gabungkan jadi input vector
    feature_list = [
        ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
        get_value(Gender, gender_dict), get_fvalue(Married),
        class_0, class_1, class_2, class_3,
        get_value(Education, edu), get_fvalue(Self_Employed),
        Rural, Urban, Semiurban
    ]
    single_sample = np.array(feature_list).reshape(1, -1)

    # Tombol prediksi
    if st.button("Click to Predict"):
        # Animasi GIF
        with open("6m-rain.gif", "rb") as f:
            data_url = base64.b64encode(f.read()).decode("utf-8")

        with open("green-cola-no.gif", "rb") as f:
            data_url_no = base64.b64encode(f.read()).decode("utf-8")

        # Load model (joblib)
        loaded_model = load('Random_Forest.joblib')

        # Prediksi
        prediction = loaded_model.predict(single_sample)

        # Output
        if prediction[0] == 0:
            st.error('According to our calculations, you will NOT get the loan from the bank.')
            st.markdown(f'<img src="data:image/gif;base64,{data_url_no}" alt="rejected gif">', unsafe_allow_html=True)
        else:
            st.success('Congratulations! You will get the loan from the bank.')
            st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="approved gif">', unsafe_allow_html=True)
