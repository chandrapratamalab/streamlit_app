import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

# 1. Load dataset
df = pd.read_csv('loan_dataset.csv')

# 2. Bersihkan data (hapus missing)
df.dropna(inplace=True)

# 3. Encode target
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# 4. Encode fitur kategorikal
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 2})
df['Married'] = df['Married'].map({'No': 1, 'Yes': 2})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 2})
df['Self_Employed'] = df['Self_Employed'].map({'No': 1, 'Yes': 2})
df['Property_Area'] = df['Property_Area'].map({'Rural': 1, 'Urban': 2, 'Semiurban': 3})

# 5. One-hot encoding manual untuk Dependents
df['class_0'] = df['Dependents'].apply(lambda x: 1 if x == '0' else 0)
df['class_1'] = df['Dependents'].apply(lambda x: 1 if x == '1' else 0)
df['class_2'] = df['Dependents'].apply(lambda x: 1 if x == '2' else 0)
df['class_3'] = df['Dependents'].apply(lambda x: 1 if x == '3+' else 0)

# 6. One-hot encoding untuk Property_Area
df['Rural'] = df['Property_Area'].apply(lambda x: 1 if x == 1 else 0)
df['Urban'] = df['Property_Area'].apply(lambda x: 1 if x == 2 else 0)
df['Semiurban'] = df['Property_Area'].apply(lambda x: 1 if x == 3 else 0)

# 7. Fitur yang dipakai (harus sesuai Streamlit)
features = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Gender', 'Married',
    'class_0', 'class_1', 'class_2', 'class_3',
    'Education', 'Self_Employed',
    'Rural', 'Urban', 'Semiurban'
]

X = df[features]
y = df['Loan_Status']

# 8. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 10. Simpan model ke file joblib
dump(model, 'Random_Forest.joblib')

print("✅ Model berhasil disimpan sebagai 'Random_Forest.joblib'")
