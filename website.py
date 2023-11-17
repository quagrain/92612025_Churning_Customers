import pandas as pd
import streamlit
from streamlit import radio, number_input
import pickle 
from keras.models import load_model
import numpy as np

streamlit.set_page_config(
    page_title="Customer Churn Predictor",
    layout="centered"
)

streamlit.header("Customer Churn Predict", divider=True)
online_security = radio('Online Security', ['Yes', 'No', 'No internet service'])
online_backup = radio('Online Backup', ['No', 'Yes', 'No internet service'])
tech_support = radio('TechSupport', ['No', 'Yes', 'No internet service'])
contract = radio('Contract', ['Month-to-month', 'Two year', 'One year'])
paperless_billing = radio('Paperless Billing', ['Yes', 'No'])
monthly_charges = number_input('Monthly Charges')
total_charges = number_input('Total Charges')
tenure = number_input('Tenure')

submit = streamlit.button("Submit")

inputs = [tenure, online_security, online_backup, tech_support, contract, paperless_billing, monthly_charges, total_charges]
data = pd.DataFrame(np.array(inputs).reshape(1,-1), columns=['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'Contract', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'tenure'])


model = load_model('best_model.h5')
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encode.pkl", "rb"))
print(encoder)

for i in data.columns:
    if i in encoder.keys():
        data[i] = encoder[i].transform(data[i])

data = scaler.transform(data)



if submit:
    pred = model.predict(data)
    if pred[0][0] >= 0.5:
        streamlit.write('Customer will churn')
    else:
        streamlit.write('Customer will not churn')