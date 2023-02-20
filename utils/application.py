import numpy as np
import pandas as pd 
import streamlit as st
from utils.reader_config import config_reader 
import pickle
import os
from datetime import date
from sklearn.preprocessing  import LabelEncoder, MinMaxScaler

# Import of parameters
config = config_reader('config/config.json')

# loading saved model
with open(os.path.join(config.model_path, 'model_rf_opt.pkl'), 'rb') as f:
    model = pickle.load(f)
    
# Loading current date    
today = date.today().isoformat().split('-')
yyyy , mm, dd = today

st.write("""
# Whether client will deposit money after the marketing campaign ?
""")

# st.sidebar.header('User Input Parameters')
date_full = st.sidebar.date_input( "Input contact date", date(int(yyyy), int(mm), int(dd)))
Age = st.sidebar.slider('Age', 18, 95, 20)
Job = st.sidebar.selectbox(
    "Select customer job", 
    ('admin.', 'technician', 'services', 'management','blue-collar',  'entrepreneur', 'housemaid', 'self-employed', 'student','retired', 'unemployed')
)
Marital = st.sidebar.radio("Select Marital status", ('married', 'single', 'divorced'), horizontal=True)
Education = st.sidebar.selectbox(
    "Select education", 
    ('basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')
)
Contact = st.sidebar.radio("Contact type", ('cellular', 'telephone', 'unknown'), horizontal=True)
#Month = st.radio( "Select month",  ('jan','feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))

Balance = st.sidebar.number_input('Balance', -3000, 5000, 0)
st.sidebar.markdown("---")

Default = st.sidebar.radio('Has credit in default?', ('no','yes', 'unknown'), horizontal=True)
Housing = st.sidebar.radio('Has housing loan?', ('no','yes', 'unknown'), horizontal=True)
Loan = st.sidebar.radio('Has personal loan?', ('no','yes', 'unknown'), horizontal=True)
Campaign = st.sidebar.number_input('Number of contacts during this campaign', 1, 65, 1)
Pdays = st.sidebar.number_input('Days since the client was contacted while previous campaign', -1, 800, 1)
Previous= st.sidebar.number_input('Number of contacts performed during this campaign', -1, 800, 1)
Poutcome = st.sidebar.radio('Previous campaign outcome', ('unknown', 'other', 'failure', 'success'), horizontal=True)


day = st.write(date_full.day)
month = st.write(date_full.month)

data = {
    'Age': Age,
    'Job': Job,
    
    #'Duration':Duration,
    'Marital':Marital,
    'Education':Education,
    'Contact':Contact,
    'Month': month,
    'Day': day,
    'Balance': Balance,
    'Default': Default,  
    'Housing': Housing,
    'Loan': Loan,
    'Campaign': Campaign,
    'Pdays':Pdays,
    'Previous':Previous,
    'Poutcome':Poutcome     
}

df = pd.DataFrame(data, index=[0]) 

st.subheader('User Input parameters')
st.write(df)

# preprocessing---------------------------
le = LabelEncoder()
df['education'] = le.fit_transform(df['Education'])
df['age_group'] = pd.cut(df.Age, [0,30,40,50,60,9999], labels = ['<30','30-40','40-50','50-60','60+'])

df['age_group'] = le.fit_transform(df['age_group'])

data_encoded = pd.get_dummies(df) #[['job', 'marital', 'contact', 'month', 'poutcome']]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(data_encoded) 

# Prediction of probabilities:
y_test_prob = model.predict_proba(df_scaled)
print(y_test_prob) #.round(2)


st.subheader('Predicted customer status:')
res = 'Deposit' if np.argmax(y_test_prob) == 1 else 'Not deposit'
st.write('👉', res)

#st.subheader('Probability of exit')
#st.write(y_new_proba_predict[0][1].round(2)*100, '%')
