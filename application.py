import pandas as pd 
import streamlit as st
 
import pickle
from joblib import load
import os, sys
from datetime import date
from sklearn.preprocessing  import StandardScaler 
sys.path.insert(1, "./config/")

from utils.reader_config import config_reader
# Import of parameters
config = config_reader('config/config.json')
path_models = './models/'

for file in os.listdir(path_models):
    if file.endswith("reduced.pkl"):
        path_reduced_model = os.path.join(path_models, file)
        
# loading saved model
with open(path_reduced_model, 'rb') as f:
    model = pickle.load(f)

    
# Loading current date    
today = date.today().isoformat().split('-')
yyyy , mm, dd = today

st.write("""
## Result prediction of the marketing campaign
""")

#st.markdown('**How to use this app**')
#st.markdown("Select test mode or interactive and see how prediction changes")

st.sidebar.header('User Input Parameters')

st.subheader('Selected parameters')


# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = True


st.sidebar.checkbox("Test mode ON", key="disabled", help="Select test mode or interactive and see how prediction changes")

test_mode = st.sidebar.selectbox(
    "Mode", 
    ('Test_yes','Test_no', 'Interactive mode')
)    

date_full = st.sidebar.date_input( "Input contact date", date(int(yyyy), int(mm), int(dd)),  disabled=st.session_state.disabled)
Age = st.sidebar.slider('Age', 18, 95, 20, disabled=st.session_state.disabled)
Job = st.sidebar.selectbox(
    "Select customer job", 
    ('admin.', 'technician', 'services', 'management','blue-collar',  'entrepreneur', 'housemaid', 'self-employed', 'student','retired', 'unemployed'), disabled=st.session_state.disabled
)
Marital = st.sidebar.radio("Select Marital status", ('married', 'single', 'divorced'), horizontal=True, disabled=st.session_state.disabled)
Education = st.sidebar.radio("Select education", ('primary', 'secondary','tertiary', 'unknown'), horizontal=True, disabled=st.session_state.disabled)

Contact = st.sidebar.radio("Contact type", ('cellular', 'telephone', 'unknown'), horizontal=True, disabled=st.session_state.disabled)

Balance = st.sidebar.number_input('Balance', -3000, 5000, 0, disabled=st.session_state.disabled)
st.sidebar.markdown("---")

Default = st.sidebar.radio('Has credit in default?', ('no','yes', 'unknown'), horizontal=True, disabled=st.session_state.disabled)
Housing = st.sidebar.radio('Has housing loan?', ('no','yes', 'unknown'), horizontal=True, disabled=st.session_state.disabled)
#Loan = st.sidebar.radio('Has personal loan?', ('no','yes', 'unknown'), horizontal=True)
Campaign = st.sidebar.number_input('Number of contacts during this campaign', 1, 65, 1, disabled=st.session_state.disabled)
Pdays = st.sidebar.number_input('Days since the client was contacted while previous campaign', -1, 800, 1, disabled=st.session_state.disabled)
Previous= st.sidebar.number_input('Number of contacts performed during this campaign', -1, 800, 1, disabled=st.session_state.disabled)
Poutcome = st.sidebar.radio('Previous campaign outcome', ('unknown', 'other', 'failure', 'success'), horizontal=True, disabled=st.session_state.disabled)

day = date_full.day
month = date_full.month


data = {
    'age': Age,
    'job': Job,
    'marital':Marital,
    'Education':Education,
    'default': Default,
    'balance': Balance,
    'housing': Housing,
    'Loan': 'no',
    'contact':Contact,
    'month': month,
    'day': day,
    'campaign': Campaign,
    'pdays':Pdays,
    'previous':Previous,
    'poutcome':Poutcome     
}


data_yes = {'age': {2541: 42},
    'job': {2541: 'admin.'},
    'marital': {2541: 'married'},
    'education': {2541: 'primary'},
    'default': {2541: 'no'},
    'balance': {2541: -306.0},
    'housing': {2541: 'yes'},
    'loan': {2541: 'no'},
    'contact': {2541: 'cellular'},
    'day': {2541: 17},
    'month': {2541: 'aug'},
    'campaign': {2541: 1},
    'pdays': {2541: 459},
    'previous': {2541: 1}, 
    'poutcome': {2541: 'unknown'},
}

data_no = {'age': {1865: 47},
    'job': {1865: 'retired'},
    'marital': {1865: 'married'},
    'education': {1865: 'primary'},
    'default': {1865: 'no'},
    'balance': {1865: 1374},
    'housing': {1865: 'no'},
    'loan': {1865: 'yes'},
    'contact': {1865: 'telephone'},
    'day': {1865: 8},
    'month': {1865: 'may'},
    'campaign': {1865: 479},
    'pdays': {1865: 359},
    'previous': {1865: 3},
    'poutcome': {1865: 'failure'},
}

#test_df = pd.DataFrame(data) #.items()

limits = pd.DataFrame({
    'age': [18, 95],
    'balance': [-2049.0, 4062.0],
    'day':[1.0	,31.0],
    #'duration':	[2.0, 3881.0],
    'campaign':	[1.0,43.0],
    'pdays'	:[-1.0, 854.0],
    'previous':[0, 58]}
)

if test_mode == 'Test_yes':
    test_df = pd.DataFrame(data_yes)
    #test_df['duration'] = (test_df['duration']-(limits['duration'].min()))/(limits['duration'].max()-(limits['duration'].min()))
    
elif test_mode == 'Test_no':
    test_df = pd.DataFrame(data_no) #.items()
    #test_df['duration'] = (test_df['duration']-(limits['duration'].min()))/(limits['duration'].max()-(limits['duration'].min()))
else:
    test_df = pd.DataFrame(data, index=[0]) #.items()


st.write(test_df)
print(test_df)

# preprocessing---------------------------

from joblib import load
scaler = load(os.path.join(path_models, 'scaler_ss.joblib'))
columns_to_process = ['age', 'balance']
transformed_data = scaler.fit_transform(test_df[columns_to_process])

columns_to_process = ['age', 'balance']
temp = pd.DataFrame(transformed_data, columns = columns_to_process, index=[test_df.index[0]]) 

test_df.drop(columns_to_process, axis=1, inplace=True)
test_df = test_df.merge(temp, left_index=True, right_index=True)

test_df.drop(['job', 'marital', 'education', 'default', 'housing', 'loan', 'day', 'month', 'campaign', 'previous'], axis=1, inplace=True)
test_df.columns

for i in ['success', 'unknown']:
    test_df[f'poutcome_success'] = test_df['poutcome'].apply(lambda x: 1 if x==i else 0)

for i in ['cellular', 'unknown']:
    test_df[f'contact_{i}'] = test_df['contact'].apply(lambda x: 1 if x==i else 0)

test_df.drop(['poutcome', 'contact_cellular'], axis=1, inplace=True)
test_df = test_df[['poutcome_success', 'balance', 'contact_unknown', 'age','pdays']]

y_pred, y_prob= model.predict(test_df), model.predict_proba(test_df)[:,1]


# for j in ['balance',  'pdays', 'previous', 'campaign']: #'duration',
#     test_df[j] = (test_df[j]-(limits[j].min()))/(limits[j].max()-(limits[j].min()))

# """
# for i in ['mar', 'may', 'oct', 'sep']:
#     test_df[f'month_{i}'] = test_df['month'].apply(lambda x: 1 if x==i else 0)

# test_df['contact_cellular'] = test_df['contact'].apply(lambda x: 1 if x=='cellular' else 0)

# test_df['contact_unknown'] = test_df['contact'].apply(lambda x: 1 if x=='unknown' else 0) 

# for i in ['success', 'unknown']:
#     test_df[f'poutcome_{i}'] = test_df['poutcome'].apply(lambda x: 1 if x==i else 0)

# for i in ['housing', 'loan']:
#     test_df[i] = test_df[i].apply(lambda x: 1 if x=='yes' else 0)
# """ 

# #test_df['housing'] = test_df['housing'].apply(lambda x: 1 if x=='yes' else 0)

# #test_df['age_group_60+'] = test_df['age'].apply(lambda x: 1 if 60<=x<=95 else 0)
# #test_df = test_df.drop(labels=['age']) # for series

# test_df['age_group'] = pd.cut(test_df['age'], [0,30,40,50,60,9999], labels = ['<30','30-40','40-50','50-60','60+'])

# df_encoded = pd.get_dummies(test_df)


# df_encoded.drop(['job', 'marital','default', 'day', 'contact', 'month','poutcome'], axis=1, inplace=True) #'age_group','education', 'loan','deposit'

# # Set columns order 
# test_df = test_df[['default', 'housing', 'loan', 'campaign', 'pdays', 'previous','job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid','job_management', 'job_retired', 'job_self-employed', 'job_services','job_student', 'job_technician', 'job_unemployed', 'marital_divorced','marital_married', 'marital_single', 'education_primary','education_secondary', 'education_tertiary', 'contact_cellular','contact_telephone', 'contact_unknown', 'month_apr', 'month_aug','month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun','month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep','poutcome_failure', 'poutcome_other', 'poutcome_success','poutcome_unknown', 'age_group_<30', 'age_group_30-40','age_group_40-50', 'age_group_50-60', 'age_group_60+', 'day_of_week_Mon', 'day_of_week_Tue', 'day_of_week_wed', 'day_of_week_Thu', 'day_of_week_Fri', 'day_of_week_Sat','day_of_week_Sun', 'age', 'balance']]


# Prediction of probabilities:
predict_prob = model.predict_proba(test_df)[:,1]
predict_class = model.predict(test_df)

#st.subheader('Suggestion:')
res = 'Contact' if predict_class[0] == 1 else 'No'
st.write('### Forecasted outcome:','👉', res)