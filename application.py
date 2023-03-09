import pandas as pd 
import streamlit as st
 
import pickle
import os, sys
from datetime import date

sys.path.insert(1, "./config/")

from utils.reader_config import config_reader
# Import of parameters
config = config_reader('config/config.json')

for file in os.listdir("../models/"):
    if file.endswith(".pkl"):
        #print(os.path.join("../models/", file))
        best_model = os.path.join("../models/", file)
        
# loading saved model
with open(best_model), 'rb') as f:
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
    #'duration':224, # убрать
    'Loan': 'no',
    'contact':Contact,
    'month': month,
    'day': day,
    'campaign': Campaign,
    'pdays':Pdays,
    'previous':Previous,
    'poutcome':Poutcome     
}


data_yes = {'age': {1: 55},
    'job': {1: '	services'},
    'marital': {1: 'married'},
    'education': {1: 'secondary'},
    'default': {1: 'no'},
    'balance': {1: 2476.0},
    'housing': {1: 'yes'},
    'loan': {1: 'no'},
    'contact': {1: 'unknown'},
    'day': {1: 5},
    'month': {1: 'may'},
    #'duration': {1: 1201},
    'campaign': {1: 1},
    'pdays': {1: -1},
    'previous': {1: 0},
    'poutcome': {1: 'unknown'},
 }

data_no = {'age': {6001: 26},
    'job': {6001: 'technician'},
    'marital': {6001: 'married'},
    'education': {6001: 'tertiary'},
    'default': {6001: 'no'},
    'balance': {6001: 8.0},
    'housing': {6001: 'yes'},
    'loan': {6001: 'yes'},
    'contact': {6001: 'unknown'},
    'day': {6001: 13},
    'month': {6001: 'may'},
    #'duration': {6001: 262},
    'campaign': {6001: 2},
    'pdays': {6001: -1},
    'previous': {6001: 0},
    'poutcome': {6001: 'unknown'},
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
for j in ['balance',  'pdays', 'previous', 'campaign']: #'duration',
    test_df[j] = (test_df[j]-(limits[j].min()))/(limits[j].max()-(limits[j].min()))

for i in ['mar', 'may', 'oct', 'sep']:
    test_df[f'month_{i}'] = test_df['month'].apply(lambda x: 1 if x==i else 0)

test_df['contact_cellular'] = test_df['contact'].apply(lambda x: 1 if x=='cellular' else 0)

test_df['contact_unknown'] = test_df['contact'].apply(lambda x: 1 if x=='unknown' else 0) 

for i in ['success', 'unknown']:
    test_df[f'poutcome_{i}'] = test_df['poutcome'].apply(lambda x: 1 if x==i else 0)

for i in ['housing', 'loan']:
    test_df[i] = test_df[i].apply(lambda x: 1 if x=='yes' else 0)
#test_df['housing'] = test_df['housing'].apply(lambda x: 1 if x=='yes' else 0)

test_df['age_group_60+'] = test_df['age'].apply(lambda x: 1 if 60<=x<=95 else 0)
#test_df = test_df.drop(labels=['age']) # for series

test_df.drop(['age', 'job', 'marital','default', 'day', 'contact', 'month','poutcome'], axis=1, inplace=True) #'age_group','education', 'loan','deposit'

# Set columns order 
test_df = test_df[['balance', 'housing', 'loan', 'campaign', 'pdays', 'previous', 'contact_cellular', 'contact_unknown', 'month_mar', 'month_may', 'month_oct', 'month_sep', 'poutcome_success', 'poutcome_unknown', 'age_group_60+']] #'duration', 


# Prediction of probabilities:
predict_prob = model.predict_proba(test_df)
predict_class = model.predict(test_df)

#st.subheader('Suggestion:')
res = 'Contact' if predict_class[0] == 1 else 'No'
st.write('### Forecasted outcome:','👉', res)