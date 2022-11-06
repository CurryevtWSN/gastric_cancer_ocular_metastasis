import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
#应用标题
st.set_page_config(page_title='Prediction model for ocular metastasis of gastric cancer')
st.title('Prediction Model of Ocular Metastases in Gastric Adenocarcinoma: Machine Learning–Based Development and Interpretation Study')
st.sidebar.markdown('## Variables')
Pathological_type = st.sidebar.selectbox('Pathological_type',('Highly differentiated adenocarcinoma','Moderately differentiated adenocarcinoma',
                                                              'Poorly differentiated adenocarcinoma','Others'),index=2)
TG = st.sidebar.slider("TG", 0.00, 20.00, value=3.39, step=0.01)
TC = st.sidebar.slider("TC", 0.00, 100.00, value=2.07, step=0.01)
HDL = st.sidebar.slider("HDL", 0.00, 20.00, value=1.03, step=0.01)
LDL = st.sidebar.slider("LDL", 0.00, 20.00, value=5.36, step=0.01)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'Highly differentiated adenocarcinoma':1,
       'Moderately differentiated adenocarcinoma':2,
       'Poorly differentiated adenocarcinoma':3, 
       'Others':4}
Pathological_type =map[Pathological_type]

# 数据读取，特征标注
hp_train = pd.read_csv('Gastric_cancer.csv')
hp_train['M'] = hp_train['M'].apply(lambda x : +1 if x==1 else 0)
features =["Pathological_type","TG","TC","HDL","LDL"]
target = 'M'
random_state_new = 50
data = hp_train[features]
X_data = data
X_ros = np.array(X_data)
y_ros = np.array(hp_train[target])
oversample = SMOTE(random_state = random_state_new)
X_ros, y_ros = oversample.fit_resample(X_ros, y_ros)
XGB_model = XGBClassifier(n_estimators=360, max_depth=2, learning_rate=0.1,random_state = random_state_new)
XGB_model.fit(X_ros, y_ros)
# mlp = MLPClassifier(hidden_layer_sizes=(100,), 
#                     activation='relu', solver='lbfgs',
#                     alpha=0.0001,
#                     batch_size='auto',
#                     learning_rate='constant',
#                     learning_rate_init=0.01,
#                     power_t=0.5,
#                     max_iter=200,
#                     shuffle=True, 
#                     random_state=random_state_new)
# mlp.fit(X_ros, y_ros)
sp = 0.5
#figure
is_t = (XGB_model.predict_proba(np.array([[Pathological_type,TG,TC,HDL,LDL]]))[0][1])> sp
prob = (XGB_model.predict_proba(np.array([[Pathological_type,TG,TC,HDL,LDL]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk Ocular metastasis'
else:
    result = 'Low Risk Ocular metastasis'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Ocular metastasis':
        st.balloons()
    st.markdown('## Probability of High risk Ocular metastasis group:  '+str(prob)+'%')

