import numpy as np
import pandas as pd
import streamlit as st
import joblib
from joblib import load
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from collections import Counter
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
# Sayfa Ayarları
st.set_page_config(
    page_title="Kredi karti dolandiricişigi",
    page_icon="file:///C:/Users/Alimturk/Downloads/kredi-karti-dolandiricilig%CC%86i1.jpeg.html",
    
)
st.title("Kredi karti dolandiriciligi")
st.markdown("Bu projede bize gönderdiğiniz g kuvveti değerleri ile sizin adım sayınızı hesaplayacağız.")
st.markdown("![Alt Text](https://r.resimlink.com/va1lCL7TI_.gif)")
uploaded_file = st.file_uploader("Choose a file")
dataframe = pd.read_csv(uploaded_file)
st.write(dataframe)
df = pd.read_csv("creditcard.csv")
df["Hours"] = round(df['Time']/3600)
df=df.drop(columns="Time")
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
X = np.array(df.iloc[:, df.columns != 'Class'])
y = np.array(df.iloc[:, df.columns == 'Class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_smote, y_smote = SMOTE().fit_resample(X_train, y_train)
xgb = XGBClassifier()
xgb.fit(X_smote,y_smote)
if uploaded_file is not None:
  
  input_df = pd.DataFrame(
    {
    'V1': [dataframe[0]],
    'V2': [dataframe[1]],
    'V3': [dataframe[2]],
    'V4': [dataframe[3]],
    'V5': [dataframe[4]],
    'V6': [dataframe[5]],
    'V7': [dataframe[6]],
    'V8': [dataframe[7]],
    'V9': [dataframe[8]],
    'V10': [dataframe[9]],
    'V11': [dataframe[10]],
    'V12': [dataframe[11]],
    'V13': [dataframe[12]],
    'V14': [dataframe[13]],
    'V15': [dataframe[14]],
    'V16': [dataframe[15]],
    'V17': [dataframe[16]],
    'V18': [dataframe[17]],
    'V19': [dataframe[18]],
    'V20': [dataframe[19]],
    'V21': [dataframe[20]],
    'V22': [dataframe[21]],
    'V23': [dataframe[22]],
    'V24': [dataframe[23]],
    'V25': [dataframe[24]],
    'V26': [dataframe[25]],
    'V27': [dataframe[26]],
    'V28': [dataframe[27]],
    'Amount': [dataframe[28]],
    'Hour': [dataframe[29]]
})
 
  pred = xgb.predict(input_df.values)
  st.write(pred)
  
