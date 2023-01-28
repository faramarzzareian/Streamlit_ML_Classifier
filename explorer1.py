import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.title("ML GUI")
Data_n=  st.sidebar.selectbox("Select Dataset",("Wine dataset","Breast Cancer"))
classifier_n = st.sidebar.selectbox("Select Classifier",("KNN", "SVM"))





def get_data(Data_n):
    if Data_n=="Breast_Cancer":
        data=datasets.load_Breast_Cancer()
 
    else: data=datasets.load_wine()
    X=data.data
    y=data.target
    return X,y

X,y= get_data(Data_n)



def add_p(classifier_name):
    params=dict()
    if classifier_name=="KNN":
        k=st.sidebar.slider("k", 1, 10)
        params["k"]=k

    elif classifier_name=="SVM":
        C= st.sidebar.slider("C" , 0.01, 10.0)
        params["C"]= C
    

    return params
params = add_p(classifier_n)

def get_classifier(classifier_name, params):
    if classifier_name=="KNN":
        Classifier=KNeighborsClassifier(n_neighbors=params["k"])

    elif classifier_name=="SVM":
        Classifier= SVC(C=params["C"])
  
    return Classifier
Classifier=get_classifier(classifier_n, params)


X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.2 , random_state= 1234)
Classifier.fit(X_train, y_train)
y_pred= Classifier.predict(X_test)
acc=accuracy_score(y_test,y_pred)
st.write("Dataset is :", Data_n)
st.warning( f"classifier: {classifier_n}")
st.warning(f"Accuracy : {acc}")




