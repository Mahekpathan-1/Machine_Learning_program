import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Stp 1: load dataset
#----------------------------------------------------------
# Function Name : LoadData
# Description :   Load the data from csv
# Input :         Name of csv file
# Output :        Data Frame
# Author :        Mahek Firoj Pathan
# Date :          16/08/2026
#----------------------------------------------------------

def LoadData(filename):
    
    df  = pd.read_csv(filename)
    
    print("Dataset loaded successfully")
    print(df.head())
    
    return df

# step 2 : Data preprocesiing
#----------------------------------------------------------
# Function Name : PreprocessData
# Description :   It performs data analytics
# Input :         Data frame
# Output :        Updated Data frame
# Author :        Mahek Firoj Pathan
# Date :          16/08/2026
#----------------------------------------------------------

def PreprocessData(df):
    
    df= df.drop([
        'Passengerid',
        'Zero',
        'name'
    ],
    errors = "ignore"
    )
    
    # HAndle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    
    # Convert catagorical to numeric data
    
    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first= True,
        dtype = int
    )
    print(df.head())
    
    print("Data preprocessing completed")
    return df
    
# step 2 :  split Data
#----------------------------------------------------------
# Function Name : SplitData
# Description :   It performs splitting activity
# Input :         Data frame
# Output :        4 subsets for training and testing
# Author :        Mahek Firoj Pathan
# Date :          16/08/2026
#----------------------------------------------------------

def SplitData(df):
    
    X = df.drop("Survived", axis=1)
    Y = df["Survived"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )
    
    print("Data set splitting completed successfully")
    
    return X_train, X_test, Y_train, Y_test

# step 4 :  Train the model
#----------------------------------------------------------
# Function Name : TrainModel
# Description :   It performs model training
# Input :         Training features and labels
# Output :        Trained model
# Author :        Mahek Firoj Pathan
# Date :          16/08/2026
#----------------------------------------------------------
    
def TrainModel(X_train, Y_train):
    model = LogisticRegression(max_iter=1000)
    
    model = model.fit(X_train, Y_train)
    
    print("Model trained sucesfully")
    
    return model

# step 5 :  Evaluate model
#----------------------------------------------------------
# Function Name : EvaluateModel
# Description :   It performs model testing
# Input :         model, testing data (features, labels)
# Output :        none
# Author :        Mahek Firoj Pathan
# Date :          16/08/2026
#----------------------------------------------------------
    
def EvaluateModel(model, X_test, Y_test):
    
    Y_pred = model.predict(X_test)
    
    Accuracy = accuracy_score(Y_test,Y_pred)
    
    print("Accuracy is :", Accuracy)
    
    print(confusion_matrix(Y_test,Y_pred))
    
# step 6 :  Preserved model
#----------------------------------------------------------
# Function Name : Preservedmodel
# Description :   It performs preservation into .pkl file
# Input :         model
# Output :        none
# Author :        Mahek Firoj Pathan
# Date :          16/08/2026
#----------------------------------------------------------
    
def Preservedmodel(model,filename):
    joblib.dump(model,filename)
    
    print("Model preserved with name:", filename) 
    
#----------------------------------------------------------
# Function Name : main
# Description :   Entry point function
# Input :         None
# Output :        None
# Author :        Mahek Firoj Pathan
# Date :          16/08/2026
#----------------------------------------------------------

def main():
    
    # step 1
    df = LoadData("MarvellousTitanicDataset.csv")
    
    # step 2
    df = PreprocessData(df)
    
    # step 3
    X_train, X_test, Y_train, Y_test= SplitData(df)
    
    # step 4
    model = TrainModel(X_train, Y_train)
    
    # step 5
    EvaluateModel(model,X_test,Y_test)
    
    # step 6
    Preservedmodel(model,"MarvellousTitanic.pkl")

if __name__ == "__main__":
    main()