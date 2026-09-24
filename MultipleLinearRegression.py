import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def MarvellousRegression(Datapath):
    Border = "-"*40
    
    # step 1: Load the Data
    print(Border)
    print(" # step 1: Load the Data")
    print(Border)
    
    df = pd.read_csv(Datapath)
    
    print(df.head())
    
    # step 2: Removed unwanted columns
    
    print(Border)
    print(" # step 2: Removed unwanted columns")
    print(Border)
    
    if "Unnamed: 0" in df.columns:
        df= df.drop(columns=["Unnamed: 0"])
        
    print(df.head())
    
    # step 3 : check missing values 
    
    print(Border)
    print("# step 3 : check missing values ")
    print(Border)

    print("Total missing values :")
    print(Border)
    print(df.isnull().sum())
    
    # Step 4 : Statistical summary 
    
    print(Border)
    print("# Step 4 : Statistical summary")
    print(Border)
    
    print(df.describe())
    
    # Step 5: Correlation
    
    print(Border)
    print("# Step 5: Correlation")
    print(Border)
    
    print(df.corr())
    
    # step 6 : Separate Independent and Dependent variable
    
    print(Border)
    print("# step 6 : Separate Independent and Dependent variable")
    print(Border)
    
    X = df[["TV", "radio","newspaper"]]
    Y = df['sales']
    
    print("Independent variables :")
    print(X.head())
    
    print("Dependent variables :")
    print(Y.head())
    
    # step 7 : Split the dataset
    
    print(Border)
    print("# step 7 : Split the dataset")
    print(Border)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,Y,
        test_size=0.2, 
        random_state= 42
        )
    
    print("Training data :", X_train.shape)
    print("testing data :", X_test.shape)
    
    # Step 8 : create and train the model 
    
    print(Border)
    print("# Step 8 : create and train the model ")
    print(Border)
    
    model = LinearRegression()
    
    model = model.fit(X_train, Y_train)
    
    print("Model Trained successfully")
    
    # Step 9 : Test the model 
    
    print(Border)
    print("# Step 9 : Test the model ")
    print(Border)
    
    Y_pred = model.predict(X_test)
    
    print("Expected answer :")
    print(Y_test[:3])
    
    print("Predicted answer :")
    print(Y_pred[:3])
    
    # Step 10 : Evaluate the model 
    
    print(Border)
    print("# Step 10 : Evaluate the model ")
    print(Border)
    
    MSE = mean_squared_error(Y_test,Y_pred)
    
    RMSE = np.sqrt(MSE)
    
    R2 = r2_score(Y_test,Y_pred)
    
    print("MSE :", MSE)
    print("RMSE :", RMSE)
    print("R2 :", R2)
    
    # Step 11 : Display Coefficient  
        
    print(Border)
    print(" # Step 11 : Display Coefficient  ")
    print(Border)
        
    print("TV coefficient :", model.coef_[0])
    print("radio coefficient :", model.coef_[1])
    print("newspaper coefficient :", model.coef_[2])
    
    print("Intercept :", model.intercept_)
    
def main():
    
    MarvellousRegression("Advertising.csv")

if __name__ =="__main__":
    main()