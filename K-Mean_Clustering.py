import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    
     # Step 1 : load the data 
     
    df = pd.read_csv("Mall_Customers.csv")
    
    print("Dataset loaded with values ")
    print(df.head())
    
    print("Missing values")
    print(df.isnull().sum())
    
    # step 2: feature selection 
    
    X = df[["AnnualIncome", "SpendingScore"]]
    
    print("Selected features :")
    print(X.head())
    
    # step 3 : Scalled the data 
    
    scalar = StandardScaler()
    
    X_scaled = scalar.fit_transform(X)
    
    print("Scaled Data : ")
    print(X_scaled[:5])
    
    # step 4 : Elbow method
    
    WCSS = []
    
    for k in range(1,11):
        model = KMeans(
            n_clusters=  k,
            random_state= 42,
            n_init= 10
        )
        
        model.fit(X_scaled)
        
        WCSS.append(model.inertia_)
        
    print("VAlues of WCSS : ")
    
    for i in range(len(WCSS)):
        print(f"{i+1}: {WCSS[i]}")
        
    # step 5 : Visualization
    
    plt.plot(range(1,11), WCSS, marker = "o")
    plt.xlabel("Number of clusters : k")
    plt.ylabel("WCSS")
    plt.title("Marvellous Elbow method")
    
    plt.grid(True)
    plt.show()
    
    # step 6: Final model
    
    model = KMeans(
                n_clusters=  4,
                random_state= 42,
                n_init= 10
            )
    
    clusters = model.fit_predict(X_scaled)
    
    df["cluster"] = clusters
    
    print("Data set with clusters")
    print(df.head(100))
    
if __name__ == "__main__":
    main()