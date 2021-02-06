import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

#load data
data = pd.read_csv(r"C:\Users\frank\Desktop\project\Diabetes App\Diabetes\data\diabetes.csv")

#build model
X = data.drop("Outcome", axis=1)
Y = data["Outcome"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#fit model
model = LogisticRegression()
model.fit(X_train, Y_train)
  
# Creating a pickle file for the classifier
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))