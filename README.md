# PRODIGY_ML_001
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


train_data = pd.read_csv(r"C:\Users\aishwarya amin\house-prices-advanced-regression-techniques\train.csv")
test_data = pd.read_csv(r"C:\Users\aishwarya amin\house-prices-advanced-regression-techniques\train.csv")





for column in train_data.columns:
    if train_data[column].dtype == 'object':
        #fill missing values with the mode for categorical features
        train_data[column].fillna(train_data[column].mode()[0])
        if column in test_data.columns:
            test_data[column].fillna(test_data[column].mode()[0])
    else:
        #fill the missing value with the mean for numeric features
        train_data[column].fillna(train_data[column].mean())
        if column in test_data.columns:
            test_data[column].fillna(test_data[column].mean())

# Select features
features=['GrLivArea','BedroomAbvGr','FullBath','HalfBath','TotRmsAbvGrd']
X = train_data[features]
y = train_data['SalePrice']



#Split the training data for validation 
X_train,X_val,y_train,y_val = train_test_split(X, y,test_size=0.2,random_state=42)

# Train the model
model=LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val,y_pred)
mse = mean_squared_error(y_val,y_pred)
r2 = r2_score(y_val, y_pred)


plt.figure(figsize=(10,6))
plt.scatter(y_val,y_pred,alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Prediction Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.show()
