https://github.com/2004Prashanthini/Ibm/blob/main/ADS_phase5.pdf
Model training:
Data Preprocessing and Feature Selection:
Preprocess the dataset by performing the feature engineering steps mentioned 
earlier.
Select the features you want to include in your model and define the target 
variable (IMDb scores).
import pandas as pd
# Load and preprocess your dataset (feature engineering)
# Define your features and target variable
Split the Data:
Split the dataset into a training set and a testing set. This allows you to train 
the model on one subset and evaluate it on another.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
Choose a Model:
Choose a regression model to predict IMDb scores. For simplicity, let's use a 
Linear Regression model in this example. You can explore more advanced
models if needed.
from sklearn.linear_model import LinearRegression
model = LinearRegression()
Train the chosen model using the training data.
model.fit(X_train, y_train)
Model Evaluation:
Evaluate the model using appropriate regression metrics, such as Mean 
Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
