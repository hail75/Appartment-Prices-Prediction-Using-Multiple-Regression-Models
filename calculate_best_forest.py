import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('dataset.csv')

# Separate the features (x) and target variable (y)
x = data.drop('price', axis=1)
y = data['price']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [None, 5, 10],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
}

# Create the Random Forest Regression model
rfr = RandomForestRegressor()

# Perform grid search to find the best parameters
grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)

# Print the best parameters and best MSE
print("Best Parameters: ", grid_search.best_params_)

# Use the best model for predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

# Calculate the MSE of the best model
mse = mean_squared_error(y_test, y_pred)
print("MSE of the Best Model: ", mse)
