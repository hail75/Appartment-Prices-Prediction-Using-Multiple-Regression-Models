import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('dataset.csv')

# Separate the features (x) and target variable (y)
x = data.drop('price', axis=1)
y = data['price']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define a range of k values to evaluate
k_values = range(1, 21)  # Example range: 1 to 20

# Initialize dictionaries to store MSE values for weighted and unweighted KNN
weighted_knn_mse = {}
unweighted_knn_mse = {}

# Iterate over the k values
for k in k_values:
    # Initialize weighted and unweighted KNN regressors
    weighted_knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    unweighted_knn = KNeighborsRegressor(n_neighbors=k)
    
    # Fit the models
    weighted_knn.fit(x_train, y_train)
    unweighted_knn.fit(x_train, y_train)
    
    # Predict on the test set
    weighted_knn_preds = weighted_knn.predict(x_test)
    unweighted_knn_preds = unweighted_knn.predict(x_test)
    
    # Calculate mean squared error for weighted and unweighted KNN
    weighted_knn_mse[k] = mean_squared_error(y_test, weighted_knn_preds)
    unweighted_knn_mse[k] = mean_squared_error(y_test, unweighted_knn_preds)

# Find the best k for weighted KNN
best_weighted_k = min(weighted_knn_mse, key=weighted_knn_mse.get)
best_weighted_mse = weighted_knn_mse[best_weighted_k]

# Find the best k for unweighted KNN
best_unweighted_k = min(unweighted_knn_mse, key=unweighted_knn_mse.get)
best_unweighted_mse = unweighted_knn_mse[best_unweighted_k]

# Print the best k and corresponding MSE for weighted and unweighted KNN
print("Best k for Weighted KNN:", best_weighted_k)
print("MSE for Weighted KNN (k={}) : {}".format(best_weighted_k, best_weighted_mse))
print("Best k for Unweighted KNN:", best_unweighted_k)
print("MSE for Unweighted KNN (k={}) : {}".format(best_unweighted_k, best_unweighted_mse))
