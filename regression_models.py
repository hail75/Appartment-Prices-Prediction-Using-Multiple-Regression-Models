import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
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

# Initialize the models
linear_regression = LinearRegression()
ridge = Ridge(alpha=0.5)
lasso = Lasso(alpha=0.5)
random_forest = RandomForestRegressor(random_state=42, max_depth=5, min_samples_leaf=2, min_samples_split=10,n_estimators= 100)
kneighbors = KNeighborsRegressor(n_neighbors=15)
weighted_kneighbors = KNeighborsRegressor(n_neighbors=15, weights='distance')

# Fit the models
linear_regression.fit(x_train, y_train)
ridge.fit(x_train, y_train)
lasso.fit(x_train, y_train)
random_forest.fit(x_train, y_train)
kneighbors.fit(x_train, y_train)
weighted_kneighbors.fit(x_train, y_train)

# Predict on the test set
linear_regression_preds = linear_regression.predict(x_test)
ridge_preds = ridge.predict(x_test)
lasso_preds = lasso.predict(x_test)
random_forest_preds = random_forest.predict(x_test)
kneighbors_preds = kneighbors.predict(x_test)
weighted_kneighbors_preds = weighted_kneighbors.predict(x_test)

# Calculate Mean Squared Errors
linear_regression_mse = mean_squared_error(y_test, linear_regression_preds)
ridge_mse = mean_squared_error(y_test, ridge_preds)
lasso_mse = mean_squared_error(y_test, lasso_preds)
random_forest_mse = mean_squared_error(y_test, random_forest_preds)
kneighbors_mse = mean_squared_error(y_test, kneighbors_preds)
weighted_kneighbors_mse = mean_squared_error(y_test, weighted_kneighbors_preds)

# Create a dictionary of MSE for each method
mse_data = {
    'Method': ['Linear Regression', 'Ridge', 'Lasso', 'KNeighbors', 'Weighted KNeighbors', 'Random Forest'],
    'MSE (negative)': [-linear_regression_mse, -ridge_mse, -lasso_mse, -kneighbors_mse, -weighted_kneighbors_mse, -random_forest_mse]
}

# Create a DataFrame from the MSE data
mse_df = pd.DataFrame(mse_data)

# Print the MSE table
print(mse_df)

# Box plot of negative squared errors
methods = ['Linear Regression', 'Ridge', 'Lasso', 'KNeighbors', 'Weighted KNeighbors', 'Random Forest']
errors = [-(linear_regression_preds - y_test)**2, -(ridge_preds - y_test)**2, -(lasso_preds - y_test)**2,
          -(kneighbors_preds - y_test)**2, -(weighted_kneighbors_preds - y_test)**2, -(random_forest_preds - y_test)**2]

fig, ax = plt.subplots(figsize=(10, 24))
bp = ax.boxplot(errors, patch_artist=True, medianprops={'color': 'black'})

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_xticklabels(methods, rotation=45)
ax.set_xlabel('Methods')
ax.set_ylabel('Negative Squared Error')
ax.set_title('Comparison of Negative Squared Error for Different Methods')

# Show the plot
plt.show()
