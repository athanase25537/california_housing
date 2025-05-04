import pandas as pd
import os
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import math

path = "/home/andriamasy/.cache/kagglehub/datasets/camnugent/california-housing-prices/versions/1"
filename = "housing.csv"

df = pd.read_csv(os.path.join(path, filename))

# check missing value
cols_with_missing_value = [col for col in df.columns
                                if df[col].isnull().any()]
cpt = 0
for v in df[cols_with_missing_value[0]]:
    if math.isnan(v): cpt += 1

# clear row with missing value
df = df.dropna(axis=0)

# Encode ocen_proximity to numeric value
encoder = LabelEncoder()
df['ocean_proximity_encoded'] = encoder.fit_transform(df['ocean_proximity'])

target = 'median_house_value'
y = df[target]
X = df.drop([target, 'ocean_proximity'], axis=1)

X_train, X_valid, y_train, y_valid =  train_test_split(X, y)

# # linear regression model
# model_linear = LinearRegression()

# model_linear.fit(X_train, y_train)

# y_preds = model_linear.predict(X_valid)

# score1 = model_linear.score(X_valid, y_valid)
# mae1 = mean_absolute_error(y_valid, y_preds)

# # decision tree regressor model
# model_decision_tree = DecisionTreeRegressor(random_state=1)

# model_decision_tree.fit(X_train, y_train)

# y_preds = model_decision_tree.predict(X_valid)

# score2 = model_decision_tree.score(X_valid, y_valid)
# mae2 = mean_absolute_error(y_valid, y_preds)

# random forest model
model_random_forest = RandomForestRegressor(random_state=1)

model_random_forest.fit(X_train, y_train)

y_preds = model_random_forest.predict(X_valid)

score3 = model_random_forest.score(X_valid, y_valid)
# mae3 = mean_absolute_error(y_valid, y_preds)

# print("Scores")
# print(f"Linear Regression: {score1*100:.2f}%")
# print(f"Decision Tree Regressor: {score2*100:.2f}%")
# print(f"Random Forest Regressor: {score3*100:.2f}%")

# print("Mean Absolute Error")
# print(f"Linear Regression: {mae1}")
# print(f"Decision Tree Regressor: {mae2}")
# print(f"Random Forest Regressor: {mae3}")

joblib.dump(model_random_forest, "california_housing.pkl")