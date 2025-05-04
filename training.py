import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
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


