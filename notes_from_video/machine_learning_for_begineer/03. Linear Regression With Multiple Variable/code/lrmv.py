import pandas as pd
import numpy as np
from sklearn import linear_model
import math

path = "notes_from_video/machine_learning_for_begineer/03. Linear Regression With Multiple Variable/code\homeprices.csv"

df = pd.read_csv(path)
print(df)

# for empty value, use median of that column is saver

# calculate median

df['bedrooms'] = pd.to_numeric(df['bedrooms'],errors='coerce')

bedroom_median = df['bedrooms'].median()
print(f"bedroom median = {bedroom_median}")

df.bedrooms.fillna(bedroom_median,inplace=True)

print(df)

reg = linear_model.LinearRegression()
reg.fit(df[["area","bedrooms","age"]],df.price)
print(f"Coefficient {reg.coef_}")
print(f"Intercept {reg.intercept_}")
print(f"Predict {reg.predict([[4000,5, 8]])}")



