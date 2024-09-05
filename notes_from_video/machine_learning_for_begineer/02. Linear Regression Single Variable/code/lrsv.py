import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

path = "E://Testdocml/Keras-learn/notes_from_video/machine_learning_for_begineer/02. Linear Regression Single Variable/code/homeprice.csv"
path1 = "E://Testdocml/Keras-learn/notes_from_video/machine_learning_for_begineer/02. Linear Regression Single Variable/code/areas.csv"
path2 = "E://Testdocml/Keras-learn/notes_from_video/machine_learning_for_begineer/02. Linear Regression Single Variable/code/prediction.csv"

df = pd.read_csv(path)
print(df)

plt.xlabel("area (sq ft)")
plt.ylabel("price $")
plt.scatter(df.area,df.price,color='red',marker='*')
#plt.show()


reg = linear_model.LinearRegression()
fit = reg.fit(df[['area']],df.price)
print(fit)
prediction = reg.predict([[4000]])

plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()


print(f"prediction is {prediction[0]:,.2f}")
print(f"Cofficient is {reg.coef_}")
print(f"Intercept is {reg.intercept_}")

## y= m*x + b 
## 135.78767123 * 4000 + 180616.43835616432

area_d = pd.read_csv(path1)
print(f"Areas CSV file ${area_d.head(3)}")

p = reg.predict(area_d)
area_d['prices'] = p
area_d.to_csv(path2,index=False)
print(p)







# import numpy as np
# from sklearn.linear_model import LinearRegression

# # Sample data
# X = np.array([[1500], [2500], [3000], [3500], [4000]])
# y = np.array([300000, 500000, 600000, 700000, 800000])

# # Fit the model
# reg = LinearRegression()
# reg.fit(X, y)

# # Reshape the input to a 2D array
# input_value = np.array([[3300]])  # 2D array with a single value

# # Make prediction
# prediction = reg.predict(input_value)
# print(prediction)
