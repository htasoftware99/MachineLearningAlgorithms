import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("linear-regression-dataset.csv",sep = ";")
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

b0 = linear_reg.predict([[0]])
print("b0: ", b0)

b0_ = linear_reg.intercept_
print("b0_: ", b0_)

b1 = linear_reg.coef_
print("b1: ",b1) #eğim-slope

print(linear_reg.predict([[11]]))

maas_yeni = 1663+1138*11
print(maas_yeni)

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
plt.scatter(x, y)
plt.show()

y_head = linear_reg.predict(array)

plt.plot(array, y_head, color="red")





