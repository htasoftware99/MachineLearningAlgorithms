import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression # doğrsusal
from sklearn.preprocessing import PolynomialFeatures # polinomsal


df = pd.read_csv("polinomsal_regresyon_veriseti.csv", sep = ";")
print(df.head())

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("Araba Maksimum Hızı")
plt.xlabel("Araba Fiyatı")
plt.title("Araba hız-fiyat ilişkisi")
plt.grid(True)
plt.show()

# Doğrusal regresyon deneyelim(uygun değil)
lr = LinearRegression()
print(lr.fit(x,y))

y_tahmin = lr.predict(x)

plt.scatter(x,y)
plt.plot(x,y_tahmin,color = "red")
plt.ylabel("Araba Maksimum hız")
plt.xlabel("Araba Fiyat")
plt.title("Araba hız-fiyat ilişkisi")
plt.grid(True)
plt.show()

araba_fiyat = 10000
print(lr.predict((np.array([araba_fiyat]).reshape(1,-1))))

# polinomsal regresyon
polinom_regresyon = PolynomialFeatures(degree=4)

x_polinom = polinom_regresyon.fit_transform(x)
print(x_polinom)

lr2 = LinearRegression()
print(lr2.fit(x_polinom,y))

y_tahmin2 = lr2.predict(x_polinom)
plt.scatter(x,y)
plt.plot(x,y_tahmin,color = "red", label = "Doğrusal")
plt.plot(x,y_tahmin2,color = "green", label = "Polinomsal")
plt.ylabel("Araba Maksimum hız")
plt.xlabel("Araba Fiyat")
plt.title("Araba hız-fiyat ilişkisi")
plt.grid(True)
plt.show()



















