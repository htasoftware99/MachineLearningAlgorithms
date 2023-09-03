import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# veri okuma
df = pd.read_csv("dogrusal_regresyon_veriseti.csv", sep = ";")
print(df.head())

# veri görselleştirme
plt.scatter(df.deneyim,df.maas)
plt.xlabel("Deneyim (Yıl)")
plt.ylabel("Maaş (TL)")
plt.title("Deneyim Maaş İlişkisi")
plt.grid(True)
plt.show()


# Dğrusal Regresyon Modeli
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

# Doğrusal Regresyon Modeli
print(linear_reg.fit(x,y))

# y eksenini kestiği nokta intercept bulunması
y_ekseni_kesisim = np.array([0]).reshape(1,-1)
b0 = linear_reg.predict(y_ekseni_kesisim)
print("b0: ", b0)

# y eksenini kestiği nokta (intercept)
b0_ = linear_reg.intercept_
print("b0_:", b0_)

# Eğim(slope) bulunması
b1 = linear_reg.coef_
print("b1: ", b1)

# maas = 1663 + 1138 * deneyim
# y eksenini kestiği nokta ve eğime göre doğrusal model oluşturulur

deneyim = 11 #11 yıllık deneyim
# 11 yıllık deneyime sahip birinin maaşı tahmin edilir
maas_yeni = 1663 + 1138*deneyim
print(maas_yeni)

# 11 yıllık deneyime sahip birinin maaşı predict metodu ile tahmin edilir
sonuc = linear_reg.predict(np.array([deneyim]).reshape(1,-1))
print("11 yıllık deneyime sahip birinin maaşı: {} TL".format(sonuc[0]))


# Doğrusal Regresyon Modeli ile Test/Tahmin/Görselleşirme
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.figure()
plt.scatter(x, y)

# 0-15 yılları arasında deneyime sahip insanların maaşı tahmin edilir
y_head = linear_reg.predict(array) #y_head=maas
plt.plot(array, y_head,color = "red")
plt.xlabel("Deneyim (Yıl)")
plt.ylabel("Maaş (TL)")
plt.title("Deneyim Maaş İlişkisi")
plt.grid(True)
plt.show()


















