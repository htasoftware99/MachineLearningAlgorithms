import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("coklu_dogrusal_regresyon_veriseti.csv", sep = ";")
print(df.head())

x = df.iloc[:,[0,2]].values #deneyim ve yaş bağımsız değişkenlerdir
print(x)

y = df.maas.values.reshape(-1,1) #maas bağımlı değişkendir
print(y)

# Çoklu Doğrusal Regresyon Modeli Eğitimi

# Çoklu Doğrusal Regresyon Modeli
coklu_dogrusal_regresyon = LinearRegression()
# Doğrusal Regresyon Eğitimi
coklu_dogrusal_regresyon.fit(x,y)

#test1
test_verisi1 = np.array([[10,35]]) #deneyim=10 ve yaş=35
test_sonucu1 = coklu_dogrusal_regresyon.predict(test_verisi1)
print("10 yıllık deneyim ve 35 yaş sonucu çıkan maaş: {} TL".format(test_sonucu1[0]))


#test2
test_verisi2 = np.array([[5,35]]) #deneyim=10 ve yaş=35
test_sonucu2 = coklu_dogrusal_regresyon.predict(test_verisi2)
print("5 yıllık deneyim ve 35 yaş sonucu çıkan maaş: {} TL".format(test_sonucu2[0]))