import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # eğitim-test bölünmesi
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri.csv")
print(data.head())

# Sınıf sayılarını hesaplayarak yeni bir Seri oluşturalım
class_counts = data["class"].value_counts()

# Sınıf sayılarını içeren Seriyi kullanarak görselleştirme yapın
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Sınıf Dağılımı")
plt.xlabel("Class")
plt.ylabel("Output")
plt.xticks(rotation=45)
plt.show()

#abnormal = 1, normal = 0
data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
data.head()
print(data.info())

y = data["class"].values
x_data = data.drop(["class"],axis=1)

sns.pairplot(x_data)
plt.show()

# Veriyi normalize etmeliyiz her bir veriyi 0 ile 1 arasında sıkıştırmalıyız
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
print(x)

# %85 trainig, %15 test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.15, random_state=42)

# transpose alıyoruz
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)

# eğitim
lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)
print(lr)

# test
test_dogrulugu = lr.score(x_test.T, y_test.T)
print("Test Doğruluğu: {}".format(test_dogrulugu))











