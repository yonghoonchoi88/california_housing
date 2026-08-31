import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


# 1. 데이터 로드
data = fetch_california_housing(as_frame = True)
X = data.data
y = data.target

df = X.copy()
df["MedHouseVal"] = y
df.to_csv("california_housing.csv", index=False)

# # 2. 데이터 정보 확인.
print("Shape:")
print(X.shape)
print(y.shape)

print(X.head())
print(y.head())

print("\nDescribe:")
print(X.describe())
print(y.describe())

print("\nInfo:")
print(X.info())


# # 3. 결측치 확인
print("\nNull values:")
print(X.isna().sum())
print(y.isna().sum())


# 4. 데이터 탐색
plt.figure(figsize=(10, 10))
sns.histplot(x=y, bins=50)
plt.title("Distribution of MedHouseVal", fontsize=20)
plt.show()


# 5. 데이터 학습 준비.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
m = LinearRegression()

pipe = make_pipeline(scaler, m)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)



# 6. 평가
MAE = mean_absolute_error(y_test, y_pred)
print("MAE Score : ",MAE)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE Value : ",RMSE)
R2_val = r2_score(y_test, y_pred)
print("R2 Score : ",R2_val)
