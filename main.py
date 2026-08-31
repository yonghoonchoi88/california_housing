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
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

df = X.copy()
df["MedHouseVal"] = y
df.to_csv("california_housing.csv", index=False)

# # # 2. 데이터 정보 확인.
# print("Shape:")
# print(df.shape)
# print(df.shape)
# print(df.head())
#
# print("\nDescribe:")
# print(df.describe())
#
# print("\nInfo:")
# print(df.info())
#
# # # 3. 결측치 확인
# print("\nNull values:")
# print(df.isna().sum())
#
# # 4. 데이터 탐색
# print("\nCorr with 'MedHouseVal':")
# print(df.corr()["MedHouseVal"].drop("MedHouseVal").sort_values(key=abs, ascending=False))

# plt.figure(figsize=(10, 10))
# sns.histplot(x=y, bins=50)
# plt.title("Distribution of MedHouseVal", fontsize=20)
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# sns.histplot(x=df["AveRooms"], ax=axes[0, 0], bins=50)
# sns.histplot(x=df["AveBedrms"], ax=axes[0, 1], bins=50)
# sns.histplot(x=df["AveOccup"], ax=axes[1, 0], bins=50)
# sns.histplot(x=df["Population"], ax=axes[1, 1], bins=50)
# plt.tight_layout()
# plt.show()
# fetures = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
# print(df[fetures].describe(percentiles=[.5, .9, .95, .99, .999]).round(2))


# 5-1. 전처리; (파생변수; 침실비율(여유공간 레벨), 1인당방(여유도), 가구수(동네 규모)
df["BedrmRatio"] = df["AveBedrms"] / df["AveRooms"]
df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
df["Households"] = df["Population"] / df["AveOccup"]

# 5-2. dis to LA & SF // 도심까지의 거리
SF, LA = (37.77, -122.42), (34.05, -118.24)
df["dist_SF"] = np.sqrt((df["Latitude"] - SF[0]) ** 2 + (df["Longitude"] - SF[1]) ** 2)
df["dist_LA"] = np.sqrt((df["Latitude"] - LA[0]) ** 2 + (df["Longitude"] - LA[1]) ** 2)
df["dist_city"] = df[["dist_SF", "dist_LA"]].min(axis=1)

# print(df.corr()["MedHouseVal"].drop("MedHouseVal").sort_values(key=abs, ascending=False))

# 6. 데이터 // 준비 // 학습 // 예측
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
m = LinearRegression()
pipe = make_pipeline(scaler, m)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# 7. 평가
MAE = mean_absolute_error(y_test, y_pred)
print("MAE Score : ", MAE)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE Value : ", RMSE)
R2_val = r2_score(y_test, y_pred)
print("R2 Score : ", R2_val)


# 8. README 작성
with open("README.md", "w", encoding="utf-8") as f:
    f.write(f"""# California Housing Price Prediction

캘리포니아 주택가격 예측 (선형회귀 // LinearRegression)

## 평가 결과

 MAE : {MAE:.4f}

 RMSE : {RMSE:.4f}

 R² : {R2_val:.4f}

""")


