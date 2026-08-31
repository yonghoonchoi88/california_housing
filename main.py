import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn


# 1. 데이터 로드
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 2. 데이터 정보 확인.
# print("Shape:")
# print(train.shape)
# print(test.shape)
#
# print("\nDescribe:")
# print(train.describe())
# print(test.describe())
#
# print("\nInfo:")
# print(train.info())
# print(test.info())

# 3. 결측치 확인
# print("\nNull values:")
# print(train.isna().sum())
# print(test.isna().sum())

# 4. 데이터 탐색
# fig, axes = plt.subplots(figsize=(10, 10))
# sns.histplot(x=train["SalePrice"], bins=50)
# plt.title("Distribution of SalePrice", fontsize=20)
# plt.show()


