# Cycle 1 — 載入與初步認識資料
# 程式 1：載入套件 + 讀取資料
import pandas as pd
import numpy as np
import os

# 印出目前程式的工作目錄（確認真的在同一資料夾）
print("目前工作目錄:", os.getcwd())

# 讀取CSV
df = pd.read_csv("YRBS_2007.csv")

print(df.head())
df["SexLabel"] = df["WhatIsYourSex"].map({1: "Male", 2: "Female"})


# ==============================
# 載入套件
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# ==============================
# 讀取資料
# ==============================
df = pd.read_csv("YRBS_2007.csv")
print("資料欄位:", df.columns)
print(df.head())

# ==============================
# 性別欄位轉文字
# ==============================
df["SexLabel"] = df["WhatIsYourSex"].map({1: "Male", 2: "Female"})

# ==============================
# 1️⃣ SadOrHopeless 分析
# ==============================
# 重新編碼成二元變數: 1=Yes, 0=No
df["SadOrHopeless_bin"] = df["SadOrHopeless"].map({1:1, 2:0})

print("\nSadOrHopeless 二元變數比例:")
print(df["SadOrHopeless_bin"].value_counts(normalize=True)*100)

# 繪圖 - 整體
os.makedirs("outputs/figures", exist_ok=True)
plt.figure(figsize=(6,4))
sns.countplot(x="SadOrHopeless_bin", data=df)
plt.title("SadOrHopeless (Binary)")
plt.xlabel("SadOrHopeless: 0=No, 1=Yes")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/figures/SadOrHopeless_binary.png")
plt.show()

# 繪圖 - 分性別
plt.figure(figsize=(6,4))
sns.countplot(x="SadOrHopeless_bin", hue="SexLabel", data=df)
plt.title("SadOrHopeless by Gender")
plt.xlabel("SadOrHopeless: 0=No, 1=Yes")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/figures/SadOrHopeless_by_gender.png")
plt.show()

# ==============================
# 2️⃣ BMIPCT 分析
# ==============================
bmi = df["BMIPCT"].dropna()
print("\nBMIPCT 描述性統計:")
print(bmi.describe())

# 直方圖
plt.figure(figsize=(7,4))
sns.histplot(bmi, kde=True, bins=30)
plt.title("BMI Percentile Distribution")
plt.xlabel("BMI Percentile")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/figures/BMIPCT_histogram.png")
plt.show()

# 箱型圖
plt.figure(figsize=(6,4))
sns.boxplot(x=bmi)
plt.title("BMI Percentile Boxplot")
plt.xlabel("BMI Percentile")
plt.tight_layout()
plt.savefig("outputs/figures/BMIPCT_boxplot.png")
plt.show()

# 分性別箱型圖
plt.figure(figsize=(7,4))
sns.boxplot(x="SexLabel", y="BMIPCT", data=df)
plt.title("BMI Percentile by Gender")
plt.tight_layout()
plt.savefig("outputs/figures/BMIPCT_by_gender.png")
plt.show()