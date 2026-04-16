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


# 程式 2：查看資料大小與欄位
# 查看資料筆數與欄位數
print("資料形狀:", df.shape)

# 查看所有欄位名稱
print("欄位名稱:")


# 程式 3：查看資料型態與缺失值
# 查看每個欄位的資料型態與缺失值
print(df.info())


# Cycle 2 — 基本統計
# 程式 4：基本統計摘要
# 數值欄位的統計資訊
print(df.describe())

# Cycle 3 — 缺失值分析（Data Cleaning 開始）
# 程式 5：計算缺失值數量
# 計算每個欄位的缺失值數量
missing = df.isnull().sum()

# 排序顯示缺失最多的欄位
missing = missing.sort_values(ascending=False)

print(missing.head(20))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# =====================================================
# Cycle 1 — 讀取資料
# =====================================================
df = pd.read_csv("YRBS_2007.csv")

# =====================================================
# Cycle 2 — 類別資料轉換 (Data Cleaning)
# =====================================================
# 性別轉文字
df["SexLabel"] = df["WhatIsYourSex"].map({1: "Male", 2: "Female"})

# 年級轉文字
grade_map = {1:"9th Grade", 2:"10th Grade", 3:"11th Grade", 4:"12th Grade"}
df["GradeLabel"] = df["GradeLevel"].map(grade_map)

df["GradeLabel"] = df["InWhatGradeAreYou"].map(grade_map)

# 年齡轉文字
age_map = {
    1: "≤12",
    2: "13",
    3: "14",
    4: "15",
    5: "16",
    6: "17",
    7: "18+"
}
df["AgeLabel"] = df["HowOldAreYou"].map(age_map)

# =====================================================
# Cycle 3 — 性別分布分析
# =====================================================
print("\n性別人數:")
print(df["SexLabel"].value_counts())

print("\n性別比例:")
print(df["SexLabel"].value_counts(normalize=True)*100)

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="SexLabel")
plt.title("Gender Distribution")
plt.tight_layout()
plt.show(block=False)
plt.pause(2)
plt.close()

# =====================================================
# Cycle 4 — 年級分布分析
# =====================================================
print("\n年級分布:")
print(df["GradeLabel"].value_counts())

plt.figure(figsize=(7,4))
sns.countplot(data=df, x="GradeLabel", order=["9th Grade","10th Grade","11th Grade","12th Grade"])
plt.title("Grade Distribution")
plt.tight_layout()
plt.show(block=False)
plt.pause(2)
plt.close()

# =====================================================
# Cycle 5 — 年齡分布分析
# =====================================================
print("\n年齡分布:")
print(df["AgeLabel"].value_counts())

plt.figure(figsize=(7,4))
sns.countplot(data=df, x="AgeLabel", order=["≤12","13","14","15","16","17","18+"])
plt.title("Age Distribution")
plt.tight_layout()
plt.show(block=False)
plt.pause(2)
plt.close()


# =====================================================
# Cycle 2 EDA - Role 2
# 分析變數: SadOrHopeless (比例) & BMIPCT (連續)
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# 讀取資料
df = pd.read_csv("YRBS_2007.csv")

# ==============================
# 資料清理 & 類別標籤
# ==============================
# 性別
df["SexLabel"] = df["WhatIsYourSex"].map({1: "Male", 2: "Female"})

# 年級
grade_map = {1:"9th Grade", 2:"10th Grade", 3:"11th Grade", 4:"12th Grade"}
df["GradeLabel"] = df["InWhatGradeAreYou"].map(grade_map)

# 年齡
age_map = {1:"≤12", 2:"13", 3:"14", 4:"15", 5:"16", 6:"17", 7:"18+"}
df["AgeLabel"] = df["HowOldAreYou"].map(age_map)

# ==============================
# 1️⃣ 比例變數: SadOrHopeless
# ==============================
print("=== SadOrHopeless 原始代碼統計 ===")
print(df["SadOrHopeless"].value_counts(dropna=False))

# 缺失值檢查
print("SadOrHopeless 缺失值數量:", df["SadOrHopeless"].isnull().sum())

# 重編碼成二元變數: 成功=1, 失敗=2
df["SadOrHopeless_bin"] = df["SadOrHopeless"].map({1:1, 2:0})

# 成功/失敗比例
print("\n=== SadOrHopeless 二元變數比例 ===")
print(df["SadOrHopeless_bin"].value_counts())
print(df["SadOrHopeless_bin"].value_counts(normalize=True)*100)

import os

# 確保資料夾存在
os.makedirs("outputs/figures", exist_ok=True)

# 條形圖 - 整體
plt.figure(figsize=(6,4))
sns.countplot(x="SadOrHopeless_bin", data=df)

plt.title("SadOrHopeless (Binary)")
plt.xlabel("SadOrHopeless: 0=No, 1=Yes")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/figures/SadOrHopeless_binary.png")
plt.show()

# 條形圖 - 分性別
plt.figure(figsize=(6,4))
sns.countplot(x="SadOrHopeless_bin", hue="SexLabel", data=df)
plt.title("SadOrHopeless by Gender")
plt.xlabel("SadOrHopeless: 0=No, 1=Yes")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/figures/SadOrHopeless_by_gender.png")
plt.show()

# 口頭觀察建議:
print("\n觀察文字示例:")
print("1. 約 XX% 學生曾感到悲傷或絕望 (SadOrHopeless=1)。")
print("2. 男生與女生比例差異可能不大，但女生略高。")

# ==============================
# 2️⃣ 連續變數: BMIPCT
# ==============================
print("\n=== BMIPCT 描述性統計 ===")
bmi = df["BMIPCT"].dropna()
print(bmi.describe())

# 缺失值
print("BMIPCT 缺失值數量:", df["BMIPCT"].isnull().sum())

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

# 補充探索: 分性別
plt.figure(figsize=(7,4))
sns.boxplot(x="SexLabel", y="BMIPCT", data=df)
plt.title("BMI Percentile by Gender")
plt.tight_layout()
plt.savefig("outputs/figures/BMIPCT_by_gender.png")
plt.show()