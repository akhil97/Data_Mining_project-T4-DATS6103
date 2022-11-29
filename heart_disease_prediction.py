#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

 #%% Reading the dataset
df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')
print("First 5 rows:", df.head())
print("Information about dataset:", df.info())
print("Columns of the dataset:", df.columns.tolist())
print("Number of rows in dataset:", len(df))

#%% Checking for any null values
df.isnull().sum()


# %% Does age have an effect on heart disease?
print(df['HeartDiseaseorAttack'].unique().tolist())
print("Minimum age:", min(df['Age']))
print("Maximum age:", max(df['Age']))

#Fraction of values with heart disease 
print("Percentage of people with no heart disease: ", (len(df[df['HeartDiseaseorAttack'] == 0])/len(df))*100)
print("Percentage of people with a heart disease: ",(len(df[df['HeartDiseaseorAttack'] == 1])/len(df))*100)

# Distribution of Age
fig, axes = plt.subplots(2, figsize=(20, 30))
sns.histplot(ax = axes[0] ,data = df, x='Age')
plt.xlabel("Age (in years)")
plt.ylabel("Count")

# Barplot of Age vs Heart Disease
sns.barplot(ax = axes[1], x = 'HeartDiseaseorAttack', y = 'Age', data = df)
plt.title("Age vs Heart Disease")
plt.xlabel("Heart Disease")
plt.ylabel("Age (in years)")



#%% Does having high BP have an effect on heart disease?
print(df['HighBP'].unique().tolist())

#Fraction of values with High BP
print("Percentage of people with low BP: ", (len(df[df['HighBP'] == 0])/len(df))*100)
print("Percentage of people with high BP: ",(len(df[df['HighBP'] == 1])/len(df))*100)

#Frequency table
my_crosstab = pd.crosstab(index=df["HeartDiseaseorAttack"], columns=df["HighBP"], margins=True)   # Include row and column totals
print(my_crosstab)

# Catplot for HighBP vs Heart Disease
sns.catplot(x='HighBP', hue='HeartDiseaseorAttack', data=df, kind='count')
plt.title("HighBP vs Heart Disease")


# %%
