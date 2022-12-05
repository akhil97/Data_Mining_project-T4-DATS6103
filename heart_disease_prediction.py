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

#%%% 
df.describe()

#%% data statistics
print("The unique values in each feature are: {}".format(df.nunique()))

#%% How the target variable looks like
sns.countplot(x = df["HeartDiseaseorAttack"], data = df).set(title = "Countplot of the target variable")

print('The number of no heart disease or attack in the dataset are: ', round(df['HeartDiseaseorAttack'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('The number of heart disease or attack in the dataset are: ', round(df['HeartDiseaseorAttack'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
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


# %% Does heavy alcohol consumption cause heart attack or disease?
print(df['HvyAlcoholConsump'].unique().tolist())

# Fraction of the values with Heavy Alcohol Consumption
print("Percentage of people who do not consume alcohol hevaily: ", (len(df[df['HvyAlcoholConsump'] == 0])/len(df))*100)
print("Percentage of people who do consume alcohol heavily: ",(len(df[df['HvyAlcoholConsump'] == 1])/len(df))*100)

# Frequency tab;e
alcohol_freq = pd.crosstab(index = df["HeartDiseaseorAttack"],
                            columns = df["HvyAlcoholConsump"],
                            margins = True)
print(alcohol_freq)

# catplot for HvyAlcoholConsump vs HeartDiseaseorAttack
sns.catplot(x = "HvyAlcoholConsump",
            hue = "HeartDiseaseorAttack",
            data = df,
            kind = "count")
plt.title("HvyAlcoholConsump vs HeartDiseaseorAttack")

#%%%[markdown]
# From all the above information we can see that people who consume alcohol heavily and have a heart attack are only 848 through out the dataset which clearly indicates that the dataset is imbalanced.
# %%
