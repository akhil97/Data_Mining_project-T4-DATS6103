#%%[markdown]
### Section - 1 
## EDA

#%%
# importing the needed libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import researchpy as rp

# %%
# Reading the dataset
df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')

# Prrinting the first 5 rows of the dataset
df.head()

# About the dataset
df.info()

print("Columns of the dataset:", df.columns.tolist())
print("Number of rows in dataset:", len(df))

# %%
# Checking for any null values
print("To check if any null values are present in the dataset", df.isnull().sum())

# %%
# Statistical understanding of the data
df.describe()

# %%
# Unique values present in each feature
print("The unique values in each feature are: {}".format(df.nunique()))

# %%
# Confusion matrix to check for correlation
correlation_matrix = df.corr()
k = 22 # number of variables for heatmap
cols = correlation_matrix.nlargest(k,'HeartDiseaseorAttack')['HeartDiseaseorAttack'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(18,18))  # Sample figsize in inches
hm = sns.heatmap(cm, cbar=True, annot=True, cmap = "Blues",
                square=True, fmt='.01f', 
                annot_kws={'size': 10}, 
                yticklabels=cols.values, 
                xticklabels=cols.values,ax=ax)
plt.title("Correlation Matrix")
plt.show()

# %%
# Target variable HeartDiseaseorAttack distribution
sns.countplot(x = df["HeartDiseaseorAttack"], 
                data = df,
                palette = "rocket").set(title = "Countplot of the target variable")

print('The number of no heart disease or attack in the dataset are: ', round(df['HeartDiseaseorAttack'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('The number of heart disease or attack in the dataset are: ', round(df['HeartDiseaseorAttack'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

# %%[markdown]
## Smart Questions

# %%[markdown]
### SMART - 1: Does age influence heart disease or attack?

#%%
# Countplot for the feature age
sns.countplot(x = df["Age"], 
              data = df, palette = "rocket").set(title = "Distribution of Age in the dataset")

# Unique values present in age
print("Unique values present in age: ",df["Age"].unique())

# Number of people within each category of age
print("Number of people in the age group 1: ", df["Age"].value_counts()[1])
print("Number of people in the age group 2: ", df["Age"].value_counts()[2])
print("Number of people in the age group 3: ", df["Age"].value_counts()[3])
print("Number of people in the age group 4: ", df["Age"].value_counts()[4])
print("Number of people in the age group 5: ", df["Age"].value_counts()[5])
print("Number of people in the age group 6: ", df["Age"].value_counts()[6])
print("Number of people in the age group 7: ", df["Age"].value_counts()[7])
print("Number of people in the age group 8: ", df["Age"].value_counts()[8])
print("Number of people in the age group 9: ", df["Age"].value_counts()[9])
print("Number of people in the age group 10: ", df["Age"].value_counts()[10])
print("Number of people in the age group 11: ", df["Age"].value_counts()[11])
print("Number of people in the age group 12: ", df["Age"].value_counts()[12])

#%%
# HeartDiseaseorAttack vs Age
sns.boxplot(x = 'HeartDiseaseorAttack', 
            y = 'Age', data = df, palette = "rocket")
plt.title("HeartDiseaseorAttack vs Age")
plt.xlabel("Heart Disease or Attack")
plt.ylabel("Age (in years)")

#%%
# ttest of means of age for having heart disease or not
rp.ttest(group1= df["Age"][df["HeartDiseaseorAttack"] == 0], group1_name= "0",
        group2= df["Age"][df["HeartDiseaseorAttack"] == 1], group2_name= "1")

# %%
# Unique values in HighBP
print("Unique values in HighBP ", df["HighBP"].unique())

# Number of people within each group of High BP
print("Number of people in the high bp group 0: ", df["HighBP"].value_counts()[0])
print("Number of people in the high bp group 1: ", df["HighBP"].value_counts()[1])

# HighBP vs Age
sns.boxplot(x = "HighBP",
            y = "Age", data = df,
            palette = "rocket").set(title = "HighBP vs Age")

# %%
# Unique values in HighChol
print("Unique values in HighChol ", df["HighChol"].unique())

# Number of people within each group of High Cholesterol
print("Number of people in the high cholesterol group 0: ", df["HighChol"].value_counts()[0])
print("Number of people in the high cholesterol group 1: ", df["HighChol"].value_counts()[1])

# HighBP vs Age
sns.boxplot(x = "HighChol",
            y = "Age", data = df,
            palette = "rocket").set(title = "HighChol vs Age")

#%%[markdown]
### SMART - 2: Does having high BP influence heart disease or attack?

#%%
# Countplot for the feature High BP
sns.countplot(x = df["HighBP"], 
              data = df, palette = "rocket").set(title = "Distribution of HighBP in the dataset")

# Unique values present in High BP
print("Unique values present in age: ",df["HighBP"].unique())

#%%
# HighBP and HeartDiseaseorAttack
sns.catplot(x = "HighBP", hue="HeartDiseaseorAttack", 
            data=df, kind="count", palette = "rocket")
plt.title("HighBP and HeartDiseaseorAttack")
plt.xlabel("High BP")
plt.ylabel("Count")

#%%
# ttest of heart disease for having high blood pressure or not
rp.ttest(group1= df["HeartDiseaseorAttack"][df["HighBP"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["HighBP"] == 1], group2_name= "1")

# %%
# Unique values in Diabetes
print("Unique values present in diabetes: ",df["Diabetes"].unique())

# Number of people within each category of diabetes
print("Number of people in diabetes group 0", df["Diabetes"].value_counts()[0])
print("Number of people in diabetes group 2", df["Diabetes"].value_counts()[1])
print("Number of people in diabetes group 3", df["Diabetes"].value_counts()[2])

# Countplot for each group in diabetes
sns.countplot(x = df["Diabetes"],
            data = df, palette = "rocket").set(title = "Distribution of diabetes")
plt.xlabel("Diabetes categories")
plt.ylabel("Count")

# %%
# Diabetes and HighBP
sns.catplot(x = "Diabetes", hue="HighBP", 
            data=df, kind="count", palette = "rocket")
plt.title("Diabetes and HighBP")
plt.xlabel("Diabetes categories")
plt.ylabel("Count")

# %%
# Unique values in HighChol
print("Unique values present in high cholesterol: ",df["HighChol"].unique())

# Number of people within each category of High Cholesterol
print("Number of people in high cholesterol group 0: ", df["HighChol"].value_counts()[0])
print("Number of people in high cholesterol group 1: ", df["HighChol"].value_counts()[1])

# Countplot for high cholesterol
sns.countplot(x = df["HighChol"],
            data = df, palette = "rocket").set(title = "Distribution of high cholesterol")
plt.xlabel("High Cholesterol")
plt.ylabel("Count")

# %%
# HighChol and HighBP
sns.catplot(x = "HighChol", hue="HighBP", 
            data=df, kind="count", palette = "rocket")
plt.title("HighChol and HighBP")
plt.xlabel("High Cholesterol")
plt.ylabel("Count")

#%%[markdown]
### SMART - 3: Does heavy alcohol consumption influence having a heart disease or attack?

#%%
# Unique values in Heavy Alcohol Consumption
print("Unique values in Heavy Alcohol Consumption:", df['HvyAlcoholConsump'].unique())

# Number of people within each category of Heavy Alcohol Consumption
print("Number of people in heavy alcohol consumption group 0: ", df["HvyAlcoholConsump"].value_counts()[0])
print("Number of people in heavy alcohol consumption group 1: ", df["HvyAlcoholConsump"].value_counts()[1])

# Countplot for heavy alcohol consumption
sns.countplot(x = df["HvyAlcoholConsump"],
            data = df, palette = "rocket")

# %%
# HeartDiseaseorAttack and HvyAlcoholConsump
sns.catplot(x = "HeartDiseaseorAttack", hue="HvyAlcoholConsump", 
            data=df, kind="count", palette = "rocket")
plt.title("HeartDiseaseorAttack and HvyAlcoholConsump")
plt.xlabel("Heart Disease or Attack")
plt.ylabel("Heavy Alcohol Consumption")

#%%
# ttest of heart disease for different levels of alcohol consumption
rp.ttest(group1= df["HeartDiseaseorAttack"][df["HvyAlcoholConsump"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["HvyAlcoholConsump"] == 1], group2_name= "1")

#%%[markdown]
# SMART - 4: Does Income influence heart disease or attack?

# %%
# Count plot for Income
sns.countplot(x = df["Income"], 
              data = df, palette = "rocket").set(title = "Distribution of Income in the dataset")

# Unique values in Income
print("Unique values in Income: ", df["Income"].unique())

# Number of people within each category of Income
print("Number of people in the income group 1", df["Income"].value_counts()[1])
print("Number of people in the income group 2", df["Income"].value_counts()[2])
print("Number of people in the income group 3", df["Income"].value_counts()[3])
print("Number of people in the income group 4", df["Income"].value_counts()[4])
print("Number of people in the income group 5", df["Income"].value_counts()[5])
print("Number of people in the income group 6", df["Income"].value_counts()[6])
print("Number of people in the income group 7", df["Income"].value_counts()[7])
print("Number of people in the income group 8", df["Income"].value_counts()[8])

#%%
# HeartDiseaseorAttack vs Income
sns.boxplot(x = df["HeartDiseaseorAttack"], y = df["Income"],
            data = df, palette = "rocket").set(title = "HeartDiseaseorAttack vs Income")

#%%
# ttest of income level for having heart disease or not
rp.ttest(group1= df["Income"][df["HeartDiseaseorAttack"] == 0], group1_name= "0",
        group2= df["Income"][df["HeartDiseaseorAttack"] == 1], group2_name= "1")

# %%
# Unique values in education
print("Unique values in Education: ", df["Education"].unique())

print("Number of people in the education group 1", df["Education"].value_counts()[1])
print("Number of people in the education group 2", df["Education"].value_counts()[2])
print("Number of people in the education group 3", df["Education"].value_counts()[3])
print("Number of people in the education group 4", df["Education"].value_counts()[4])
print("Number of people in the education group 5", df["Education"].value_counts()[5])
print("Number of people in the education group 6", df["Education"].value_counts()[6])

# Countplot for education
sns.countplot(x = df["Education"], 
            data = df,
            palette = "rocket").set(title = "Distribution of education")

# %%
# HeartDiseaseorAttack vs Education
sns.boxplot(x = df["HeartDiseaseorAttack"],
            y = df["Education"],
            data = df,
            palette = "rocket")
plt.title("HeartDiseaseorAttack vs Education")

#%%[markdown]
# SMART - 5: How BMI influences heart disease or attack?

#%%
# Unique values in BMI
print("Unique values in BMI: ", df["BMI"].unique()) 

# Countplot for BMI
fig = px.histogram(df, x="BMI", 
                color="Sex", 
                pattern_shape="HeartDiseaseorAttack")
fig.update_layout(yaxis_range=[-1000,25000])
fig.show()

# %%
# HeartDiseaseorAttack vs BMI
sns.boxplot(x = df["HeartDiseaseorAttack"],
            y = df["BMI"],
            data = df,
            palette = "rocket")
plt.title("HeartDiseaseorAttack vs BMI")

# %%
# ttest of BMI for having heart disease or not
rp.ttest(group1= df["BMI"][df["HeartDiseaseorAttack"] == 0], group1_name= "0",
        group2= df["BMI"][df["HeartDiseaseorAttack"] == 1], group2_name= "1")

#%%[markdown]
# SMART - 6: How consuming veggies and fruits can affect heart disease or attack?

# %%
# Unique values in Fruits
print("Unique values in fruits: ", df["Fruits"].unique())

# Number of people in each group of fruits
print("Number of people in fruit group 0: ", df["Fruits"].value_counts()[0])
print("Number of people in fruit group 1: ", df["Fruits"].value_counts()[1])

# Countplot of fruits
sns.countplot(x = df["Fruits"],
            data = df,
            palette = "rocket").set(title = "Distribution of fruits in the dataset")

# %%
# Fruits and Heart disease or attack
sns.catplot(x = "Fruits", hue="HeartDiseaseorAttack", 
            data=df, kind="count", palette = "rocket")
plt.title("Fruits and HeartDiseaseorAttack")
plt.xlabel("Fruits")
plt.ylabel("Count")

#%%
rp.ttest(group1= df["HeartDiseaseorAttack"][df["Fruits"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["Fruits"] == 1], group2_name= "1")

# %%
# Unique values in Veggies
print("Unique values in fruits: ", df["Veggies"].unique())

# Number of people in each group of veggies
print("Number of people in veggie group 0: ", df["Veggies"].value_counts()[0])
print("Number of people in veggie group 1: ", df["Veggies"].value_counts()[1])

# Fruits and Veggies
sns.catplot(x = "Fruits", hue="Veggies", 
            data=df, kind="count", palette = "rocket")
plt.title("Fruits and Veggies")
plt.xlabel("Fruits")
plt.ylabel("Count")

# %%
# Veggies and Heart disease or attack
sns.catplot(x = "Veggies", hue="HeartDiseaseorAttack", 
            data=df, kind="count", palette = "rocket")
plt.title("Veggies and HeartDiseaseorAttack")
plt.xlabel("Veggies")
plt.ylabel("Count")

# %%
# ttest of heart disease for eating vegetables or not
rp.ttest(group1= df["HeartDiseaseorAttack"][df["Veggies"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["Veggies"] == 1], group2_name= "1")

#%%
# Unique values in Different Walks
print("Unique values in Different Walks: ", df["DiffWalk"].unique())

# Veggies vs DiffWalk
sns.barplot(x = df["Veggies"],
            y = df["DiffWalk"],
            data = df,
            palette = "rocket")
plt.title("Veggies vs DiffWalk")

#%%[markdown]
# SMART - 7: How does the level of physical activity contribute to one having a heart disease or attack?

# %%
# Unique values in physical activity
print("Unique values in Physical Activity: ", df["PhysActivity"].unique())

# Number of people in each group of veggies
print("Number of people in physical activity group 0: ", df["PhysActivity"].value_counts()[0])
print("Number of people in physical activity group 1: ", df["PhysActivity"].value_counts()[1])

# Countplot for PhysActivity
sns.countplot(x = df["PhysActivity"],
            data = df,
            palette = "rocket")
plt.title("Countplot for Physical Activity in the dataset")

# %%
# Plot of Physical Activity and Heart Disease or attack
sns.catplot(x = "PhysActivity", 
            hue="HeartDiseaseorAttack", 
            data=df, kind="count", palette = "rocket")
plt.xlabel("Physical Activity")
plt.ylabel("Count")
plt.title("Physical Activity and Heart Disease or Attack")

#%%
# ttest of heart disease for different levels of physical activity
rp.ttest(group1= df["HeartDiseaseorAttack"][df["PhysActivity"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["PhysActivity"] == 1], group2_name= "1")

# %%
# Unique values in general health
print("Unique values in General Health: ", df["GenHlth"].unique())

# Number of people in different groups of general health
print("Number of people in general health group 1: ", df["GenHlth"].value_counts()[1])
print("Number of people in general health group 2: ", df["GenHlth"].value_counts()[2])
print("Number of people in general health group 3: ", df["GenHlth"].value_counts()[3])
print("Number of people in general health group 4: ", df["GenHlth"].value_counts()[4])
print("Number of people in general health group 5: ", df["GenHlth"].value_counts()[5])

# Countplot for general health
sns.countplot(x = df["GenHlth"],
            data = df,
            palette = "rocket")
plt.title("Countplot for general health")

# %%
sns.boxplot(x = df["PhysActivity"],
            y = df["GenHlth"],
            data = df,
            palette = "rocket")
plt.title("Boxplot for Physical activity and General health")
plt.xlabel("Physical Activity")
plt.ylabel("General Health")

#%%[markdown]
# SMART - 8: Does having high cholesterol impact one having heart disease or attack? 

# %%
# Unique values in high cholesterol
print("Unique values in High cholesterol: ", df["HighChol"].unique())

# Number of people in each group of veggies
print("Number of people in high cholesterol group 0: ", df["HighChol"].value_counts()[0])
print("Number of people in high cholesterol group 1: ", df["HighChol"].value_counts()[1])

# Countplot for High cholesterol
sns.countplot(x = df["HighChol"],
            data = df,
            palette = "rocket")
plt.title("Countplot for High cholesterol in the dataset")
plt.xlabel("High Cholesterol")

# %%
# Plot of High cholesterol and Heart Disease or attack
sns.catplot(x = "HighChol", hue="HeartDiseaseorAttack", 
            data=df, kind="count", palette = "rocket")
plt.xlabel("High Cholesterol")
plt.ylabel("Count")
plt.title("High Cholesterol and Heart Disease or Attack")

# %%
# ttest of heart disease for having high cholesterol or not
rp.ttest(group1= df["HeartDiseaseorAttack"][df["HighChol"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["HighChol"] == 1], group2_name= "1")

#%%
