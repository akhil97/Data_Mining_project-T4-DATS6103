#%%
#importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import researchpy as rp

#%% Reading the dataset
df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')
print("First 5 rows:", df.head())
print("Information about dataset:", df.info())
print("Columns of the dataset:", df.columns.tolist())
print("Number of rows in dataset:", len(df))

#%% Checking for any null values
print("To check if any null values are present in the dataset", df.isnull().sum())

#%%% 
df.describe()

#%% data statistics
print("The unique values in each feature are: {}".format(df.nunique()))

#%%
# Confusion matrix to check for correlation
correlation_matrix = df.corr()
k = 22 # number of variables for heatmap
cols = correlation_matrix.nlargest(k,'HeartDiseaseorAttack')['HeartDiseaseorAttack'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(18,18))  # Sample figsize in inches
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.01f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,ax=ax)
plt.title("Correlation Matrix")
plt.show()

#%% How the target variable looks like
sns.countplot(x = df["HeartDiseaseorAttack"], data = df).set(title = "Countplot of the target variable")

print('The number of no heart disease or attack in the dataset are: ', round(df['HeartDiseaseorAttack'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('The number of heart disease or attack in the dataset are: ', round(df['HeartDiseaseorAttack'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

# %% Does age have an effect on heart disease?
print(df['HeartDiseaseorAttack'].unique().tolist())
print("Minimum age:", min(df['Age']))
print("Maximum age:", max(df['Age']))

# Distribution of Age
fig, axes = plt.subplots(2, figsize=(20, 30))
sns.histplot(ax = axes[0] ,data = df, x='Age')
plt.xlabel("Age (in years)")
plt.ylabel("Count")

# Barplot of Age vs Heart Disease
sns.boxplot(ax = axes[1], x = 'HeartDiseaseorAttack', y = 'Age', data = df, palette = "rocket").set(title = "Age vs Heart disease")
plt.xlabel("Heart Disease")
plt.ylabel("Age (in years)")

#Fraction of values with heart disease 
print("Percentage of people with no heart disease: ", (len(df[df['HeartDiseaseorAttack'] == 0])/len(df))*100)
print("Percentage of people with a heart disease: ",(len(df[df['HeartDiseaseorAttack'] == 1])/len(df))*100)


#%% Does having high BP have an effect on heart disease?
print(df['HighBP'].unique().tolist())

#Fraction of values with High BP
print("Percentage of people with low BP: ", (len(df[df['HighBP'] == 0])/len(df))*100)
print("Percentage of people with high BP: ",(len(df[df['HighBP'] == 1])/len(df))*100)

#Frequency table
my_crosstab = pd.crosstab(index=df["HeartDiseaseorAttack"], 
                        columns=df["HighBP"], 
                        margins=True)   # Include row and column totals
print(my_crosstab)

# Catplot for HighBP vs Heart Disease
sns.catplot(x='HighBP', hue='HeartDiseaseorAttack', data=df, kind='count')
plt.title("HighBP vs Heart Disease")

# %% Does heavy alcohol consumption cause heart attack or disease?
print("The unique values under Heavy Alcohol Consumption are:", df['HvyAlcoholConsump'].unique().tolist())

# Fraction of the values related to the feature heavy alcohol consumption
print("Percentage of people who do not consume alcohol hevaily: ", (len(df[df['HvyAlcoholConsump'] == 0])/len(df))*100)
print("Percentage of people who do consume alcohol heavily: ",(len(df[df['HvyAlcoholConsump'] == 1])/len(df))*100)

# Frequency table
alcohol_freq = pd.crosstab(index = df["HeartDiseaseorAttack"],
                            columns = df["HvyAlcoholConsump"],
                            margins = True)
print(alcohol_freq)

# Countplot for feature heavy alcohol consumption
sns.countplot(x = df["HvyAlcoholConsump"], 
              data = df).set(title = "Countplot of feature HvyAlcoholConsump")

# catplot for HvyAlcoholConsump vs HeartDiseaseorAttack
sns.catplot(x = "HvyAlcoholConsump",
            hue = "HeartDiseaseorAttack",
            data = df,
            kind = "count")
plt.title("HvyAlcoholConsump vs HeartDiseaseorAttack")


#%%%[markdown]
# From all the above information we can see that people who consume alcohol heavily and have a heart attack are only 848 through out the dataset which clearly indicates that the dataset is imbalanced.

#%%
# Does having variable income cause heart attack or disease?
print("The unique values under the feature Income are: ", df['Income'].unique().tolist())

# There are a total of 8 different categories in the column Income
# So the categories represent the following:
# 1.0 -> less than $10,000
# 2.0 -> $20,000
# 3.0 -> $30,000
# 4.0 -> $40,000
# 5.0 -> $50,000
# 6.0 -> $60,000
# 7.0 -> $70,000
# 8.0 -> $75,000 more

# Countplot for the feature income
sns.countplot(x = df["Income"], 
              data = df,
             palette = "rocket").set(title = "Countplot of feature Income")

# Fraction of the values of the feature Income
print("Percentage of people who have income below $10,000: ", (len(df[df['Income'] == 1.0])/len(df))*100)
print("Percentage of people who have income of $20,000: ",(len(df[df['Income'] == 2.0])/len(df))*100)
print("Percentage of people who have income of $30,000: ", (len(df[df['Income'] == 3.0])/len(df))*100)
print("Percentage of people who have income of $40,000: ",(len(df[df['Income'] == 4.0])/len(df))*100)
print("Percentage of people who have income of $50,000: ", (len(df[df['Income'] == 5.0])/len(df))*100)
print("Percentage of people who have income of $60,000: ",(len(df[df['Income'] == 6.0])/len(df))*100)
print("Percentage of people who have income of $70,000: ", (len(df[df['Income'] == 7.0])/len(df))*100)
print("Percentage of people who have income of more than $75,000: ",(len(df[df['Income'] == 8.0])/len(df))*100)

# Frequency table
income_freq = pd.crosstab(index = df["HeartDiseaseorAttack"],
                            columns = df["Income"],
                            margins = True)
print(income_freq)

# catplot for HvyAlcoholConsump vs HeartDiseaseorAttack
sns.catplot(x = "Income",
            hue = "HeartDiseaseorAttack",
            data = df,
            kind = "count")
plt.title("Income vs HeartDiseaseorAttack")


#%%
# Checking to see if Heavy alcohol consumption has an impact on income
sns.countplot(data = df, 
              x = "HvyAlcoholConsump", 
              hue ='Income',
             palette = "rocket")

# %%

#How BMI can affect the health of the heart or heart condition? 

bmi_freq = pd.crosstab(index = df["HeartDiseaseorAttack"],
                            columns = df["BMI"],
                            margins = True)
print(bmi_freq)

#Histogram for the BMI vs HeartDiseaseorAttack
sns.histplot(data = df, x='BMI', hue='HeartDiseaseorAttack', multiple="stack", binwidth=3)

#Histogram for the BMI vs HeartDiseaseorAttack based on gender
fig = px.histogram(df, x="BMI", 
                color="Sex", 
                pattern_shape="HeartDiseaseorAttack")
fig.update_layout(yaxis_range=[-1000,25000])
fig.show()

#How consuming veggies and fruits can affect the heart condition?
#Fruits :-
fruits_cons = pd.crosstab(index = df["HeartDiseaseorAttack"],
                            columns = df["Fruits"],
                            margins = True)
print(fruits_cons)

print("Percentage of people who do not eat fruits: ", (len(df[df['Fruits'] == 0])/len(df))*100)
print("Percentage of people who do eat fruits: ",(len(df[df['Fruits'] == 1])/len(df))*100)

#Histogram for Fruits vs HeartDiseaseorAttack
sns.catplot(x='Fruits', hue='HeartDiseaseorAttack', data=df, kind='count')
plt.title("Consumption of Fruits vs Heart Disease")

#Veggies :-
veggies_cons = pd.crosstab(index = df["HeartDiseaseorAttack"],
                            columns = df["Veggies"],
                            margins = True)
print(veggies_cons)

print("Percentage of people who do not eat veggies: ", (len(df[df['Veggies'] == 0])/len(df))*100)
print("Percentage of people who do eat veggies: ",(len(df[df['Veggies'] == 1])/len(df))*100)

#Histogram for Veggies vs HeartDiseaseorAttack
sns.catplot(x='Veggies', hue='HeartDiseaseorAttack', data=df, kind='count')
plt.title("Consumption of Veggies vs Heart Disease")

#Histogram to check how the consumption of Fruits and Veggies affect the BMI :-
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=df['Fruits'],
    histnorm='percent',
    name='Fruits',
    marker_color='#EB89B5',
    opacity=0.75
))
fig.add_trace(go.Histogram(
    x=df['Veggies'],
    histnorm='percent',
    name='Veggies',
    marker_color='#330C73',
    opacity=0.75
))
fig.update_layout(
    xaxis_title_text='Consumption of Fruits and Veggies',
    yaxis_title_text='BMI',
    bargap=0.2,
    bargroupgap=0.1
)
fig.show()


# %%
#EDA for physical activity


# Plot of counts for physical activity
df["PhysActivity"].value_counts().plot(kind= "bar")
plt.xlabel("Physical Activity")
plt.ylabel("Count")
plt.title("Counts of Physical Activity")

# %%
# Plot of Physical Activity and Heart Disease
sns.catplot(x = "PhysActivity", hue="HeartDiseaseorAttack", data=df, kind="count")
plt.xlabel("Physical Activity")
plt.ylabel("Count")
plt.title("Heart Disease versus Physical Activity")


# %%
#EDA for high cholesterol

# Plot of counts for high cholesterol
df["HighChol"].value_counts().plot(kind= "bar")
plt.xlabel("High Cholesterol")
plt.ylabel("Count")
plt.title("Counts of High Cholesterol")

# %%
# Plot of High Cholesterol and Heart Disease
sns.catplot(x = "HighChol", hue="HeartDiseaseorAttack", data=df, kind="count")
plt.xlabel("High Cholesterol")
plt.ylabel("Count")
plt.title("Heart Disease versus High Cholesterol")


#%%
#Smoker vs HeartDiseaseorAttack
sns.catplot(x='Smoker', hue='HeartDiseaseorAttack', data=df, kind='count')
plt.title("Smoking vs Heart Disease")

#As we can see here the person who dont smoke have good chance of not getting any heart disease but for the person who smokes graph shows the chance of not getting 
# heart disease decreases and also the the chance of getting heart disease increases with respect to the persone who dont smoke but still can get the heart disease.

#%%
#Smoker vs Sex
sns.countplot(x='Sex', data=df, hue='Smoker')

#Here we can see that the sex=0 have more number of smokers and less number of non smoker whereas the sex=1 have equal number of smokers and non smokers. 
# Also the smokers are more for sex=1 than sex=0.

#So now we are checking the gender wise chance of getting heart disease.

#%%
#Sex vs HeartDiseaseorAttack
sns.countplot(x='Sex',data=df, hue='HeartDiseaseorAttack')

#As we saw in previous plot that the sex=1 have more number of smokers, and according to the current plot we can see the chance of getting heart disease for sex=1 is more than that of sex=0. 
# So we can say that smoking can increase of getting heart disease which we can clearly see in below plot.

#Smoker vs HeartDiseaseorAttack
sns.countplot(x='Smoker',data=df, hue='HeartDiseaseorAttack')

#%%
#Now lets see plot between Diabetes and HeartDiseaseorAttack
#Diabetes vs HeartDiseaseorAttack
sns.catplot(x='Diabetes', hue='HeartDiseaseorAttack', data=df, kind='count')
plt.title("Diabetes vs Heart Disease")

#%%
#Next lets see the plot between Physical Activities vs HeartDiseaseorAttack
#PhysActivity vs HeartDiseaseorAttack
sns.countplot(x='PhysActivity', data=df, hue='HeartDiseaseorAttack')

#As plot shows, person involved in any type of physical activity have better chance of not getting any hear disease than the person who is not doing any physical activity. 
# But it also increase the chance of getting heart disease compared to the person who is not doing any physical activity.

#%%
#Plot for all the variables vs HeartDiseaseorAttack

catcol = ['HighBP', 'HighChol', 'CholCheck',
       'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'DiffWalk', 'Sex', 'Education',
       'Income']

plt.figure(figsize=(15,50))
for i,column in enumerate(catcol[1:]):
    plt.subplot(len(catcol), 2, i+1)
    plt.suptitle("Plot Value Count VS HeartAttack", fontsize=20, x=0.5, y=1)
    sns.countplot(data=df, x=column, hue='HeartDiseaseorAttack', palette="rocket")
    plt.title(f"{column}")
    plt.tight_layout()


# T-TESTS

# %%
# ttest of heart disease for having high blood pressure or not
rp.ttest(group1= df["HeartDiseaseorAttack"][df["HighBP"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["HighBP"] == 1], group2_name= "1")

# %%
# ttest of heart disease for having high cholesterol or not
rp.ttest(group1= df["HeartDiseaseorAttack"][df["HighChol"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["HighChol"] == 1], group2_name= "1")

# %%
# ttest of heart disease for different levels of CholCheck
rp.ttest(group1= df["HeartDiseaseorAttack"][df["CholCheck"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["CholCheck"] == 1], group2_name= "1")

# %%
# ttest of BMI for having heart disease or not
rp.ttest(group1= df["BMI"][df["HeartDiseaseorAttack"] == 0], group1_name= "0",
        group2= df["BMI"][df["HeartDiseaseorAttack"] == 1], group2_name= "1")

# %%
# ttest of heart disease for being a smoker or not
rp.ttest(group1= df["HeartDiseaseorAttack"][df["Smoker"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["Smoker"] == 1], group2_name= "1")

# %%
# ttest of heart disease for having a stroke or not
rp.ttest(group1= df["HeartDiseaseorAttack"][df["Stroke"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["Stroke"] == 1], group2_name= "1")

# %%
# ttest of heart disease for having diabetes or not
rp.ttest(group1= df["HeartDiseaseorAttack"][df["Diabetes"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["Diabetes"] == 2], group2_name= "1")

# %%
# ttest of heart disease for different levels of physical activity
rp.ttest(group1= df["HeartDiseaseorAttack"][df["PhysActivity"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["PhysActivity"] == 1], group2_name= "1")

# %%
# ttest of heart disease for eating vegetables or not
rp.ttest(group1= df["HeartDiseaseorAttack"][df["Veggies"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["Veggies"] == 1], group2_name= "1")

# %%
# ttest of heart disease for eating fruits or not
rp.ttest(group1= df["HeartDiseaseorAttack"][df["Fruits"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["Fruits"] == 1], group2_name= "1")

# %%
# ttest of heart disease for different levels of AnyHealthcare
rp.ttest(group1= df["HeartDiseaseorAttack"][df["AnyHealthcare"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["AnyHealthcare"] == 1], group2_name= "1")

# %%
# ttest of heart disease for different levels of NoDocbcCost
rp.ttest(group1= df["HeartDiseaseorAttack"][df["NoDocbcCost"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["NoDocbcCost"] == 1], group2_name= "1")

# %%
# ttest of GenHlth for having heart disease or not
rp.ttest(group1= df["GenHlth"][df["HeartDiseaseorAttack"] == 0], group1_name= "0",
        group2= df["GenHlth"][df["HeartDiseaseorAttack"] == 1], group2_name= "1")

# %%
# ttest of MentHlth for having heart disease or not
rp.ttest(group1= df["MentHlth"][df["HeartDiseaseorAttack"] == 0], group1_name= "0",
        group2= df["MentHlth"][df["HeartDiseaseorAttack"] == 1], group2_name= "1")

# %%
# ttest of PhysHlth for having heart disease or not
rp.ttest(group1= df["PhysHlth"][df["HeartDiseaseorAttack"] == 0], group1_name= "0",
        group2= df["PhysHlth"][df["HeartDiseaseorAttack"] == 1], group2_name= "1")

# %%
# ttest of heart disease for different levels of DiffWalk
rp.ttest(group1= df["HeartDiseaseorAttack"][df["DiffWalk"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["DiffWalk"] == 1], group2_name= "1")

# %%
# ttest of heart disease for each sex
rp.ttest(group1= df["HeartDiseaseorAttack"][df["Sex"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["Sex"] == 1], group2_name= "1")

# %%
# ttest of means of age for having heart disease or not
rp.ttest(group1= df["Age"][df["HeartDiseaseorAttack"] == 0], group1_name= "0",
        group2= df["Age"][df["HeartDiseaseorAttack"] == 1], group2_name= "1")

# %%
# ttest of heart disease for different levels of alcohol consumption
rp.ttest(group1= df["HeartDiseaseorAttack"][df["HvyAlcoholConsump"] == 0], group1_name= "0",
        group2= df["HeartDiseaseorAttack"][df["HvyAlcoholConsump"] == 1], group2_name= "1")

# %%
# ttest of education level for having heart disease or not
rp.ttest(group1= df["Education"][df["HeartDiseaseorAttack"] == 0], group1_name= "0",
        group2= df["Education"][df["HeartDiseaseorAttack"] == 1], group2_name= "1")

# %%
# ttest of income level for having heart disease or not
rp.ttest(group1= df["Income"][df["HeartDiseaseorAttack"] == 0], group1_name= "0",
        group2= df["Income"][df["HeartDiseaseorAttack"] == 1], group2_name= "1")
