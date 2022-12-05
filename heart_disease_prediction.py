#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

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

#How BMI can affect the health of the heart or heart condition? 

bmi_freq = pd.crosstab(index = df["HeartDiseaseorAttack"],
                            columns = df["BMI"],
                            margins = True)
print(bmi_freq)

#Histogram for the BMI vs HeartDiseaseorAttack
sns.histplot(data = df, x='BMI', hue='HeartDiseaseorAttack', multiple="stack", binwidth=3)

#Histogram for the BMI vs HeartDiseaseorAttack based on gender
fig = px.histogram(df, x="BMI", color="Sex", pattern_shape="HeartDiseaseorAttack")
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