#%%
#As we have already done the EDA on the variables which we are using in our smart questions, so I have done the EDA on the remaining ones based on the correlation.
#In confusion matrix income and edication are showing negative correlation.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    sns.countplot(data=df, x=column, hue='HeartDiseaseorAttack')
    plt.title(f"{column}")
    plt.tight_layout()