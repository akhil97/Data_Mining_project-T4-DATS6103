#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#%%
df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')
df.head()
df["HeartDiseaseorAttack"] = df["HeartDiseaseorAttack"].astype(int)

# %%
# Drop duplicated rows
duplicated_rows = df[df.duplicated()]
print("There are a total of {} number of duplicated rows.".format(duplicated_rows.shape[0]))

df.loc[df.duplicated(), :]

# Dropping the duplicated values
df.drop_duplicates(inplace = True)
print("Data shape after dropping the duplicated rows is {}".format(df.shape))

# %%
# import SMOTE and other over-sampling techniques
from collections import Counter
# pip install imblearn
from imblearn.over_sampling import SMOTE

# %%
# Separating the independent and dependent variables
X = df.iloc[:, :-1]
y = df.iloc[:,-1]

#%%
# Split train-test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 30)

# Data Scaling
from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
X_train[X_train.columns] = rs.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = rs.transform(X_test[X_test.columns])

# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# define oversampling strategy
SMOTE = SMOTE("minority")

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
print("After oversampling: ", Counter(y_train_SMOTE))

# %%
# SMOTE - Naive Bayes
# Basic Naive Bayes 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
Gnb = GaussianNB()
Gnb.fit(X_train_SMOTE, y_train_SMOTE)
y_pred = Gnb.predict(X_test)
print("Naive Bayes Classifier Accuracy: ",accuracy_score(y_test, y_pred))

#%%
# Under sampling - RandomForest
# Under sampling is over-fitting the data
from imblearn.under_sampling import RandomUnderSampler
sampler = RandomUnderSampler(random_state=11)
xs, ys = sampler.fit_resample(X, y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(xs, ys, test_size = 0.3)
forest = RandomForestClassifier(random_state = 11)
forest.fit(xtrain, ytrain)

ypred_tr = forest.predict(xtrain)
ypred_ts = forest.predict(xtest)

print("Training Results:\n")
print(classification_report(ytrain, ypred_tr))

print("\n\nTesting Results:\n")
print(classification_report(ytest, ypred_ts))

# %%
# Over sampling - RandomForest
rs = RandomForestClassifier(random_state = 11)
rs.fit(X_train_SMOTE, y_train_SMOTE)

rs_ypred_tr = rs.predict(X_train_SMOTE)
rs_ypred_ts = rs.predict(X_test)

print("Training Results:\n")
print(classification_report(y_train_SMOTE, rs_ypred_tr))

print("\n\nTesting Results:\n")
print(classification_report(y_test, rs_ypred_ts))

# %%
print("RandomForest Accuracy:", accuracy_score(y_test, rs_ypred_ts))

# %%
# Under Sampling - LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_SMOTE, y_train_SMOTE)
lr_ypred = lr.predict(X_test)

print("Logistic Regression Accuracy is:", accuracy_score(y_test, lr_ypred))

# %%
