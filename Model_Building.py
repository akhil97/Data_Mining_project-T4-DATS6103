#%%
# importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#%%
df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')
df["HeartDiseaseorAttack"] = df["HeartDiseaseorAttack"].astype(int)
print("Looking at the head of the dataset: ", df.head())
print("Information about the dataset: ", df.info())

#%%
# Drop duplicated rows
duplicated_rows = df[df.duplicated()]
print("There are a total of {} number of duplicated rows.".format(duplicated_rows.shape[0]))

df.loc[df.duplicated(), :]

# Dropping the duplicated values
df.drop_duplicates(inplace = True)
print("Data shape after dropping the duplicated rows is {}".format(df.shape))

#%%
# Plot for the target variable before balancing
sns.countplot(x = df["HeartDiseaseorAttack"],
             palette = "Blues").set(title = "Count plot for the target variable HeartDiseaseorAttack before balancing")
plt.grid(True)

#%%
# Correlation plot for the dataset to see which variables are correlated to each other
correlation_matrix = df.corr()
k = 22 # number of variables for heatmap
cols = correlation_matrix.nlargest(k,'HeartDiseaseorAttack')['HeartDiseaseorAttack'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1)
fig, ax = plt.subplots(figsize = (10, 11))  # Sample figsize in inches
hm = sns.heatmap(cm, cbar=True, cmap = "Blues",
                 annot=True, square=True, 
                 fmt='.01f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values,ax=ax)
plt.title("Correlation Matrix")
plt.show()

#%%
# Dropping columns which are highly correlated
df = df.drop(['Veggies','HvyAlcoholConsump',
              'AnyHealthcare','NoDocbcCost',
              'BMI','CholCheck','Fruits'], axis = 1)
print("Information of the dataset after dropping columns which are not correlated: ", df.info())

#%%
# import SMOTE and other over-sampling techniques
from collections import Counter
# pip install imblearn
from imblearn.over_sampling import SMOTE

#%%
# Separating the target feature from other features
X = df.iloc[:,1:]
y = df.iloc[:,0]

#%%
# Split train-test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# define oversampling strategy
SMOTE = SMOTE()

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
print("After oversampling: ", Counter(y_train_SMOTE))

#%%
resampled_df = pd.concat([X_train_SMOTE, y_train_SMOTE], axis = 1)
print("Information of the dataset after balancing: ",resampled_df.info())

#%%
# Target variable distribution after balancing the data
sns.countplot(x = resampled_df["HeartDiseaseorAttack"],
             palette = "Blues").set(title = "Count plot of the target variable HeartDiseaseorAttack after balancing.")
plt.grid(True)

#%%
# Scaling the data after balancing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
print(X_train_SMOTE.shape)
print(y_train_SMOTE.shape)
X_train_SMOTE = sc.fit_transform(X_train_SMOTE)
X_test = sc.transform(X_test)

#%%
# SMOTE - Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
Gnb = GaussianNB()
Gnb.fit(X_train_SMOTE, y_train_SMOTE)

gnb_ypred_tr = Gnb.predict(X_train_SMOTE)
gnb_ypred_ts = Gnb.predict(X_test)

print("Training Results:\n")
print(classification_report(y_train_SMOTE, gnb_ypred_tr))

print("\n\nTesting Results:\n")
print(classification_report(y_test, gnb_ypred_ts))

print("Naive Bayes Classifier Accuracy: ",accuracy_score(y_test, gnb_ypred_ts))

#%%
# Confusion Matrix
from sklearn.metrics import confusion_matrix 
print("Confusion Matrix for Decision Tree Classifier", confusion_matrix(y_test, gnb_ypred_ts))

#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#gnb_cm = confusion_matrix(y_test, gnb_ypred_ts)
#gnb_disp = ConfusionMatrixDisplay(gnb_cm)
#gnb_disp.plot()

#%%
# ROC AUC curve
gnb_tpr, gnb_fpr, gnb_th = roc_curve(y_test, Gnb.predict(X_test))
plt.plot(gnb_tpr,gnb_fpr)
print("The AUC value is: ", roc_auc_score(y_test, Gnb.predict(X_test)))
plt.title("ROC curve for Naive Bayes Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Under Sampling - LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_SMOTE, y_train_SMOTE)

lr_ypred_tr = lr.predict(X_train_SMOTE)
lr_ypred_ts = lr.predict(X_test)

print("Training Results:\n")
print(classification_report(y_train_SMOTE, lr_ypred_tr))

print("\n\nTesting Results:\n")
print(classification_report(y_test, lr_ypred_ts))

print("Logistic Regression Accuracy is:", accuracy_score(y_test, lr_ypred_ts))

#%%
# ROC AUC curve 
lr_tpr, lr_fpr, lr_th = roc_curve(y_test, lr.predict(X_test))
plt.plot(lr_tpr,lr_fpr)
print("The AUC value is: ", roc_auc_score(y_test, lr.predict(X_test)))
plt.title("ROC curve for LogisticRegression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for Decision Tree Classifier", confusion_matrix(y_test, lr_ypred_ts))

#%%
# Over sampling - RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

rs = RandomForestClassifier(random_state = 11, n_jobs = -1)
rs.fit(X_train_SMOTE, y_train_SMOTE)

rs_ypred_tr = rs.predict(X_train_SMOTE)
rs_ypred_ts = rs.predict(X_test)

print("Training Results:\n")
print(classification_report(y_train_SMOTE, rs_ypred_tr))

print("\n\nTesting Results:\n")
print(classification_report(y_test, rs_ypred_ts))

print("RandomForest Accuracy:", accuracy_score(y_test, rs_ypred_ts))

#%%
# ROC AUC curve
rs_tpr, rs_fpr, rs_th = roc_curve(y_test, rs_ypred_ts)
plt.plot(rs_tpr,rs_fpr)
print("The AUC value is: ", roc_auc_score(y_test, rs_ypred_ts))
plt.title("ROC curve for RandomForest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for Decision Tree Classifier", confusion_matrix(y_test, rs_ypred_ts))

#%%
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=5, random_state=1)
dtc.fit(X_train_SMOTE,y_train_SMOTE)

dtc_ypred_tr = dtc.predict(X_train_SMOTE)
dtc_ypred_ts = dtc.predict(X_test)

# Evaluate test-set accuracy
print("Training Results:\n")
print(classification_report(y_train_SMOTE, rs_ypred_tr))

print("\n\nTesting Results:\n")
print(classification_report(y_test, rs_ypred_ts))

print("RandomForest Accuracy:", accuracy_score(y_test, rs_ypred_ts))

# %%
# ROC AUC curve
dtc_tpr, dtc_fpr, dtc_th = roc_curve(y_test, dtc_ypred_ts)
plt.plot(dtc_tpr,dtc_fpr)
print("The AUC value is: ", roc_auc_score(y_test, dtc_ypred_ts))
plt.title("ROC curve for DecisionTree Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for Decision Tree Classifier", confusion_matrix(y_test, dtc_ypred_ts))

#%%
#XGBoost Classifier
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train_SMOTE, y_train_SMOTE)

xgbc_ypred_tr = xgbc.predict(X_train_SMOTE)
xgbc_ypred_ts = xgbc.predict(X_test)

# Evaluate test-set accuracy
print("Training Results:\n")
print(classification_report(y_train_SMOTE, xgbc_ypred_tr))

print("\n\nTesting Results:\n")
print(classification_report(y_test, xgbc_ypred_ts))

print("RandomForest Accuracy:", accuracy_score(y_test, xgbc_ypred_ts))

#%%
xgbc_tpr, xgbc_fpr, xgbc_th = roc_curve(y_test, xgbc_ypred_ts)
plt.plot(xgbc_tpr, xgbc_fpr)
print("The AUC value is: ", roc_auc_score(y_test, xgbc_ypred_ts))
plt.title("ROC curve for XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for Decision Tree Classifier", confusion_matrix(y_test, xgbc_ypred_ts))

# %%
