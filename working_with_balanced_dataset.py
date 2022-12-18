#%%
# importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from mlxtend.evaluate import bias_variance_decomp

warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#%%
df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')
df["HeartDiseaseorAttack"] = df["HeartDiseaseorAttack"].astype(int)
df.head()
df.info()

#%%
# Drop duplicated rows
duplicated_rows = df[df.duplicated()]
print("There are a total of {} number of duplicated rows.".format(duplicated_rows.shape[0]))

df.loc[df.duplicated(), :]

# Dropping the duplicated values
df.drop_duplicates(inplace = True)
print("Data shape after dropping the duplicated rows is {}".format(df.shape))

#%%
# Distribution of the target variable - HeartDiseaseorAttack
sns.countplot(x = df["HeartDiseaseorAttack"],
             palette = "Blues").set(title = "Distribution of the target variable")

# Distribution in terms of percentage of the target variable
print('The number of no heart disease or attack in the dataset are: ', round(df['HeartDiseaseorAttack'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('The number of heart disease or attack in the dataset are: ', round(df['HeartDiseaseorAttack'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

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
# Checking for multicollinearity
# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]]
y = df[["HeartDiseaseorAttack"]]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["variables"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print(vif_data)

#%%
# Dropping columns with multicollinearity
df = df.drop(["CholCheck", "BMI", "AnyHealthcare",
             "GenHlth", "Age", "Education", "Income",
             "Veggies"], axis = 1)
df.info()

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
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)

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

#%%
# SMOTE - Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
Gnb = GaussianNB()
Gnb.fit(X_train_SMOTE, y_train_SMOTE)

gnb_ypred_train = Gnb.predict(X_train_SMOTE)
gnb_ypred_valid = Gnb.predict(X_valid)

print("Training Results:\n")
print(classification_report(y_train_SMOTE, gnb_ypred_train))

print("\n\n Validation Results:\n")
print(classification_report(y_valid, gnb_ypred_valid))

print("Naive Bayes Classifier Accuracy: ",accuracy_score(y_valid, gnb_ypred_valid))

#%%
# ROC AUC curve
gnb_tpr, gnb_fpr, gnb_th = roc_curve(y_valid, Gnb.predict(X_valid))
plt.plot(gnb_tpr,gnb_fpr)
print("The AUC value is: ", roc_auc_score(y_valid, Gnb.predict(X_valid)))
plt.title("ROC curve for Naive Bayes Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
from sklearn.metrics import confusion_matrix 
print("Confusion Matrix for Naive Bayes Classifier", confusion_matrix(y_valid, gnb_ypred_valid))

# LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_SMOTE, y_train_SMOTE)

lr_ypred_train = lr.predict(X_train_SMOTE)
lr_ypred_valid = lr.predict(X_valid)

print("Training Results:\n")
print(classification_report(y_train_SMOTE, lr_ypred_train))

print("\n\n Validation Results:\n")
print(classification_report(y_valid, lr_ypred_valid))

print("Logistic Regression Accuracy is:", accuracy_score(y_valid, lr_ypred_valid))

#%%
# ROC AUC curve 
lr_tpr, lr_fpr, lr_th = roc_curve(y_valid, lr.predict(X_valid))
plt.plot(lr_tpr,lr_fpr)
print("The AUC value is: ", roc_auc_score(y_valid, lr.predict(X_valid)))
plt.title("ROC curve for LogisticRegression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for Decision Tree Classifier", confusion_matrix(y_valid, lr_ypred_valid))

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=3, random_state=1)
dtc.fit(X_train_SMOTE,y_train_SMOTE)

dtc_ypred_train = dtc.predict(X_train_SMOTE)
dtc_ypred_valid = dtc.predict(X_valid)

# Evaluate test-set accuracy
print("Training Results:\n")
print(classification_report(y_train_SMOTE, dtc_ypred_train))

print("\n\n Validation Results:\n")
print(classification_report(y_valid, dtc_ypred_valid))

print("Decision Tree Accuracy:", accuracy_score(y_valid, dtc_ypred_valid))

# %%
# ROC AUC curve
dtc_tpr, dtc_fpr, dtc_th = roc_curve(y_valid, dtc_ypred_valid)
plt.plot(dtc_tpr,dtc_fpr)
print("The AUC value is: ", roc_auc_score(y_valid, dtc_ypred_valid))
plt.title("ROC curve for DecisionTree Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for Decision Tree Classifier", confusion_matrix(y_valid, dtc_ypred_valid))

#%%
# RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

rs = RandomForestClassifier(random_state = 11, n_jobs = -1, max_depth=3)
rs.fit(X_train_SMOTE, y_train_SMOTE)

rs_ypred_train = rs.predict(X_train_SMOTE)
rs_ypred_valid = rs.predict(X_valid)

print("Training Results:\n")
print(classification_report(y_train_SMOTE, rs_ypred_train))

print("\n\n Validation Results:\n")
print(classification_report(y_valid, rs_ypred_valid))

print("RandomForest Accuracy:", accuracy_score(y_valid, rs_ypred_valid))

#%%
# ROC AUC curve
rs_tpr, rs_fpr, rs_th = roc_curve(y_valid, rs_ypred_valid)
plt.plot(rs_tpr,rs_fpr)
print("The AUC value is: ", roc_auc_score(y_valid, rs_ypred_valid))
plt.title("ROC curve for RandomForest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for Random Forest", confusion_matrix(y_valid, rs_ypred_valid))

#%%
# XGBoost Classifier
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train_SMOTE, y_train_SMOTE)

xgbc_ypred_train = xgbc.predict(X_train_SMOTE)
xgbc_ypred_valid = xgbc.predict(X_valid)

# Evaluate test-set accuracy
print("Training Results:\n")
print(classification_report(y_train_SMOTE, xgbc_ypred_train))

print("\n\n Validation Results:\n")
print(classification_report(y_valid, xgbc_ypred_valid))

print("XGBoost Accuracy:", accuracy_score(y_valid, xgbc_ypred_valid))

#%%
xgbc_tpr, xgbc_fpr, xgbc_th = roc_curve(y_valid, xgbc_ypred_valid)
plt.plot(xgbc_tpr, xgbc_fpr)
print("The AUC value is: ", roc_auc_score(y_valid, xgbc_ypred_valid))
plt.title("ROC curve for XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for XGBoost", confusion_matrix(y_valid, xgbc_ypred_valid))

#%%
# Naive Bayes

gnb_ypred_test = Gnb.predict(X_test)

print("Training Results:\n")
print(classification_report(y_train_SMOTE, gnb_ypred_train))

print("\n\n Testing Results:\n")
print(classification_report(y_test, gnb_ypred_test))

print("Naive Bayes Classifier Accuracy for testing data: ",accuracy_score(y_test, gnb_ypred_test))

#%%
# ROC AUC curve
gnb_tpr_test, gnb_fpr_test, gnb_th_test = roc_curve(y_test, Gnb.predict(X_test))
plt.plot(gnb_tpr_test,gnb_fpr_test)
print("The AUC value is: ", roc_auc_score(y_test, Gnb.predict(X_test)))
plt.title("ROC curve for Naive Bayes Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
from sklearn.metrics import confusion_matrix 
print("Confusion Matrix for Naive Bayes Classifier", confusion_matrix(y_test, gnb_ypred_test))

#%%
# estimate bias and variance For Naive Bayes
mse, bias, var = bias_variance_decomp(Gnb, X_train_SMOTE.values, y_train_SMOTE.values, X_test.values, y_test.values, loss='0-1_loss', num_rounds=50, random_seed=123)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)

#%%
# LogisticRegression

lr_ypred_test = lr.predict(X_test)

print("Training Results:\n")
print(classification_report(y_train_SMOTE, lr_ypred_train))

print("\n\n Testing Results:\n")
print(classification_report(y_test, lr_ypred_test))

print("Logistic Regression Accuracy for Testing data:", accuracy_score(y_test, lr_ypred_test))

#%%
# ROC AUC curve 
lr_tpr_test, lr_fpr_test, lr_th_test = roc_curve(y_test, lr.predict(X_test))
plt.plot(lr_tpr_test,lr_fpr_test)
print("The AUC value is: ", roc_auc_score(y_test, lr.predict(X_test)))
plt.title("ROC curve for LogisticRegression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for Logistic Regression", confusion_matrix(y_test, lr_ypred_test))

#%%
# estimate bias and variance For Logistic Regression
mse, bias, var = bias_variance_decomp(lr, X_train_SMOTE.values, y_train_SMOTE.values, X_test.values, y_test.values, loss='0-1_loss', num_rounds=50, random_seed=123)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)

#%%
# Decision Tree Classifier

dtc_ypred_test = dtc.predict(X_test)

# Evaluate test-set accuracy
print("Training Results:\n")
print(classification_report(y_train_SMOTE, dtc_ypred_train))

print("\n\n Testing Results:\n")
print(classification_report(y_test, dtc_ypred_test))

print("Decision Tree Accuracy on Testing data:", accuracy_score(y_test, dtc_ypred_test))

# %%
# ROC AUC curve
dtc_tpr_test, dtc_fpr_test, dtc_th_test = roc_curve(y_test, dtc_ypred_test)
plt.plot(dtc_tpr,dtc_fpr)
print("The AUC value is: ", roc_auc_score(y_test, dtc_ypred_test))
plt.title("ROC curve for DecisionTree Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for Decision Tree Classifier", confusion_matrix(y_test, dtc_ypred_test))

#%%
# estimate bias and variance For Decision Tree
mse, bias, var = bias_variance_decomp(dtc, X_train_SMOTE.values, y_train_SMOTE.values, X_test.values, y_test.values, loss='0-1_loss', num_rounds=50, random_seed=123)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)

#%%
# RandomForest

rs_ypred_test = rs.predict(X_test)

print("Training Results:\n")
print(classification_report(y_train_SMOTE, rs_ypred_train))

print("\n\n Testing Results:\n")
print(classification_report(y_test, rs_ypred_test))

print("RandomForest Accuracy on Testing data:", accuracy_score(y_test, rs_ypred_test))

#%%
# ROC AUC curve
rs_tpr_test, rs_fpr_test, rs_th_test = roc_curve(y_test, rs_ypred_test)
plt.plot(rs_tpr_test, rs_fpr_test)
print("The AUC value is: ", roc_auc_score(y_test, rs_ypred_test))
plt.title("ROC curve for RandomForest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for Random Forest", confusion_matrix(y_test, rs_ypred_test))

#%%
# estimate bias and variance For Random Forest
mse, bias, var = bias_variance_decomp(rs, X_train_SMOTE.values, y_train_SMOTE.values, X_test.values, y_test.values, loss='0-1_loss', num_rounds=50, random_seed=123)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)

#%%
#XGBoost Classifier

xgbc_ypred_test = xgbc.predict(X_test)

print("Training Results:\n")
print(classification_report(y_train_SMOTE, xgbc_ypred_train))

print("\n\n Testing Results:\n")
print(classification_report(y_test, xgbc_ypred_test))

print("XGBoost Accuracy on Testing data:", accuracy_score(y_test, xgbc_ypred_test))

#%%
xgbc_tpr_test, xgbc_fpr_test, xgbc_th_test = roc_curve(y_test, xgbc_ypred_test)
plt.plot(xgbc_tpr_test, xgbc_fpr_test)
print("The AUC value is: ", roc_auc_score(y_test, xgbc_ypred_test))
plt.title("ROC curve for XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%%
# Confusion Matrix
print("Confusion Matrix for XGBoost", confusion_matrix(y_test, xgbc_ypred_test))

#%%
# estimate bias and variance For XGBoost
mse, bias, var = bias_variance_decomp(xgbc, X_train_SMOTE.values, y_train_SMOTE.values, X_test.values, y_test.values, loss='0-1_loss', num_rounds=50, random_seed=123)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)
