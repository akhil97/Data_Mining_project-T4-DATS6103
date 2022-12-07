#%%
import pandas as pd
from statsmodels.formula.api import ols
#%%
heart_data = pd.read_csv(r"C:\Users\riles\Desktop\heart_disease_health_indicators_BRFSS2015.csv")
heart_data.head(5)
# %%
phyact_model = ols(formula='HeartDiseaseorAttack ~ PhysActivity', data=heart_data)
phyact_model_fit = phyact_model.fit()
print(phyact_model_fit.summary())
# %%
highchol_model = ols(formula='HeartDiseaseorAttack ~ CholCheck', data=heart_data)
highchol_model_fit = highchol_model.fit()
print(highchol_model_fit.summary())
