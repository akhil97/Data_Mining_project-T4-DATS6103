#%%
import pandas as pd
from statsmodels.formula.api import ols
#%%
data = pd.read_csv(r"C:\Users\riles\Desktop\heart_disease_health_indicators_BRFSS2015.csv")
data.head(5)
# %%
phyact_model = ols(formula='HeartDiseaseorAttack ~ PhysActivity', data=data)
phyact_model_fit = phyact_model.fit()
print(phyact_model_fit.summary())
# %%
