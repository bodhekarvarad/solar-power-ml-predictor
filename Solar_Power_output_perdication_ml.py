#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
gen_Data=pd.read_csv(r"D:\Collage\Sipna\5_sem\edunet\solar-power-ml-predictor\Plant_1_Generation_Data.csv")
weather_Data=pd.read_csv(r"D:\Collage\Sipna\5_sem\edunet\solar-power-ml-predictor\Plant_1_Weather_Sensor_Data.csv")
gen_Data


# In[2]:


weather_Data


# In[3]:


## Display the first 5 rows of the 'salary' DataFrame
gen_Data.head()
weather_Data.head()


# In[4]:


weather_Data.head()


# In[5]:


gen_Data.info()


# In[6]:


weather_Data.info()


# In[7]:


gen_Data.isnull().sum()


# In[8]:


#check an null values
gen_Data.isna().sum()


# In[9]:


weather_Data.isnull().sum()


# In[10]:


gen_Data.columns


# In[11]:


weather_Data['DATE_TIME'] = pd.to_datetime(weather_Data['DATE_TIME'], dayfirst=True, errors='coerce')
gen_Data['DATE_TIME'] = pd.to_datetime(gen_Data['DATE_TIME'], dayfirst=True, errors='coerce')



# In[12]:


#merge the datasets
merged_Data=pd.merge(gen_Data,weather_Data,on=['DATE_TIME','PLANT_ID'])
merged_Data


# In[13]:


selected_features=merged_Data[['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD',
                                 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]


# In[14]:


# Remove all-NaN columns
selected_features=selected_features.dropna(axis=1, how='all')
# Fill NaNs with column mean
selected_features = selected_features.fillna(selected_features.mean())


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(selected_features.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# In[ ]:


sns.pairplot(selected_features[['DC_POWER', 'AC_POWER', 'IRRADIATION', 'MODULE_TEMPERATURE']])
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


X = selected_features[['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']]
y = selected_features['DC_POWER']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


#train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[ ]:


y_test_pred = rf_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)



# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5)

# Reference line (ideal case: predicted = actual)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

plt.xlabel("Actual DC Power")
plt.ylabel("Predicted DC Power")
plt.title("Actual vs Predicted DC Power (Random Forest)")
plt.grid(True)
plt.show()


# In[ ]:


import joblib
# Save your trained model
joblib.dump(rf_model, r"D:\Collage\Sipna\5_sem\edunet\solar_power_model.pkl")


# In[ ]:


import numpy as np
import pandas as pd
import joblib
#load the model
model = joblib.load(r"D:\Collage\Sipna\5_sem\edunet\solar_power_model.pkl")
#take input from user
irradiation = float(input("Enter Irradiation (W/m²): "))
ambient_temp = float(input("Enter Ambient Temperature (°C): "))
module_temp = float(input("Enter Module Temperature (°C): "))
#create a DataFrame with correct column names
input_df = pd.DataFrame([[irradiation, ambient_temp, module_temp]],
                        columns=['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE'])
#predict
prediction = model.predict(input_df)
#output
print(f"Predicted DC Power Output: {prediction[0]:.2f} kW")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




