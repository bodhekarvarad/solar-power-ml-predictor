#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
gen_Data=pd.read_csv(r"D:\Collage\Sipna\5_sem\edunet\solar-power-ml-predictor\Plant_1_Generation_Data.csv")
weather_Data=pd.read_csv(r"D:\Collage\Sipna\5_sem\edunet\solar-power-ml-predictor\Plant_1_Weather_Sensor_Data.csv")
print(gen_Data)


# In[2]:


print(weather_Data)


# In[3]:


## Display the first 5 rows of the 'salary' DataFrame
print(gen_Data.head())
# weather_Data.head()


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


print(gen_Data.columns)


# In[11]:


weather_Data['DATE_TIME'] = pd.to_datetime(weather_Data['DATE_TIME'], dayfirst=True, errors='coerce')
gen_Data['DATE_TIME'] = pd.to_datetime(gen_Data['DATE_TIME'], dayfirst=True, errors='coerce')



# In[12]:


#merge the datasets
merged_Data=pd.merge(gen_Data,weather_Data,on=['DATE_TIME','PLANT_ID'])
print(merged_Data)


# In[13]:


selected_features=merged_Data[['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD',
                                 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]


# In[14]:


#remove all-NaN columns
selected_features=selected_features.dropna(axis=1, how='all')
#fill NaNs with column mean
selected_features = selected_features.fillna(selected_features.mean())


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(selected_features.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# In[16]:


sns.pairplot(selected_features[['DC_POWER', 'AC_POWER', 'IRRADIATION', 'MODULE_TEMPERATURE']])
plt.show()


# In[17]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[18]:


X = selected_features[['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']]
y = selected_features['DC_POWER']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


#train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[28]:


y_test_pred = rf_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)


# In[29]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5)

#reference line (ideal case: predicted = actual)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

plt.xlabel("Actual DC Power")
plt.ylabel("Predicted DC Power")
plt.title("Actual vs Predicted DC Power (Random Forest)")
plt.grid(True)
plt.show()


# In[30]:


test_mse = mean_squared_error(y_test, y_test_pred)
print(test_mse)


# In[31]:


test_r2 = r2_score(y_test, y_test_pred)
print(test_r2)


# In[32]:


import joblib
#save your trained model
joblib.dump(rf_model, r"D:\Collage\Sipna\5_sem\edunet\solar_power_model.pkl")


# In[36]:


import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Load the model
model = joblib.load(r"D:\Collage\Sipna\5_sem\edunet\solar_power_model.pkl")

st.title("Solar Power Output Prediction")

# Take input from user
irradiation = st.text_input("Enter Irradiation (W/m²): ")
ambient_temp = st.text_input("Enter Ambient Temperature (°C): ")
module_temp = st.text_input("Enter Module Temperature (°C): ")

# Predict only if all inputs are provided
if irradiation and ambient_temp and module_temp:
    try:
        # Convert inputs to float
        irradiation = float(irradiation)
        ambient_temp = float(ambient_temp)
        module_temp = float(module_temp)

        # Create a dataframe
        input_df = pd.DataFrame([[irradiation, ambient_temp, module_temp]],
                                columns=['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE'])

        # Predict
        prediction = model.predict(input_df)

        # Output
        st.success(f"Predicted DC Power Output: {prediction[0]:.2f} kW")

    except ValueError:
        st.error("Please enter valid numeric values.")
else:
    st.warning("Please enter all input values.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




