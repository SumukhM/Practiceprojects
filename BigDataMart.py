#!/usr/bin/env python
# coding: utf-8

# In[336]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')


# In[337]:


#Importing the train data set
df= pd.read_csv("bigdatamart_Train.csv")
df.head()


# In[338]:


df.shape


# The dataset has 8523 observations and 12 columns

# In[339]:


df.dtypes


# In[340]:


df.nunique()


# In[341]:


df['Outlet_Type'].unique()


# In[342]:


df['Outlet_Location_Type'].unique()


# In[343]:


df['Outlet_Size'].unique()


# In[344]:


df.isnull().sum()


# In[345]:


df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())


# In[346]:


df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])


# In[347]:


df.isnull().sum()


# In[348]:


df['Item_Fat_Content'].unique()


# In[349]:



df['Item_Type'].unique()


# In[350]:


df['Item_Fat_Content'].replace(['low fat','Low Fat','reg'],['LF','LF','Regular'],inplace = True)


# In[351]:


df['Item_Fat_Content'].unique()


# In[352]:


sn.distplot(df['Item_Weight'])


# In[353]:


sn.distplot(df['Item_Visibility'])


# In[354]:


sn.distplot(df['Item_Outlet_Sales'])


# In[355]:


sn.boxplot(df['Item_Weight'])


# In[356]:


sn.boxplot(df['Item_Visibility'])


# In[357]:



sn.boxplot(df['Item_Outlet_Sales'])


# In[358]:


sn.pairplot(df, hue = "Item_Fat_Content")


# In[359]:


plt.figure(figsize=[10,6])
plt.title("Sales based on Item visibility")
sn.scatterplot(df['Item_Visibility'],df['Item_Outlet_Sales'])


# In[360]:


plt.figure(figsize=[10,6])
plt.title("Sales based on Item visibility")
sn.scatterplot(df['Item_MRP'],df['Item_Outlet_Sales'])


# In[361]:


sn.countplot(x="Outlet_Location_Type", data = df)


# In[362]:


sn.countplot(x="Item_Fat_Content", data = df)


# In[363]:


sn.countplot(x="Outlet_Size", data = df)


# In[364]:


df.skew()


# In[365]:


df.dtypes


# In[366]:


cor=df.corr()
cor=df.corr()
plt.figure(figsize=(10,5))
sn.heatmap(df.corr(),annot=True)


# In[367]:


Q1 = df['Item_Visibility'].quantile(0.25)
Q3 = df['Item_Visibility'].quantile(0.75)
IQR = Q3 - Q1
lower_lim = Q1 - 1.5*IQR
upper_lim = Q3 + 1.5*IQR
df_iqr = df[(df['Item_Visibility']>lower_lim)&(df['Item_Visibility']<upper_lim)]
df_iqr.shape


# In[368]:


sn.boxplot(df_iqr['Item_Visibility'])


# In[369]:


Q1 = df['Item_Outlet_Sales'].quantile(0.25)
Q3 = df['Item_Outlet_Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_lim = Q1 - 1.5*IQR
upper_lim = Q3 + 1.5*IQR
df_iqr = df[(df['Item_Outlet_Sales']>lower_lim)&(df['Item_Outlet_Sales']<upper_lim)]
df_iqr.shape


# In[370]:


sn.boxplot(df_iqr['Item_Outlet_Sales'])


# In[371]:


df_iqr.skew()


# In[372]:


df_iqr.head()


# We can see that the sales is not depending on the values of Item_Identifier, Outlet_Identifier.
# So we can get rid of these columns.

# In[373]:


new_df = df_iqr.drop(columns=[ 'Item_Identifier', 'Outlet_Identifier','Outlet_Establishment_Year',])
new_df.head()


# In[374]:


from sklearn.preprocessing import LabelEncoder
columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
le = LabelEncoder()
for i in columns:
    new_df[i] = le.fit_transform(new_df[i])
new_df.head()


# In[375]:


new_df.skew()


# In[376]:


from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer()
dfpt=pt.fit_transform(new_df)
new_df=pd.DataFrame(dfpt,columns=new_df.columns)


# In[377]:


new_df.skew()


# We can see that the Item_Visibility skewness is now removed and since Item_Fat_Content is a categorical variabe we will not try to remove its skewness

# In[378]:



y = new_df['Item_Outlet_Sales']
X= new_df.drop(columns=['Item_Outlet_Sales'])


# In[379]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[380]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.25, random_state=42)


# In[381]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
lr = LinearRegression()
lr.fit(X_train,y_train)
pred= lr.predict(X_test)
r2=r2_score(y_test,pred)
mae=mean_absolute_error(pred,y_test)
mse=mean_squared_error(pred,y_test)
rmse=np.sqrt(mean_squared_error(pred,y_test))
print('r2score=',r2)
print('Mean absolute error = ',mae)
print('Mean Squared error = ',mse)
print('Root Mean Sqaured Error= ',rmse)


# In[382]:


from sklearn.model_selection import cross_val_score


# In[383]:


cv_score = cross_val_score(lr,X_scaled,y,cv=5)
cv_mean = cv_score.mean()
cv_mean


# In[384]:


from sklearn.tree import DecisionTreeRegressor 
dtr = DecisionTreeRegressor()
dtr.fit(X_train,y_train)
pred= dtr.predict(X_test)
r2=r2_score(y_test,pred)
mae=mean_absolute_error(pred,y_test)
mse=mean_squared_error(pred,y_test)
rmse=np.sqrt(mean_squared_error(pred,y_test))
print('r2score=',r2)
print('Mean absolute error = ',mae)
print('Mean Squared error = ',mse)
print('Root Mean Sqaured Error= ',rmse)


# In[385]:


cv_score = cross_val_score(dtr,X_scaled,y,cv=5)
cv_mean = cv_score.mean()
cv_mean


# In[386]:


from sklearn.ensemble import RandomForestRegressor
  
rfr = RandomForestRegressor(n_estimators = 100, random_state = 12)
rfr.fit(X_train, y_train) 
pred= rfr.predict(X_test)
r2=r2_score(y_test,pred)
mae=mean_absolute_error(pred,y_test)
mse=mean_squared_error(pred,y_test)
rmse=np.sqrt(mean_squared_error(pred,y_test))
print('r2score=',r2)
print('Mean absolute error = ',mae)
print('Mean Squared error = ',mse)
print('Root Mean Sqaured Error= ',rmse)


# In[387]:


cv_score = cross_val_score(rfr,X_scaled,y,cv=5)
cv_mean = cv_score.mean()
cv_mean


# In[388]:


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(X_train, y_train) 
pred= knr.predict(X_test)
r2=r2_score(y_test,pred)
mae=mean_absolute_error(pred,y_test)
mse=mean_squared_error(pred,y_test)
rmse=np.sqrt(mean_squared_error(pred,y_test))
print('r2score=',r2)
print('Mean absolute error = ',mae)
print('Mean Squared error = ',mse)
print('Root Mean Sqaured Error= ',rmse)


# In[389]:


cv_score = cross_val_score(knr,X_scaled,y,cv=5)
cv_mean = cv_score.mean()
cv_mean


# In[390]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha= 0.001,random_state= 0)
ridge.fit(X_train, y_train) 
pred= ridge.predict(X_test)
r2=r2_score(y_test,pred)
mae=mean_absolute_error(pred,y_test)
mse=mean_squared_error(pred,y_test)
rmse=np.sqrt(mean_squared_error(pred,y_test))
print('r2score=',r2)
print('Mean absolute error = ',mae)
print('Mean Squared error = ',mse)
print('Root Mean Sqaured Error= ',rmse)


# In[391]:


cv_score = cross_val_score(ridge,X_scaled,y,cv=5)
cv_mean = cv_score.mean()
cv_mean


# In[392]:


from sklearn.linear_model import Lasso

lasso = Lasso(alpha= 0.1,random_state= 0)
lasso.fit(X_train, y_train) 
pred= lasso.predict(X_test)
r2=r2_score(y_test,pred)
mae=mean_absolute_error(pred,y_test)
mse=mean_squared_error(pred,y_test)
rmse=np.sqrt(mean_squared_error(pred,y_test))
print('r2score=',r2)
print('Mean absolute error = ',mae)
print('Mean Squared error = ',mse)
print('Root Mean Sqaured Error= ',rmse)


# In[393]:


cv_score = cross_val_score(lasso,X_scaled,y,cv=5)
cv_mean = cv_score.mean()
cv_mean


# In[394]:


pip install xgboost


# In[395]:



from xgboost.sklearn import XGBRegressor
XGB = XGBRegressor()
XGB.fit(X_train, y_train)
pred = XGB.predict(X_test)
r2=r2_score(y_test,pred)
mae=mean_absolute_error(pred,y_test)
mse=mean_squared_error(pred,y_test)
rmse=np.sqrt(mean_squared_error(pred,y_test))
print('r2score=',r2)
print('Mean absolute error = ',mae)
print('Mean Squared error = ',mse)
print('Root Mean Sqaured Error= ',rmse)


# In[396]:


cv_score = cross_val_score(XGB,X_scaled,y,cv=5)
cv_mean = cv_score.mean()
cv_mean


# Watching all the above data we can see that the minimum difference between r2 score cv score is for the XGB Regressor.
# Hence that is our best performing model

# In[397]:


from sklearn.model_selection import GridSearchCV


# In[398]:


parameter = { 
             'max_depth' : np.arange(2,10),
             'eta' : [0.01,0.1,0.2,0.3],
             'alpha' : [0,0.1,1,10]
             }


# In[399]:


GCV = GridSearchCV(XGBRegressor(),parameter,cv =5)


# In[400]:


GCV.fit(X_train,y_train)


# In[401]:


GCV.best_params_


# In[402]:


Final_mod = XGBRegressor(alpha=1,eta=0.1,max_depth = 2)
Final_mod.fit(X_train,y_train)
pred = Final_mod.predict(X_test)
r2=r2_score(y_test,pred)
cv_score = cross_val_score(Final_mod,X_scaled,y,cv=5)
cv_mean = cv_score.mean()
print(r2,cv_mean)


# We see that the r2 score has improved after Hyper parameter tuning

# <b>Serialization<b>

# In[403]:


#Saving the model
import joblib
joblib.dump(Final_mod,"BigDataMart.pkl")


# In[449]:


#Importing the test data set
df_test= pd.read_csv("bigdatamart_Test.csv")
df_test.head()


# In[450]:


df_test.nunique()


# In[451]:


df_test['Outlet_Type'].unique()


# We see that even in the test there are four types of outlet.

# In[452]:


df_test.shape


# There are 5681 rows and 11 columns

# In[453]:


df_test.dtypes


# In[454]:


df_test.isnull().sum()


# We see that there are missing values under Outlest size and Item weight

# In[455]:


df_test['Item_Weight'] = df_test['Item_Weight'].fillna(df_test['Item_Weight'].mean())
df_test['Outlet_Size'] = df_test['Outlet_Size'].fillna(df_test['Outlet_Size'].mode()[0])
df_test.head()


# In[456]:


df_test.isnull().sum()


# We can now see that all the missing values have been imputed

# In[457]:


df_test['Item_Fat_Content'].unique()


# In[458]:


df_test['Item_Fat_Content'].replace(['low fat','Low Fat','reg'],['LF','LF','Regular'],inplace = True)


# In[459]:


df_test['Item_Fat_Content'].unique()


# We have replaced the mispelled words with the same word for Item_Fat_Content

# In[460]:


df_test.skew()


# In[461]:


from sklearn.preprocessing import LabelEncoder
columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
le = LabelEncoder()
for i in columns:
    df_test[i] = le.fit_transform(df_test[i])
df_test.head()


# In[462]:


df_test = df_test.drop(columns=[ 'Item_Identifier', 'Outlet_Identifier','Outlet_Establishment_Year',])


# In[463]:


df_test.head()


# In[464]:


pt=PowerTransformer()
dfpt=pt.fit_transform(df_test)
df_test=pd.DataFrame(dfpt,columns=df_test.columns)


# In[465]:


df_test.skew()


# The skewness of Item visibility is still high

# In[466]:


scaler = StandardScaler()
df_test_scaled = scaler.fit_transform(df_test)
df_test_scaled


# In[467]:


prediction_test = Final_mod.predict(df_test)
prediction_test


# In[ ]:




