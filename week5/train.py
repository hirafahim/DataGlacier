#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd                  # For data manupulation using dataframes
import numpy as np                   # For Statistical Analysis
import seaborn as sns                # for statistical Data Visualisation
import matplotlib. pyplot as plt     # For Data Visualisation


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Import Dataset

# In[3]:


d=pd.read_csv("gender.csv")
d.head()     # display columns and 5 rows of dataset        


# In[4]:


d.shape # shape of the dataset(no. of rows and columns)


# In[5]:


d.info()  # Information about datset


# In[6]:


d.describe()  #Description about dataset


# # Exploratory Data Analysis

# In[7]:


d.isnull().sum()


# In[8]:


#Drop Unnecessary columns
d=d.iloc[:,:-1]
d


# In[9]:


# Drop duplicate rows and row with null values
d=d.dropna(how="all")
d=d.drop_duplicates()
d.shape


# In[10]:


#invalid parsing will be set as NaN
d['Age']=pd.to_numeric(d['Age'],errors='coerce')
d['Height_cm']=pd.to_numeric(d['Height_cm'],errors='coerce')
d['Weight_kg']=pd.to_numeric(d['Weight_kg'],errors='coerce')
d['Income_USD']=pd.to_numeric(d['Income_USD'],errors='coerce')
#Datatype of categorical columns define as category
d['Gender']=d['Gender'].astype('category')
d['Occupation']=d['Occupation'].astype('category')
d['Education_Level']=d['Education_Level'].astype('category')
d['Marital_Status']=d['Marital_Status'].astype('category')


# In[11]:


d.info()  # Dataset information after data cleaning and feature selection


# # Univariate Analysis for Continuous Columns

# In[12]:


d.hist()


# In[13]:


d.boxplot()


# In[ ]:





# In[14]:


d['Gender'].value_counts() # Numbers of records of each category of discrete column


# In[15]:


d['Occupation'].value_counts() # Numbers of records of each category of discrete column


# In[16]:


d['Education_Level'].value_counts()  # Numbers of records of each category of discrete column


# In[17]:


d['Marital_Status'].value_counts() # Numbers of records of each category of discrete column


# In[ ]:





# In[18]:


sns.pairplot(d)   #visuals representation of Correlation between all continuous columns.


# In[19]:


d.corr()


# In[20]:


sns.heatmap(d.corr(),annot=True)


# # Encoding

# In[21]:


# Encoding of discrete categorical columns to discrete numerical columns
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
d['Gender']=le.fit_transform(d['Gender'])
d['Occupation']=le.fit_transform(d['Occupation'])
d['Education_Level']=le.fit_transform(d['Education_Level'])
d['Marital_Status']=le.fit_transform(d['Marital_Status'])
d.head()


# # Evaluation

# In[22]:


from sklearn.model_selection import train_test_split   #splitting dataset into train and test dataset


# In[23]:


X=d.iloc[:,:-1] # Independent variables
Y=d.iloc[:,-1]  #Output/ Dependent variable variables


# In[24]:


X


# In[25]:


Y


# In[26]:


X_train,X_test,Y_train,Y_test =train_test_split(X,Y, test_size=0.2,random_state=40) 
# Splitting dataset into 20% test dataset and 80% train dataset


# In[27]:


X_train.shape


# In[28]:


X_test.shape


# # Building Machine Learning Model

# In[29]:


from sklearn.ensemble import GradientBoostingRegressor #imnport XGboost regressor from sklearn


# In[30]:


params ={"n_estimators": 300,             # parameters for the model building
        "max_depth":4,
        "min_samples_split":5,
        "learning_rate": 0.01,
        "loss":"squared_error",}
reg=GradientBoostingRegressor(**params)   #initialize the model
reg.fit(X_train,Y_train)                  # Training the model


# In[31]:


y_pred=reg.predict(X_test)


# In[32]:


print ("Accuracy of All dataset: ", (reg.score(X,Y)))                # score of complete Dataset
print ("Accuracy of Train dataset: " ,(reg.score(X_train,Y_train)))  # score of Train Dataset
print ("Accuracy of Test dataset: " ,(reg.score(X_test,Y_test)))     # score of test Dataset


# In[33]:


from sklearn import metrics


# In[34]:


print('MAE:', metrics.mean_absolute_error(Y_test,y_pred))
print('MSE:', metrics.mean_squared_error(Y_test,y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))


# In[ ]:





# # Save the Model

# In[35]:


# import pickle library
import pickle # its used for seriealizing and de-seriealizing a python object Structure
pickle.dump(reg, open('model.pkl','wb'))       # open the file for writing
model = pickle.load(open('model.pkl','rb'))    # dump an object to file object


# In[36]:


print(model.predict([[0,35,175,70,16,1,1]]))


# In[ ]:





# In[ ]:





# In[ ]:




