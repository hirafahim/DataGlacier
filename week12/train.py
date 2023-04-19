#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing Campaign

# In[ ]:





# In[1]:


# Import Libraries
import pandas as pd  # For data manupulation using dataframes
import numpy as np   # For Statistical Analysis


# In[2]:


d= pd.read_csv("bank-full.csv")


# In[3]:


d.head()


# In[4]:


# no of rows and columns
d.shape


# In[5]:


# Datatypes of columns and non-null values
d.info()


# In[6]:


# Function to identify numeric features
def numeric_features(dataset):
    numeric_col = dataset.select_dtypes(include=['number']).columns
    return numeric_col

# Function to identify categorical features
def categorical_features(dataset):
    categorical_col = dataset.select_dtypes(exclude=['number']).columns
    return categorical_col


# In[7]:


# display numeric and categorical features
def display_numeric_categoric_feature(dataset):
    numeric_columns = numeric_features(dataset)
    print("Numeric Features:")
    print(numeric_columns)
    print("===="*20)
    categorical_columns = categorical_features(dataset)
    print("Categorical Features:")
    print(categorical_columns)


# In[8]:


display_numeric_categoric_feature(d)


# In[9]:


# total null values in the dataset
d.isnull().sum()


# There is no null record in the data.

# In[10]:


# Description of numerical columns
d.describe()


# In[11]:


# list of columns 
d.columns


# # Exploratory Data Analysis

# In[12]:


# Remove duplicate rows
d=d.drop_duplicates()
d


# In[13]:


# The duration is not known before a call is performed. Also, after the end of the call y is obviously known. 
#Thus, this input should be discarded for a realistic predictive model.
d= d.drop(['duration'], axis=1)


# In[14]:


d.shape


# In[15]:


# change datatype of categorical columns into "category"
d["job"]=d["job"].astype("category")
d["marital"]=d["marital"].astype("category")
d["education"]=d["education"].astype("category")
d["default"]=d["default"].astype("category")
d["housing"]=d["housing"].astype("category")
d["loan"]=d["loan"].astype("category")
d["contact"]=d["contact"].astype("category")
d["month"]=d["month"].astype("category")
d["poutcome"]=d["poutcome"].astype("category")
d["y"]=d["y"].astype("category")


# In[16]:


d.info()


# In[17]:


# Number of counts of categorical features
d['job'].value_counts()


# In[18]:


# Number of counts of categorical features
d['marital'].value_counts()


# In[19]:


# Number of counts of categorical features
d['education'].value_counts()


# In[20]:


# Number of counts of categorical features
d['default'].value_counts()


# In[21]:


# Number of counts of categorical features
d['housing'].value_counts()


# In[22]:


# Number of counts of categorical features
d['loan'].value_counts()


# In[23]:


# Number of counts of categorical features
d['contact'].value_counts()


# In[24]:


# Number of counts of categorical features
d['month'].value_counts()


# In[25]:


# Number of counts of categorical features
d['poutcome'].value_counts()


# In[26]:


# Number of counts of Target variable
d['y'].value_counts()


# In[ ]:





# # Visualisation and Pre-processing

# In[27]:


import matplotlib.pyplot as plt       # For Data Visualisation
import seaborn as sns                 # for statistical Data Visualisation
import warnings
warnings.filterwarnings('ignore')


# # Univariate Analysis

# In[28]:


# Function to plot boxplots
def plot_box_plots(dataframe):
    numeric_columns = numeric_features(dataframe)
    dataframe = dataframe[numeric_columns]
    
    for i in range(0,len(numeric_columns),2):
        if len(numeric_columns) > i+1:
            plt.figure(figsize=(10,4))
            plt.subplot(121)
            sns.boxplot(dataframe[numeric_columns[i]])
            plt.subplot(122)            
            sns.boxplot(dataframe[numeric_columns[i+1]])
            plt.tight_layout()
            plt.show()

        else:
            sns.boxplot(dataframe[numeric_columns[i]])


# In[29]:


plot_box_plots(d)


# In[30]:


# Function to plot histograms
def plot_continuous_columns(dataframe):
    numeric_columns = numeric_features(dataframe)
    dataframe = dataframe[numeric_columns]
    
    for i in range(0,len(numeric_columns),2):
        if len(numeric_columns) > i+1:
            plt.figure(figsize=(10,4))
            plt.subplot(121)
            sns.distplot(dataframe[numeric_columns[i]], kde=False)
            plt.subplot(122)            
            sns.distplot(dataframe[numeric_columns[i+1]], kde=False)
            plt.tight_layout()
            plt.show()

        else:
            sns.distplot(dataframe[numeric_columns[i]], kde=False)


# In[31]:


plot_continuous_columns(d)


# In[32]:


# Encoding target variable to check correlation between target and input variables
from sklearn import preprocessing
le=preprocessing.LabelEncoder()

d['y']=le.fit_transform(d['y'])


# In[33]:


sns.pairplot(d) # checking the relationship 


# There is no strong correlation between target variable and input variables.

# In[34]:


d.corr()


# In[35]:


# Categorical columns
d['job'].value_counts().plot.bar()


# In[36]:


d['marital'].value_counts().plot.bar()


# In[37]:


d['education'].value_counts().plot.bar()


# In[38]:


d['default'].value_counts().plot.bar()


# In[39]:


d['housing'].value_counts().plot.bar()


# In[40]:


d['loan'].value_counts().plot.bar()


# In[41]:


d['contact'].value_counts().plot.bar()


# In[42]:


d['month'].value_counts().plot.bar()


# In[43]:


d['poutcome'].value_counts().plot.bar()


# In[44]:


d['y'].value_counts().plot.bar()


# # Bivariate Analysis

# In[45]:


sns.displot(d, x="age", hue="y", element="step")


# Here we see peolpe between age 30-40 are main contributor towards deposit subscription

# In[46]:


sns.catplot(
    data=d, y="job", hue="y", kind="count",
    palette="pastel", edgecolor=".6",)


# Here we see people with job related to 'management, blue-collar and technician' have subscribed for deposit 

# In[47]:


sns.catplot(
    data=d, y="marital", hue="y", kind="count",
    palette="pastel", edgecolor=".6",)


# Married people are main contributor for deposit scheme

# In[48]:


sns.catplot(
    data=d, y="education", hue="y", kind="count",
    palette="pastel", edgecolor=".6",)


# People with secondary and tertiary educational background are main contributors.

# In[49]:


sns.catplot(
    data=d, y="default", hue="y", kind="count",
    palette="pastel", edgecolor=".6",)


# People with good credit history are main contributors.

# In[50]:


sns.catplot(
    data=d, y="month", hue="y", kind="count",
    palette="pastel", edgecolor=".6",)


# In the month of 'May',we see more interest of client in term deposit

# In[51]:


sns.catplot(data=d, x="y", y="balance")


# People with balance around 20,000 are main contributors

# In[52]:


sns.catplot(
    data=d, y="housing", hue="y", kind="count",
    palette="pastel", edgecolor=".6",)


# most of the People with no housing scheme have subcribed for the deposit.

# In[53]:


sns.catplot(
    data=d, y="loan", hue="y", kind="count",
    palette="pastel", edgecolor=".6",)


# Those who have no personal loan have subscribed for the deposit.

# In[54]:


sns.catplot(
    data=d, y="poutcome", hue="y", kind="count",
    palette="pastel", edgecolor=".6",)


# From the Outcome of previous Campaign, if the outcome is Failure, then there is a less chance that client will subscribe to the term deposit. whereas if the outcome of previous Campaign is Success, then it is more likely that Client will subscribe to the term deposit.

# In[55]:


sns.heatmap(d.corr(),annot=True)


# In[ ]:





# # Encoding

# In[56]:


from sklearn import preprocessing


# In[57]:


le=preprocessing.LabelEncoder()
d['job']=le.fit_transform(d['job'])
d['marital']=le.fit_transform(d['marital'])
d['education']=le.fit_transform(d['education'])
d['default']=le.fit_transform(d['default'])
d['housing']=le.fit_transform(d['housing'])
d['loan']=le.fit_transform(d['loan'])
d['contact']=le.fit_transform(d['contact'])
d['month']=le.fit_transform(d['month'])
d['poutcome']=le.fit_transform(d['poutcome'])


# In[58]:


d.head()


# number of category counts in categorical column after encoding

# In[59]:


d['job'].value_counts()


# In[60]:


d['marital'].value_counts()


# In[61]:


d['education'].value_counts()


# In[62]:


d['default'].value_counts()


# In[63]:


d['housing'].value_counts()


# In[64]:


d['loan'].value_counts()


# In[65]:


d['contact'].value_counts()


# In[66]:


d['month'].value_counts()


# In[67]:


d['poutcome'].value_counts()


# In[ ]:





# In[ ]:





# # Normalization

# In[68]:


from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler


# In[69]:


d1=d.iloc[:,:-1]
d1


# In[70]:


array=d1.values
scaler=MinMaxScaler(feature_range=(0,1))
rescaledX=scaler.fit_transform(array)

set_printoptions(precision=2)
print(rescaledX[0:5,:])


# In[71]:


d2=pd.DataFrame(rescaledX,columns=["age","job","marital","education","default","balance","housing","loan","contact","day",
                                   "month","campaign","pdays","previous","poutcome"])
d2


# In[ ]:





# In[72]:


d2['job'].value_counts()


# In[73]:


d2['marital'].value_counts()


# In[74]:


d2['education'].value_counts()


# In[75]:


d2['default'].value_counts()


# In[76]:


d2['housing'].value_counts()


# In[77]:


d2['loan'].value_counts()


# In[78]:


d2['contact'].value_counts()


# In[79]:


d2['month'].value_counts()


# In[80]:


d2['poutcome'].value_counts()


# In[81]:


d['y'].value_counts()


# In[ ]:





# # Evaluation

# In[82]:


from sklearn.model_selection import train_test_split


# In[83]:


X=d2
Y=d.iloc[:,-1]


# In[84]:


X.head()


# In[85]:


Y.head()


# In[86]:


Y.value_counts()


# # Balance the dataset

# In[87]:


# balance the dataset using SMOTE
from imblearn.over_sampling import SMOTE
from collections import Counter


# In[88]:


sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X, Y)
print('Resampled dataset shape %s' % Counter(Y_res))


# In[89]:


Y_res.value_counts()


# In[90]:


# splitting dataset in 70% train dataset and 30% test dataset
X_train,X_test,Y_train,Y_test =train_test_split(X_res,Y_res, test_size=0.3,random_state=0)


# In[91]:


X_train.shape


# In[92]:


X_test.shape


# In[ ]:





# # Model Building

# In[ ]:





# # GradientBoostingClassifier

# In[107]:


from sklearn.ensemble import GradientBoostingClassifier


# In[108]:


model=GradientBoostingClassifier(n_estimators=300, learning_rate=1.0, max_depth=2, random_state=40)
model.fit(X_train,Y_train) # Fit the model to the training data


# In[109]:


Y2_pred=model.predict(X_test) # Predict the classes on the test data
Y2_pred


# In[110]:


np.mean(Y2_pred==Y_test)


# In[111]:


pd.crosstab(Y_test,Y2_pred)


# In[112]:


gbc_data=model.score(X,Y)
gbc_train=model.score(X_train,Y_train)
gbc_test=model.score(X_test,Y_test)


# In[113]:


print ("Accuracy of All dataset: " ,(gbc_data))
print ("Accuracy of Train dataset: " ,(gbc_train))
print ("Accuracy of Test dataset: " ,(gbc_test))


# In[ ]:





# In[ ]:





# # Save the Model

# In[145]:


# import pickle library
import pickle # its used for seriealizing and de-seriealizing a python object Structure
pickle.dump(model, open('model.pkl','wb'))       # open the file for writing
model = pickle.load(open('model.pkl','rb'))    # dump an object to file object


# In[146]:


print(model.predict([[0.5,0.36,0.5,0.667,0.0,0.09,1.0,0.0,1.0,0.133,0.727,0.0,0.0,0.0,1.0]]))


# In[ ]:




