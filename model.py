


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train=pd.read_csv("Data_Train.csv")
test=pd.read_csv("Test_set.csv")


# In[3]:


test.head()


# In[4]:


train.isnull().sum()


# In[5]:


train[train["Route"].isnull()==True]


# In[6]:


train.dropna(inplace=True)


# In[7]:


train.shape


# # Exploratory Data Analysis

# ### Train Set

# ***Airline***
# - As Airline is a categorical and nominal we will use One-Hot-Encoding using pandas get_dummies()

# In[8]:


plt.figure(figsize=(10,5))
sns.set_theme(style="darkgrid")
sns.countplot(y="Airline",data=train,palette="coolwarm")


# In[9]:


train["Airline"].value_counts()


# In[10]:


Airline=pd.get_dummies(train["Airline"],drop_first=True)


# In[11]:


Airline.head()


# ***Source and Destination***
# - As Source and Destination is a categorical and nominal we will use One-Hot-Encoding using pandas get_dummies()

# In[12]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle('Visualization of Source and Destination')
sns.set_theme(style="darkgrid")
sns.countplot(x="Source",data=train,ax=axes[0],palette="coolwarm")
sns.countplot(x="Destination",data=train,ax=axes[1],palette="coolwarm")


# In[13]:


Source=pd.get_dummies(train[["Source"]],drop_first=True)
Destination=pd.get_dummies(train[["Destination"]],drop_first=True)


# In[14]:


Destination.head()


# As Route and Total_stops imply the same thing we can drop the Route feature.
# 

# In[15]:


train.drop(["Route"],axis=1,inplace=True)
train.head()


# ***Additional_Info***

# In[16]:


info=train["Additional_Info"].value_counts()


# In[17]:


sum=0
for i in info:
    sum=sum+i
print("The percent missing is:", (info[0]/sum)*100)    


# As about 80% of data is NO INFO we will drop this column

# In[18]:


train.drop(["Additional_Info"],axis=1,inplace=True)


# In[19]:


train.head()


# ***Total_Stops***

# As from the data we see as stop increases fare increases so it plays a vital role and so we need to Label encode it.

# In[20]:


train["Total_Stops"].value_counts()


# In[21]:


train.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[22]:


train.head()


# In[23]:


plt.figure(figsize=(15,8))
sns.set_theme(style="darkgrid")
plt.figure(figsize=(15,8))
sns.boxplot(x="Total_Stops",y="Price",data=train,palette="coolwarm")
plt.title("Visualization of Price and Total Stops")


# ***Date Of Journey***

# In[24]:


train["Journey_day"] = pd.to_datetime(train.Date_of_Journey, format="%d/%m/%Y").dt.day
train["Journey_month"] = pd.to_datetime(train["Date_of_Journey"], format = "%d/%m/%Y").dt.month
train.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[25]:


train.head()


# ***Dep_Time and Arrival_Time***

# In[26]:


# Dep_Time
train["Dep_hour"] = pd.to_datetime(train["Dep_Time"]).dt.hour
train["Dep_min"] = pd.to_datetime(train["Dep_Time"]).dt.minute
train.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
train["Arrival_hour"] = pd.to_datetime(train.Arrival_Time).dt.hour
train["Arrival_min"] = pd.to_datetime(train.Arrival_Time).dt.minute
train.drop(["Arrival_Time"], axis = 1, inplace=True)


# In[27]:


train.head()


# ***Duration***

# In[28]:


duration = list(train["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
train["Duration_hours"] = duration_hours
train["Duration_mins"] = duration_mins
train.drop(["Duration"], axis = 1, inplace = True)


# In[29]:


train["Duration"]=(train["Duration_hours"]*60)+train["Duration_mins"]


# In[30]:


train.drop(["Duration_hours","Duration_mins"],inplace=True,axis=1)
train.head()


# In[31]:


final_train = pd.concat([train, Airline, Source, Destination], axis = 1)


# In[32]:


final_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
final_train.head()


# In[33]:


pd.set_option('display.max_columns', None)
final_train.head()


# In[34]:


final_train.iloc[:,2:9].columns


# In[35]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled=scaler.fit_transform(final_train.iloc[:,2:9])
final_train['Journey_day']=scaled[:,0]
final_train["Journey_month"]=scaled[:,1]
final_train["Dep_hour"]=scaled[:,2]
final_train['Dep_min']=scaled[:,3]
final_train["Arrival_hour"]=scaled[:,4]
final_train["Arrival_min"]=scaled[:,5]
final_train["Duration"]=scaled[:,6]


# In[36]:


final_train.head()


# ### Test Set

# In[37]:


# Preprocessing

print("Test data Info")
print("-"*75)
print(test.info())

print()
print()

print("Null values :")
print("-"*75)
test.dropna(inplace = True)
print(test.isnull().sum())

# EDA

# Date_of_Journey
test["Journey_day"] = pd.to_datetime(test.Date_of_Journey, format="%d/%m/%Y").dt.day
test["Journey_month"] = pd.to_datetime(test["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test["Dep_hour"] = pd.to_datetime(test["Dep_Time"]).dt.hour
test["Dep_min"] = pd.to_datetime(test["Dep_Time"]).dt.minute
test.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test["Arrival_hour"] = pd.to_datetime(test.Arrival_Time).dt.hour
test["Arrival_min"] = pd.to_datetime(test.Arrival_Time).dt.minute
test.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test["Duration_hours"] = duration_hours
test["Duration_mins"] = duration_mins
test.drop(["Duration"], axis = 1, inplace = True)
test["Duration"]=(test["Duration_hours"]*60)+test["Duration_mins"]
test.drop(["Duration_hours","Duration_mins"],inplace=True,axis=1)


# Categorical data

print("Airline")
print("-"*75)
print(test["Airline"].value_counts())
Airline = pd.get_dummies(test["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test["Source"].value_counts())
Source = pd.get_dummies(test["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test["Destination"].value_counts())
Destination = pd.get_dummies(test["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test + Airline + Source + Destination
final_test = pd.concat([test, Airline, Source, Destination], axis = 1)

final_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", final_test.shape)


# In[38]:


final_test.head()


# In[39]:


scaled=scaler.transform(final_test.iloc[:,1:8])
final_test['Journey_day']=scaled[:,0]
final_test["Journey_month"]=scaled[:,1]
final_test["Dep_hour"]=scaled[:,2]
final_test['Dep_min']=scaled[:,3]
final_test["Arrival_hour"]=scaled[:,4]
final_test["Arrival_min"]=scaled[:,5]
final_test["Duration"]=scaled[:,6]


# # Feature Importance

# In[40]:


X=final_train.drop(["Price"],axis=1)
X.head()


# In[41]:


y=final_train.Price
y.head()


# In[42]:


plt.figure(figsize = (12,12))
sns.heatmap(train.corr(), annot = True, cmap = "viridis")
plt.show()


# In[43]:


from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)
plt.figure(figsize = (10,8))
feature = pd.Series(selection.feature_importances_, index=X.columns)
feature.nlargest(20).plot(kind='barh')
plt.show()


# # Model Building 

# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[46]:


from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics



# In[50]:


from sklearn.ensemble import RandomForestRegressor
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

reg_rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42)


rf_random.fit(X_train,y_train)


# In[51]:


rf_random.best_params_


# In[52]:


prediction = rf_random.predict(X_test)

plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()


# In[53]:


plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")


# In[54]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[55]:


metrics.r2_score(y_test,prediction)


# # Save Model

# In[62]:


import pickle
with open("flight_prices.pkl",'wb') as f:
    pickle.dump(rf_random,f)

with open("flight_prices.pkl",'rb') as f:
    model=pickle.load(f)


# In[63]:


y_prediction = model.predict(X_test)
metrics.r2_score(y_test,prediction)


# # Conclusions

# **Best model: Random Forest Regressor**
# - MAE: 1143.70131
# - MSE: 3524552.2059
# - RMSE: 1877.37907
# - R-squared: 0.83653
# 
# **Most important features**
# - Total_Stops
# - Journey_Day
# - Jet Airways

# # Future Aspects

# Use of other models.
# Deployment of Model using Flask.
