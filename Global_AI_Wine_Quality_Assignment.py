#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


Wine_dataset = pd.read_csv('file:///Users/mac/Downloads/ML_Course_19-11-20-main/Project/winequality.csv')


# In[3]:


print("Wine_Data shape: ",Wine_dataset.shape)


# In[4]:


#pd.options.display.max_columns = None #Display all columns
#pd.options.display.max_rows = None #Display all rows


# In[5]:


Wine_dataset.head()


# In[6]:


Wine_dataset.columns


# In[7]:


Wine_dataset.info()


# In[8]:


Wine_dataset.nunique()


# In[9]:


Wine_dataset.describe()


# In[10]:


#Check is there is any NULL value in the data set
Wine_dataset.isnull().sum()


# In[11]:


#Check is there is any NULL value in the data set
Wine_dataset.isnull().any().any() 


# In[12]:


Wine_dataset.rename(columns={'fixed acidity': 'fixed_acidity',
                     'citric acid':'citric_acid',
                     'volatile acidity':'volatile_acidity',
                     'residual sugar':'residual_sugar',
                     'free sulfur dioxide':'free_sulfur_dioxide',
                     'total sulfur dioxide':'total_sulfur_dioxide'},
            inplace=True)


# In[13]:


Wine_dataset.head()


# In[14]:


Wine_dataset['quality'].unique()


# In[15]:


Wine_dataset.quality.value_counts().sort_index()


# In[16]:


#Visualize the count for distribution of wine quality
sns.countplot(Wine_dataset['quality'])

plt.title('Distribution of the Quality')


# In[17]:


#Visualize the quality count for the alcohol
sns.countplot(x='pH', hue='quality', data = Wine_dataset)


# In[18]:


# calculate correlation matrix
corr = Wine_dataset.corr()# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, 
            yticklabels=corr.columns, annot=True, 
            cmap='RdPu')
            #cmap=sns.diverging_palette(220, 20, as_cmap=True))
#bottom, top = ax.get_ylim()
#ax.set_ylim(bottom + 0.1, top - 0.1);


# In[19]:


Wine_dataset.plot(kind='scatter', x='quality', y='citric_acid')


# In[20]:


sns.pairplot(Wine_dataset)


# In[21]:


#Wine_dataset['quality'].plot(kind='hist', bins=20, figsize=(12,6), facecolor='grey', edgecolor='black')Wine_dataset['sulphates'].plot(kind='hist', bins=20, figsize=(12,6), facecolor='grey',edgecolor='black')
Wine_dataset['quality'].plot(kind='hist', bins=20, figsize=(12,6), facecolor='green',edgecolor='blue')


# In[22]:


plt.figure(figsize=(10,15))

for pl,col in enumerate(list(Wine_dataset.columns.values)):
    plt.subplot(4,3,pl+1)
    sns.set()
    sns.boxplot(col,data=Wine_dataset)
    plt.tight_layout()


# In[23]:


Wine_dataset.corr()['quality'].sort_values()


# In[24]:


Wine_data_Cor = Wine_dataset.drop(['fixed_acidity', 'volatile_acidity', 'density', 'residual_sugar', 'chlorides','total_sulfur_dioxide'], axis=1)


# In[25]:


sns.pairplot(Wine_data_Cor,hue = 'quality');


# In[26]:


#analyse each attribute; alcohol
Wine_dataset['alcohol'].describe()


# In[27]:


#analyse each attribute; alcohol
Wine_dataset['sulphates'].describe()


# In[28]:


#analyse each attribute; alcohol
Wine_dataset['free_sulfur_dioxide'].describe()


# In[29]:


#analyse each attribute; alcohol
Wine_dataset['pH'].describe()


# In[30]:


#Removing the quality column
#Wine_dataset.iloc[:,:11].head()
#Removing the quality column
Wine_data = Wine_dataset.drop(["quality"], axis = 1)


# In[31]:


plt.figure(figsize=(10,15))

for pl,col in enumerate(list(Wine_dataset.iloc[:,:11].columns.values)):
    plt.subplot(4,3,pl+1)
    sns.violinplot(y= Wine_dataset[col],x='quality',data=Wine_dataset, scale='count')
    plt.title(f'quality/{col}')
    plt.tight_layout()
    


# In[32]:


#This plots a 2d scatter plot with a regression line. Easily showing the correlation, distribution, and outliers!

for col in (Wine_dataset.iloc[:,:11].columns.values):
 
    sns.lmplot(x='quality',y=col,data=Wine_dataset, fit_reg=False)
  
    plt.title(f'quality/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    sns.lmplot(x='quality',y=col,data=Wine_dataset)
  
    plt.title(f'quality/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    print('   ')


# In[33]:


#Setting the condition for good and bad ratings
condition = [(Wine_dataset['quality']>6),(Wine_dataset['quality']<=4)]

rating = ['good','bad']

Wine_dataset['rating'] = np.select(condition,rating,default='average')

Wine_dataset.rating.value_counts()


# In[34]:


Wine_dataset.head(20)


# In[35]:


for col in Wine_dataset.iloc[:,:11].columns.values:
 
    
    sns.set()
    sns.violinplot(y= col ,x='rating',data=Wine_dataset, scale='count')
    plt.title(f'rating/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    sns.set()
    sns.swarmplot(x='rating',y=col,data=Wine_dataset)
    plt.title(f'rating/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    print('   ')
    


# In[36]:


Wine_dataset[[('rating'),('quality')]].head(20)


# In[37]:


Wine_dataset.groupby('rating')['quality'].value_counts()


# In[38]:


Wine_dataset['rating'].value_counts()


# In[39]:


print(Wine_dataset['rating'].value_counts(0)[0])
print(Wine_dataset['rating'].value_counts(0)[1])
print(Wine_dataset['rating'].value_counts(0)[2])


# In[40]:


average = Wine_dataset['rating'].value_counts(0)[0]
good = Wine_dataset['rating'].value_counts(0)[1]
bad = Wine_dataset['rating'].value_counts(0)[2]


# In[41]:


#Print the percentage of bad qualitywine
print(bad /(bad + average + good) * 100 , "% of bad quality wine.")


#Print the percentage of average quality wine
print(average /(bad + average + good) * 100 , "% of average quality wine.")

#Print the percentage of good quality wine
print(good /(bad + average + good) * 100 , "% of good quality wine.")


# In[42]:


#This changes the quality from numbers to ratings between good and bad

bins = (2, 4, 9)
group_names = ['bad', 'good']
Wine_dataset['quality'] = pd.cut(Wine_dataset['quality'], bins = bins, labels = group_names)
Wine_dataset.head(20)


# In[43]:


Wine_dataset[[('rating'),('quality')]].head(20)


# In[44]:


#This basically maps all good values to 1 and all bad values to 0 in the quality column

dataset = np.array(Wine_dataset['quality'])

dataset = pd.DataFrame(dataset)

Wine_dataset['quality'] = dataset.apply(lambda x: x.map({'good':1,'bad':0}))

Wine_dataset.head(30)


# In[45]:


Wine_dataset[[('rating'),('quality')]].head(30)


# In[46]:


#Setting the values of X and Y

X =  Wine_dataset[['alcohol','density','sulphates','pH','free_sulfur_dioxide','citric_acid']]
y =  Wine_dataset['quality']


# In[47]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[49]:


X_train.shape, X_test.shape


# In[50]:


y_train.shape, y_test.shape 


# In[51]:


stds= StandardScaler()

X_train = stds.fit_transform(X_train)
X_test = stds.fit_transform(X_test)


# In[52]:


#The functions below will be used to measure the accuracy of the model

def generateClassificationReport_Train(y_true,y_pred):
    '''Train data accuracy'''
    print(classification_report(y_true,y_pred));
    print(confusion_matrix(y_true,y_pred));
    print('\n\nTrain Accuracy is: ',
          round(100*accuracy_score(y_true,y_pred),3),'%\n');
    
def generateClassificationReport_Test(y_true,y_pred):
    '''Test data accuracy'''
    print(classification_report(y_true,y_pred));
    print(confusion_matrix(y_true,y_pred));
    print('\n\nTest Accuracy is: ',
          round(100*accuracy_score(y_true,y_pred),3),'%\n');


# In[53]:


#LOGISTIC REGRESSION

logreg = LogisticRegression(max_iter=1000);
logreg.fit(X_train, y_train);


# In[54]:


#TRAIN DATA

y_train_pred = logreg.predict(X_train)
generateClassificationReport_Train(y_train, y_train_pred)


# In[55]:


#TEST DATA

y_test_pred = logreg.predict(X_test)
generateClassificationReport_Test(y_test,y_test_pred)


# In[56]:


#DESCISION TREE

DecisionTree = DecisionTreeClassifier()
DecisionTree.fit(X_train, y_train);


# In[57]:


#TRAIN DATA

y_train_pred = DecisionTree.predict(X_train)
generateClassificationReport_Train(y_train, y_train_pred)


# In[58]:


#TEST DATA

y_test_pred = DecisionTree.predict(X_test)
generateClassificationReport_Test(y_test,y_test_pred)


# In[59]:


#STOCHASTIC GRADIENT DESCENT

SGD = SGDClassifier()
SGD.fit(X_train, y_train);


# In[60]:


#TRAIN DATA

y_train_pred = SGD.predict(X_train)
generateClassificationReport_Train(y_train, y_train_pred)


# In[61]:


#TEST DATA

y_test_pred = SGD.predict(X_test)
generateClassificationReport_Test(y_test,y_test_pred)


# In[62]:


#GAUSSIAN NORMAL DISTRIBUTION 

Gaussian = GaussianNB()
Gaussian.fit(X_train, y_train);


# In[63]:


#TRAIN DATA

y_train_pred = Gaussian.predict(X_train)
generateClassificationReport_Train(y_train, y_train_pred)


# In[64]:


#TEST DATA

y_test_pred = Gaussian.predict(X_test)
generateClassificationReport_Test(y_test,y_test_pred)


# In[65]:


#RANDOM FOREST

RandomForest = RandomForestClassifier(n_estimators=100)
RandomForest.fit(X_train, y_train);


# In[66]:


#TRAIN DATA

y_train_pred = RandomForest.predict(X_train)
generateClassificationReport_Train(y_train, y_train_pred)


# In[67]:


#TEST DATA

y_test_pred = RandomForest.predict(X_test)
generateClassificationReport_Test(y_test,y_test_pred)


# In[72]:


#Five different models; 
#Logistic Regression, DecisionTree, RandomForest,GAUSSIAN NORMAL DISTRIBUTION, and STOCHASTIC GRADIENT DESCENT
#were used and each of them gave an accuracy relatively close to 100%
#Random Forest, Logistic Regression, SGD yielded the best results with an acuracy of 95.25%
#GAUSSIAN yielded a close accuracy of 94.5% while DecisionTree gave the lowest accuracy of 90.75%


# In[ ]:


#Name: Ogundeji Olawale Abiodun

