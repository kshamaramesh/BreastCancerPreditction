
# coding: utf-8

# # Investigating Breast Cancer
# 
# Breast cancer that forms in the cells of the breasts. I get data from UCI repository and apply machine learning on it to get prediction model to detect breast cancer.Each record represents follow-up data for one breast cancer case. These are consecutive patients seen by Dr. Wolberg since 1984, and include only those cases exhibiting invasive breast cancer and no evidence of distant metastases at the time of diagnosis. 
# 
# ### Context
# Data From: UCI Machine Learning Repository http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.names
# 
# 

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


df = pd.read_csv('data 2.csv')
print 'df.shape = ',df.shape
df


# In[3]:


from sklearn.model_selection import train_test_split
df_clean = df.drop('id',axis=1)
y_train = df_clean['diagnosis']
X_train = df_clean.drop('diagnosis',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.20)


# In[4]:


from sklearn.decomposition import PCA


# In[5]:


pca = PCA(n_components=2)
pca_value = pca.fit_transform(X_train)


# In[6]:


malignant  = pca_value[y_train=='M']
benign = pca_value[y_train=='B']
plt.scatter(malignant[::,0],malignant[::,1],label="Malignant",color="red",alpha=0.5)
plt.scatter(benign[::,0],benign[::,1],label="Benign",color="green",alpha=0.5)
plt.legend(loc="best")
plt.show()


# In[7]:


# lets check pca with non-linearity 
from sklearn.manifold import Isomap
isomap = Isomap(n_components=2,n_neighbors=6)
pca_value_iso = isomap.fit_transform(X_train)

malignant_iso  = pca_value_iso[y_train=='M']
benign_iso = pca_value_iso[y_train=='B']
plt.scatter(malignant_iso[::,0],malignant_iso[::,1],label="Malignant")
plt.scatter(benign_iso[::,0],benign_iso[::,1],label="Benign")
plt.legend(loc="best")
plt.show()


# In[8]:


# k-folds of data
X_train1,X_test1,y_train1,y_test1=train_test_split(X_train,y_train,test_size=0.30)
X_train2,X_test2,y_train2,y_test2=train_test_split(X_train,y_train,test_size=0.30)
X_train3,X_test3,y_train3,y_test3=train_test_split(X_train,y_train,test_size=0.30)


# In[9]:


from sklearn.svm import SVC


# In[10]:


for gamma_value in [0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3]:
    for C_value in[0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000,3000,10000,30000]:
        clf1 = SVC(C = C_value,gamma = gamma_value)
        clf2 = SVC(C = C_value,gamma = gamma_value)
        clf3 = SVC(C = C_value,gamma = gamma_value)

        clf1.fit(X_train1,y_train1)
        clf2.fit(X_train2,y_train2)
        clf3.fit(X_train3,y_train3)

        model1_score = clf1.score(X_train1,y_train1)
        test1_score = clf1.score(X_test1,y_test1)
        model2_score = clf2.score(X_train2,y_train2)
        test2_score = clf2.score(X_test2,y_test2)
        model3_score = clf3.score(X_train3,y_train3)
        test3_score = clf3.score(X_test3,y_test3)

        print 'C=',C_value,' gamma=',gamma_value
        print 'M1 : ',model1_score,' M2 : ',model2_score,' M3 : ',model3_score ,'M avg',(model1_score+model2_score+model3_score)/3 
        print 'T1 : ',test1_score,' T2 : ',test2_score,' T3 : ',test3_score ,'T avg',(test1_score+test2_score+test3_score)/3


# In[11]:


# main classifier with best value
C_value = 1000
gamma_value = 1e-5
clf = SVC(C=C_value,gamma=gamma_value)
clf.fit(X_train,y_train)
print 'train score : ',clf.score(X_train,y_train)
print 'test score : ',clf.score(X_test,y_test)

