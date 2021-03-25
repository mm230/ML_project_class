#!/usr/bin/env python
# coding: utf-8

# ## Projet Machine Learning

# ## Chargement des données modifiéees de Boston Housing fourni par le Prof

# In[1]:



# Load library
import pandas as pd
# Load data as a dataframe
dataframe = pd.read_csv("Boston_Data.csv", index_col="index")
# Show first 5 rows
dataframe.head(15)


# # Les caracteristiques des données et leurs dimensions

# In[3]:


Y = dataframe['Y']
X = dataframe.drop('Y', axis = 1)
#X = dataframe.iloc[:,:-1]    
# Success
print("L'ensemble de données sur le logement à Boston comporte {} points de données avec {} variables chacun.".format(*dataframe.shape))


# # Les statistiques du jeu de données

# In[4]:


import numpy as np

Y.describe()


# In[5]:


X.describe()


# # Observation des caracteristiques

# La science des données est le processus qui consiste à faire des hypothèses sur les données et à les tester en effectuant certaines tâches. Au départ, nous pourrions faire les hypothèses intuitives suivantes pour chaque caractéristique :

# In[6]:


# Dimension de X
X.shape


# In[7]:


# Dimension de Y
Y.shape


# # Visualisation des donnees

# In[8]:


# Using pyplot
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))

# i: index
for i, col in enumerate(X.columns):
    # 3 plots here hence 1, 3
    plt.subplot(2, 13, i+1)
    x = dataframe[col]
    y = Y
    plt.plot(x, y, 'o')
    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Y')


# # Implementation: melange  et  séparation des  données

# In[8]:


# Import 'train_test_split'
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# melange  et  séparation des  données
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Success
print("Training and testing split was successful.")


# # Le rapport donnees de Test et Train

# In[9]:


# Check if split is actually correct
# We can see it's roughly 80% train and 20% train
# So we can proceed!
print(X.shape[0])
print(float(X_train.shape[0]) / float(X.shape[0]))
print(float(X_test.shape[0]) / float(X.shape[0]))


#  Dimension donnees de Train:

# In[10]:


X_train.shape


# Dimension donnees de Test:

# In[11]:


X_test.shape


# # Normalisation des donnees de test et d'entrainement

# In[12]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train_ac = scalar.fit_transform(X_train)
X_test_ac = scalar.transform(X_test)


# # Entrainer et tester le modèle
# Nous utilisons la LinearRegression de Scikit-learn pour entraîner notre modèle à la fois sur les ensembles d'entrainement et de test.

# ## Insert intercept data train

# In[13]:


n_train, _ = X_train.shape
X_train_ac_df = pd.DataFrame(X_train_ac, columns = X_train.columns)
X_train_ac_df.insert(0, 'intercept', np.ones(n_train), True)
X_train_ac_df.columns


# ## Insert intercept data test

# In[14]:


n_test, _ = X_test.shape
X_test_ac_df = pd.DataFrame(X_test_ac, columns = X_test.columns)
X_test_ac_df.insert(0, 'intercept', np.ones(n_test), True)
X_test_ac_df.columns


# In[15]:


print(X_train_ac_df.columns)
print(X_test_ac_df.columns)


# In[67]:


#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#X_train_constant = sm.add_constant(X_train_ac)


# ## Elimination des caracteristiques pour les p-values superieure à 0.05 en commençant par la plus grande valeur de p-value

# In[16]:


## c'est les caracteristiques supprimés
X_train_ac_df_reduit = X_train_ac_df.drop(['R1', 'R2', 'R3', 'R4', 'R5', 'X2', 'X3', 'X7'], axis=1)


# In[17]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
model = sm.OLS(np.asarray(Y_train), X_train_ac_df_reduit)
res_train = model.fit()
#résultats détaillés
print(res_train.summary())


# In[478]:


### drop features for  training data =
#X_train_ac_df_reduit = 
#X_train_ac_df_reduit = X_train_ac_df.drop(['R1', 'R2', 'R3', 'R4', 'R5', 'X2', 'X3', 'X7'], axis=1)


# In[18]:


### drop features for  test data =
X_test_ac_df_reduit = X_test_ac_df.drop(['R1', 'R2', 'R3', 'R4', 'R5', 'X2', 'X3', 'X7'], axis=1)
X_test_ac_df_reduit


# In[19]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
model = sm.OLS(np.asarray(Y_test), X_test_ac_df_reduit)
res_test = model.fit()
#résultats détaillés
#print(res_test.summary())


# # Calculs intermédiaires

# In[20]:


#paramètres estimés train data
print(res_train.params)
#le R2
print(res_train.rsquared)
#calcul manuel du F à partir des carrés moyens
F = res_train.mse_model / res_train.mse_resid
print(F)
#F fourni par l'objet res
print(res_train.fvalue)


# In[21]:


#valeurs estimées par la régression en resubstitution train
print(res_train.fittedvalues)


# In[22]:


#paramètres estimés test
print(res_test.params)
#le R2
print(res_test.rsquared)
#calcul manuel du F à partir des carrés moyens
F = res_test.mse_model / res_test.mse_resid
print(F)
#F fourni par l'objet res
print(res_test.fvalue)


# In[23]:


#valeurs estimées par la régression en resubstitution test
print(res_test.fittedvalues)


# In[24]:


from sklearn.metrics import mean_squared_error  


mse_train = mean_squared_error(Y_train, res_train.fittedvalues, squared=False)
mse_test = mean_squared_error(Y_test, res_test.fittedvalues, squared=False)


# In[ ]:





# ## Resultats Methode de sélection de variable pas-à-pas descendante::

# In[25]:


#print('CV: ', )
print('R2_score (train): ', res_train.rsquared)
print('R2_score (test): ', res_test.rsquared)
print("RMSE: ", mse_test)


# #  La régression linéaire multiple:

# In[27]:


### Regression lineaire multiple
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from math import sqrt
from sklearn.linear_model import LinearRegression


print(" Méthode: Régression linéaire :")

model_train   = LinearRegression().fit(X_train_ac_df,Y_train)

# Predicting Cross Validation Score the Test set results
cv_linear = cross_val_score(estimator = model_train, X = X_train_ac_df, y = Y_train, cv = 10)

r2_score_train = model_train.score(X_train_ac_df,Y_train)
#print( "r-squared train = ",r2_score_train)

#print("---------------------------------------------")
model_test   = LinearRegression().fit(X_test_ac_df,Y_test)
r2_score_test = model_test.score(X_test_ac_df,Y_test)
#print( "r-squared test = ",r2_score_test)


# In[ ]:





# In[28]:


y_train_predict = model_train.predict(X_train_ac_df)


# In[29]:


y_test_predict = model_test.predict(X_test_ac_df)


# In[30]:


### RMSE
from sklearn.metrics import mean_squared_error  

#mse2_train = mean_squared_error(Y_train, y_train_predict, squared=False)
mse2_test = mean_squared_error(Y_test, y_test_predict, squared=False)


# In[493]:


#from math import sqrt
#print("RMSE train regression multiple lineaire: ", sqrt(mse2_train))
#print("RMSE test regression multiple lineaire: ", sqrt(mse2_test))


# ## Resultats  methode d'analyse regression Lineaire multiple:

# In[31]:


print('CV: ', cv_linear.mean())
print('R2_score (train): ', r2_score_train)
print('R2_score (test): ', r2_score_test)
print("RMSE: ", mse2_test)


# ## Regression de RIDGE

# In[32]:


X_train_ac_df.shape
X_test_ac_df.shape


# In[33]:


from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=3.8, fit_intercept=True))
]

ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train_ac_df, Y_train)


# In[38]:


from sklearn.metrics import r2_score

# Predicting Cross Validation Score the Test set results
cv_ridge = cross_val_score(estimator = ridge_pipe, X = X_train_ac, y = Y_train.ravel(), cv = 10)

# Train
y_pred_ridge_train = ridge_pipe.predict(X_train_ac_df)
r2_score_ridge_train = r2_score(Y_train, y_pred_ridge_train)
## Test
y_pred_ridge_test = ridge_pipe.predict(X_test_ac_df)
r2_score_ridge_test = r2_score(Y_test, y_pred_ridge_test)


# ## Resultat  methode d'analyse regression RIDGE:

# In[39]:


# Predicting RMSE the Test set results
rmse_ridge = (np.sqrt(mean_squared_error(Y_test, y_pred_ridge_test)))
print('CV: ', cv_ridge.mean())
print('R2_score (train): ', r2_score_ridge_train)
print('R2_score (test): ', r2_score_ridge_test)
print("RMSE: ", rmse_ridge)


# ## La régression LASSO

# In[40]:


from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=0.012, fit_intercept=True, max_iter=3000))
]

lasso_pipe = Pipeline(steps)
lasso_pipe.fit(X_train_ac_df, Y_train)


# In[41]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Predicting Cross Validation Score
cv_lasso = cross_val_score(estimator = lasso_pipe, X = X_train_ac_df, y = Y_train, cv = 10)


# ## Calcul des scores:

# In[42]:


# Predicting R2 Score the Test set results
y_pred_lasso_train = lasso_pipe.predict(X_train_ac_df)
r2_score_lasso_train = r2_score(Y_train, y_pred_lasso_train)

# Predicting R2 Score the Test set results
y_pred_lasso_test = lasso_pipe.predict(X_test_ac_df)
r2_score_lasso_test = r2_score(Y_test, y_pred_lasso_test)


# ## Resultats methode d'analyse LASSO:
# 

# In[43]:


# Predicting RMSE the Test set results
rmse_lasso = (np.sqrt(mean_squared_error(Y_test, y_pred_lasso_test)))
print('CV: ', cv_lasso.mean())
print('R2_score (train): ', r2_score_lasso_train)
print('R2_score (test): ', r2_score_lasso_test)
print("RMSE: ", rmse_lasso)


# In[ ]:





# # Comparaison des differentes methodes:
# 

# In[44]:


print("Methode de sélection de variable pas-à-pas descendante:")
print('R2_score (train): ', res_train.rsquared)
print('R2_score (test): ', res_test.rsquared)
print("RMSE: ", mse_test)
print('--------------------------------------------')
print("Methode Regression Lineaire: ")
print('CV: ', cv_linear.mean())
print('R2_score (train): ', r2_score_train)
print('R2_score (test): ', r2_score_test)
print("RMSE: ", mse2_test)
print('--------------------------------------------')
print("Methode Regression de RIDGE: ")
print('CV: ', cv_ridge.mean())
print('R2_score (train): ', r2_score_ridge_train)
print('R2_score (test): ', r2_score_ridge_test)
print("RMSE: ", rmse_ridge)
print('--------------------------------------------')
print("Methode Lasso: ")
print('CV: ', cv_lasso.mean())
print('R2_score (train): ', r2_score_lasso_train)
print('R2_score (test): ', r2_score_lasso_test)
print("RMSE: ", rmse_lasso)


# # Conclusion:   
# Les methode de RIDGE et du LASSO semblent etre beaucoup plus performant que les deux autres methodes.
# 

# # FIN du Projet de Machine Learning

# In[ ]:




