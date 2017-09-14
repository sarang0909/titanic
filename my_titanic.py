# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 10:05:14 2017

@author: Vani
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Get Data
dataset = pd.read_csv('train.csv')

#look at Data
dataset.info()
dataset.head()
dataset.describe()
dataset.hist()
'''Observations:
    1.Survived might not be depending on PassengerId,Name,Ticket,Cabin,Embarked,
    we'll keep PassengerId,just to identify Passenger/row'
    2.Sex is categorical,need to encode
    3.Age has null values
    4.Pclass is orederly categorical
    5.Need to check Fare will be useful to find Survived,Fare+Pclass can be combined
    6.Embarked categorical feature can be useful?No
    7.Stratified sampling required to cover all pclasses
'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import StratifiedShuffleSplit
split_object = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split_object.split(dataset,dataset["Pclass"]):
    train_set = dataset.loc[train_index] 
    test_set = dataset.loc[test_index]

#Exploratory Analysis
titanic = train_set.copy()
corr_matrix = titanic.corr()
from pandas.tools.plotting import scatter_matrix
scatter_matrix(titanic)

#Create feature and response
'''X'''
titanic = train_set.drop("Survived",axis=1)
'''y'''
titanic_labels = train_set["Survived"] 

#Clean data

not_required_attr = ['Sex','Name','Ticket','Cabin','Embarked']
titanic_num = titanic.drop(not_required_attr,axis=1)
num_attr = list(titanic_num)
cat_attr = ['Sex']


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion

cat_pipeline = Pipeline([
                         ('dataFrameSelector',DataFrameSelector(cat_attr)),
                         ('encoder',LabelBinarizer())
                        ])
num_pipeline = Pipeline([
                         ('dataFrameSelector',DataFrameSelector(num_attr)),
                         ('imputer',Imputer()),
                         ('scalar',StandardScaler())
                        ])


    
full_pipeline = FeatureUnion(transformer_list = [('num_pipeline',num_pipeline),
                                                 ('cat_pipeline',cat_pipeline)
                                                ])

titanic_prepared = full_pipeline.fit_transform(titanic)


#Fit the model
#1.Logistic Regression
from sklearn.linear_model import LogisticRegression
lg_reg = LogisticRegression(random_state=42)
lg_reg.fit(titanic_prepared,titanic_labels)

from sklearn.model_selection import cross_val_score,cross_val_predict
lg_reg_score = cross_val_score(lg_reg,titanic_prepared,titanic_labels,cv=3,scoring='accuracy')
titanic_pred = cross_val_predict(lg_reg,titanic_prepared,titanic_labels,cv=3,method='predict')

# Check performance
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,precision_recall_curve
cm = confusion_matrix(titanic_labels,titanic_pred)
precision_score(titanic_labels,titanic_pred)
recall_score(titanic_labels,titanic_pred)
lg_reg_f1 = f1_score(titanic_labels,titanic_pred)


#2.SGD Classifier
from sklearn.linear_model import SGDClassifier
sg_class = SGDClassifier(random_state=42)
sg_class.fit(titanic_prepared,titanic_labels)
sg_class_score =cross_val_score(sg_class,titanic_prepared,titanic_labels,cv=3,scoring='accuracy')
titanic_sg_pred = cross_val_predict(sg_class,titanic_prepared,titanic_labels,cv=3,method='predict')
sg_class_f1 = f1_score(titanic_labels,titanic_sg_pred)

#3KNN
from sklearn.neighbors import KNeighborsClassifier
knn_class = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn_class.fit(titanic_prepared,titanic_labels)
knn_class_score =cross_val_score(knn_class,titanic_prepared,titanic_labels,cv=3,scoring='accuracy')
titanic_knn_pred = cross_val_predict(knn_class,titanic_prepared,titanic_labels,cv=3,method='predict')
knn_class_f1 = f1_score(titanic_labels,titanic_knn_pred)
#4.Random Forest
from sklearn.ensemble import RandomForestClassifier
forest_class = RandomForestClassifier(random_state=42)
forest_class.fit(titanic_prepared,titanic_labels)
forest_class_score = cross_val_score(forest_class,titanic_prepared,titanic_labels,cv=3,scoring='accuracy')
titanic_forest_pred = cross_val_predict(forest_class,titanic_prepared,titanic_labels,cv=3,method='predict')
forest_class_f1 = f1_score(titanic_labels,titanic_forest_pred)

np.mean(forest_class_score)

# compare models
cross_score = (np.mean(lg_reg_score), np.mean(sg_class_score), np.mean(knn_class_score), np.mean(forest_class_score))
f1_score = (lg_reg_f1, sg_class_f1, knn_class_f1, forest_class_f1)
#compare_model(cross_score,f1_score)




'''From graph,Forest is best model'''
'''Precision vs Recall'''
titanic_prob = cross_val_predict(forest_class,titanic_prepared,titanic_labels,cv=3,method='predict_proba')
titanic_scores = titanic_prob[:,1]
precisions, recalls, thresholds = precision_recall_curve(titanic_labels, titanic_scores)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)


    plt.plot(recalls[:-1],precisions[:-1], "g-",  linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("precisions", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    plt.figure(figsize=(8, 4))    

#Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [90, 100, 120], 'max_features': [2, 4, 6]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [90, 100], 'max_features': [2, 3, 4]},
  ]
grid_search = GridSearchCV(forest_class,param_grid)
grid_search.fit(titanic_prepared,titanic_labels)

grid_search.best_score_
grid_search.best_index_
grid_search.best_estimator_
grid_search.cv_results_
grid_search.best_params_


#Apply model on test set
'''X'''
titanic_test = test_set.drop("Survived",axis=1)
'''y'''
titanic_test_labels = test_set["Survived"] 
titanic_test_prepared = full_pipeline.fit_transform(titanic_test)
final_model = RandomForestClassifier(random_state=42,max_features=4,n_estimators=120)
#final_model.fit(titanic_test_prepared,titanic_test_labels)
final_model_score = cross_val_score(final_model,titanic_test_prepared,titanic_test_labels,cv=3,scoring='accuracy')
final_model_pred = cross_val_predict(final_model,titanic_test_prepared,titanic_test_labels,cv=3,method='predict')
final_model_accuracy_score = accuracy_score(titanic_test_labels,final_model_pred)




#Common functions and Class
from sklearn.base import BaseEstimator,TransformerMixin
class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self, attribute_names):
       self.attribute_names = attribute_names 
    def fit(self,X,y=None):
       return self
    def transform(self,X):
       return X[self.attribute_names].values
   
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    plt.figure(figsize=(8, 4))    


def compare_model(cross_score,f1_score):
n_groups = 4
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, cross_score, bar_width,
                 alpha=opacity,
                 color='b',
                 label='cross_score')
 
rects2 = plt.bar(index + bar_width, f1_score, bar_width,
                 alpha=opacity,
                 color='g',
                 label='f1_score')
 
plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Scores by Model')
plt.xticks(index + bar_width, ('Logistic', 'SGD', 'KNN', 'Forest'))
plt.legend()
 
plt.tight_layout()
plt.show()        








