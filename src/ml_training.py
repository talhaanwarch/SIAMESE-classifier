# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 03:44:10 2021

@author: TAC
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
#model=torch.load('../model_weights/resnet50.pth')
device='cuda' if torch.cuda.is_available() else 'cpu'

from glob import glob    
from PIL import Image

#augmentation data
from dl_training import augmentation
aug=augmentation()
def get_features(df,model,dim='2D'):
    """
    

    Parameters
    ----------
    directory : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.

    Returns
    -------
    feature : TYPE
        DESCRIPTION.

    """
    feature,label=[],[]
    for path in df:
        img = Image.open(path[0])
        if dim=='2D':
            img=img.convert("L")#convert to gray scale
        img=aug(img)#augmented
        img=img.unsqueeze(0) #add another dimension at 0
        emb=model(img.to(device))
        emb=emb.cpu().detach().numpy().ravel()
        feature.append(emb)
        label.append(path[1])
    return np.array(feature),np.array(label)
    

#machine learning part
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import  VotingClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.decomposition import PCA

#svm
def svc_param_selection(X, y,pca=False):
    Cs = [ 0.0001,0.05,0.1,0.5, 1, 10,15,20,25,30,50,70,100]
    gammas = [0.0001,0.005,0.001, 0.01, 0.1,0.3,0.5, 1,1.5,2,5.10]
    pca_n=[5, 10, 15, 20, 30,40,50]
    param_grid = {'svc__C': Cs, 'svc__gamma' : gammas}
    param_grid_pca={'svc__C': Cs, 'svc__gamma' : gammas,'pca__n_components':pca_n}
    if pca:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),PCA(),SVC(kernel='rbf')), param_grid_pca, cv=5,
                               n_jobs=-1,scoring='f1_macro')
    else:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),SVC(kernel='rbf')), param_grid, cv=5,
                               n_jobs=-1,scoring='f1_macro')
    grid_search.fit(X, y)
    return grid_search.best_estimator_ ,grid_search.best_score_



#lr
def logistic_param_selection(X, y,pca=False):
    C= [0.0001,0.005,0.001,0.05,0.01,0.5,0.1, 1,3,5,8, 10,12,15]
    pca_n=[5, 10, 15, 20, 30,40,50]
    param_grid_pca = {'logisticregression__C': C,'pca__n_components':pca_n}
    param_grid = {'logisticregression__C': C}
    if pca:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),PCA(),LogisticRegression(max_iter=500)), param_grid_pca,scoring='f1_macro', cv=5,n_jobs=-1)
    else:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)), param_grid,scoring='f1_macro', cv=5,n_jobs=-1)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_estimator_ ,grid_search.best_score_




def dtree_param_selection(X,y,pca=False):
    param_grid = { 'decisiontreeclassifier__criterion':['gini','entropy'],
                  'decisiontreeclassifier__max_features':["auto", "sqrt", "log2"],
                  'decisiontreeclassifier__max_depth': np.arange(2, 20),
                  'decisiontreeclassifier__random_state':[10,20,30,40,50]}
    
    param_grid_pca = { 'decisiontreeclassifier__criterion':['gini','entropy'],
                      'decisiontreeclassifier__max_features':["auto", "sqrt", "log2"],
                      'decisiontreeclassifier__max_depth': np.arange(2, 20),
                      'decisiontreeclassifier__random_state':[10,20,30,40,50],
                   'pca__n_components':[5, 10, 15, 20, 30,40,50]}
    
    # decision tree model
    dtree_model=DecisionTreeClassifier()
    #use gridsearch to test all values
    if pca:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),PCA(),dtree_model), param_grid_pca, cv=5,n_jobs=-1,scoring='f1_macro')
    else:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),dtree_model), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    #fit model to data
    grid_search.fit(X, y)
    #print(dtree_gscv.best_score_)
    return grid_search.best_estimator_ ,grid_search.best_score_




def knn_param_selection(X, y,pca=False):
    n_neighbors  =list(range(1,10))
    weights  = ['uniform','distance']
    metric=['minkowski','manhattan','euclidean']
    pca_n=[5, 10, 15, 20, 30,40,50]
    param_grid = {'kneighborsclassifier__n_neighbors': n_neighbors, 'kneighborsclassifier__weights' : weights,'kneighborsclassifier__metric':metric}
    
    param_grid_pca = {'kneighborsclassifier__n_neighbors': n_neighbors, 'kneighborsclassifier__weights' : weights,'kneighborsclassifier__metric':metric,'pca__n_components':pca_n}
    
    if pca:
        grid_search =GridSearchCV(make_pipeline(StandardScaler(),PCA(), KNeighborsClassifier()), param_grid_pca, cv=5,n_jobs=-1,scoring='f1_macro')
    else:
        grid_search =GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_estimator_ ,grid_search.best_score_

#from catboost import CatBoostClassifier
def cat_param_selection(X,y,pca=False):
    #create a dictionary of all values we want to test
    param_grid = { 'catboostclassifier__depth':list(np.arange(2,10,2)),
                  'catboostclassifier__l2_leaf_reg':list(np.logspace(-20, -19, 3)),
                  'catboostclassifier__learning_rate': [0.1,0.001,0.05],
                  }
    # decision tree model
    cat_model=CatBoostClassifier(task_type="GPU", loss_function='MultiClass')
    #use gridsearch to test all values
    if pca:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),pca,cat_model), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    else:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),cat_model), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    #fit model to data
    grid_search.fit(X, y)
    #print(dtree_gscv.best_score_)
    return grid_search.best_params_,grid_search.best_score_

#import lightgbm as lgb
def lgbm_param_selection(X,y,pca=False):
    #create a dictionary of all values we want to test
    param_grid = { 
                    'lgbmclassifier__n_estimators':list(np.arange(50,500,50)),
                  'lgbmclassifier__boosting_type':['gbdt', 'dart'],
                  'lgbmclassifier__learning_rate': [0.1,0.001,0.05],
                  'lgbmclassifier__random_state':np.arange(0,30,10)
                  
                  }
    # decision tree model
    lgbm_model=lgb.LGBMClassifier(objective='binary')
    #use gridsearch to test all values
    if pca:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),pca,lgbm_model), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    else:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),lgbm_model), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    #fit model to data
    grid_search.fit(X, y)
    #print(dtree_gscv.best_score_)
    return grid_search.best_params_,grid_search.best_score_

#lgbm_param_selection(feature,label)

#import xgboost
def xgb_param_selection(X,y,pca=None):
    #create a dictionary of all values we want to test
    param_grid = { 
                    'xgbclassifier__n_estimators':list(np.arange(10,500,50)),
                  'xgbclassifier__boosting_type':['dart'],
                  'xgbclassifier__learning_rate': [0.1,0.001,0.05],
                  'xgbclassifier__max_depth':np.arange(1,30,5),
                  'xgbclassifier__random_state ':np.arange(0,30,10),
                  
                  }
    # decision tree model
    xgb_model=xgboost.XGBClassifier(objective='binary:logistic')
    #use gridsearch to test all values
    if pca:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),pca,xgb_model), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    else:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),xgb_model), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    #fit model to data
    grid_search.fit(X, y)
    #print(dtree_gscv.best_score_)
    return grid_search.best_params_,grid_search.best_score_
    
from autogluon.tabular import TabularPredictor
def automl_algo(X,y,train=True):
   
    data=pd.concat([pd.DataFrame(X),pd.DataFrame(y)],axis=1)
    data.columns=['data_{}'.format(i) for i in range(X.shape[1])]+['Label']
#     data.to_csv('test.csv',index=False)
    save_path = 'agModels-predictClass' 
    if train:
        predictor=TabularPredictor(label='Label',path=save_path,verbosity=0).fit(data,refit_full=True,presets='best_quality',time_limit=60*2,)
    else:
        predictor = TabularPredictor.load(save_path) 
        leaderboard = predictor.leaderboard(data,silent=True)
        test_pred = predictor.predict(data,model=leaderboard.model[0])
        return test_pred


def classifiers_tuning(option,feature,label,pca=False,skip_tuning=False):
    """
    

    Parameters
    ----------
    option : str
        Select the machine learning classifier
        Options : svm, dt, lr, knn
    feature : array
        embeddings obtained from deep learning network
    label : array
        class labels
    skip_tuning : bool, optional
        DESCRIPTION. The default is False.
        It is used to skip the grid search in each fold. Setting False
        will improve performance but increase computational time

    Returns
    -------
    clf : Classifier
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    
    
    if skip_tuning==False:
        if option=='svm':
            clf,f1=svc_param_selection(feature,label,pca)
            return clf,f1
            
        elif option=='lr':
            clf,f1=logistic_param_selection(feature,label,pca)
            return clf,f1
        elif option=='knn':
            clf,f1=knn_param_selection(feature,label,pca)
            return clf,f1
        elif option=='dt':
            clf,f1=dtree_param_selection(feature,label,pca)
            return clf,f1

       
    if skip_tuning==True:
        if option=='svm':
            clf=make_pipeline(StandardScaler(),SVC(kernel='rbf'))
            return clf,'Skip tuning'
            
        elif option=='lr':
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))
            return clf,'Skip tuning'
        elif option=='knn':
            clf=make_pipeline(StandardScaler(),KNeighborsClassifier())
            return clf,'Skip tuning'
        elif option=='dt':
            clf=make_pipeline(StandardScaler(),DecisionTreeClassifier())
            return clf,'Skip tuning'



def display_result(clf,feature,label,pca=None):
    """
    

    Parameters
    ----------
    clf :sklearn classifier
        pass an sk learn classifier 
    feature : 2-d array
        pass feature set.
    label : 1-d array
        pass labels.
    pca : TYPE, optional
        DESCRIPTION. Not implemeted. The default is None.
        

    Returns
    -------
    TYPE : pandas dataframe
        average train and test metrics.

    """
        
    if pca:
        result=cross_validate(estimator=clf, X=feature,y=label, 
                       cv=5,scoring=['accuracy','precision_macro', 'recall_macro', 'f1_macro'],return_train_score=True)
    else:
        result=cross_validate(estimator=clf, X=feature,y=label, 
                       cv=5,scoring=['accuracy','precision_macro', 'recall_macro', 'f1_macro'],return_train_score=True)
    result=pd.DataFrame.from_dict(result).T
    #print(result.mean(axis=1))
    return result.mean(axis=1)








