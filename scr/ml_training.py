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
def get_features(directory,model):
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
    feature=[]
    for path in glob(directory+'/*.jpg'):
        img = Image.open(path).convert("L")#convert to gray scale
        img=aug(img)#augmented
        img=img.unsqueeze(0) #add another dimension at 0
        emb=model(img.to(device))
        emb=emb.cpu().detach().numpy().ravel()
        feature.append(emb)
    return feature

    


# #pca
# from sklearn.decomposition import PCA

# pca = PCA().fit(feature)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');


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


#svm
def svc_param_selection(X, y,pca=None):
    Cs = [ 0.0001,0.05,0.1,0.5, 1, 10,15,20,25,30,50,70,100]
    gammas = [0.0001,0.005,0.001, 0.01, 0.1,0.3,0.5, 1]
    param_grid = {'svc__C': Cs, 'svc__gamma' : gammas}
    if pca:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),pca,SVC(kernel='rbf')), param_grid, cv=5,
                               n_jobs=-1,scoring='f1_macro')
    else:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),SVC(kernel='rbf')), param_grid, cv=5,
                               n_jobs=-1,scoring='f1_macro')
    grid_search.fit(X, y)
    return grid_search.best_params_,grid_search.best_score_



#lr
def logistic_param_selection(X, y,pca=None):
    C= [0.0001,0.005,0.001,0.05,0.01,0.5,0.1, 1,3,5,8, 10,12,15]
    param_grid = {'logisticregression__C': C}
    if pca:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),pca,LogisticRegression()), param_grid,scoring='f1_macro', cv=5,n_jobs=-1)
    else:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),LogisticRegression()), param_grid,scoring='f1_macro', cv=5,n_jobs=-1)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_,grid_search.best_score_




def dtree_param_selection(X,y,pca=None):
    #create a dictionary of all values we want to test
    param_grid = { 'decisiontreeclassifier__criterion':['gini','entropy'],
                  'decisiontreeclassifier__max_features':["auto", "sqrt", "log2"],
                  'decisiontreeclassifier__max_depth': np.arange(2, 20),
                  'decisiontreeclassifier__random_state':[10,20,30,40,50]}
    # decision tree model
    dtree_model=DecisionTreeClassifier()
    #use gridsearch to test all values
    if pca:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),pca,dtree_model), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    else:
        grid_search = GridSearchCV(make_pipeline(StandardScaler(),dtree_model), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    #fit model to data
    grid_search.fit(X, y)
    #print(dtree_gscv.best_score_)
    return grid_search.best_params_,grid_search.best_score_


def knn_param_selection(X, y,pca=None):
    n_neighbors  =list(range(1,10))
    weights  = ['uniform','distance']
    metric=['minkowski','manhattan','euclidean']
    param_grid = {'kneighborsclassifier__n_neighbors': n_neighbors, 'kneighborsclassifier__weights' : weights,'kneighborsclassifier__metric':metric}
    if pca:
        grid_search =GridSearchCV(make_pipeline(StandardScaler(),pca, KNeighborsClassifier()), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    else:
        grid_search =GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()), param_grid, cv=5,n_jobs=-1,scoring='f1_macro')
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_,grid_search.best_score_




def classifiers_tuning(option,feature,label):
    """
    

    Parameters
    ----------
    option : str
        select for which machine learning classifier , grid search should be performed
        available options are 'svm','lr','dt','knn'
    feature : 2-d array
        pass feature array
    label : 1-d array
        label array

    Returns
    -------
    clf
        tuned classifier.
    f1-score
        f1-score obtained with grid search.

    """
    if option=='svm':
        svm_param,svm_f1=svc_param_selection(feature,label)
        svm=SVC(C=svm_param['svc__C'],gamma=svm_param['svc__gamma'],probability=True)
        return svm,svm_f1
    elif option=='lr':
        lr_param,lr_f1=logistic_param_selection(feature,label)
        lr=LogisticRegression(C=lr_param['logisticregression__C'])
        return lr,lr_f1
    elif option=='knn':
        knn_param,knn_f1=knn_param_selection(feature,label)
        knn=KNeighborsClassifier(metric=knn_param['kneighborsclassifier__metric'],
                                 n_neighbors=knn_param['kneighborsclassifier__n_neighbors'],
                                 weights=knn_param['kneighborsclassifier__weights'])
        return knn,knn_f1
    elif option=='dt':
        dt_param,dt_f1=dtree_param_selection(feature,label)

        dt=DecisionTreeClassifier(criterion=dt_param['decisiontreeclassifier__criterion'],
                               max_depth=dt_param['decisiontreeclassifier__max_depth'],
                               max_features=dt_param['decisiontreeclassifier__max_features'],
                               random_state=dt_param['decisiontreeclassifier__random_state'])
        return dt,dt_f1
    

from sklearn.model_selection import cross_validate
import pandas as pd

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
        result=cross_validate(estimator=make_pipeline(StandardScaler(),pca,clf), X=feature,y=label, 
                       cv=5,scoring=['accuracy','precision_macro', 'recall_macro', 'f1_macro'],return_train_score=True)
    else:
        result=cross_validate(estimator=make_pipeline(StandardScaler(),clf), X=feature,y=label, 
                       cv=5,scoring=['accuracy','precision_macro', 'recall_macro', 'f1_macro'],return_train_score=True)
    result=pd.DataFrame.from_dict(result).T
    #print(result.mean(axis=1))
    return result.mean(axis=1)


# fracture=get_features('../Wrist Fracture/Fracture',model)
# normal= get_features('../Wrist Fracture/Normal',model)

# feature=np.array(fracture+normal)
# label=np.array([0]*len(normal)+[1]*len(fracture))

# classifiers('svm',feature,label)

#pca=PCA(5)    
#print_results
# dt_result=display_result(dt)
# svm_result=display_result(svm)
# lr_result=display_result(lr)
# knn_result=display_result(knn)

# df=pd.DataFrame(zip(dt_result,svm_result,lr_result,knn_result),index=dt_result.index,columns=['dt','svm','lr','knn'])

# df.to_clipboard()









