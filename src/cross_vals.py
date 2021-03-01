# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:18:49 2021

@author: TAC
"""


from sklearn.model_selection import KFold
from ml_training import *
from dl_training import *
from loss import ContrastiveLoss

criterion=ContrastiveLoss()
from functools import reduce
import numpy as np
import gc
import numpy as np

def weight_reset(m):
    """Reset model weights"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def report_summary(clf_report):
    report_list={'Decision Tree':[],'SVM':[],"logistic regression":[],'K Nearest neighbors':[]}
    acc_clf=[] #accuracies of all classifier
    from functools import reduce
    for j in clf_report.keys():
        report=[]
        acc_report=[]#accuracy of all folds
        for i in range(5):
            splited = [' '.join(x.split()) for x in clf_report[j][i].split('\n\n')]
            header = [x for x in splited[0].split(' ')]
            
            data = np.array(splited[1].split(' ')).reshape(-1, len(header) + 1)
            data = np.delete(data, 0, 1).astype(float)[:,0:-1]
            acc=splited[2].split(' ')[1:2]
            macro=np.array(splited[2].split(' ')[5:8])
            weight=np.array(splited[2].split(' ')[-4:-1])
            index=['Class 0','Class 1','macro','weighted']
            df = pd.DataFrame(np.concatenate((data, np.expand_dims(macro,0),np.expand_dims(weight,0))), columns=header[0:-1],index=index)
            report.append(df)
            acc_report.append(acc)
        #mean #https://stackoverflow.com/a/25058102/11170350
        df_concat = pd.concat(report).astype(float)
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = by_row_index.mean()
        report_list[j].append(df_means)
        acc_clf.append(pd.DataFrame(acc_report))
    accs=pd.concat(acc_clf,axis=1)
    accs.columns=['DT','SVM','LR','KNN']
    accs.loc['mean']=accs.astype('float').mean()
    return report_list,accs

def kfoldcv(model,data,epochs=50,dim='2D',n_splits=5,lr=0.0001,batchsize=8,skip_tuning=False,aug=1):
    kf = KFold(n_splits)
    fold=0
    train_cv=[]
    clf_report={'Decision Tree':[],'SVM':[],"logistic regression":[],'K Nearest neighbors':[]}

    for train_index, test_index in kf.split(data.img):
        opt=torch.optim.AdamW(params=model.parameters(),lr=lr)
        
        train=data.iloc[train_index,:].values
        test=data.iloc[test_index,:].values
        #load images
        train_loader=load_data(train,batchsize,aug=aug,image_D=dim)

        #train on all train images
        model=train_dl(train_loader,epochs,model,"cuda",criterion,opt)
        train_features,train_labels=get_features(train,model,dim)
         #now get embeddings of test data
        test_features,test_labels=get_features(test,model,dim)
        #reset the model
        model.apply(weight_reset)
        #hyper parameter tuning of train data
        dt_clf,_=classifiers_tuning('dt',train_features,train_labels,skip_tuning=skip_tuning)
        svm_clf,_=classifiers_tuning('svm',train_features,train_labels,skip_tuning=skip_tuning)
        knn_clf,_=classifiers_tuning('knn',train_features,train_labels,skip_tuning=skip_tuning)
        lr_clf,_=classifiers_tuning('lr',train_features,train_labels,skip_tuning=skip_tuning)

        #append result of best tuned 5cv 
        if skip_tuning==False:
            dt_result=display_result(dt_clf,train_features,train_labels)
            svm_result=display_result(svm_clf,train_features,train_labels)
            knn_result=display_result(knn_clf,train_features,train_labels)
            lr_result=display_result(lr_clf,train_features,train_labels)
            df=pd.DataFrame(zip(dt_result,svm_result,lr_result,knn_result,),\
                            index=dt_result.index,columns=['dt','svm','lr','knn'])
            train_cv.append(df)
        else:
            dt_clf.fit(train_features,train_labels)
            svm_clf.fit(train_features,train_labels)
            knn_clf.fit(train_features,train_labels)
            lr_clf.fit(train_features,train_labels)
            train_cv='Tuning is skipped'

       
        for clf_model,clf_name in zip([dt_clf,svm_clf,lr_clf,knn_clf],\
                                  ['Decision Tree','SVM',"logistic regression",'K Nearest neighbors']):
           
            test_pred=clf_model.predict(test_features)
            clf_report[clf_name].append(classification_report(test_labels,test_pred))
    del model
    gc.collect()
    return train_cv,report_summary(clf_report)