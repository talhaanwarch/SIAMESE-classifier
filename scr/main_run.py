# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:20:51 2021

@author: TAC
"""


from loss import ContrastiveLoss
from model import *
import torch
criterion=ContrastiveLoss()
model=SqueezeNetV3()
opt=torch.optim.Adam(params=model.parameters(),lr=0.0001)

#train model
from dl_training import train_dl
trained_model=train_dl(50,model,"cuda",criterion,opt)

#extract features
from ml_training import *
fracture=get_features('../Wrist Fracture/train/Fracture',trained_model)
normal= get_features('../Wrist Fracture/train/Normal',trained_model)

feature=np.array(fracture+normal)
label=np.array([0]*len(normal)+[1]*len(fracture))


dt_clf,_=classifiers_tuning('dt',feature,label)
svm_clf,_=classifiers_tuning('svm',feature,label)
knn_clf,_=classifiers_tuning('knn',feature,label)
lr_clf,_=classifiers_tuning('lr',feature,label)

dt_result=display_result(dt_clf,feature,label)
svm_result=display_result(svm_clf,feature,label)
knn_result=display_result(knn_clf,feature,label)
lr_result=display_result(lr_clf,feature,label)


df=pd.DataFrame(zip(dt_result,svm_result,lr_result,knn_result),index=dt_result.index,columns=['dt','svm','lr','knn'])
print(df)


print("-------------------------------------------------------")
test_fracture=get_features('../Wrist Fracture/test/Fracture',trained_model)
test_normal= get_features('../Wrist Fracture/test/Normal',trained_model)

test_feature=np.array(test_fracture+test_normal)
test_label=np.array([0]*len(test_normal)+[1]*len(test_fracture))

scaler=StandardScaler()
feature=scaler.fit_transform(feature)
test_feature=scaler.transform(test_feature)


dt_clf.fit(feature,label)
test_pred=dt_clf.predict(test_feature)
print(classification_report(test_label,test_pred))

svm_clf.fit(feature,label)
test_pred=svm_clf.predict(test_feature)
print(classification_report(test_label,test_pred))

lr_clf.fit(feature,label)
test_pred=lr_clf.predict(test_feature)
print(classification_report(test_label,test_pred))

knn_clf.fit(feature,label)
test_pred=knn_clf.predict(test_feature)
print(classification_report(test_label,test_pred))











