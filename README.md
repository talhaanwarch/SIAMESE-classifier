# SIAMESE-classifier


This is the main function
```
train_cv,clf_report=kfoldcv(model=densenet,epochs=epoch,batchsize=batchsize,data=data,lr=0.001,skip_tuning=False,aug=1,dim='3D')
for clf in ['Decision Tree','SVM',"logistic regression",'K Nearest neighbors']:
    print('------------------',clf,'----------------')
    print(clf_report[0][clf][0])
    
print('--------------------------Accuracy Table-----------------------------------')
clf_report[1]
```
kfoldcv got following parameters
* dim='3D' or '2D' for 3D images and 2D iamges. 
* skip_tuning=False, no grid search applied for hyper-parameter tuning, 
* Augmentation
-- aug=1-> pytorch augmentation
-- aug=0-> no augmentation
-- aug=2-> albumentation augmentation
-- aug=3-> rand augmentation
