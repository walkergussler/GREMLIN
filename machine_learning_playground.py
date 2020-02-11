from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, RFE
# from seaborn import heatmap
import matplotlib.pyplot as plt 
# import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Input, Dropout, Flatten, Reshape
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.optimizers import SGD

def list2str(x):
    return ','.join(map(str,x))


def mlp(drop_rate=.05, neurons=128):
    model=Sequential()
    model.add(Dense(neurons,activation='relu',input_dim=xlen,kernel_initializer='normal'))
    model.add(Dropout(drop_rate))
    # model.add(Dense(neurons,activation='relu',kernel_initializer='lecun_uniform'))
    # model.add(Dropout(drop_rate))
    model.add(Dense(neurons,activation='tanh',kernel_initializer='lecun_uniform'))
    model.add(Dropout(drop_rate))
    model.add(Dense(1,activation='sigmoid',kernel_initializer='normal'))
    # sgd=SGD(lr=.01,decay=1e-6,momentum=.9,nesterov=True)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

data_end=-3
# sourcefiledic={0:'FINAL2.csv',1:'pelin.csv',2:'final_whoops.csv'}
# sourcefiledic={0:'FINAL2.csv'}
sourcefiledic={0:'17',1:'16',2:'13',3:'12',4:'10_1',5:'10_2',6:'5',7:'pelin'}
modeldic={
    0:'random forest',
    1:'extra_random_forest',
    2:'svm',
    3:'logistic regression',
    4:'naive_bayes'
}
groupdic={
    0:'genotype',
    1:'10 hamming'
}

total_storage=[]

# print("processing 1step data!")
# with open('normalized_1step.csv') as f:

num_models=5
num_groups=2
for i in range(len(sourcefiledic)):
    cv_10x=np.zeros([num_groups,num_models])
    group_x10cv=np.zeros([num_groups,num_models])
    for modelid in range(num_models):
        for groupid in range(num_groups):
            X=[]
            y=[]
            groups=[]
            names=[]
            c=0
            groups_col=(groupid+1)*-1
            regular_arr=[]
            file=sourcefiledic[i]+'_model.csv'
            with open(file) as f:
                for line in f.readlines():
                    c+=1
                    if c!=1:
                    # if not line.startswith('mean'):
                        s=line.strip().split(',') 
                        #X.append(list(map(float,s[4:-2])))
                        # X.append(list(map(float,s[1:-2])))
                        
                        #if there are 4 groups
                        #letters - 1
                        #genotype - 2
                        #20 hamming - 3
                        #10 hamming - 4
                        xx=s[1:data_end]
                        # for i in xx:
                            # print(i,type(i))
                        # print(xx)
                        X.append(list(map(float,xx)))
                        y.append(int(s[data_end]))
                        groups.append(int(s[groups_col]))
                    else:
                        s=line.strip().split(',')
                        names=s[1:data_end]
                        groupname=s[groups_col]
            # from random import shuffle
            # shuffle(y)
            # print(names)
            ######################################## for groupkfold
            group_kfold=GroupKFold(n_splits=10)
            group_kfold.get_n_splits(X,y,groups)
            xlen=len(names)
            for train_index,test_index in group_kfold.split(X,y,groups):
                X_train=[]
                X_test=[]
                y_train=[]
                y_test=[]
                # print(train_index)
                # print(test_index)
                for id in train_index:
                    X_train.append(X[id])
                for id in test_index:
                    X_test.append(X[id])
                for id in train_index:
                    y_train.append(y[id])
                for id in test_index:
                    y_test.append(y[id])

            # if True:
                # X_train,X_test,y_train,y_test=train_test_split(X,y)    
                # print(np.shape(X))
                if modelid==0:
                    clf=RandomForestClassifier()
                elif modelid==1:
                    clf=ExtraTreesClassifier()
                elif modelid==2:
                    clf=SVC(kernel='linear')
                elif modelid==3:
                    clf=GaussianNB()
                elif modelid==4:
                    clf=LogisticRegression(solver='liblinear')
                clf.fit(X_train,y_train)
                # print(groupdic[groupid],modeldic[modelid],np.shape(X_test),np.shape(X_train))
                qwer=clf.score(X_test,y_test)
                regular_arr.append(qwer)
                # print('accuracy='+','+str(qwer)+'\n')
                # print(groupdic[groupid],modeldic[modelid],np.shape(X_test),np.shape(X_train))
            scores=cross_val_score(clf,X,y,cv=10)
            # print(np.mean(scores))
            # print(np.mean(regular_arr))
            group_x10cv[groupid,modelid]=np.mean(regular_arr)
            cv_10x[groupid,modelid]=np.mean(scores)
    total_storage.append(group_x10cv)
    total_storage.append(cv_10x)
# total_items=['fulldata_cv10x','fulldata_regular3x','onestep_cv10x','onestep_regular3x','pelin_cv10x','pelin_regular3x','reduced_onestep_cv10x','reduced_onestep_regular3x']
# jdic={0:'group_x10cv',1:'random_x10cv'}
for i in sourcefiledic:
    data=[]
    print(sourcefiledic[i])
    for j in range(2):
        index=i*2+j
        data.append(total_storage[index])
        # print(data)
    print("group_x10cv,Random forest,Extra Random Forest,SVM,logistic regression,naive bayes,average,,random_x10cv,Random forest,Extra Random Forest,SVM,logistic regression,naive bayes,average")
    s=[[],[]]
    for k in range(2):
        item=data[k]
        for j in range(2):
            avg=np.mean(item[j])
            l=list(map(str,list(item[j])))
            l.insert(0,groupdic[j])
            l.append(str(avg))
            l.append('')
            s[j].append(','.join(l))
    for x in s:
        print(list2str(x))
    print()
exit()
########selectkbest
bestfeatures=SelectKBest(score_func=chi2,k=10)
fit=bestfeatures.fit(X,y)
scores=fit.scores_
for i in range(len(names)):
    print(names[i],clf.feature_importances_)
sfm=SelectFromModel(clf)
sfm.fit(X_train,y_train)
for f in sfm.get_support(indices=True):
    print(names[f])
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)
clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
clf_important.fit(X_important_train, y_train)

acc=accuracy_score(y_test, y_pred)

y_important_pred = clf_important.predict(X_important_test)
acc2=accuracy_score(y_test, y_important_pred)
print(acc,acc2)
selector=RFE(clf,3)
selector=selector.fit(X,y)
sup=selector.support_
r=selector.ranking_

for i in range(len(r)):
    print(names[i]+','+str(sup[i])+','+str(r[i]))
    # print(names[i]+','+str(scores[i])+','+str(q[i]),str(coef2[i]))
# x2=np.array(X)    
# g=heatmap(x2.corr())

# df=pd.read_csv('for_pandas.csv')
# ax=heatmap(df.corr())
