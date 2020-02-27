from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, RFE
# from seaborn import heatmap
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from collections import defaultdict
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Input, Dropout, Flatten, Reshape
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.optimizers import SGD
from random import choice

def list2str(x):
    return ','.join(map(str,x))

def oversample(X,y,names):
    total_allowed=sum(y)
    legal_rows=range(len(y))
    out_X=[]
    out_y=[]
    out_names=[]
    acutes_seen=0
    chronics_seen=0
    # print(len(names))
    while True:
        id=choice(legal_rows)
        # print(id)
        # print(names[id])
        recency_val=y[id]
        if recency_val==0:
            if acutes_seen!=total_allowed:
                acutes_seen+=1
                out_X.append(X[id])
                out_y.append(recency_val)
                out_names.append(names[id])
                # print('acute,'+names[id])
            else:
                if chronics_seen==total_allowed:
                    return out_X,out_y,out_names
        else:
            if chronics_seen!=total_allowed:
                chronics_seen+=1
                out_X.append(X[id])
                out_y.append(recency_val)
                out_names.append(names[id])
                # print('chronic,'+names[id])
            else:
                if acutes_seen==total_allowed:
                    return out_X,out_y,out_names

def custom_tt_split(X_train,y_train):
    y_train=np.reshape(np.array(y_train),[len(y_train),1])
    data=np.concatenate([X_train,y_train],axis=1)
    acutes=[]
    chronics=[]
    for row in data:
        yval=int(row[-1])
        if yval==1:
            chronics.append(row)
        else:
            acutes.append(row)
    num_chron=len(chronics)
    num_acute=len(acutes)
    if num_chron<num_acute:
        chronics_o,acutes_o=oversample(chronics,acutes)
        chronics_u,acutes_u=undersample(chronics,acutes)
    elif num_chron>num_acute:
        acutes_o,chronics_o=oversample(acutes,chronics)
        acutes_u,chronics_u=undersample(acutes,chronics)
    data_o=np.concatenate([acutes_o,chronics_o])
    data_u=np.concatenate([acutes_u,chronics_u])
    # print(len(y_train))
    # print(sum(y_train)[0]*2)
    # print((len(y_train)-sum(y_train)[0])*2)
    # print(np.shape(data_o))
    # print(np.shape(data_u))
    # print('under')
    # for row in data_u:
        # print(list2str(row))
    # print("===")
    # print('over')
    # print("===")
    # for row in data_o:
        # print(list2str(row))
    # exit()
    return data_o[:,:-1],data_o[:,-1],data_u[:,:-1],data_u[:,-1]
    
def undersample(small_class,big_class):
    random_bigclass=np.random.permutation(big_class)
    out=[]
    for i in range(len(small_class)):
        out.append(random_bigclass[i])
    return small_class,out
        
def oversample(small_class,big_class):
    num_2_add=len(big_class)-len(small_class)
    out=small_class[:]
    for i in range(num_2_add):
        out.append(choice(small_class))
    return out,big_class
    
data_end=-3
# sourcefiledic={0:'17',1:'16',2:'13',3:'12',4:'10_1',5:'10_2',6:'5',7:'pelin',8:'6',9:'9'}
# sourcefiledic={0:'11_model.csv'}

# sourcefiledic={0:'17',1:'5',2:'pelin',3:'11',4:'11_no_phacelia'}
# sourcefiledic={0:'11_model.csv',1:'11_no_phacelia_model.csv',2:'pelin.csv',3:'phacelia_only.csv'}
sourcefiledic={0:'pelin_model.csv',1:'11_no_phacelia_model.csv',2:'11_model.csv',3:'phacelia_only.csv'}
# sourcefiledic={0:'tmp.csv'}
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

data_end=-3
for i in range(len(sourcefiledic)):
    reg=[]
    over=[]
    under=[]
    qq=sourcefiledic[i]
    # print(qq)
    # print(type(qq))
    data=pd.read_csv(qq,index_col=0)
    X=data.iloc[:,0:data_end]
    y=data['status']
    groups=data['10_hamming']
    var_names=list(data)
    sample_names=data.index.values
    # X,y,names,groups,samples=ml_data_parser(source_file)
    x_train,_,y_train,_=train_test_split(X,y)
    x_over_train,y_over_train,x_under_train,y_under_train=custom_tt_split(x_train,y_train) #make sure train and test don't have the same guy
    for modelid in range(len(modeldic)):
        if modelid==0:
            clf=RandomForestClassifier(n_estimators=100)
            clf_over=RandomForestClassifier(n_estimators=100)
            clf_under=RandomForestClassifier(n_estimators=100)
        elif modelid==1:
            clf=ExtraTreesClassifier(n_estimators=100)
            clf_over=ExtraTreesClassifier(n_estimators=100)
            clf_under=ExtraTreesClassifier(n_estimators=100)
        elif modelid==2:
            clf=SVC(kernel='linear')
            clf_over=SVC(kernel='linear')
            clf_under=SVC(kernel='linear')
        elif modelid==3:
            clf=GaussianNB()
            clf_over=GaussianNB()
            clf_under=GaussianNB()
        elif modelid==4:
            clf=LogisticRegression(solver='liblinear')
            clf_over=LogisticRegression(solver='liblinear')
            clf_under=LogisticRegression(solver='liblinear')
        clf.fit(x_train,y_train)
        clf_over.fit(x_over_train,y_over_train)
        clf_under.fit(x_under_train,y_under_train)
        scores=cross_val_score(clf,X,y,cv=10)
        scores_over=cross_val_score(clf_over,X,y,cv=10)
        scores_under=cross_val_score(clf_under,X,y,cv=10)
        reg.append(np.mean(scores))
        over.append(np.mean(scores_over))
        under.append(np.mean(scores_under))
    total_storage.append([under,reg,over])
over_under_names=['Undersampling','Regular sampling','Oversampling']
sourcefiledic={0:'pelin_model.csv',2:'11_no_phacelia_model.csv',3:'11_model.csv',1:'phacelia_only.csv'}
desired_names=['GSU','Phacelia','DORIS w/o Phacelia','DORIS']
x=np.zeros([3,4])
for i in sourcefiledic:
    print(desired_names[i]+",Random forest,Extra Random Forest,SVM,logistic regression,naive bayes,average")
    item=total_storage[i]
    for samp_id in range(3):
        avg=np.mean(item[samp_id])
        l=item[samp_id]
        val=l[1]
        l.insert(0,over_under_names[samp_id])
        l.append(str(avg))
        x[samp_id,i]=val
        print(list2str(l))
    print()

print(','+list2str(desired_names))
for a in range(3):
    qw=list(x[a])
    qw.insert(0,over_under_names[a])
    print(list2str(qw))
    