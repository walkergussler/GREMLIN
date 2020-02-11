from sklearn.model_selection import train_test_split, GroupKFold
import numpy as np
from sys import argv
# from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.cluster import DBSCAN
from skrebate import MultiSURF
from scipy.stats import pearsonr
# from matplotlib import pyplot
# from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.svm import SVC

def ml_data_parser(file):
    # print(file)
    data_end=-3
    groups_col=-2
    c=0
    X=[]
    y=[]
    groups=[]
    with open(file) as f:
        for line in f.readlines():
            c+=1
            if c!=1:
                s=line.strip().split(',') 
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
    return np.array(X),np.array(y),names,groups,groupname

def list2str(x):
    return ','.join(map(str,x))

def get_correlation_graph(X,names,num_vars,t):
    g=nx.Graph()
    for i in range(num_vars):
        v1=X[:,i]
        n1=names[i]
        for j in range(i,num_vars):
            n2=names[j]
            v2=X[:,j]
            cor,_=pearsonr(v1,v2)
            x=abs(cor)
            if x>t:
                g.add_edge(i,j)
    return g

def which_vars(g,feature_importance,t_quantile,names):
    ml_list=list(feature_importance.values())
    model_t=np.quantile(ml_list,t_quantile)
    end_vars=[]
    # x=0
    for clique in nx.find_cliques(g):
        print(clique)
        m=0
        # x+=1
        for var in clique:
            challenger=feature_importance[names[var]]
            if challenger>m:
                m=challenger
                chosen=var
        if m>model_t:
            end_vars.append(chosen)
    # print('cliques='+str(x))
    return list(set(end_vars))

def which_vars_2(g,num_dic,t_quantile,names): #TODO: fix model_t/ why is chosen not being set?
    ml_list=list(feature_importance.values())
    model_t=np.quantile(ml_list,t_quantile)
    # print()
    end_vars=[]
    cliques_list=list(nx.find_cliques(g))
    clique_dict={}
    allowed=list(num_dic.keys())
    for clique in cliques_list:
        dic_key=tuple(clique)
        dic_val=0
        for i in clique:
            val=num_dic[i]
            if val>0:
                dic_val+=val
        clique_dict[dic_key]=dic_val
    for clique, score in sorted(clique_dict.items(), key=lambda item: item[1]):
        m=0
        for var in clique:
            challenger=num_dic[var]
            if challenger>m:
                m=challenger
                chosen=var
        if m>model_t:
            if chosen in allowed:
                # print('chosen = '+names[chosen])
                # print(str(chosen)+':'+list2str(clique))
                # print(list2str(clique))
                # print(list2str(allowed))
                end_vars.append(chosen)
                for var in clique:
                    if var in allowed:
                        allowed.remove(var)
        # print(end_vars)
    # exit()
    return list(set(end_vars))
    
model_list=['rf','rf_extra','svm','nb','lr']

def group_test_2(pre_x,kmeans_labels,names,num_dic,groups,num_vars,meta_i):
    # print('meta-i='+str(meta_i))
    chosen_vars=np.zeros(meta_i)
    chosen_values=np.zeros(meta_i)
    # print('===')
    for i in range(num_vars): #clean this routine up? check for errors?
        old_val=0
        new_val=num_dic[i]
        clust=kmeans_labels[i]        
        old_val=chosen_values[clust]
        if old_val<new_val:
            # print(names[i],old_val,new_val)
            chosen_vars[clust]=int(i)
            chosen_values[clust]=new_val
    # print(chosen_vars)
    # print(type(chosen_vars))
    chosen_works=[]
    chosen_names=[]
    for qq in list(chosen_vars):
        chosen_names.append(names[int(qq)])
        chosen_works.append(int(qq))
    X=pre_x[:,chosen_works]
    # print(chosen_names)
    # x_train,x_test,y_train,y_test=train_test_split(new_x,y,test_size=.3) 
    group_kfold=GroupKFold(n_splits=3)
    group_kfold.get_n_splits(X,y,groups)
    acc_arr=[]
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

        # if model=='svm':
            # clf=SVC().fit(X_train,y_train)
        # elif model=='rf_extra':
            # clf=ExtraTreesClassifier().fit(X_train,y_train)
        # elif model=='rf':
            # clf=RandomForestClassifier().fit(X_train,y_train)
        # elif model=='nb':
            # clf=GaussianNB().fit(X_train,y_train)
        # elif model=='lr':
            # clf=LogisticRegression().fit(X_train,y_train)
        clf=RandomForestClassifier().fit(X_train,y_train)
        # score=np.mean(cross_val_score(svm,X,y,cv=10))
        # print(groupdic[groupid],modeldic[modelid],np.shape(X_test),np.shape(X_train))
        tmp_score=clf.score(X_test,y_test)
        acc_arr.append(tmp_score)
        # print('accuracy='+','+str(qwer)+'\n')
    return np.mean(acc_arr),chosen_names
    
def group_test_3(pre_x,kmeans_labels,names,num_dic,groups,num_vars,meta_i):
    chosen_vars=np.zeros(meta_i)
    chosen_values=np.zeros(meta_i)-1 #subtract one so that decent negative values can be chosen
    # print('===')
    for j in range(meta_i):
        # old_val=np.inf*-1
        # print(j)
        for i in range(num_vars): #clean this routine up? check for errors?
            clust=kmeans_labels[i]        
            if clust==j:
                # print(names[i])
                new_val=num_dic[i]
                old_val=chosen_values[clust]
                if old_val<new_val:
                    # print(names[i],old_val,new_val)
                    chosen_vars[clust]=int(i)
                    chosen_values[clust]=new_val
    # print(chosen_vars)
    # print(type(chosen_vars))
    chosen_works=[]
    chosen_names=[]
    for qq in list(chosen_vars):
        chosen_names.append(names[int(qq)])
        chosen_works.append(int(qq))
    X=pre_x[:,chosen_works]
    group_kfold=GroupKFold(n_splits=10) #make new func
    group_kfold.get_n_splits(X,y,groups)
    acc_arr=[]
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
        clf=RandomForestClassifier().fit(X_train,y_train)
        tmp_score=clf.score(X_test,y_test)
        acc_arr.append(tmp_score)
    return np.mean(acc_arr),chosen_names    
    
try:
    X,y,names,groups,_=ml_data_parser(argv[1])
except IndexError:
    X,y,names,groups,_=ml_data_parser('30_data.csv')
# X,y,names,groups,_=ml_data_parser('74_data.csv')
# print('graph_threshold,multisurf_threshold,model_type,model_acc,num_vars')
X=np.array(X)
y=np.array(y)
num_vars=len(names)
fs = MultiSURF().fit(X,y)
ms_array = list(fs.feature_importances_)
feature_importance={}
num_dic={}
trans_x=np.transpose(X)
max_val=0
for i in range(num_vars):
    feature_importance[names[i]]=ms_array[i]
    num_dic[i]=ms_array[i]
    if max_val<num_dic[i]:
        max_val=num_dic[i]
        best_feature=i
for a in range(10):
    x1=X[:,best_feature].reshape(-1,1)
    group_kfold=GroupKFold(n_splits=10)
    group_kfold.get_n_splits(x1,y,groups)
    acc_arr=[]
    for train_index,test_index in group_kfold.split(X,y,groups):
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        for id in train_index:
            X_train.append(x1[id])
        for id in test_index:
            X_test.append(x1[id])
        for id in train_index:
            y_train.append(y[id])
        for id in test_index:
            y_test.append(y[id])
        clf=RandomForestClassifier().fit(X_train,y_train)
        tmp_score=clf.score(X_test,y_test)
        acc_arr.append(tmp_score)
    qwer=[1,np.mean(acc_arr),names[best_feature]]
    print(list2str(qwer))
    for i in range(2,num_vars):
    # for i in range(2,12):
    # i=2
    # if True:
        kmeans=KMeans(n_clusters=i).fit(trans_x)
        k_labels=kmeans.labels_
        # print(k_labels)
        # print(len(k_labels))
        # kmeans=KMeans(n_clusters=i).fit(X)
        # k_labels=kmeans.labels_
        # num_var=len(set(k_labels))
        # print(num_var,i)
        # for j in range(num_vars):
            # print(names[j],k_labels[j])
        acc,vars=group_test_3(X,k_labels,names,num_dic,groups,num_vars,i)
        # printout=[i,acc]
        # print(i,acc)
        vars.insert(0,acc)
        vars.insert(0,i)
        # vars.insert(0,'model_here')
        print(list2str(vars))
    # for var, score in sorted(feature_importance.items(), key=lambda item: item[1]):
        # print(list2str([var,score]))
    print()