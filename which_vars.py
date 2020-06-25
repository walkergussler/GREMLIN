from sklearn.model_selection import train_test_split, GroupKFold
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.cluster import DBSCAN
from skrebate import MultiSURF
from scipy.stats import pearsonr
# from matplotlib import pyplot
# from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import networkx as nx
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sys import argv
from mlxtend.feature_selection import ExhaustiveFeatureSelector


def ml_data_parser(file):
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
    for clique, score in sorted(clique_dict.items(), key=lambda item: item[1],reverse=True):
        m=-100
        for var in clique:
            challenger=num_dic[var]
            if challenger>m:
                m=challenger
                chosen=var
        if m>model_t:
            if chosen in allowed:
                removed=[]
                for var in clique:
                    if var in allowed:
                        allowed.remove(var)
                        removed.append(var)
                end_vars.append(chosen)
                # print()
                # print('chosen = %.2f,cliqscore=%.2f,varscore=%.2f' %(chosen,score,m))
                # print('removed,'+list2str(sorted(removed)))
                # print('clique,'+list2str(sorted(clique)))
                # print('allowed,'+list2str(sorted(allowed)))
                # print('endvars,'+list2str(sorted(end_vars)))
    # print(list2str(end_vars))
    # if sorted(end_vars)!=sorted(list(set(end_vars))):
        # print(end_vars)
        # print('fail')
        # exit()
    # exit()
    return end_vars
    
model_list=['rf','rf_extra','svm','nb','lr']

def test_varset(X,chosen_vars,y,model):
    new_x=X[:,chosen_vars]
    x_train,x_test,y_train,y_test=train_test_split(new_x,y,test_size=.3)
    if model=='svm':
        svm=SVC().fit(x_train,y_train)
    elif model=='rf_extra':
        svm=ExtraTreesClassifier().fit(x_train,y_train)
    elif model=='rf':
        svm=RandomForestClassifier().fit(x_train,y_train)
    elif model=='nb':
        svm=GaussianNB().fit(x_train,y_train)
    elif model=='lr':
        svm=LogisticRegression().fit(x_train,y_train)
    score=np.mean(cross_val_score(svm,new_x,y,cv=10))
    return score

def group_test(pre_x,chosen_vars,y,model,groups):
    X=pre_x[:,chosen_vars]
    # x_train,x_test,y_train,y_test=train_test_split(new_x,y,test_size=.3) 
    group_kfold=GroupKFold(n_splits=10)
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

        if model=='svm':
            clf=SVC(gamma='auto').fit(X_train,y_train)
        elif model=='rf_extra':
            clf=ExtraTreesClassifier(n_estimators=100).fit(X_train,y_train)
        elif model=='rf':
            clf=RandomForestClassifier(n_estimators=100).fit(X_train,y_train)
        elif model=='nb':
            clf=GaussianNB().fit(X_train,y_train)
        elif model=='lr':
            clf=LogisticRegression(solver='lbfgs').fit(X_train,y_train)
        # score=np.mean(cross_val_score(svm,X,y,cv=10))
        # print(groupdic[groupid],modeldic[modelid],np.shape(X_test),np.shape(X_train))
        tmp_score=clf.score(X_test,y_test)
        acc_arr.append(tmp_score)
        # print('accuracy='+','+str(qwer)+'\n')
    return np.mean(acc_arr)

def forward_selection(X, y, initial_list=[], threshold_in=0.01, verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        ...  # Check the entire function in my github page. 
    return included


X,y,names,groups,_=ml_data_parser('test.csv')
X=np.array(X)
y=np.array(y)
clf=RandomForestClassifier()
efs1 = ExhaustiveFeatureSelector(clf, 
           min_features=1,
           max_features=15,
           scoring='accuracy',
           print_progress=True,
           cv=5)


efs1 = efs1.fit(X, y)

print('Best accuracy score: %.2f' % efs1.best_score_)
print('Best subset (indices):', efs1.best_idx_)
print('Best subset (corresponding names):', efs1.best_feature_names_)

exit()

# for i in range(len(ms_array)):
#     print(names[i]+','+str(ms_array[i]))
feature_importance={}
num_dic={}
for i in range(num_vars):
    feature_importance[names[i]]=ms_array[i]
    feature_importance_rf[names[i]]=rf_array[i]
    num_dic[i]=ms_array[i]

corr_list=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
# model_list=['rf','rf_extra','svm','nb','lr']
model_list=['rf','rf_extra']


g=get_correlation_graph(X,names,num_vars,corr_threshold)
chosen_vars=which_vars_2(g,num_dic,model_threshold,names)
model_acc=group_test(X,chosen_vars,y,model,groups)
var_names=[]
for var in chosen_vars:
    var_names.append(names[var])
q=[corr_threshold,model_threshold,model,model_acc,len(var_names)]


# for model in model_list:
#     out_mat=np.zeros([9,9])
#     if model=='rf':
#         num_vars_mat=np.zeros([9,9])
#         corr_threshold=corr_list[i]
#         for j in range(len(corr_list)):
#             model_threshold=corr_list[j]
#             if len(var_names)!=len(set(var_names)):
#                 print(var_names)
#                 exit()
#             q.extend(var_names)
#             lines_place.append(list2str(q))
#             out_mat[i,j]=model_acc
#             if model=='rf':
#                 num_vars_mat[i,j]=len(var_names)
#     if model=='rf':
#         squares_house.append(num_vars_mat)        
#     squares_house.append(out_mat)
# for square_ind in range(len(squares_house)):
#     mat=squares_house[square_ind]
#     print()
#     print(square_ids[square_ind]+',multisurf_threshold=.1,multisurf_threshold=.2,multisurf_threshold=.3,multisurf_threshold=.4,multisurf_threshold=.5,multisurf_threshold=.6,multisurf_threshold=.7,multisurf_threshold=.8,multisurf_threshold=.9')
#     for i in range(len(mat)):
#         row=mat[i]
#         s=['graph_link_threshold='+str(corr_list[i])]
#         for a in row:
#             s.append(a)
#         print(list2str(s))
# print('graph_threshold,multisurf_threshold,model_type,model_acc,num_vars')
# for line in lines_place:
#     print(line)
# for k in feature_importance:
#     print(k+','+str(feature_importance[k]))
