from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import numpy as np
from skrebate import MultiSURF
from scipy.stats import pearsonr
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import time
from sys import argv

def list2str(x):
    return ','.join(map(str,x))

def ml_data_parser(file):
    data=pd.read_csv(file)
    y=data['status']
    X=data.drop(['file','10_hamming','genotype','status'],axis=1)
    # X=data.drop(['sample','status'],axis=1)
    return X,np.array(y)
    
def select_from_groups(X,scores,num_groups=3):
    """
    Step 1
    Select only one variable from each of the user defined groups of related variables
    """
    names=[]; values=[]
    for i in range(num_groups):
        names.append('')
        values.append(0)
    
    for i,j in zip(X.columns,scores):
        if '#' in i:
            index=int(i.split('#')[1])
            winner_val=values[index]
            if j>winner_val:
                names[index]=i
                values[index]=j
    for i,j in zip(X.columns,scores):
        if '#' in i:
            index=int(i.split('#')[1])
            if i!=names[index]:
                X=X.drop(i,axis=1)
    return X

def skrebate_selection(X,scores):
    """
    Step 2
    Remove approximately half of the features according to scores
    """
    tf=scores>0 #change sensitivity here
    return X.iloc[:,pd.Series(tf).values],tf 
    
def correl_reduce(X,scores,t):
    """
    Step 3
    Loop through all combinations of variables, if the correlation between two of them is high, remove the less informative one
    """
    names=list(X.columns)
    tmp_varnum=len(names)
    corr_elim=np.ones(tmp_varnum)
    X_arr=np.array(X)
    for i in range(tmp_varnum):
        v1=X_arr[:,i]
        for j in range(i+1,tmp_varnum):
            v2=X_arr[:,j]
            cor,_=pearsonr(v1,v2)
            r_sq=cor*cor
            trigger=corr_elim[i]+corr_elim[j]==2
            if r_sq>t and trigger:
                if scores[i]<scores[j]:
                    corr_elim[i]=0
                else:
                    corr_elim[j]=0    
    tf2=corr_elim!=0
    return X.iloc[:,pd.Series(tf2).values]#step 2 varnum reduced here
    
def only_useful_vars(X,y,correl_treshold=.8):
    """
    Perform initial feature selection on large feature set in three steps.
    1) Use only one variable from each of the user-defined groups (atchley vals, protein kmer vals, nuc kmer vals).
    2) Use skrebate's agnostic feature importance algorithms to remove roughly half the variables.
    3) Remove any feature which is highly correlated with a better feature.
    """
    #TODO: would it be better if we measured importance of these variables when we have a candidate model at the end rather than at the beginning
    print("starting number of variables from csv: %i"%X.shape[1])
    #step 1
    scores=MultiSURF().fit(np.array(X),y).feature_importances_
    X=select_from_groups(X,scores)#step 1 varnum reduced here
    print("number of variables after 1st reduction: %i"%X.shape[1])
    #step 2
    scores=MultiSURF().fit(np.array(X),y).feature_importances_
    X,tf=skrebate_selection(X,scores)
    print("number of variables after 2nd reduction: %i"%X.shape[1])
    #step 3
    scores=MultiSURF().fit(np.array(X),y).feature_importances_
    X=correl_reduce(X,scores,correl_treshold)
    print("number of variables after 3rd reduction: %i"%X.shape[1])
    out_names=list(X.columns)
    return X,out_names

def run_sfs(X,y,num_features):
    feature_searcher = SFS(RandomForestClassifier(), 
               k_features=num_features,
               forward=True,
               floating=False,
               scoring='accuracy',
               verbose=0,
               cv=5)
    return feature_searcher.fit(X, y)

def main(data_source):
    #arguments
    MIN_FEATURES=6 #exhaustive search options
    MAX_FEATURES=12 #exhaustive search options
    VERBOSE=True
    
    starttime=time.time()
    X,y=ml_data_parser(data_source)    
    # X,y=ml_data_parser('tmp.csv')    
    X,names=only_useful_vars(X,y)
    
    if VERBOSE:
        print('selected %i variables for search:'%len(names))
        print(', '.join(names))
        print("===")
        print("beginning exhaustive search...")
        x=time.time()-starttime
        print('time: %.2f seconds'% x)
        print("===")
    
    models=[]
    scores=[]
    for i in range(MIN_FEATURES,MAX_FEATURES+1):
        model=run_sfs(X,y,i)
        score=model.k_score_
        models.append(model)
        scores.append(score)
        if VERBOSE:
            print(i,score)
    max_score=max(scores)
    
    chosen_model=models[scores.index(max(scores))]
    if VERBOSE:
        print('\nsubsets:')
        for i in chosen_model.subsets_:
            item=chosen_model.subsets_[i]            
            print('model %i=%.3f'%(i,item['avg_score']))
            print(list2str(item['feature_names']))
        print('Best subset (indices):', chosen_model.k_feature_idx_)
    print('Best subset (names):', chosen_model.k_feature_names_)
    print('Score: %.3f' % chosen_model.k_score_)
    x=time.time()-starttime
    print("completed successfully in %.2f seconds. exiting" %x)

if __name__=="__main__":
    #TODO: export other args
    try:
        data_source=sys.argv[1]
    except:
        data_source='values.csv'
    main(data_source)