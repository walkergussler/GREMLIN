from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import numpy as np
from skrebate import MultiSURF
from scipy.stats import pearsonr
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import time
from sys import argv

def list2str(x):
    return ','.join(map(str,x))
    
def select_from_groups(X,scores,keywords=['atchley','nuc_kmer','aa_kmer']):
    """
    Step 1
    Select only one variable from each of the user defined groups of related variables
    """    
    all_names=list(X.columns)
    valuable={}
    for keyword in keywords:
        valuable[keyword]=['',-100]

    #get names of 3 important variables from each group
    for name,score in zip(all_names,scores):
        for keyword in keywords:
            if keyword in name:
                current_name,current_val=valuable[keyword]
                if score>current_val:
                    valuable[keyword]=[current_name,current_val]
    #remove all in-group vars which aren't the important one
    for name in all_names:
        for keyword in keywords:
            if keyword in name:
                winner=valuable[keyword][0]
                if name!=winner:
                    X=X.drop(name,axis=1)
    return X

def skrebate_selection(X,scores):
    """
    Step 2
    Remove approximately half of the features according to scores
    """
    tf=scores>=0 #change sensitivity here
    return X.iloc[:,pd.Series(tf).values]
    
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
            if corr_elim[i]+corr_elim[j]==2:
                v2=X_arr[:,j]
                cor,_=pearsonr(v1,v2)   
                r_sq=cor*cor
                if r_sq>t:
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
    X=skrebate_selection(X,scores)
    print("number of variables after 2nd reduction: %i"%X.shape[1])
    #step 3
    scores=MultiSURF().fit(np.array(X),y).feature_importances_
    X=correl_reduce(X,scores,correl_treshold)
    print("number of variables after 3rd reduction: %i"%X.shape[1])
    #REMOVE #step 2 (again) for testing 
    scores=MultiSURF().fit(np.array(X),y).feature_importances_
    X=skrebate_selection(X,scores)
    print("number of variables after 4th reduction: %i"%X.shape[1])
    out_names=list(X.columns)
    return X,out_names

def run_sfs(X,y,num_features): #TODO: is this function implemented sub-optimally? can we just take max features and use the models which sequentially built the largest model
    feature_searcher = SFS(RandomForestClassifier(), 
               k_features=num_features,
               forward=True,
               floating=False,
               scoring='accuracy',
               verbose=0,
               cv=5)
    return feature_searcher.fit(X, y)

def normalize_input(X):
    #nancheck: X.isnull().values.any()
    a=X.values
    inds = np.where(np.isnan(a))
    col_means = np.nanmean(a, axis=0)
    a[inds] = np.take(col_means, inds[1])
    a=MinMaxScaler().fit_transform(a)
    X[:]=a
    if X.isnull().values.any():
        print('NAN value in data! exiting')
        a=X.values
        inds = np.where(np.isnan(a))
        print(inds)
        exit()
    return X

def group_kfold_retest(X,y,models,groups):
    real_scores={}
    models_ref={}
    for model in models:
        scores=[]
        num_features=len(model.k_feature_idx_)
        models_ref[num_features]=model
        names=list(model.k_feature_names_)
        data_tmp=X[names]
        group_kfold=GroupKFold()
        group_kfold.get_n_splits(data_tmp,y,groups)
        for train_index,test_index in group_kfold.split(data_tmp,y,groups):
            print('train: ',train_index)
            print('test: ',test_index)
            X_train=data_tmp.iloc[train_index,:]
            X_test=data_tmp.iloc[test_index,:]
            y_train=y[train_index]
            y_test=y[test_index]
            clf=RandomForestClassifier().fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            scores.append(accuracy_score(y_pred,y_test))
        real_scores[num_features]=np.mean(scores)
    maxscore=max(real_scores.values())
    for var_num in real_scores:
        if real_scores[var_num]==maxscore:
            print('returning',var_num)
            return models_ref[var_num]            

def build_model(data):
    
    #arguments
    MIN_FEATURES=6 
    MAX_FEATURES=7 
    VERBOSE=True
    y=np.array(data['status'])
    groups=np.array(data['group'])
    X=data.drop(['group','status'],axis=1)
    X=normalize_input(X)
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
    # scores=[]
    for i in range(MIN_FEATURES,MAX_FEATURES+1):
        model=run_sfs(X,y,i)
        # score=model.k_score_
        models.append(model)
        # scores.append(score)
        # if VERBOSE:
        #     print(i,score)
    chosen_model=group_kfold_retest(X,y,models,groups)
    #chosen_model=group(models,groups)#TODO:add groups to calculate parameters
    # max_score=max(scores)#TODO: push useful small python scripts 
    #     print('\nsubsets:')
    #     for i in chosen_model.subsets_:
    #         item=chosen_model.subsets_[i]            
    #         print('model %i=%.3f'%(i,item['avg_score']))
    #         print(list2str(item['feature_names']))
    #     print('Best subset (indices):', chosen_model.k_feature_idx_)
    final_names=chosen_model.k_feature_names_
    if VERBOSE:
        print('Best subset (names):', final_names)
        print('Score: %.3f' % chosen_model.k_score_)
    out_data=X[names]
    out_data['status']=pd.Series(y)
    return out_data, chosen_model

def main(data_file):
    output_name='tmp'
    starttime=time.time()
    data=pd.read_csv(data_file,index_col=0)
    model_data,model=build_model(data)
    out_data.to_csv(output_name+'.csv')
    pickle.dump(chosen_model,open(output_name+'.pkl','wb'))
    x=time.time()-starttime
    print("completed successfully in %.2f seconds. exiting" %x)


if __name__=="__main__":
    #TODO: export other args
    try:
        data_source=argv[1]
    except:
        data_source='attempt_2.csv'
        # data_source='random_values.csv'
    main(data_source)
