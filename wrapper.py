from final_echlin import main_1file
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

def list2str(x):
    return ','.join(map(str,x))

def main(recent,chronic,test,output,fullfile):
    """Makes HCV duration of infection predictions on test folder samples
    after building model based off data from samples contained in recent and chronic folders"""
    X=[]
    y=[]
    status=0
    vars=['phacelia_score',
    'atchley_wa0',
    'meanConsensus',
    'std_dev',
    'inscape_nuc_kmer_7',
    'inscape_prot_kmer_3',
    'degree_assortativity',
    'VonEntropy',
    'KSEntropy',
    'degreeEntropy',
    'corrPageRankfreq']
    print(list2str(vars))
    for file in os.listdir(recent):
        if file.endswith('fas') or file.endswith('fasta') or file.endswith('fa'):
            params=main_1file('recent/'+file,fullfile)
            if params==0:
                exit('error in parsing for parameter calculation for %s!'%file)
            X.append(params)
            y.append(status)
    status=1
    for file in os.listdir(chronic):
        if file.endswith('fas') or file.endswith('fasta') or file.endswith('fa'):
            params=main_1file('chronic/'+file,fullfile)
            if params==0:
                exit('error in parsing for parameter calculation for %s!'%file)
            X.append(params)
            y.append(status)
    scaler=MinMaxScaler().fit(X)
    x=scaler.transform(X)
    x_train,x_test,y_train,y_test=train_test_split(X,y)
    clf=ExtraTreesClassifier(n_estimators=100)
    clf.fit(x_train,y_train)
    scores=cross_val_score(clf,X,y,cv=2)
    x_test=[]
    files_test=[]
    with open(output,'w') as f:
        for i in range(len(scores)):
            score=scores[i]
            f.write('Accuracy score for fold %i: %.2f\n' %(i,score))
        for file in os.listdir(test):
            if file.endswith('fas') or file.endswith('fasta') or file.endswith('fa'):
                files_test.append(file)
                params=main_1file('test/'+file,fullfile)
                if params==0:
                    exit('error in parsing for parameter calculation for %s!'%file)
                # fixed_params=np.array(params).reshape(-1,1)
                x_test.append(params)
        predictions=clf.predict(x_test)
        for i in range(len(x_test)):
            prediction=predictions[i]
            file=files_test[i]
            f.write(file+':'+str(prediction)+'\n')

if __name__=="__main__":
    import argparse # possible arguments to add: delta, nIter
    parser = argparse.ArgumentParser(description='DORIS: predict duration of HCV infection')
    parser.add_argument('-r','--recent', 
        type=str, required=False, default="recent",
        help="Path to input folder with recent samples")
    parser.add_argument('-c','--chronic', 
        type=str, required=False, default="chronic",
        help="Path to input folder with chronic samples")
    parser.add_argument('-t','--test', 
        type=str, required=False, default="test",
        help="Path to input folder with test samples")
    parser.add_argument('-o','--output', 
        type=str, required=False, default="DORIS_output.txt",
        help="Desired output file name")
    parser.add_argument('-f',  '--fullfile', 
        action='store_true', default=False,
        help="Pass this as an argument to process the whole file rather than the largest 1-step connected component. Shown to be less accurate.")

    args = parser.parse_args()
    main(args.recent,args.chronic,args.test,args.output,args.fullfile)