import os, itertools, math
import numpy as np
import networkx as nx
from Bio import SeqIO, Seq
from codons import codons
from classify import recency_bin
from tempfile import NamedTemporaryFile
from pyseqdist import hamming as ghosthamm
from subprocess import check_call
from collections import defaultdict
from scipy.stats import entropy, pearsonr
from scipy.sparse.csgraph import connected_components,csgraph_from_dense

def parse_input(input): #get sequences from a file
    seqs=defaultdict(int)
    seqlens=defaultdict(int)
    with open(input,'r') as f:
        for record in SeqIO.parse(f,'fasta'):
            freq = int(record.id.split('_')[-1])
            seq=record.seq.upper()
            seqs[seq]+=freq
            seqlens[len(seq)]+=freq
    return seqs, seqlens

def is_there_stop(codon_list):
    for codon in codon_list:
        if codons[codon]=='STOP':
            return 1
    return 0
    
def remove_blanks(seqs,seqlen): #expects aligned sequences
    cons=consensus_seq(seqs,seqlen)
    out={}
    blanks=['-','_','N']
    for seq in seqs:
        o=[]
        for i in range(len(seq)):
            cons_char=cons[i]
            if cons_char not in blanks:
                chr=seq[i]
                if chr in blanks:
                    o.append(cons[i])
                else:
                    o.append(chr)
        for char in blanks:
            if char in o:
                sys.exit('error! did not get all blanks removed?')
        out[''.join(o)]=seqs[seq]
    return out
    
def consensus_seq(dict,seqlen):
    blanks=['-','_','N']
    if len(dict)==1:
        for item in dict:
            return item
    else:
        arr=np.zeros([5,seqlen])
        order={'A':0,'T':1,'C':2,'G':3,'-':4,'N':5,'_':6}
        border={0:'A',1:'T',2:'C',3:'G',4:'-',5:'N',6:'_'}
        for seq in dict: 
            freq=int(dict[seq])
            for id in range(seqlen):
                try:
                    char=seq[id]
                    if char in blanks:
                        arr[order[char],id]+=freq/2
                    else:
                        arr[order[char],id]+=freq
                except IndexError:
                    continue
        out=[]
        for a in range(seqlen):
            slice=list(arr[:,a])
            out.append(border[slice.index(max(slice))])
        return ''.join(out)
    
def remove_n(seqs,seqlens): #does not expect aligned sequences
    m=0
    for s in seqlens:
        if seqlens[s]>m:
            seqlen=s
    cons=consensus_seq(seqs,seqlen)
    out={}
    i=0
    for seq in seqs:
        o=[]
        for i in range(len(seq)):
            try:
                cons_char=cons[i]
                if cons_char!='N' and cons_char!='-':
                    chr=seq[i]
                    if chr=='N' or chr=='-':
                        o.append(cons[i])
                    else:
                        o.append(chr)
            except IndexError:
                chr=seq[i]
                if chr!='N' and chr!='-':
                    o.append(chr)
        if 'N' in o or '-' in o:
            sys.exit('error! did not get all blanks removed?')
        i+=1
        out[''.join(o)]=seqs[seq]
    return out
    
def split_seq(seq, n=3):
    '''Returns sequence split into chunks of n characters, default is codons'''
    start = [seq[i:i + n] for i in range(0, len(seq), n)]
    if len(start[-1])!=n:
        return start[:-1]
    else:
        return start
    
def get_good_seqs(file): #find appropriate reading frame, return sequences with no stop codons in frame
    preseqs,seqlens=parse_input(file)
    seqs=remove_n(preseqs,seqlens)
    zeros=defaultdict(int)
    ones=defaultdict(int)
    twos=defaultdict(int)
    i=0
    for seq in seqs:
        i+=1
        # print(">seq_"+str(i))
        # print(seq)
        freq=seqs[seq]
        seq0=seq
        seq1=seq[1:]
        seq2=seq[2:]
        codons0=split_seq(seq0)
        codons1=split_seq(seq1)
        codons2=split_seq(seq2) 
        # print([(codons[q], q) for q in codons2])
        if not is_there_stop(codons0):
            zeros[seq]+=freq
        if not is_there_stop(codons1):
            ones[seq1]+=freq
        if not is_there_stop(codons2):
            twos[seq2]+=freq
    z=sum(zeros.values())
    o=sum(ones.values())
    t=sum(twos.values())
    m=max([z,o,t])
    if z==0 and o==0 and t==0:
        print("no viable reading frames in "+file)
        return []
    if z==m:
        return zeros
    elif o==m:
        return ones
    elif t==m:
        return twos
    else:
        sys.exit('why has this happened?')

def align(seqs):
    with NamedTemporaryFile(delete=False, mode='w') as seqdump:
        catname=seqdump.name
        for seq in seqs:
            seqdump.write('>seq_'+str(seqs[seq])+'\n'+str(seq)+'\n')
    with NamedTemporaryFile(delete=False) as aligned:
        alignname=aligned.name
        check_call(['mafft', '--quiet', '--auto', '--thread', '20', '--preservecase', catname], stdout=aligned)
    os.unlink(catname)
    seqs,_=parse_input(alignname)
    if type(seqs)==bool:
        print('error parsing aligned seqs!')
        sys.exit()
    return seqs

#######################################
## End sequence parsing portion
## 
## Enter data analysis portion
#######################################
    
def get_dvec(seqs,seqnum,seqlen):
    """Calculate distance matrix return upper triangle as well"""
    DM=calc_distance_matrix(seqs)
    triu_index=np.triu_indices(seqnum,k=1)
    return (DM[triu_index], DM)
        
def calc_distance_matrix(finalSeqs): #calculate distance matrix from the 1-step list
    """Calculate distance matrix over the set of sequences in the sample"""
    l=len(finalSeqs)
    arr=np.zeros([l,l])
    hdist=ghosthamm(finalSeqs,finalSeqs)
    for id in range(len(hdist)):
        item=hdist[id]
        arr[:,id]=item[:,0]
    return arr        
        
def degree_distribution(g):
    x=[]
    for i in g.nodes():
        x.append(g.degree(i))
    return entropy(x,base=2)
        
def phacelia_API(file):
    """make phacelia prediction weighted average over whole file"""
    seqsList=[]
    with open(file) as f:
        for record in SeqIO.parse(f,'fasta'):
            splitid=record.id.split('_')                
            record.annotations['freq']=int(splitid[-1])
            record.annotations['sample']=file
            record.annotations['genotype']=splitid[1]
            record.seq=Seq.Seq(str(record.seq).replace('-','').upper())
            seqsList.append(record)
    seqs=iter(seqsList)
    pdic={}
    for item in recency_bin([seqs]):
        pdic[item[2]]=item[0]
    winner=max(pdic.keys())
    return winner
        
def dna_to_protein(codon):
    '''Returns single letter amino acid code for given codon'''
    return codons[codon]
        
def translate(seq):
    """Translate a DNA sequence into the 1-letter amino acid sequence"""
    return "".join([dna_to_protein(codon) for codon in split_seq(seq)])
        
def get_proteins(seqs):
    proteins=defaultdict(int)
    for seq in seqs:
        prot=translate(seq)
        if 'STOP' not in prot: #discard sequences with unresolvable stop codon in frame
            proteins[prot]+=seqs[seq] 
    return proteins
        
def get_wa0(proteins,total_reads):
    # atchley_dict={'A':[-0.591,-1.302,-0.733,1.570,-0.146],'C':[1.343,0.465,-0.862,-1.020,-0.255],'D':[1.050,0.302,-3.656,-0.259,-3.242],'E':[1.357,-1.453,1.477,0.113,-0.837],'F':[-1.006,-0.590,1.891,-0.397,0.412],'G':[-0.384,1.652,1.330,1.045,2.064],'H':[0.336,-0.417,-1.673,-1.474,-0.078],'I':[-1.239,-0.547,2.131,0.393,0.816],'K':[1.831,-0.561,0.533,-0.277,1.648],'L':[-1.019,-0.987,-1.505,1.266,-0.912],'M':[-0.663,-1.524,2.219,-1.005,1.212],'N':[0.945,0.828,1.299,-0.169,0.933],'P':[0.189,2.081,-1.628,0.421,-1.392],'Q':[0.931,-0.179,-3.005,-0.503,-1.853],'R':[1.538,-0.055,1.502,0.440,2.897],'S':[-0.228,1.399,-4.760,0.670,-2.647],'T':[-0.032,0.326,2.213,0.908,1.313],'V':[-1.337,-0.279,-0.544,1.242,-1.262],'W':[-0.595,0.009,0.672,-2.128,-0.184],'Y':[0.260,0.830,3.097,-0.838,1.512]}
    atchley_dict={'A':-0.591,'C':1.343,'D':1.050,'E':1.357,'F':-1.006,'G':-0.384,'H':0.336,'I':-1.239,'K':1.831,'L':-1.019,'M':-0.663,'N':0.945,'P':0.189,'Q':0.931,'R':1.538,'S':-0.228,'T':-0.032,'V':-1.337,'W':-0.595,'Y':0.260}
    wa0=0
    for seq in proteins:
        rel_freq=proteins[seq]/total_reads
        for char in seq:
            wa0+=atchley_dict[char]*rel_freq
            if char=='*':
                sys.exit("Error! Stop codon in protein sequences, exiting")
    return wa0

def degree_corell(g):
    v1=[]
    v2=[]
    # print(g.edges())
    for edge in g.edges():
        i=edge[0]
        j=edge[1]
        v1.append(g.degree(i))
        v2.append(g.degree(j))
    # print(len(v1),len(v2))
    corr,_=pearsonr(v1,v2)
    return corr
    
def nuc44_consensus(seqs,seqlen,seqnum):
    ScoringMatrix=np.array([[5,-4,-4,-4],[-4,5,-4,-4],[-4,-4,5,-4],[-4,-4,-4,5]])
    prof=seq_profile(seqs,seqlen,seqnum)
    X=np.matmul(ScoringMatrix,prof)
    s=[]
    for j in range(seqlen):
        myman=(np.tile(X[:,j],[1,4])).reshape([4,4])
        myman2=np.transpose(myman)-ScoringMatrix
        item=sum(myman2**2)**.5
        item2=prof[:,j]
        s.append(np.dot(item,item2))
    return np.mean(s)
    
def seq_profile(seqs,seqlen,seqnum):
    order={'A':0,'C':1,'G':2,'T':3}
    out=np.zeros([4,seqlen])
    c=0
    for b in range(len(seqs)):
        seq=seqs[b]
        for col in range(seqlen):
            if seq[col]!='-':
                id=order[seq[col]]
                out[id,col]+=1
            else:
                c+=1
    return np.divide(out,seqnum,dtype='float')
    
def get_adj(DM,thr_dist,seqnum):
    return 1*(DM <= thr_dist) - np.eye(seqnum)    
    
def get_std_dist(dvec):
    return np.std(dvec,ddof=1)

def kmer_entropy_inscape(seqs,k):# calculate kmer entropy 
    kmers=get_kmers(seqs,k)
    totalKmers=float(sum(kmers.values()))
    kmer_rel_freq=np.divide(list(kmers.values()),sum(kmers.values()),dtype='float')
    return entropy(kmer_rel_freq,base=2)
    
def get_kmers(seqs,kmer_len):
    kmers=defaultdict(int)
    for seq in seqs:
        for item in window(seq,kmer_len):
            kmers[''.join(item)]+=seqs[seq]
    return kmers    
    
def window(seq, n): # https://docs.python.org/release/2.3.5/lib/itertools-example.html 
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result
    
def get_comp_freqs(adj,rel_freq):
    sparseMatrix = csgraph_from_dense(adj)
    connected = connected_components(sparseMatrix, directed=False, connection='weak', return_labels=True)
    comp_num = connected[0]
    comp_list = connected[1]
    comps_freqlist=np.zeros(comp_num)
    comps_info=[]
    for i in range(comp_num):
        comps_info.append([])
    for i in range(len(rel_freq)):
        freq=rel_freq[i]
        comp=comp_list[i]
        comps_freqlist[comp]+=freq
        comps_info[comp].append(i)
    max_comp=max(comps_freqlist)
    return comps_freqlist,list(comps_freqlist).index(max_comp),comps_info

def smaller_adj(adj,allowed): #this function could liekly be replaced
    a=len(allowed)
    s=np.zeros([a,a])
    i=0
    for row in range(len(adj)):
        if row in allowed:
            j=0
            for col in range(len(adj)):
                if col in allowed:
                    s[i,j]=adj[row,col]
                    j+=1
            i+=1
    return s, a
    
def trimfh(file):
    return os.path.splitext(os.path.basename(file))[0]

    
def process_component(only_seqs,only_freqs,component,comps_info,adj,dists):
    nodes_real_names=comps_info[component]
    adj_comp,comp_size=smaller_adj(adj,nodes_real_names)
    sparseMatrixComp = csgraph_from_dense(adj_comp)
    path_dists = shortest_path(sparseMatrixComp, method='auto', directed=False, return_predecessors=False, unweighted=True, overwrite=False)    
    links=[]
    for p in range(comp_size-1):
        for q in range(p+1, comp_size):
            realp=nodes_real_names[p]
            realq=nodes_real_names[q]
            s=[p,q,dists[realp,realq],adj_comp[p][q],path_dists[p][q],only_freqs[realp],only_freqs[realq]]
            links.append(s)
    comp_seqs={}
    for k in nodes_real_names:
        seq = str(only_seqs[k])
        freq=only_freqs[k]
        comp_seqs[seq]=freq
    return links,comp_seqs,adj_comp,comp_size
    
def get_pagerank_broken(g,only_freqs): #pagerank and only_freqs are in a different order
    page_rank = list(nx.pagerank_numpy(g).values())
    print(page_rank)
    print(only_freqs)
    corr,_=pearsonr(page_rank,only_freqs)
    return corr
    
def one_away(seq1,seq2):
    if len(seq1)!=len(seq2):
        exit('error why are seqs different len')
    d=0
    for a,b in zip(seq1,seq2):
        if a!=b:
            d+=1
        if d>1:
            return False
    return True
    
def get_pagerank(seqs,only_freqs):
    g=nx.Graph()
    for seq1,seq2 in itertools.combinations(seqs,2):
        if one_away(seq1,seq2):
            g.add_edge(seq1,seq2)
    
    ##################################################################
    #this part just fixes only_freqs to be congruent with the graph
    # onestep={}
    # for seq in seqs:
        # if seq in g.nodes():
            # onestep[seq]=seqs[seq]
    # only_freqs=list(onestep.values())
    ##################################################################
    # freqs=[]
    # for node in g.nodes():
        # freqs.append(seqs[node])
    page_rank = list(nx.pagerank_numpy(g).values())
    # print(len(seqs))
    # print(len(only_freqs))
    # print(len(page_rank))
    corr,_=pearsonr(only_freqs,page_rank)
    return corr
    # print(freqs)
    # print(page_rank)
    # corr,_=pearsonr(page_rank,freqs)
    # return corr
        
    
def boxStats(boxNet): #fordavid other three calculated here? 
    boxNodes = len(boxNet)
    boxMat = nx.to_numpy_matrix(boxNet)
    ##boxNet characteristics
    degreeRaw = list(boxNet.degree())
    degreeBox = []
    for i in degreeRaw:
        degreeBox.append(i)
    degreeNormBox = np.divide(degreeBox, np.sum(degreeBox), dtype = float)
    eValsBox = np.linalg.eigvals(boxMat)
    spectralRadiusAdjBox = max(abs(eValsBox))
    lapMatBox= np.asarray(nx.to_numpy_matrix(nx.from_scipy_sparse_matrix(nx.laplacian_matrix(boxNet))))
    degreeSumBox = np.sum(degreeBox)
    lapMatBoxNorm =  np.divide(lapMatBox, degreeSumBox, dtype = float)
    eValsLapBoxNorm = np.linalg.eigvals(lapMatBoxNorm)
    eValsLapNonZeroBoxNorm = []
    for i in eValsLapBoxNorm:
        j = abs(i)
        if j > 0:
            eValsLapNonZeroBoxNorm.append(j)
    vonEntropyBox = np.divide(entropyCalc(eValsLapNonZeroBoxNorm), math.log(boxNodes,2), dtype = float)
    degreeEntropyBox = np.divide(entropyCalc(degreeNormBox), math.log(boxNodes,2), dtype = float)
    KSEntropyBox = np.divide(math.log(spectralRadiusAdjBox, 2), math.log(boxNodes-1,2), dtype = float)
    return vonEntropyBox, KSEntropyBox, degreeEntropyBox
    
def entropyCalc(freqM):#different results than scipy.stats.entropy - how to resolve?
    productVectorM = 0
    for i in freqM:
        if i > 0:
            productVectorM = productVectorM + (i*math.log(i, 2))
    entropy = -1*productVectorM
    return entropy
   
def list2str(x):
    return ','.join(map(str,x))
   
def main(files):
    one_step=False
    # vars=['VonEntropy','KSEntropy','degreeEntropy','std_dev','meanConsensus','degree_assortativity','inscape_nuc_kmer_7','inscape_prot_kmer_3','phacelia_score','atchley_wa0','corrPageRankfreq']
    
    vars=['phacelia_score','atchley_wa0','meanConsensus','std_dev','inscape_nuc_kmer_7','inscape_prot_kmer_3','degree_assortativity','degree_entropy_me','VonEntropy','KSEntropy','degreeEntropy','corrPageRankfreq']
    print('file,'+','.join(vars))
    num_samples=len(files)
    for i in range(num_samples):
        file=files[i]
        # print(file)
        preseqs=get_good_seqs(file)
        if len(preseqs)==0:
            print(file+',reading frame error')
            continue
        ali=align(preseqs)
        seqlen=len(list(ali.keys())[0])
        seqs=remove_blanks(ali,seqlen)
        seqlen=len(list(seqs.keys())[0]) #may have changed if a blank was removed from alignment
        if type(seqs)==bool:
            print(file+',error!')
            continue
        only_seqs=list(seqs.keys())
        only_freqs=list(seqs.values())
        seqnum=len(seqs)
        if seqnum==0:
            print(file+'error')
            continue
        total_reads=float(sum(only_freqs))
        rel_freq=np.divide(list(only_freqs),total_reads,dtype='float')
        dvec,DM=get_dvec(only_seqs,seqnum,seqlen) 
        adj_1=get_adj(DM,1,seqnum)
        g=nx.from_numpy_matrix(adj_1)
        if one_step:
            comps_freqlist,major_comp,comps_info=get_comp_freqs(adj_1,rel_freq)
            links,seqs,adj_1,seqnum=process_component(only_seqs,only_freqs,major_comp,comps_info,adj_1,DM)
            only_seqs=list(seqs.keys())
            only_freqs=list(seqs.values())
            total_reads=float(sum(only_freqs))
            rel_freq=np.divide(list(only_freqs),total_reads,dtype='float')
            dvec,DM=get_dvec(only_seqs,seqnum,seqlen) 
            g=nx.from_numpy_matrix(adj_1)
        
        ###############calculate features###############
        corr_page_rank_freq=get_pagerank(seqs,only_freqs)
        von_entropy, ks_entropy, degree_entropy = boxStats(g)
        phacelia_score=phacelia_API(file)
        proteins=get_proteins(seqs) 
        atchley_wa0=get_wa0(proteins,total_reads)
        meanConsensus=nuc44_consensus(only_seqs,seqlen,seqnum)  
        std_dev=get_std_dist(dvec)
        inscape_nuc_kmer_7=kmer_entropy_inscape(seqs,7)
        inscape_prot_kmer_3=kmer_entropy_inscape(proteins,3)
        if sum(sum(adj_1))<=2:
            degree_assortativity=0
            degree_entropy_me=0
        else:
            degree_assortativity=degree_corell(g) #1
            degree_entropy_me=degree_distribution(g) #2
        s=[phacelia_score,atchley_wa0,meanConsensus,std_dev,inscape_nuc_kmer_7,inscape_prot_kmer_3,degree_assortativity,degree_entropy_me,von_entropy,ks_entropy,degree_entropy,corr_page_rank_freq]
        print(trimfh(file)+','+','.join(map(str,s)))
        
if __name__=='__main__':
    done=[]
    # with open('raw_8.csv') as f:
        # for line in f.readlines():
            # done.append(line.split(",")[0]+'.fas')
    files=[]
    for file in os.listdir(os.getcwd()):
        if file.endswith('fas') or file.endswith('fasta') or file.endswith('fa'):
            if file not in done:
                files.append(file)
    main(files)
    
 