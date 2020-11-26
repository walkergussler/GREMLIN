from __future__ import print_function, division
import numpy as np
import pandas as pd
import networkx as nx
import itertools, sys, os, inspect, re
from math import log, floor, isclose
from fractions import Fraction
# from phacelia import recency_bin
from scipy.sparse.csgraph import connected_components,csgraph_from_dense, shortest_path
from scipy.linalg import eig
from scipy.stats import pearsonr, entropy, linregress, variation
from Bio import SeqIO, Seq, BiopythonWarning, pairwise2
from collections import defaultdict, Counter
from tempfile import NamedTemporaryFile
from subprocess import check_call
# import ghost
import warnings
from sklearn.metrics import mean_squared_error
# from pyseqdist import hamming as ghosthamm
warnings.simplefilter('ignore', BiopythonWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#del viable_fraction
codons = {"TTT": "F","TTC": "F","TTA": "L","TTG": "L","TCT": "S","TCC": "S","TCA": "S","TCG": "S","TAT": "Y","TAC": "Y","TAA": "STOP","TAG": "STOP","TGT": "C","TGC": "C","TGA": "STOP","TGG": "W","CTT": "L","CTC": "L","CTA": "L","CTG": "L","CCT": "P","CCC": "P","CCA": "P","CCG": "P","CAT": "H","CAC": "H","CAA": "Q","CAG": "Q","CGT": "R","CGC": "R","CGA": "R","CGG": "R","ATT": "I","ATC": "I","ATA": "I","ATG": "M","ACT": "T","ACC": "T","ACA": "T","ACG": "T","AAT": "N","AAC": "N","AAA": "K","AAG": "K","AGT": "S","AGC": "S","AGA": "R","AGG": "R","GTT": "V","GTC": "V","GTA": "V","GTG": "V","GCT": "A","GCC": "A","GCA": "A","GCG": "A","GAT": "D","GAC": "D","GAA": "E","GAG": "E","GGT": "G","GGC": "G","GGA": "G","GGG": "G"}
    
def list2str(x):
    return ','.join(map(str,x))
    
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
    
def get_min_seqlen(seqs):
    msl=0
    for seq in seqs:
        seqlen=len(seq)
        msl=min(seqlen,msl)
    return msl
    
def all_same(items):
    return all(x == items[0] for x in items)
    
def get_mutation_freq(dict,seqs,freq,seqlen): #this could maybe be sped up with hamming (dist from major to all other seq)
    major,non_majors=get_major_sequence(dict,freq)
    if all_same(list(freq)):
        seq_of_interest=consensus_seq(dict,seqlen)
        seqs2=seqs
    else:
        seq_of_interest=major
        seqs2=list(non_majors.keys())
    #build array with [freq,distance_to_major] (use in dnds? for speedup) (also speed up with pyseqdist.hamming?)
    total=0
    freqsum=0
    for seq in seqs2: 
        freqtmp=dict[seq]
        d=sum(0 if a==b else 1 for a,b in zip(seq,seq_of_interest))/float(seqlen)
        if d!=0:
            total+=d*freqtmp
            freqsum+=freqtmp
    return total/freqsum

def get_atchley(char):
    atchley_dict={'A':[-0.591,-1.302,-0.733,1.570,-0.146],'C':[1.343,0.465,-0.862,-1.020,-0.255],'D':[1.050,0.302,-3.656,-0.259,-3.242],'E':[1.357,-1.453,1.477,0.113,-0.837],'F':[-1.006,-0.590,1.891,-0.397,0.412],'G':[-0.384,1.652,1.330,1.045,2.064],'H':[0.336,-0.417,-1.673,-1.474,-0.078],'I':[-1.239,-0.547,2.131,0.393,0.816],'K':[1.831,-0.561,0.533,-0.277,1.648],'L':[-1.019,-0.987,-1.505,1.266,-0.912],'M':[-0.663,-1.524,2.219,-1.005,1.212],'N':[0.945,0.828,1.299,-0.169,0.933],'P':[0.189,2.081,-1.628,0.421,-1.392],'Q':[0.931,-0.179,-3.005,-0.503,-1.853],'R':[1.538,-0.055,1.502,0.440,2.897],'S':[-0.228,1.399,-4.760,0.670,-2.647],'T':[-0.032,0.326,2.213,0.908,1.313],'V':[-1.337,-0.279,-0.544,1.242,-1.262],'W':[-0.595,0.009,0.672,-2.128,-0.184],'Y':[0.260,0.830,3.097,-0.838,1.512]}
    return atchley_dict[char]
    
def calc_distance_matrix_ghost(seqs): #calculate distance matrix from the 1-step list
    l=len(seqs)
    arr=np.zeros([l,l])
    hdist=ghosthamm(seqs,seqs)
    for id in range(len(hdist)):
        item=hdist[id]
        arr[:,id]=item[:,0]
    return arr

def kolmogorov(s,n):
    c=1
    l=1
    i=0
    k=1
    k_max=1
    stop=0
    while stop==0:
        if s[i+k-1]!=s[l+k-1]:
            if k>k_max:
                k_max=k
            i+=1
            if i==l:
                c+=1
                l+=k_max
                if l+1>n:
                    stop=1
                else:
                    i=0
                    k=1
                    k_max=1
            else:
                k=1
        else:
            k+=1
            if l+k>n:
                c+=1
                stop=1
    return c/(n/log(n,2))
    
def kolmogorov_wrapper(seqs,seqlen): #TODO rel_freq
    total=0
    s=0
    for seq in seqs:
        freq=seqs[seq]
        total+=freq
        s+=kolmogorov(seq,seqlen)*freq
    return s/float(total)
    
def calc_distance_matrix(seqs): 
    l=len(seqs)
    arr=np.zeros([l,l])
    for id1,id2 in itertools.combinations(range(len(seqs)),2):
        seq1=seqs[id1]
        seq2=seqs[id2]
        dist=sum(0 if a==b else 1 for a,b in zip(seq1,seq2))
        arr[id1,id2]=dist
        arr[id2,id1]=dist
    return arr
    
def get_dvec(seqs,num_seqs,seqlen):
    """Calculate distance matrix return upper triangle as well"""
    DM=calc_distance_matrix(seqs)
    triu_index=np.triu_indices(num_seqs,k=1)
    return (DM[triu_index], DM)

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

def get_kmers(seqs,kmer_len):
    kmers=defaultdict(int)
    for seq in seqs:
        for item in window(seq,kmer_len):
            kmers[''.join(item)]+=seqs[seq]
    return kmers
   
def kmer_entropy_inscape(seqs,k):# calculate kmer entropy 
    kmers=get_kmers(seqs,k)
    totalKmers=float(sum(kmers.values()))
    kmer_rel_freq=np.divide(list(kmers.values()),sum(kmers.values()),dtype='float')
    return entropy(kmer_rel_freq,base=2)

def calc_ordered_frequencies(num_seqs,seqlen,seqs,byFreq): 
    freqCount = np.zeros((seqlen, 5))
    productVector = np.zeros((seqlen, 5))
    
    try:
        total_reads=0
        for read in seqs:
            if byFreq:
                freq=seqs[read]
            else:
                freq=1
            total_reads+=freq
            for pos in range(seqlen):
                if read[pos] == 'A':
                    freqCount[pos, 0] = freqCount[pos, 0] + freq
                elif read[pos] == 'C':
                    freqCount[pos, 1] = freqCount[pos, 1] + freq
                elif read[pos] == 'G':
                    freqCount[pos, 2] = freqCount[pos, 2] + freq
                elif read[pos] == 'T':
                    freqCount[pos, 3] = freqCount[pos, 3] + freq
                elif read[pos] == '-':
                    freqCount[pos, 4] = freqCount[pos, 4] + freq
        freqRel = np.divide(freqCount, float(total_reads), dtype = 'float')
    
    except IndexError:
        print("Your files are not aligned and it caused an error! Try again with -a")
    
    for pos in range(seqlen):
        for i in range(5):
            freqPos = freqRel[pos, i]
            if freqPos > 0:
                logFreqRel = log(freqPos, 2)
                productVector[pos, i] = -1*(np.multiply(freqPos, logFreqRel, dtype = 'float'))
    return np.sum(productVector, axis = 1)

def calc_dumb_epistasis(std_dev,seqs,seqlen,num_seqs):
    seq_pert = perturbSeq(seqs,seqlen)
    dvec_pert,_ = get_dvec(seq_pert,num_seqs,seqlen)
    return std_dev/float(np.std(dvec_pert,ddof=1))
    
def nuc_entropy_inscape(seqs,num_seqs,seqlen): # calculate nucleotide entropy
    ent=calc_ordered_frequencies(num_seqs,seqlen,seqs,True)
    return sum(ent)/len(ent)
    
def nuc_div_inscape(freqs,mat,seqlen,num_seqs):#calculate nucleotide diversity
    totalFreq=float(sum(freqs))
    nucDiv=0
    for a,b in itertools.combinations(range(num_seqs),2):
        nucDiv+=freqs[a]*freqs[b]*mat[a,b]*2/(float(seqlen)*totalFreq**2)
    return nucDiv

def order_positions(hVector,seqs,haploSize): #order positions for faster building of k-step network
    invH =  np.multiply(-1, hVector, dtype = 'float')
    ordH = np.argsort(invH)
    # reorder the sequences by entropy
    ordSeqs = []

    for q in seqs:
        i=str(q)
        newOne = ''
        for p in range(haploSize):
            newOne = ''.join([newOne, i[ordH[p]]])
        ordSeqs.append(newOne)
    return ordSeqs

def calc_1step_entropy(adjMatrix,counts): 
    sparseMatrix = csgraph_from_dense(adjMatrix)
    connected = connected_components(sparseMatrix, directed=False, connection='weak', return_labels=True)
    rel_freq=np.divide(counts,sum(counts),dtype='float')
    v=np.zeros(connected[0])
    for i in range(len(connected[1])):
        ele=connected[1][i]
        v[ele]+=rel_freq[i]
    haplo_freq=entropy(v,base=2)
    return entropy(v,base=2)

def seq_profile(seqs,seqlen,num_seqs):
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
    return np.divide(out,num_seqs,dtype='float')

def phacelia_API(file): #TODO: add this back in, removed temporarily
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
    
def nuc44_consensus(seqs,seqlen,num_seqs):
    ScoringMatrix=np.array([[5,-4,-4,-4],[-4,5,-4,-4],[-4,-4,5,-4],[-4,-4,-4,5]])
    prof=seq_profile(seqs,seqlen,num_seqs)
    X=np.matmul(ScoringMatrix,prof)
    s=[]
    for j in range(seqlen):
        myman=(np.tile(X[:,j],[1,4])).reshape([4,4])
        myman2=np.transpose(myman)-ScoringMatrix
        item=sum(myman2**2)**.5
        item2=prof[:,j]
        s.append(np.dot(item,item2))
    return np.mean(s)

def perturbSeq(seqs,seqlen): #iterate through sequences (1::seqlen) and randomly permute each column
    tmp=[]
    for seq in seqs:
        tmp.append(list(seq))
    seqMat=np.array(tmp)
    seqnew = np.zeros(np.shape(seqMat),dtype='S1')
    for i in range(seqlen):
        col = seqMat[:,i]
        seqnew[:,i] = np.random.permutation(col)
    seqout=[]
    for seqarr in seqnew:
        item=np.ndarray.tostring(seqarr).decode('UTF-8')
        seqout.append(''.join(item))
    return seqout

def kmer_entropy_pelin(seqs,seqlen,k): #TODO: compare entropy functions? is this the best one?
    count=0
    kmerEntropy=np.zeros([1,seqlen-k+1])[0]
    for m in range(seqlen-k+1):
        kmerVec=[]
        for n in range(len(seqs)):
            seq=seqs[n]
            kmer=seq[m:k+count]
            kmerVec.append(str(kmer))
        count+=1
        total=0 
        occurence=list(np.unique(kmerVec))
        occurenceCounts=[]
        a=defaultdict(int)
        ub=[]
        for item in kmerVec:
            ub.append(occurence.index(item))
        for item in ub:
            a[item]+=1
        for item in a:
            occurenceCounts.append(a[item])
        sumOcc=float(sum(occurenceCounts))
        for p in range(len(occurenceCounts)):
            freq_kmer=occurenceCounts[p]/sumOcc
            total=total-(freq_kmer*log(freq_kmer,2))
        kmerEntropy[m]=total
    return sum(kmerEntropy)/len(kmerEntropy)

def get_freq_corr(adj,freqs):
    thr_comp = 10
    [S, C] = connected_components(csgraph_from_dense(adj))
    C=list(C)
    corrdeg_i = []
    j=max(set(C),key=C.count)
    idx=[]
    for id in range(len(C)):
        item=C[id]
        if item==j:
            idx.append(id)
    tmp=adj[idx,:]
    A_comp=tmp[:,idx]
    nComp = len(A_comp)
    if nComp < thr_comp:
        return 0
    freq_comp = np.array(freqs)[idx]
    [_,V] = eig(A_comp) #is this the same as matlab eig? did we break things here?
    V = np.real(V)[:,0]
    V = 100*V/float(sum(V))
    corr_i,_=pearsonr(V,freq_comp)
    return corr_i
    
def get_std_dist(dvec):
    return np.std(dvec,ddof=1)

def get_cluster_coeff(adj,num_seqs,g): # is it faster to use adjacency matrix vs networkx?
    deg=sum(adj)
    coeff=2
    C=[]
    for i in range(num_seqs):
        if deg[i]<=1:
            C.append(0)
        else:
            row=adj[i]
            neighbors=[]
            for j in range(num_seqs):
                if row[j]!=0:
                    neighbors.append(j)
            sg=g.subgraph(neighbors)
            edges_s=len(sg.edges())
            C.append(coeff*edges_s/(deg[i]*(deg[i]-1)))
    return sum(C)/num_seqs
        
def get_transver_mut(seqs,seqlen): #list of seqs w/o freqs
    ar=[]
    v=0
    for i in range(seqlen):
        item=[]
        for seq in seqs:
            item.append(seq[i])
        ar.append(item)
    for item in ar:
        s=set(item)
        if ('A' in s or 'G' in s) and ('C' in s or 'T' in s):
            v+=1
    return v/float(seqlen)
    
def get_cv_dist(dvec,std_dev):
    return std_dev/float(np.mean(dvec))
    
def seqAverage(seq,seqlen):
    translation={'C':1,'T':2,'A':3,'G':4}
    counts=dict(Counter(seq))
    t=0
    for nuc in counts:
        c=counts[nuc]
        t+=c*translation[nuc]
    return t/seqlen
    
def seq2num(seq,seqlen):
    average=seqAverage(seq,seqlen)
    translation={'C':1,'T':2,'A':3,'G':4}
    out=[]
    for char in seq:
        out.append(translation[char]-average)
    return out
    
def get_pca_components(seqs,num_seqs,seqlen): #TODO move noblanks preprocessing to main
    #fill gaps (for matlab consistency)
    centered_num=[]
    for preseq in seqs:
        if 'N' in preseq or '_' in preseq or '-' in preseq:
            seq=fill_blanks(preseq)
        else:
            seq=preseq
        #make covariance 
        seq_num=seq2num(seq,seqlen)
        centered_num.append(seq_num)
    
    cov_matrix=np.cov(np.transpose(centered_num))
    w=np.linalg.eig(cov_matrix) 
    eigvec=w[1]
    eigval=np.real(w[0]) # this is not quite the same as matlab's flip(eigval) which it is meant to mimic, however it is close
    norm_eig=eigval/sum(eigval)
    cs=list(np.cumsum(norm_eig))
    for a in cs:
        if a>=.5:
            return (cs.index(a)+1)/len(norm_eig)

def get_s_metric(g,num_seqs):
    s=0
    for edge in g.edges():
        i=edge[0]
        j=edge[1]
        s+=(g.degree(i)*g.degree(j))*2
    return s/(num_seqs**4)

def how_many_codons(seq):
    return floor(len(seq)/3)

def seq_to_codons(seq):
    out=[]
    i=0
    for i in range(how_many_codons(seq)): #this could be optimized perhaps
        q=i*3
        try:
            out.append(seq[q:q+3])
        except IndexError:
            return out
    return out

def is_there_stop(codon_list):
    for codon in codon_list:
        if codons[codon]=='STOP':
            return 1
    return 0
    
def how_many_stop(codon_list):
    i=0
    for codon in codon_list:
        if codons[codon]=='STOP':
            i+=1
    return i
    
def simply_trim_stop(seq1,seq2): #not used, implemented for concordance with matlab's DNDS
    codons_1=seq_to_codons(seq1)
    codons_2=seq_to_codons(seq2)
    out1=[]
    out2=[]
    for i in range(len(codons_1)):
        nt1=codons_1[i]
        nt2=codons_2[i]
        if '-' not in nt1 and '_' not in nt1 and 'N' not in nt1 and '-' not in nt2 and '_' not in nt2 and 'N' not in nt2:
            if codons[nt1]!='STOP' and codons[nt2]!='STOP':
                out1.append(nt1)
                out2.append(nt2)
    r1=''.join(out1)
    r2=''.join(out2)
    return r1,r2
    
def remove_stop(s1,s2,indices):
    outs1=''
    outs2=''
    num_codons=min(floor(len(s1)/3),floor(len(s2)/3))
    for i in range(num_codons):
        if i not in indices:
            q=i*3
            outs1+=s1[q:q+3]
            outs2+=s2[q:q+3]
    return outs1,outs2
    
def dnds_codon(codon):
    '''Returns list of synonymous counts for a single codon.
    http://sites.biology.duke.edu/rausher/DNDS.pdf
    '''
    BASES={'A','C','T','G'}
    syn_list = []
    for i in range(len(codon)):
        base = codon[i]
        other_bases = BASES - {base}
        syn = 0
        for new_base in other_bases:
            new_codon = codon[:i] + new_base + codon[i + 1:]
            syn += int(is_synonymous(codon, new_codon))
        syn_list.append(Fraction(syn, 3))
    return syn_list


def dnds_codon_pair(codon1, codon2):
    """Get the dN/dS for the given codon pair"""
    return average_list(dnds_codon(codon1), dnds_codon(codon2))

def substitutions(seq1, seq2):
    """Returns number of synonymous and nonsynonymous substitutions"""
    dna_changes = hamming(seq1, seq2)
    codon_list1 = split_seq(seq1)
    codon_list2 = split_seq(seq2)
    syn = 0
    for i in range(len(codon_list1)):
        codon1 = codon_list1[i]
        codon2 = codon_list2[i]
        syn += codon_subs(codon1, codon2)
    return (syn, dna_changes - syn)
    
def substitutions_extra(seq1, seq2):
    """Returns number of synonymous and nonsynonymous substitutions"""
    dna_changes = hamming(seq1, seq2)
    codon_list1 = split_seq(seq1)
    codon_list2 = split_seq(seq2)
    syn = 0
    for i in range(len(codon_list1)):
        codon1 = codon_list1[i]
        codon2 = codon_list2[i]
        if hamming(codon1,codon2)==1:
            if codons[codon1]==codons[codon2]:
                a=1 #n=0
            else:   
                a=0 #n=0
        elif hamming(codon1,codon2)==3:
            a=synonymous_diff(codon1,codon2)
        else:
            a=codon_subs(codon1, codon2)
        syn += a
    return (syn, dna_changes - syn)

def split_seq(seq, n=3):
    '''Returns sequence split into chunks of n characters, default is codons'''
    start = [seq[i:i + n] for i in range(0, len(seq), n)]
    if len(start[-1])!=n:
        return start[:-1]
    else:
        return start

def average_list(l1, l2):
    """Return the average of two lists"""
    return [(i1 + i2) / 2 for i1, i2 in zip(l1, l2)]

def dna_to_protein(codon):
    '''Returns single letter amino acid code for given codon'''
    return codons[codon]

def translate(seq):
    """Translate a DNA sequence into the 1-letter amino acid sequence"""
    return "".join([dna_to_protein(codon) for codon in split_seq(seq)])

def is_synonymous(codon1, codon2):
    '''Returns boolean whether given codons are synonymous'''
    return dna_to_protein(codon1) == dna_to_protein(codon2)

def syn_sum(seq1, seq2):
    """Get the sum of synonymous sites from two DNA sequences"""
    syn = 0
    codon_list1 = split_seq(seq1)
    codon_list2 = split_seq(seq2)
    for i in range(len(codon_list1)):
        codon1 = codon_list1[i]
        codon2 = codon_list2[i]
        dnds_list = dnds_codon_pair(codon1, codon2)
        syn += sum(dnds_list)
    return syn

def hamming(s1, s2):
    """Return the hamming distance between 2 DNA sequences"""
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2)) + abs(len(s1) - len(s2))

def codon_subs(codon1, codon2):
    """Returns number of synonymous substitutions in provided codon pair
    Methodology for multiple substitutions from Dr. Swanson, UWashington
    https://faculty.washington.edu/wjs18/dnds.ppt
    """
    diff = hamming(codon1, codon2)
    if diff < 1:
        return 0
    elif diff == 1:
        return int(translate(codon1) == translate(codon2))

    syn = 0
    for i in range(len(codon1)):
        base1 = codon1[i]
        base2 = codon2[i]
        if base1 != base2:
            new_codon = codon1[:i] + base2 + codon1[i + 1:]
            syn += int(is_synonymous(codon1, new_codon))
            syn += int(is_synonymous(codon2, new_codon))
    return syn / diff

def clean_sequence(seq):
    """Simply remove instances of blanks, whitespace, ambiguous characters."""
    tmp=seq.replace('-', '')
    tmp2=tmp.replace('_', '')
    tmp3=tmp2.replace('N', '')
    return tmp3.replace(' ', '')

def fill_blanks(seq):
    """Replace instances of blanks, whitespace, ambiguous characters with most common character in sequence."""
    st=str(seq)
    nuc=Counter(st).most_common(1)[0][0]
    noN=st.replace('N',nuc)
    nohyphen=noN.replace('-',nuc)
    return nohyphen.replace('_',nuc)

def synonymous_diff(c1,c2):
    if hamming(c1,c2)==3:
        a1=codons[c2[0]+c1[1:]]
        a2=codons[c1[0]+c2[1]+c1[2]]
        a3=codons[c1[:2]+c2[2]]
        b1=codons[c1[0]+c2[1:]]
        b2=codons[c2[0]+c1[1]+c2[2]]
        b3=codons[c2[:2]+c1[2]]
        aa1=codons[c1]
        aa2=codons[c2]
        s=[]
        if a3!='STOP' and b1!='STOP':
            s.append([aa1,a3,b1,aa2])
        if b2!='STOP' and a3!='STOP':
            s.append([aa1,a3,b2,aa2])
        if a2!='STOP' and b1!='STOP':
            s.append([aa1,a2,b1,aa2])
        if a2!='STOP' and b3!='STOP':
            s.append([aa1,a2,b3,aa2])
        if a1!='STOP' and b2!='STOP':
            s.append([aa1,a1,b2,aa2])
        if a1!='STOP' and b3!='STOP':
            s.append([aa1,a1,b3,aa2])
        syn=0
        non=0
        for i in range(3):
            for row in s:
                
                if row[i]==row[i+1]:
                    syn+=1
                else:
                    non+=1
        sd=syn/len(s)
        nd=non/len(s)
        return sd
    else:
        exit('error: synonymous_diff expects two codons which have no common characters. exiting')

def dnds_2seq(s1,s2): #in1 is same every time
    S=syn_sum(s1,s2)
    N=len(s1)-S
    sd,nd=substitutions(s1,s2)
    pN=nd/N
    pS=sd/S
    try: #domain error for log functions for d_hat_(s|n) 
        d_hat_s=-.75*log((1-((4/3)*pS)))
        d_hat_n=-.75*log((1-((4/3)*pN)))
        try:
            return d_hat_n/d_hat_s
        except ZeroDivisionError: #d_hat_s==0 (how should I handle this?)
            return d_hat_n
    except ValueError:
        return np.nan


def get_major_sequence(seqsDict,freqs):
    m=max(freqs)
    c=Counter(freqs)
    s=c[max(c)]
    # if s!=1:
    for seq in seqsDict:
        if seqsDict[seq]==m:
            chosen=seq
            break
    non_majors={}
    for seq in seqsDict:
        if seq!=chosen:
            non_majors[seq]=seqsDict[seq]
    return chosen,non_majors

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
        exit('error parsing aligned seqs!')
    return seqs

def find(s, ch):
    return [i for i, x in enumerate(s) if x == ch]
    
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
                exit('error! did not get all blanks removed?')
        out[''.join(o)]=seqs[seq]
    return out

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
            exit('error! did not get all blanks removed?')
        i+=1
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
        
def reduce_seqs(preseqs,threshold):
    seqs={}
    for seq in preseqs:
        if preseqs[seq]<=threshold:
            seqs[seq]=preseqs[seq]
    return seqs

def get_good_seqs(file): #find appropriate reading frame, return sequences with no stop codons in frame
    preseqs,seqlens=parse_input(file)
    seqs=remove_n(preseqs,seqlens)
    zeros=defaultdict(int)
    ones=defaultdict(int)
    twos=defaultdict(int)
    i=0
    for seq in seqs:
        i+=1
        freq=seqs[seq]
        seq0=seq
        seq1=seq[1:]
        seq2=seq[2:]
        codons0=split_seq(seq0)
        codons1=split_seq(seq1)
        codons2=split_seq(seq2) 
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
        exit('why has this happened?')
 
def dnds_wrapper(seqs,freqs):
    total_freq=sum(freqs)
    major,non_majors=get_major_sequence(seqs,freqs)
    dnds=0 
    prot_count=0
    proteins=defaultdict(int)
    i=0
    for seq in seqs:
        prot=translate(seq)
        if 'STOP' not in prot: #discard sequences with unresolvable stop codon in frame
            prot_count+=1
            if seq!=major:
                freq=seqs[seq]
                v=dnds_2seq(major,seq)
                dnds+=v*freq/total_freq
            i+=1
            if prot in proteins:
                proteins[prot]+=seqs[seq] 
            else:
                proteins[prot]=seqs[seq] 
    return dnds,proteins,prot_count

def get_adj(DM,thr_dist,num_seqs):
    return 1*(DM <= thr_dist) - np.eye(num_seqs)    
    
def atchley(proteins,rel_freq): #TODO: fix, atchley_dist4==atchley_dist_cat
    atchley_all=[]
    # rel_freq=[]
    sd=np.zeros(5) #standard deviation
    sd_calc=[[],[],[],[],[]]
    for seq in proteins:
        seq_vals=np.array(list(map(get_atchley,list(seq))))
        atchley_all.append(np.transpose(seq_vals))
    
    if len(proteins)==1: #if there's only one sequence, we can't calculate distances
        d_out=[0,0,0,0,0,0]        
        a=atchley_all[0]
        rf_i=rel_freq[0]
        row_sums=sum(np.transpose(a))
        wa=row_sums*rf_i
        sd_row_out=np.std(row_sums)
        for i in range(5):
            sd[i]=np.std(a[i])
    else:
        wa=np.zeros(5) #weighted average
        sd_rowav=[]
        dists=[[],[],[],[],[],[]]
        for i1 in range(len(atchley_all)):
            a=atchley_all[i1]
            rf_i=rel_freq[i1]
            row_sums=sum(np.transpose(a))
            wa+=row_sums*rf_i
            sd_rowav.append(np.std(row_sums))
            for i in range(5):
                sd_calc[i].append(a[i])
            for i2 in range(i1+1,len(atchley_all)):
                b=atchley_all[i2]
                for i in range(5):
                    va=a[i]
                    vb=b[i]
                    dists[i].append(np.linalg.norm(va-vb))
                cata=va.flatten()
                catb=vb.flatten()
                dists[5].append(np.linalg.norm(cata-catb))    
        d_out=[]
        for i in range(6):  
            if i!=5:
                sd[i]=np.std(sd_calc[i])
            d_out.append(np.mean(dists[i]))
        sd_row_out=np.mean(sd_rowav)
    return sd,wa,d_out,sd_row_out
    
def normalize_vec(v):
    """Force values into [0,1] range and replace nan values with mean"""
    if all_same(v):
        return v
    x=max(v)
    m=min(v)
    d=x-m
    out=[]
    n=np.nanmean(v)
    for i in v:
        if np.isnan(i):
            val=(n-m)/d
        else:
            val=(i-m)/d
        out.append(val)
    return out
    
def degree_corell(g):
    v1=[]
    v2=[]
    for edge in g.edges():
        i=edge[0]
        j=edge[1]
        v1.append(g.degree(i))
        v2.append(g.degree(j))
    corr,_=pearsonr(v1,v2)
    return corr
    
def degree_distribution(g):
    x=[]
    for i in g.nodes():
        x.append(g.degree(i))
    return entropy(x,base=2)
    
def get_cols(seqs):
    """returns transpose of list of sequences in which the first sequence returned is all the sequences 1st position, etc"""
    return [''.join(s) for s in zip(*seqs)]
    
def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else 0

def trimfh(file):
    return os.path.splitext(os.path.basename(file))[0]

def x_highest(dic,num): 
    """get key in dictionary with <num> highest value"""
    if num==1:
        val=max(dic.values())
    elif num==2:
        val=second_largest(dic.values())
    for item in dic: #fix to complain when it finds multiple instances of dic max /second largest value
        if dic[item]==val:
            return item    
    
def david_and_pavel_epistasis(seqs,freqs,ent_array,num_seqs,seqlen,total_reads):
    david_total=0
    pavel_total=0
    cols=get_cols(seqs)
    for i in range(seqlen):
        for j in range(seqlen):
            if i!=j:
                c1=cols[i]
                c2=cols[j]
                dinucs=defaultdict(int)
                c1_freq={}
                c2_freq={}
                for a in range(num_seqs):
                    dinuc=c1[a]+c2[a]
                    dinucs[dinuc]+=freqs[a]
                    if c1[a] in c1_freq:
                        c1_freq[c1[a]]+=freqs[a]
                    else:
                        c1_freq[c1[a]]=freqs[a]
                    if c2[a] in c2_freq:
                        c2_freq[c2[a]]+=freqs[a]
                    else:
                        c2_freq[c2[a]]=freqs[a]
                
                dinuc_freqlist=np.divide(list(dinucs.values()),total_reads)
                david_total+=ent_array[i]+ent_array[j]-entropy(dinuc_freqlist)
                c1_major=x_highest(c1_freq,1)
                c2_major=x_highest(c2_freq,1)
                c1_minor=x_highest(c1_freq,2)
                c2_minor=x_highest(c2_freq,2)
                if c1_minor==None:
                    f_00=0
                    f_01=0
                    if c2_minor==None:
                        f_10=0
                    else:
                        f_10=dinucs[c1_major+c2_minor]
                else:
                    f_01=dinucs[c1_minor+c2_major]
                    if c2_minor==None:
                        f_00=0
                        f_10=0
                    else:
                        f_00=dinucs[c1_minor+c2_minor]
                        f_10=dinucs[c1_major+c2_minor]
                f_11=dinucs[c1_major+c2_major]
                newterm=(f_00+f_11-f_10-f_11)/total_reads
                pavel_total+=newterm
    return david_total,pavel_total
   
def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

            
def graphEntropyCalc(colors): 
    colorList = list(set(list(colors.values())))
    nodeList = list(set(list(colors.keys())))
    nodeNum = len(nodeList)
    p = []
    equalFreq = np.divide(1, nodeNum, dtype= float)
    for i in range(nodeNum):
        p.append(equalFreq)
    colorFreq = []
    for j in colorList:
        colorTemp = 0
        for i in nodeList:
            if colors[i] == j:
                colorTemp = colorTemp + 1
        colorFreq.append(np.divide(colorTemp, nodeNum, dtype = float))
    colorEntropy = []
    for j in colorFreq:
        hTemp = []
        for i in p:
            hTemp.append(i*log(np.divide(1, j, dtype = float),2))
            
        colorEntropy.append(sum(hTemp))
    
    graphEntropy = min(colorEntropy)
    return graphEntropy

def entropyCalc(freqM):#different results than scipy.stats.entropy - how to resolve?
    productVectorM = 0
    for i in freqM:
        if i > 0:
            productVectorM = productVectorM + (i*log(i, 2))
    entropy = -1*productVectorM
    return entropy

def boxStats(boxNet): #fordavid other three calculated here?
    ## matrices    
    boxNodes = len(boxNet)
    boxMat = nx.to_numpy_matrix(boxNet)
    boxSparse = csgraph_from_dense(boxMat)
    boxMatPath = shortest_path(boxSparse, method='auto', directed=False, return_predecessors=False, unweighted=True, overwrite=False)    
    boxPathList = []
    pairsNumBox = len(list(itertools.combinations(range(boxNodes), 2)))
    for i in range(boxNodes-1):
        for j in range(i+1, boxNodes):
            tempDist = boxMatPath[i][j]
            if tempDist > 0 and np.isfinite(tempDist):
                boxPathList.append(tempDist)
    
    ##boxNet characteristics
    degreeRaw = list(boxNet.degree())
    degreeBox = []
    for i in degreeRaw:
        degreeBox.append(i[1])
    degreeNormBox = np.divide(degreeBox, np.sum(degreeBox), dtype = float)
    
    diameterPathBox = np.max(boxPathList)
    avgPathDistBox = np.mean(boxPathList)
    nEdgesBox = np.divide(np.sum(degreeBox), 2, dtype = float)
    edgePBox = nx.density(boxNet)
    globalEfficiencyBox = np.divide(sum(np.divide(1, boxPathList, dtype = float)),pairsNumBox , dtype = float)
    radiusBox = nx.radius(boxNet)
    kCoreBox = max(list(nx.core_number(boxNet).values()))
    degreeAssortBox = nx.degree_assortativity_coefficient(boxNet)
    avgDegreeBox = np.mean(degreeBox)
    maxDegreeBox = max(degreeBox)
    eValsBox = np.linalg.eigvals(boxMat)
    spectralRadiusAdjBox = max(abs(eValsBox))
    eigenCentDictBox = nx.eigenvector_centrality_numpy(boxNet, weight=None)
    eigenCentRawBox = list(eigenCentDictBox.values())
    eigenCentBox = np.divide(eigenCentRawBox, sum(eigenCentRawBox), dtype = float)
    colorsBox = nx.coloring.greedy_color(boxNet, strategy=nx.coloring.strategy_connected_sequential_bfs)
    colorNumBox = len(list(set(list(colorsBox.values()))))
    avgClustCoeffBox = nx.average_clustering(boxNet)                        
    scaledSpectralRadiusBox = np.divide(spectralRadiusAdjBox, avgDegreeBox, dtype = float)
    freqMBox =  [0.166666667, 0.166666667, 0.166666667, 0.166666667, 0.166666667, 0.166666667]
    # network entropy
    lapMatBox= np.asarray(nx.to_numpy_matrix(nx.from_scipy_sparse_matrix(nx.laplacian_matrix(boxNet))))
    eValsLapBox = np.linalg.eigvals(lapMatBox)
    eValsLapBoxSorted = sorted(np.real(eValsLapBox))
    spectralGapBox = eValsLapBoxSorted[1]
    degreeSumBox = np.sum(degreeBox)
    lapMatBoxNorm =  np.divide(lapMatBox, degreeSumBox, dtype = float)
    eValsLapBoxNorm = np.linalg.eigvals(lapMatBoxNorm)
    eValsLapNonZeroBoxNorm = []
    for i in eValsLapBoxNorm:
        j = abs(i)
        if j > 0:
            eValsLapNonZeroBoxNorm.append(j)
    vonEntropyBox = np.divide(entropyCalc(eValsLapNonZeroBoxNorm), log(boxNodes,2), dtype = float)
    degreeEntropyBox = np.divide(entropyCalc(degreeNormBox), log(boxNodes,2), dtype = float)
    KSEntropyBox = np.divide(log(spectralRadiusAdjBox, 2), log(boxNodes-1,2), dtype = float)
    motifEntropyBox = np.divide(entropyCalc(freqMBox), log(len(freqMBox),2), dtype = float)
    popEntropyBox = np.divide(entropyCalc(eigenCentBox), log(boxNodes,2), dtype = float)
    graphEntropyBox = np.divide(graphEntropyCalc(colorsBox), log(boxNodes,2), dtype = float)
    
    return edgePBox, radiusBox, kCoreBox, degreeAssortBox, diameterPathBox, avgPathDistBox, nEdgesBox, globalEfficiencyBox, avgDegreeBox, maxDegreeBox, spectralRadiusAdjBox, spectralGapBox, scaledSpectralRadiusBox, colorNumBox, avgClustCoeffBox, freqMBox, motifEntropyBox, vonEntropyBox, graphEntropyBox, popEntropyBox, KSEntropyBox, degreeEntropyBox

def fractalCalc(dist,nodeNum,nodeList):
    pairsNum = len(dist)
    diameter = max(dist)
    ## if only one sequence
    if nodeNum < 2:
        sumBoxes, hammDb, hammRSquare, hammDbConstant, hammModularity, hammModularityConstant, hammModularityRSquare, diameterPathSelf, avgPathDistSelf, nEdgesSelf, globalEfficiencySelf, avgDegreeSelf, maxDegreeSelf, spectralRadiusAdjSelf, spectralGapSelf, popEntropySelf, scaledSpectralRadiusSelf, colorNumSelf, avgClustCoeffSelf, freqMBoxSelf, motifEntropySelf, graphEntropySelf = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        lb = 1
        nBoxesAll = []
        modLbAll = []
        # initial box size 0
        nBoxes = nodeNum
        nBoxesAll.append(nBoxes)
        modLb = 0
        boxWeightList = []
        
        ## self similarity lists
        global radiusBoxList, kCoreBoxList, degreeAssortBoxList, diameterPathBoxList, avgPathDistBoxList, nEdgesBoxList, globalEfficiencyBoxList, avgDegreeBoxList, maxDegreeBoxList, spectralRadiusAdjBoxList, spectralGapBoxList, colorNumBoxList, avgClustCoeffBoxList, freqMBoxList, motifEntropyBoxList, graphEntropyBoxList, scaledSpectralRadiusBoxList, edgePBoxList, popEntropyBoxList, vonEntropyBoxList, KSEntropyBoxList, degreeEntropyBoxList
        radiusBoxList, kCoreBoxList, degreeAssortBoxList, diameterPathBoxList, avgPathDistBoxList, nEdgesBoxList, globalEfficiencyBoxList, avgDegreeBoxList, maxDegreeBoxList, spectralRadiusAdjBoxList, spectralGapBoxList, colorNumBoxList, avgClustCoeffBoxList,     freqMBoxList, motifEntropyBoxList, graphEntropyBoxList, scaledSpectralRadiusBoxList, edgePBoxList, popEntropyBoxList, vonEntropyBoxList, KSEntropyBoxList, degreeEntropyBoxList = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        # random nodeList for growth method (it has to be outside the loop because it is done just once)
        numChosen = 100
        nodeListRandom = []
        for i in range(numChosen):
            nodeListRandom.append(np.random.choice(nodeList))
        
        while lb < diameter:
            lb = lb +1
            if lb not in dist:
                nBoxesAll.append(nBoxes)
                modLbAll.append(modLb)
            else:
                
                # make new M  graph
                edgeListStep = []
                for i in range(pairsNum):
                    if dist[i] >= lb:
                        edgeListStep.append([node1[i], node2[i]])
                M = nx.Graph()
                M.add_nodes_from(nodeList)
                M.add_edges_from(edgeListStep)
                
                # coloring
                boxes = nx.coloring.greedy_color(M, strategy=nx.coloring.strategy_saturation_largest_first)
                boxesList = list(set(list(boxes.values())))
                nBoxes = len(boxesList)
                nBoxesAll.append(nBoxes)
                withinBox = 1
                betweenBox = 1

                # box network and box Modularity
                allBoxesDict = {}                
                for boxName in boxesList:
                    allBoxesDict[boxName] = nx.Graph()
                for i in range(pairsNum):
                    if dist[i] == 1:
                        if boxes[node1[i]] == boxes[node2[i]]:
                            withinBox = withinBox + 1
                            allBoxesDict[boxes[node1[i]]].add_edge(node1[i], node2[i])
                        else:
                            betweenBox = betweenBox + 1    
                modLb = np.divide(np.divide(withinBox, betweenBox, dtype = float), nBoxes, dtype = float)
                modLbAll.append(modLb)

        degreeEntropySelf,diameterPathSelf,avgPathDistSelf,nEdgesSelf,edgePSelf,radiusSelf,kCoreSelf ,degreeAssortSelf,globalEfficiencySelf,avgDegreeSelf,maxDegreeSelf,spectralRadiusAdjSelf,spectralGapSelf,popEntropySelf,scaledSpectralRadiusSelf,colorNumSelf,avgClustCoeffSelf,freqMBoxSelf,graphEntropySelf,motifEntropySelf,vonEntropySelf,KSEntropySelf =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
            
        #final db        
        nBoxesAll.append(1)
        steps = range(1, int(diameter)+2, 1)
        x = np.log(steps)
        y = np.log(nBoxesAll)
        slope, intercept, corr, p_value, stdErr = linregress(x,y)
        fractalDb = -1*slope
        fractalDbConstant = intercept
        fractalRSquare = np.power(corr, 2, dtype = float)
        #final modularity
        steps = range(2, int(diameter)+1, 1)
        x = np.log(steps)
        y = np.log(modLbAll)
        slope, intercept, corr, p_value, stdErr = linregress(x,y)
        fractalModularity = slope
        fractalModularityConstant = intercept
        fractalModularityRSquare = np.power(corr, 2, dtype = float)
        return fractalDb, fractalRSquare, fractalDbConstant, fractalModularity, fractalModularityConstant, fractalModularityRSquare, diameterPathSelf, avgPathDistSelf, nEdgesSelf, edgePSelf, radiusSelf, kCoreSelf, degreeAssortSelf, globalEfficiencySelf, avgDegreeSelf, maxDegreeSelf, spectralRadiusAdjSelf, spectralGapSelf, popEntropySelf,scaledSpectralRadiusSelf, colorNumSelf, avgClustCoeffSelf, vonEntropySelf, KSEntropySelf, degreeEntropySelf, graphEntropySelf, motifEntropySelf, freqMBoxSelf

def fix_adj(adj):
    tmp_adj=adj
    for i in range(len(adj)):
        v=adj[i]
        if sum(v)<1: #all zeros, delete index
            tmp_adj=np.delete(np.delete(tmp_adj,i,axis=1),i,axis=0)
    return adj

def is_1step_connected(adj):
    sparse_adj = csgraph_from_dense(adj)
    pathDistMat = shortest_path(sparse_adj, method='auto', directed=False, return_predecessors=False, unweighted=True, overwrite=False)
    if np.inf in pathDistMat: #problem: parsing has created a rift between the two components
        return False
    else:
        return True
    
def get_davids_vars(nodes,adj,only_freqs,d_vec,full_file):
    #is only_freqs in the wrong order?
    var_names=['comp_size','n_edges','edgeP','radius','kCore','degreeAssort_david','corr_path_hamm_dist','RMSE_path_hamm_dist','avg_degree','max_degree','mean_hamm_dist','mean_path_dist','diameter_path','diameter_hamm','corr_degree_freq','corr_eigen_cent_freq','corr_close_cent_freq','corr_bet_cent_freq','corrPageRankfreq','genetic_load','CV_freq','local_opt_frac','viable_fraction','scaled_spectral_radius','spectral_radius_adj','spectral_gap','spectral_radius_hamm','pop_entropy','von_entropy','ks_entropy','degree_entropy','avg_clust_coeff','global_efficiency','graph_entropy','motif_entropy','color_num']
    sparse_adj = csgraph_from_dense(adj)
    pathDistMat = shortest_path(sparse_adj, method='auto', directed=False, return_predecessors=False, unweighted=True, overwrite=False)
    triu_index=np.triu_indices(len(pathDistMat),k=1)
    pathDist = pathDistMat[triu_index]
    # print(len(pathDist))
    if full_file: #doesn't work
        comps_freqlist,major_comp,comps_info,num_comp=get_comp_freqs(adj,np.divide(only_freqs,sum(only_freqs),dtype=float))
        links,seqs,adj_1,num_seqs=process_component(only_seqs,only_freqs,major_comp,comps_info,adj,DM)
        only_seqs=list(seqs.keys())
        only_freqs=list(seqs.values())
        total_reads=float(sum(only_freqs))
        rel_freq=np.divide(list(only_freqs),total_reads,dtype='float')
        dvec,DM=get_dvec(only_seqs,num_seqs,seqlen)
        
    G = nx.Graph(adj) 
    #constants
    posNum = 15
    letters = 2
    u = 0.000115
    if nodes <= 3:
        # print("Sample %s has too few(%i) nodes" %(file,nodes))
        return var_names,np.zeros(len(var_names))
    else:
        #print 'correlations'
        ## distance properties
        diameterHamm = max(d_vec)
        avgHammDist = np.mean(d_vec)
        corrPathHammDist = np.corrcoef(d_vec, pathDist)
        RMSEPathHammDist = mean_squared_error(d_vec, pathDist)
        degreeRaw = list(G.degree())
        degree = []
        for i in degreeRaw:
            degree.append(i[1])
        degreeNorm = np.divide(degree, np.sum(degree), dtype = float)
        eigenCentRaw = list(nx.eigenvector_centrality_numpy(G, weight=None).values())
        eigenCent = np.divide(eigenCentRaw, sum(eigenCentRaw), dtype = float)
        closeCent = list(nx.closeness_centrality(G).values())
        betCent = list(nx.betweenness_centrality(G).values())
        pageRank = list(nx.pagerank_numpy(G).values())
        # correlations
        corrDegreeFreq = np.corrcoef(list(only_freqs), list(degree))
        corrEigenCentFreq = np.corrcoef(only_freqs, eigenCent)
        corrCloseCentFreq = np.corrcoef(only_freqs, closeCent)
        corrBetCentFreq = np.corrcoef(only_freqs, betCent)
        corrPageRankfreq = np.corrcoef(only_freqs, pageRank)                
        edgeP, radius, kCore, degreeAssort, diameterPath, avgPathDist, nEdges, globalEfficiency, avgDegree, maxDegree, spectralRadiusAdj, spectralGap, scaledSpectralRadius, colorNum, avgClustCoeff, freqM, motifEntropy, vonEntropy, graphEntropy, popEntropy, KSEntropy, degreeEntropy = boxStats(G)
        spectralRadiusHamm = 1#max(abs(eValsH))
        nReads = sum(only_freqs)
        freqCountRel = np.divide(only_freqs, nReads, dtype = float)
        d1 = sum(degree*freqCountRel)
        neighbors = posNum*(letters-1)
        uPosNum = posNum * u
        geneticLoad = uPosNum * (1 - np.divide(scaledSpectralRadius, neighbors, dtype = float)) 
        CV = variation(freqCountRel)# coefficient of variation of the frequencies
        localOptNum = 0
        for n0 in G.nodes():#range(76,77):
            flagMax = 0
            flagMin = 0
            for n1 in G.neighbors(n0):
                #print only_freqs[n0], only_freqs[n1]
                if freqCountRel[n0] > freqCountRel[n1]:
                   flagMax = 1
                if freqCountRel[n0] < freqCountRel[n1]:
                   flagMin = 1                          
                   
            if flagMax == 1 and flagMin == 0:
                #rint n0, n1, only_freqs[n0], only_freqs[n1]
                localOptNum = localOptNum + 1 
        localOptFrac = np.divide(localOptNum, nodes, dtype = float)
        vs = 1 - uPosNum * (1 - (np.divide(degree, neighbors, dtype = float)))
        viableFraction = sum(vs*eigenCent)
        # var_names_broken=['comp_size','n_edges','edgeP','radius','kCore','degreeAssort','corr_path_hamm_dist','RMSE_path_hamm_dist','avg_degree','max_degree','mean_hamm_dist','mean_path_dist','diameter_Path','diameter_Hamm','corr_degree_freq','corr_eigen_cent_freq','corr_close_cent_freq','corr_bet_cent_freq','corrPageRankfreq','genetic_load','CV_freq','localOptFrac','viable_fraction','scaledSpectralRadius','spectralRadiusAdj','spectralGap','spectralRadiusHamm','popEntropy','VonEntropy','KSEntropy','degreeEntropy','average_clustering_coeficcient','global_efficiency','motif_1_star','motif_2_path','motif_3_cycle','motif_4_tailed_triangle','motif_5_envelope','motif_6_clique','graphEntropy','motif_entropy','colorNum','pathDb','pathDbConstant','pathRSquare','pathModularity','pathModularityConstant','pathModularityRSquare']
        var_vals=[nodes,nEdges,edgeP,radius,kCore,degreeAssort,corrPathHammDist[0][1],RMSEPathHammDist,avgDegree,maxDegree,avgHammDist,avgPathDist,diameterPath,diameterHamm,corrDegreeFreq[0][1],corrEigenCentFreq[0][1],corrCloseCentFreq[0][1],corrBetCentFreq[0][1],corrPageRankfreq[0][1],geneticLoad,CV,localOptFrac,viableFraction,scaledSpectralRadius,spectralRadiusAdj,spectralGap,spectralRadiusHamm,popEntropy,vonEntropy,KSEntropy,degreeEntropy,avgClustCoeff,globalEfficiency,graphEntropy,motifEntropy,colorNum]
        
        return var_names,var_vals
    
def get_status(sample):
    statusdic={
    'APA_new': 0, 
    'HCDE18': 0, 
    'C13ND19_new': 0, 
    'WA6': 0, 
    'HOC_P07_1b_new': 0, 
    'LBA_new': 0, 
    'C13ND21_new': 0,
	'C13ND123_new': 0,
    'C13ND132_new': 0,
	'HOC_P09_1b_new': 0,
	'C13ND6_new': 0,
	'RAM_new': 0,
	'C13ND142_new': 0,
	'HCDE10': 0,
	'WA9': 0,
	'C13ND31_new': 0,
	'MVA_new': 0,
	'EEV_new': 0,
	'WA11': 0,
	'WA8': 0,
	'WA14': 0,
	'HOC_P21_1b_new': 0,
	'C13ND126_new': 0,
	'C13ND16_new': 0,
	'DGO_new': 0,
	'C13ND14_new': 0,
	'DHE_new': 0,
	'C13ND32_new': 0,
	'LRA_new': 0,
	'C13ND70_new': 0,
	'HOC_P16_1b_new': 0,
	'NCH_new': 0,
	'HOC_P28_1b_new': 0,
	'HOC_P20_1b_new': 0,
	'HOC_P12_1b_new': 0,
	'C13ND10_new': 0,
	'CCI_new': 0,
	'MRU_new': 0,
	'WA2': 0,
	'C13ND4_new': 0,
	'MSMPAN4_new': 0,
	'C13ND20_new': 0,
	'WA10': 0,
	'C13ND121_new': 0,
	'C13ND140_new': 0,
	'HOC_P22_1b_new': 0,
	'TRO_new': 0,
	'JSA_new': 0,
	'SFGH008_new': 0,
	'C13ND22_new': 0,
	'JJO_new': 0,
	'HOC_P01_1b_new': 0,
	'HCDE9': 0,
	'MSMPAN16_new': 0,
	'VVI_new': 0,
	'C13ND33_new': 0,
	'CBA_new': 0,
	'HOC_P13_1b_new': 0,
	'AHCV03_new': 0,
	'HCDE4': 0,
	'HOC_P04_1b_new': 0,
	'C13ND170_new': 0,
	'C13ND34_new': 0,
	'C13ND15_new': 0,
	'HOC_P10_1b_new': 0,
	'TKE_new': 0,
	'HCDE14': 0,
	'HOC_P08_1b_new': 0,
	'WA7': 0,
	'C13ND130_new': 0,
	'MSMPAN19_new': 0,
	'JOP_new': 0,
	'RMA_new': 0,
	'C13ND173_new': 0,
	'SFGH005_new': 0,
	'MSMPAN3_new': 0,
	'WA5': 0,
	'HCDE5': 0,
	'C13ND7_new': 0,
	'WA13': 0,
	'WA19': 0,
	'WA17': 0,
	'MSMPAN15_new': 0,
	'C13ND117_new': 0,
	'C13ND129_new': 0,
	'HCDE2': 0,
	'CGO_new': 0,
	'C13ND11_new': 0,
	'C13ND162_new': 0,
	'C13ND18_new': 0,
	'AHCV08_new': 0,
	'HCDE20': 0,
	'HOC_P27_1b_new': 0,
	'C13ND2_new': 0,
	'HOC_P18_1b_new': 0,
	'MSMPAN11_new': 0,
	'C13ND120_new': 0,
	'C13ND25_new': 0,
	'C13ND172_new': 0,
	'HCDE17': 0,
	'SAL_new': 0,
	'C13ND125_new': 0,
	'HCDE8': 0,
	'C13ND24_new': 0,
	'ATA_new': 0,
	'C13ND12_new': 0,
	'C13ND23_new': 0,
	'WA3': 0,
	'C13ND128_new': 0,
	'C13ND127_new': 0,
	'HOC_P34_1b_new': 0,
	'HCDE1': 0,
	'HCDE11': 0,
	'HCDE13': 0,
	'C13ND1_new': 0,
	'C13ND17_new': 0,
	'C13ND131_new': 0,
	'C13ND5_new': 0,
	'NH3320_new': 0,
	'C13ND27_new': 0,
	'KOM_P141': 1,
	'LYB_P03': 1,
	'AMC_P31': 1,
	'24GLC': 1,
	'VAO_P30': 1,
	'12GLC': 1,
	'18GLC': 1,
	'LYB_P59': 1,
	'BID_P14T1': 1,
	'VAO_P01': 1,
	'KOM_P215': 1,
	'LYB_P60': 1,
	'AMC_P47': 1,
	'KOM_P146': 1,
	'SGH54': 1,
	'KOM_P059': 1,
	'LYB_P64': 1,
	'KOM_P062': 1,
	'PAK_P10': 1,
	'KOM_P065': 1,
	'14GLC': 1,
	'SGH36': 1,
	'BID_P11T1': 1,
	'SGH34': 1,
	'PAK_P13': 1,
	'VAO_P39': 1,
	'16GLC': 1,
	'KOM_P022': 1,
	'BID_P12T1': 1,
	'BID_P02T1': 1,
	'LYB_P39': 1,
	'VAO_P31': 1,
	'KOM_P060': 1,
	'LYB_P57': 1,
	'AMC_P06': 1,
	'AMC_P01': 1,
	'BID_P03T1': 1,
	'KOM_P080': 1,
	'LYB_P06': 1,
	'KOM_P188': 1,
	'VAO_P52': 1,
	'PAK_P08': 1,
	'VAO_P25': 1,
	'KOM_P297': 1,
	'VAO_P29': 1,
	'SGH23': 1,
	'PAK_P19': 1,
	'VAO_P51': 1,
	'KOM_P235': 1,
	'LYB_P48': 1,
	'AMC_P18': 1,
	'KOM_P066': 1,
	'KOM_P250': 1,
	'VAO_P15': 1,
	'BID_P04T1': 1,
	'SGH25': 1,
	'AMC_P43': 1,
	'KOM_P222': 1,
	'SGH49': 1,
	'KOM_P163': 1,
	'VAO_P06': 1,
	'BID_P01T1': 1,
	'01GLC': 1,
	'BID_P06T1': 1,
	'LYB_P01': 1,
	'AMC_P66': 1,
	'KOM_P004': 1,
	'10GLC': 1,
	'KOM_P229': 1,
	'KOM_P257': 1,
	'AMC_P44': 1,
	'LYB_P15': 1,
	'VAO_P18': 1,
	'SGH33': 1,
	'KOM_P236': 1,
	'AMC_P57': 1,
	'VAO_P27': 1,
	'VAO_P49': 1,
	'VAO_P35': 1,
	'SGH50': 1,
	'PAK_P01': 1,
	'SGH40': 1,
	'AMC_P24': 1,
	'AMC_P46': 1,
	'KOM_P135': 1,
	'LYB_P30': 1,
	'VAO_P37': 1,
	'AMC_P58': 1,
	'PAK_P12': 1,
	'06GLC': 1,
	'VAO_P07': 1,
	'KOM_P205': 1,
	'VAO_P11': 1,
	'LYB_P45': 1,
	'KOM_P199': 1,
	'LYB_P50': 1,
	'LYB_P14': 1,
	'LYB_P51': 1,
	'AMC_P45': 1,
	'AMC_P70': 1,
	'LYB_P33': 1,
	'VAO_P26': 1,
	'AMC_P12': 1,
	'03GLC': 1,
	'AMC_P02': 1,
	'KOM_P218': 1,
	'VAO_P40': 1,
	'AMC_P42': 1,
	'SGH35': 1,
	'PAK_P11': 1,
	'BID_P15T1': 1,
	'KOM_P221': 1,
	'VAO_P32': 1,
	'VAO_P02': 1,
	'LYB_P28': 1,
	'KOM_P069': 1,
	'KOM_P044': 1,
	'21GLC': 1,
	'KOM_P039': 1,
	'AMC_P04': 1,
	'04GLC': 1,
	'25GLC': 1,
	'LYB_P66': 1,
	'AMC_P16': 1,
	'AMC_P62': 1,
	'KOM_P246': 1,
	'VAO_P14': 1,
	'LYB_P54': 1,
	'LYB_P35': 1,
	'BID_P13T1': 1,
	'LYB_P27': 1,
	'SGH51': 1,
	'KOM_P248': 1,
	'KOM_P359': 1,
	'LYB_P38': 1,
	'LYB_P52': 1,
	'VAO_P46': 1,
	'KOM_P291': 1,
	'KOM_P011': 1,
	'VAO_P16': 1,
	'VAO_P50': 1,
	'AMC_P05': 1,
	'LYB_P11': 1,
	'AMC_P14': 1,
	'23GLC': 1,
	'PAK_P18': 1,
	'KOM_P084': 1,
	'SGH39': 1,
	'KOM_P033': 1,
	'PAK_P14': 1,
	'SGH31': 1,
	'SGH26': 1,
	'KOM_P003': 1,
	'KOM_P201': 1,
	'VAO_P13': 1,
	'KOM_P134': 1,
	'SGH44': 1,
	'SGH41': 1,
	'PAK_P07': 1,
	'LYB_P26': 1,
	'VAO_P05': 1,
	'KOM_P061': 1,
	'KOM_P217': 1,
	'KOM_P170': 1,
	'08GLC': 1,
	'22GLC': 1,
	'KOM_P220': 1,
	'AMC_P32': 1,
	'LYB_P62': 1,
	'SGH52': 1,
	'AMC_P39': 1,
	'KOM_P177': 1,
	'SGH27': 1,
	'02GLC': 1,
	'VAO_P48': 1,
	'VAO_P24': 1,
	'VAO_P08': 1,
	'KOM_P362': 1,
	'15GLC': 1,
	'VAO_P09': 1,
	'AMC_P49': 1,
	'09GLC': 1,
	'SGH55': 1,
	'LYB_P61': 1,
	'VAO_P45': 1,
	'LYB_P46': 1,
	'PAK_P06': 1,
	'AMC_P15': 1,
	'KOM_P063': 1,
	'VAO_P12': 1,
	'AMC_P37': 1,
	'KOM_P046': 1,
	'11GLC': 1,
	'LYB_P53': 1,
	'VAO_P44': 1,
	'AMC_P09': 1,
	'AMC_P03': 1,
	'VAO_P28': 1,
	'LYB_P31': 1,
	'AMC_P13': 1,
	'KOM_P233': 1,
	'VAO_P19': 1,
	'KOM_P032': 1,
	'LYB_P07': 1,
	'LYB_P40': 1,
	'KOM_P203': 1,
	'KOM_P204': 1,
	'AMC_P38': 1,
	'LYB_P63': 1,
	'AMC_P35': 1,
	'LYB_P44': 1,
	'AMC_P08': 1,
	'LYB_P05': 1,
	'KOM_P234': 1,
	'KOM_P241': 1,
	'KOM_P227': 1,
	'VAO_P54': 1,
	'LYB_P41': 1,
	'VAO_P22': 1,
	'SGH42': 1,
	'KOM_P354': 1,
	'PAK_P05': 1,
	'BID_P05T1': 1,
	'VAO_P36': 1,
	'SGH47': 1,
	'PAK_P04': 1,
	'LYB_P58': 1,
	'KOM_P085': 1,
	'KOM_P111': 1,
	'LYB_P37': 1,
	'LYB_P22': 1,
	'07GLC': 1,
	'BID_P10T1': 1,
	'PAK_P02': 1,
	'VAO_P41': 1,
	'LYB_P32': 1,
	'19GLC': 1,
	'17GLC': 1,
	'LYB_P20': 1,
	'VAO_P43': 1,
	'AMC_P53': 1,
	'AMC_P63': 1,
	'VAO_P20': 1,
	'AMC_P51': 1,
	'SGH30': 1,
	'VAO_P17': 1,
	'VAO_P21': 1,
	'AMC_P48': 1,
	'AMC_P30': 1,
	'SGH29': 1,
	'KOM_P239': 1,
	'VAO_P53': 1,
	'LYB_P43': 1,
	'SGH46': 1,
	'AMC_P40': 1,
	'SGH43': 1,
	'KOM_P154': 1}
    fixedsample=trimfh(sample).replace('_below10','')
    if fixedsample in statusdic:
        return statusdic[fixedsample]
    else:
        return 2
        
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
            s=[p,q,dists[realp,realq],adj[realp][realq],path_dists[p][q],only_freqs[realp],only_freqs[realq]]
            links.append(s)
    comp_seqs={}
    for k in nodes_real_names:
        seq = str(only_seqs[k])
        freq=only_freqs[k]
        comp_seqs[seq]=freq
    return links,comp_seqs,adj_comp,comp_size
        
def get_comp_freqs(adj,rel_freq):
    sparseMatrix = csgraph_from_dense(adj)
    connected = connected_components(sparseMatrix, directed=False, connection='weak', return_labels=True)
    comp_num = connected[0]
    comp_list = connected[1]
    # print(comp_list)
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
    return comps_freqlist,list(comps_freqlist).index(max_comp),comps_info,comp_num

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

def get_default_params():
    #deleted prot_ratio
    my_params=['num_haps','num_reads','mean_dist','std_dev','mean_cons','trans_mut','dnds_val','pca_comps','kol_complexity','freq_corr','s_metric','cluster_coeff','protein_nuc_entropy','atchley_sd0','atchley_sd1','atchley_sd2','atchley_sd3','atchley_sd4','atchley_wa0','atchley_wa1','atchley_wa2','atchley_wa3','atchley_wa4','atchley_dist0','atchley_dist1','atchley_dist2','atchley_dist3','atchley_dist4','atchley_dist_cat','atchley_sdd_rowsum','cv_dist','max_dist','freq_entropy','pelin_haplo_freq','nuc_div','nuc_entropy','mut_freq','one_step_entropy','num_1step_components','degree_assortativity_me','degree_entropy_me','aa_entropy_inscape']
    davids_vars=['comp_size','n_edges','edgeP','radius','kCore','degreeAssort_david','corr_path_hamm_dist','RMSE_path_hamm_dist','avg_degree','max_degree','mean_hamm_dist','mean_path_dist','diameter_path','diameter_hamm','corr_degree_freq','corr_eigen_cent_freq','corr_close_cent_freq','corr_bet_cent_freq','corrPageRankfreq','genetic_load','CV_freq','local_opt_frac','viable_fraction','scaled_spectral_radius','spectral_radius_adj','spectral_gap','spectral_radius_hamm','pop_entropy','von_entropy','ks_entropy','degree_entropy','avg_clust_coeff','global_efficiency','graph_entropy','motif_entropy','color_num']
    my_params.extend(davids_vars)
    for k in range(10):
        if k!=0:
            my_params.append('nuc_kmer_inscape_'+str(k+1))
    for k in range(10):
        if k!=0:
            my_params.append('aa_kmer_inscape_'+str(k+1))
    for k in range(10):
        my_params.append('nuc_kmer_pelin_'+str(k+1))
    for k in range(10):
        my_params.append('aa_kmer_pelin_'+str(k+1))
    return my_params


def calc_dists_inner(seq1,seq2): 
    dist=0
    for a,b in zip(seq1,seq2):
        if a!=b:
            dist+=1
        if dist==10:
            return False
    return True

def calc_dists(seq1,seqs2):
    seqs2_num=len(seqs2)
    for j in range(seqs2_num):
        seq2=seqs2[j]        
        if calc_dists_inner(seq1,seq2)<10:
            return True
    return False

def calculate_groups(data,all_seqs):
    data['group']=0
    g=nx.Graph()
    files=list(all_seqs.keys())
    for f1,f2 in itertools.combinations(files,2):
        seqs1=all_seqs[f1]
        seqs2=all_seqs[f2]
        seqs1_num=len(seqs1)
        for i in range(seqs1_num):
            seq1=seqs1[i]
            if calc_dists(seq1,seqs2):
                g.add_edge(f1,f2)
    counter=0
    for comp in nx.connected_components(g):
        counter+=1
        for file in comp:
            data.loc[file,'group']=counter
    return data

    #     for seq1,seq2 in zip()
    #         seq1=seqs[id1]
    #         seq2=seqs[id2]
    #         dist=sum(0 if a==b else 1 for a,b in zip(seq1,seq2))
    #         arr[id1,id2]=dist
    #         arr[id2,id1]=dist
    
    #     val=int(np.amin(array))
    #     print(f1+","+f2+","+str(val))
    
    # g=nx.Graph()
    # for i in range(DM):
    #     for j in range(DM):
    #         dist=DM[i,j]
    #         if dist<10:


def main(status_dic,dir_0,dir_1):
    data=pd.DataFrame()
    full_file=False#david calc script errors out if it's true
    my_params=get_default_params()
    all_seqs={}
    for val in my_params:
        data[val]=0
    for file in status_dic:
        status=status_dic[file]
        trim_file=trimfh(file)
        if status==0:
            full_path=os.path.join(dir_0,file)
        elif status==1:
            full_path=os.path.join(dir_1,file)
        data.append(pd.Series(name=trim_file))
        data.at[trim_file,'status']=status
        ###############setup###############
        preseqs=get_good_seqs(full_path)
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
        num_seqs=len(seqs)
        if num_seqs<2:
            print(file+'error') 
            continue
        total_reads=float(sum(only_freqs))
        dvec,DM=get_dvec(only_seqs,num_seqs,seqlen) 
        adj_1=get_adj(DM,1,num_seqs)
        rel_freq=np.divide(list(only_freqs),total_reads,dtype='float')
        if full_file:
            g=nx.from_numpy_matrix(adj_1)
            adj_2=get_adj(DM,2,num_seqs)
            num_comp=np.nan
        else:
            comps_freqlist,major_comp,comps_info,num_comp=get_comp_freqs(adj_1,rel_freq)
            links,seqs,adj_1,num_seqs=process_component(only_seqs,only_freqs,major_comp,comps_info,adj_1,DM)
            only_seqs=list(seqs.keys())
            only_freqs=list(seqs.values())
            total_reads=float(sum(only_freqs))
            rel_freq=np.divide(list(only_freqs),total_reads,dtype='float')
            dvec,DM=get_dvec(only_seqs,num_seqs,seqlen) 
            g=nx.from_numpy_matrix(adj_1)
            adj_2=get_adj(DM,2,num_seqs)
        ###############calculate features###############
        # print(len(only_seqs))
        if num_seqs<2:
            print(file+" only has one viable sequence! skipping")
            data=data.drop(trim_file)
            continue
        print(trim_file,num_seqs)
        all_seqs[trim_file]=only_seqs
        data.at[trim_file,'num_haps']=int(num_seqs)
        data.at[trim_file,'num_reads']=int(total_reads)
        
        mean_dist=np.mean(dvec)
        std_dev=get_std_dist(dvec)
        cv_dist=std_dev/float(mean_dist)
        data.at[trim_file,'mean_dist']=mean_dist
        data.at[trim_file,'std_dev']=std_dev
        data.at[trim_file,'cv_dist']=cv_dist
        david_var_names,david_var_vals=get_davids_vars(num_seqs,adj_1,only_freqs,dvec,full_file)
        for i in range(len(david_var_vals)):
            name=david_var_names[i]
            val=david_var_vals[i]
            data.at[trim_file,name]=val
        # corr_page_rank_freq=get_pagerank(seqs,only_freqs)
        # von_entropy, ks_entropy, degree_entropy_me = boxStats(g)        
        ent_vec=calc_ordered_frequencies(num_seqs,seqlen,seqs,True)
        trans_mut=get_transver_mut(only_seqs,seqlen)  
        data.at[trim_file,'trans_mut']=trans_mut
        # data.at[trim_file,'ent_vec']=ent_vec
        try:
            pca_comps=get_pca_components(seqs,num_seqs,seqlen)  
            data.at[trim_file,'pca_comps']=pca_comps  
        except:
            data.at[trim_file,'pca_comps']='nan'
        kol_complexity=kolmogorov_wrapper(seqs,seqlen)
        data.at[trim_file,'kol_complexity']=kol_complexity
        s_metric=get_s_metric(g,num_seqs)
        data.at[trim_file,'s_metric']=s_metric
        cluster_coeff=get_cluster_coeff(adj_1,num_seqs,g)
        data.at[trim_file,'cluster_coeff']=cluster_coeff
        freq_corr=get_freq_corr(adj_2,only_freqs)   
        data.at[trim_file,'freq_corr']=freq_corr
        max_dist=np.max(dvec)
        data.at[trim_file,'max_dist']=max_dist
        # phacelia_score=phacelia_API(file)
        # data.at[trim_file,'phacelia_score']=phacelia_score
        dnds_val,proteins,prot_count=dnds_wrapper(seqs,only_freqs) #broken?
        data.at[trim_file,'dnds_val']=dnds_val
        prot_ratio=prot_count/num_seqs #3 #broken?
        sd,wa,dists,stdrow=atchley(proteins,rel_freq)
        
        data.at[trim_file,'atchley_sdd_rowsum']=stdrow
        for i in range(len(sd)):
            sd_val=sd[i]
            wa_val=wa[i]
            d_val=dists[i]
            sd_name='atchley_sd'+str(i)
            wa_name='atchley_wa'+str(i)
            d_name='atchley_dist'+str(i)
            data.at[trim_file,sd_name]=sd_val
            data.at[trim_file,wa_name]=wa_val
            data.at[trim_file,d_name]=d_val
        data.at[trim_file,'atchley_dist_cat']=dists[-1]#'atchley_dist_cat'???
        prot_seqsonly=list(proteins.keys())
        prot_freqs=list(proteins.values())
        protnum=len(proteins)
        protlen=len(prot_seqsonly[0])
        protein_nuc_entropy=nuc_entropy_inscape(proteins,protnum,protlen)
        data.at[trim_file,'protein_nuc_entropy']=protein_nuc_entropy
        one_step_entropy=calc_1step_entropy(adj_1,only_freqs)
        data.at[trim_file,'one_step_entropy']=one_step_entropy
        data.at[trim_file,'num_1step_components']=num_comp
        freq_entropy=entropy(rel_freq,base=2)
        data.at[trim_file,'freq_entropy']=freq_entropy
        nuc_div=nuc_div_inscape(only_freqs,DM,seqlen,num_seqs)
        data.at[trim_file,'nuc_div']=nuc_div
        nuc_entropy=sum(ent_vec)/len(ent_vec)#nuc_entropy_inscape
        data.at[trim_file,'nuc_entropy']=nuc_entropy
        pelin_haplo_freq=entropy(rel_freq,base=2)/log(num_seqs,2)
        data.at[trim_file,'pelin_haplo_freq']=pelin_haplo_freq
        for k in range(1,11):
            if k!=1:
                kmer_nuc_inscape=kmer_entropy_inscape(seqs,k)
                kmer_prot_inscape=kmer_entropy_inscape(proteins,k)
                data.at[trim_file,'aa_kmer_inscape_'+str(k)]=kmer_prot_inscape
                data.at[trim_file,'nuc_kmer_inscape_'+str(k)]=kmer_nuc_inscape
            kmer_nuc_pelin=kmer_entropy_pelin(only_seqs,seqlen,k)
            kmer_prot_pelin=kmer_entropy_pelin(prot_seqsonly,protlen,k)
            data.at[trim_file,'nuc_kmer_pelin_'+str(k)]=kmer_nuc_pelin
            data.at[trim_file,'aa_kmer_pelin_'+str(k)]=kmer_prot_pelin
        mut_freq=get_mutation_freq(seqs,only_seqs,only_freqs,seqlen)
        data.at[trim_file,'mut_freq']=mut_freq
        mean_cons=nuc44_consensus(only_seqs,seqlen,num_seqs)  
        data.at[trim_file,'mean_cons']=mean_cons
        # pelin_nuc_entropy=kmer_entropy_pelin(only_seqs,seqlen,1) #
        # dumb_epistasis=calc_dumb_epistasis(std_dev,only_seqs,seqlen,num_seqs)
        # david_epis,pavel_epis=david_and_pavel_epistasis(only_seqs,only_freqs,list(ent_vec),num_seqs,seqlen,total_reads)
        prot_ent_vec=calc_ordered_frequencies(protnum,protlen,proteins,True)
        aa_entropy_inscape=sum(prot_ent_vec)/len(prot_ent_vec)
        data.at[trim_file,'aa_entropy_inscape']=aa_entropy_inscape
        # david_prot_epis,pavel_prot_epis=david_and_pavel_epistasis(prot_seqsonly,prot_freqs,list(prot_ent_vec),protnum,protlen,sum(prot_freqs))
        
        if sum(sum(adj_1))>2:
            degree_assortativity_me=degree_corell(g) #1
            degree_entropy_me=degree_distribution(g) #2
            data.at[trim_file,'degree_assortativity_me']=degree_assortativity_me
            data.at[trim_file,'degree_entropy_me']=degree_entropy_me
    return data

def main_manual(files,output_name):
    data=pd.DataFrame()
    full_file=False#TODO: david calc script errors out if it's true
    my_params=get_default_params()
    all_seqs={}
    for val in my_params:
        data[val]=0
    num_samples=len(files)
    try:
        for i in range(num_samples):
            file=files[i]
            trim_file=trimfh(file)
            data.append(pd.Series(name=trim_file))
            status=int(get_status(file))
            data.at[trim_file,'status']=status
            ###############setup###############
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
            num_seqs=len(seqs)
            if num_seqs<2:
                print(file+'error')
                continue
            total_reads=float(sum(only_freqs))
            dvec,DM=get_dvec(only_seqs,num_seqs,seqlen) 
            adj_1=get_adj(DM,1,num_seqs)
            rel_freq=np.divide(list(only_freqs),total_reads,dtype='float')
            if full_file:
                g=nx.from_numpy_matrix(adj_1)
                adj_2=get_adj(DM,2,num_seqs)
                num_comp=np.nan
            else:
                comps_freqlist,major_comp,comps_info,num_comp=get_comp_freqs(adj_1,rel_freq)
                links,seqs,adj_1,num_seqs=process_component(only_seqs,only_freqs,major_comp,comps_info,adj_1,DM)
                only_seqs=list(seqs.keys())
                only_freqs=list(seqs.values())
                total_reads=float(sum(only_freqs))
                rel_freq=np.divide(list(only_freqs),total_reads,dtype='float')
                dvec,DM=get_dvec(only_seqs,num_seqs,seqlen) 
                g=nx.from_numpy_matrix(adj_1)
                adj_2=get_adj(DM,2,num_seqs)
            ###############calculate features###############
            # print(len(only_seqs))
            if num_seqs<2:
                print(file+" only has one viable sequence! skipping")
                data.drop(trim_file)
                continue
            print(trim_file,num_seqs)
            all_seqs[trim_file]=only_seqs
            data.at[trim_file,'num_haps']=int(num_seqs)
            data.at[trim_file,'num_reads']=int(total_reads)
            
            mean_dist=np.mean(dvec)
            std_dev=get_std_dist(dvec)
            cv_dist=std_dev/float(mean_dist)
            data.at[trim_file,'mean_dist']=mean_dist
            data.at[trim_file,'std_dev']=std_dev
            data.at[trim_file,'cv_dist']=cv_dist
            david_var_names,david_var_vals=get_davids_vars(num_seqs,adj_1,only_freqs,dvec,full_file)
            for i in range(len(david_var_vals)):
                name=david_var_names[i]
                val=david_var_vals[i]
                data.at[trim_file,name]=val
            # corr_page_rank_freq=get_pagerank(seqs,only_freqs)
            # von_entropy, ks_entropy, degree_entropy_me = boxStats(g)        
            ent_vec=calc_ordered_frequencies(num_seqs,seqlen,seqs,True)
            trans_mut=get_transver_mut(only_seqs,seqlen)  
            data.at[trim_file,'trans_mut']=trans_mut
            # data.at[trim_file,'ent_vec']=ent_vec
            try:
                pca_comps=get_pca_components(seqs,num_seqs,seqlen)  
                data.at[trim_file,'pca_comps']=pca_comps  
            except:
                data.at[trim_file,'pca_comps']='nan'
            kol_complexity=kolmogorov_wrapper(seqs,seqlen)
            data.at[trim_file,'kol_complexity']=kol_complexity
            s_metric=get_s_metric(g,num_seqs)
            data.at[trim_file,'s_metric']=s_metric
            cluster_coeff=get_cluster_coeff(adj_1,num_seqs,g)
            data.at[trim_file,'cluster_coeff']=cluster_coeff
            freq_corr=get_freq_corr(adj_2,only_freqs)   
            data.at[trim_file,'freq_corr']=freq_corr
            max_dist=np.max(dvec)
            data.at[trim_file,'max_dist']=max_dist
            # phacelia_score=phacelia_API(file)
            # data.at[trim_file,'phacelia_score']=phacelia_score
            dnds_val,proteins,prot_count=dnds_wrapper(seqs,only_freqs) #broken?
            data.at[trim_file,'dnds_val']=dnds_val
            prot_ratio=prot_count/num_seqs #3 #broken?
            sd,wa,dists,stdrow=atchley(proteins,rel_freq)
            
            data.at[trim_file,'atchley_sdd_rowsum']=stdrow
            for i in range(len(sd)):
                sd_val=sd[i]
                wa_val=wa[i]
                d_val=dists[i]
                sd_name='atchley_sd'+str(i)
                wa_name='atchley_wa'+str(i)
                d_name='atchley_dist'+str(i)
                data.at[trim_file,sd_name]=sd_val
                data.at[trim_file,wa_name]=wa_val
                data.at[trim_file,d_name]=d_val
            data.at[trim_file,'atchley_dist_cat']=dists[-1]#'atchley_dist_cat'???
            prot_seqsonly=list(proteins.keys())
            prot_freqs=list(proteins.values())
            protnum=len(proteins)
            protlen=len(prot_seqsonly[0])
            protein_nuc_entropy=nuc_entropy_inscape(proteins,protnum,protlen)
            data.at[trim_file,'protein_nuc_entropy']=protein_nuc_entropy
            one_step_entropy=calc_1step_entropy(adj_1,only_freqs)
            data.at[trim_file,'one_step_entropy']=one_step_entropy
            data.at[trim_file,'num_1step_components']=num_comp
            freq_entropy=entropy(rel_freq,base=2)
            data.at[trim_file,'freq_entropy']=freq_entropy
            nuc_div=nuc_div_inscape(only_freqs,DM,seqlen,num_seqs)
            data.at[trim_file,'nuc_div']=nuc_div
            nuc_entropy=sum(ent_vec)/len(ent_vec)#nuc_entropy_inscape
            data.at[trim_file,'nuc_entropy']=nuc_entropy
            pelin_haplo_freq=entropy(rel_freq,base=2)/log(num_seqs,2)
            data.at[trim_file,'pelin_haplo_freq']=pelin_haplo_freq
            for k in range(1,11):
                if k!=1:
                    kmer_nuc_inscape=kmer_entropy_inscape(seqs,k)
                    kmer_prot_inscape=kmer_entropy_inscape(proteins,k)
                    data.at[trim_file,'aa_kmer_inscape_'+str(k)]=kmer_prot_inscape
                    data.at[trim_file,'nuc_kmer_inscape_'+str(k)]=kmer_nuc_inscape
                kmer_nuc_pelin=kmer_entropy_pelin(only_seqs,seqlen,k)
                kmer_prot_pelin=kmer_entropy_pelin(prot_seqsonly,protlen,k)
                data.at[trim_file,'nuc_kmer_pelin_'+str(k)]=kmer_nuc_pelin
                data.at[trim_file,'aa_kmer_pelin_'+str(k)]=kmer_prot_pelin
            mut_freq=get_mutation_freq(seqs,only_seqs,only_freqs,seqlen)
            data.at[trim_file,'mut_freq']=mut_freq
            mean_cons=nuc44_consensus(only_seqs,seqlen,num_seqs)  
            data.at[trim_file,'mean_cons']=mean_cons
            # pelin_nuc_entropy=kmer_entropy_pelin(only_seqs,seqlen,1) #
            # dumb_epistasis=calc_dumb_epistasis(std_dev,only_seqs,seqlen,num_seqs)
            # david_epis,pavel_epis=david_and_pavel_epistasis(only_seqs,only_freqs,list(ent_vec),num_seqs,seqlen,total_reads)
            prot_ent_vec=calc_ordered_frequencies(protnum,protlen,proteins,True)
            aa_entropy_inscape=sum(prot_ent_vec)/len(prot_ent_vec)
            data.at[trim_file,'aa_entropy_inscape']=aa_entropy_inscape
            # david_prot_epis,pavel_prot_epis=david_and_pavel_epistasis(prot_seqsonly,prot_freqs,list(prot_ent_vec),protnum,protlen,sum(prot_freqs))
            
            if sum(sum(adj_1))>2:
                degree_assortativity_me=degree_corell(g) #1
                degree_entropy_me=degree_distribution(g) #2
                data.at[trim_file,'degree_assortativity_me']=degree_assortativity_me
                data.at[trim_file,'degree_entropy_me']=degree_entropy_me
    finally:
        data=calculate_groups(data,all_seqs)
        data.to_csv(output_name)

if __name__=='__main__':
    try:
        output_name=sys.argv[1]
    except:
        output_name='tmp.csv'
    files=[]
    for file in os.listdir(os.getcwd()):    
        if file.endswith('fas') or file.endswith('fasta') or file.endswith('fa'):
            files.append(file)
    main_manual(files,output_name)