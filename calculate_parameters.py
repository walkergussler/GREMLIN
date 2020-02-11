from __future__ import print_function, division
import numpy as np
import networkx as nx
import itertools, sys, os
from math import log, floor, isclose
from fractions import Fraction
from codons import codons
from classify import recency_bin
from scipy.sparse.csgraph import connected_components,csgraph_from_dense
from scipy.linalg import eig
from scipy.stats import pearsonr, entropy
from Bio import SeqIO, Seq, BiopythonWarning, pairwise2
from collections import defaultdict, Counter
from tempfile import NamedTemporaryFile
from subprocess import check_call
import ghost
import warnings
from pyseqdist import hamming as ghosthamm
warnings.simplefilter('ignore', BiopythonWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#echlin.py - Ensemble Classification for Hcv Length of INfection
#Input sequences should have the following properties:
    #collapsed into unique haplotypes with associated frequency
    #all the same length with no gaps (we can maybe work around this)

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
    
def mutation_freq(dict,seqs,freq,seqlen,major,non_majors): #this could maybe be sped up with hamming (dist from major to all other seq)
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

    
def calc_distance_matrix(finalSeqs): #calculate distance matrix from the 1-step list
    l=len(finalSeqs)
    arr=np.zeros([l,l])
    hdist=ghosthamm(finalSeqs,finalSeqs)
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
        # print(s)
    return s/float(total)
    
def calc_distance_matrix_slow(finalSeqs): #calculate distance matrix from the 1-step list manually
    l=len(finalSeqs)
    arr=np.zeros([l,l])
    for id1,id2 in itertools.combinations(range(len(finalSeqs)),2):
        seq1=finalSeqs[id1]
        seq2=finalSeqs[id2]
        dist=sum(0 if a==b else 1 for a,b in zip(seq1,seq2))
        arr[id1,id2]=dist
        arr[id2,id1]=dist
    return arr
    
def get_dvec(seqs,seqnum,seqlen):
    """Calculate distance matrix return upper triangle as well"""
    DM=calc_distance_matrix(seqs)
    triu_index=np.triu_indices(seqnum,k=1)
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

def calc_ordered_frequencies(seqnum,seqlen,seqs,byFreq): 
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

def calc_dumb_epistasis(std_dev,seqs,seqlen,seqnum):
    seq_pert = perturbSeq(seqs,seqlen)
    dvec_pert,_ = get_dvec(seq_pert,seqnum,seqlen)
    return std_dev/float(np.std(dvec_pert,ddof=1))
    
def nuc_entropy_inscape(seqs,seqnum,seqlen): # calculate nucleotide entropy
    ent=calc_ordered_frequencies(seqnum,seqlen,seqs,True)
    return sum(ent)/len(ent)
    
def nuc_div_inscape(freqs,mat,seqlen,seqnum):#calculate nucleotide diversity
    totalFreq=float(sum(freqs))
    nucDiv=0
    for a,b in itertools.combinations(range(seqnum),2):
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
    return entropy(v,base=2),len(connected)

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
    # print('number of blanks skipped: '+c')
    return np.divide(out,seqnum,dtype='float')

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

def get_cluster_coeff(adj,seqnum,g): # is it faster to use adjacency matrix vs networkx?
    deg=sum(adj)
    coeff=2
    C=[]
    for i in range(seqnum):
        if deg[i]<=1:
            C.append(0)
        else:
            row=adj[i]
            neighbors=[]
            for j in range(seqnum):
                if row[j]!=0:
                    neighbors.append(j)
            sg=g.subgraph(neighbors)
            edges_s=len(sg.edges())
            C.append(coeff*edges_s/(deg[i]*(deg[i]-1)))
    return sum(C)/seqnum
        
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
    
def get_pca_components(seqs,seqnum,seqlen): #TODO move noblanks preprocessing to main
    #fill gaps (for matlab consistency)
    # print(len(seqs))
    centered_num=[]
    for preseq in seqs:
        if 'N' in preseq or '_' in preseq or '-' in preseq:
            seq=fill_blanks(preseq)
        else:
            seq=preseq
        #make covariance 
        seq_num=seq2num(seq,seqlen)
        centered_num.append(seq_num)
    # print(centered_num)
    # print(np.shape(centered_num))
    # input()
    cov_matrix=np.cov(np.transpose(centered_num))
    w=np.linalg.eig(cov_matrix) 
    eigvec=w[1]
    eigval=np.real(w[0]) # this is not quite the same as matlab's flip(eigval) which it is meant to mimic, however it is close
    norm_eig=eigval/sum(eigval)
    cs=list(np.cumsum(norm_eig))
    for a in cs:
        if a>=.5:
            return (cs.index(a)+1)/len(norm_eig)

def get_s_metric(g,seqnum):
    s=0
    for edge in g.edges():
        i=edge[0]
        j=edge[1]
        s+=(g.degree(i)*g.degree(j))*2
    return s/(seqnum**4)

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
        sys.exit('err')

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
        # print('Warning: proportion of synonymous changes greater than or equal to 3/4. Input sequences are too divergent, too short, or contain frame shifts')
        return np.nan


def get_major_sequence(seqsDict,freqs):
    m=max(freqs)
    c=Counter(freqs)
    s=c[max(c)]
    # if s!=1:
        # print('WARNING: major sequence frequency detected in %i sequences. This will produce strange results' %s)
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
        print('error parsing aligned seqs!')
        sys.exit()
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
                # print(o)
                sys.exit('error! did not get all blanks removed?')
        # print(''.join(o))
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
            # print(o)
            sys.exit('error! did not get all blanks removed?')
        i+=1
        # print(">seq_"+str(i))
        # print(''.join(o[2:]))
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
    # print('-')
    # print(len(preseqs))
    seqs={}
    for seq in preseqs:
        if preseqs[seq]<=threshold:
            seqs[seq]=preseqs[seq]
    # print(len(seqs))
    # print('-')
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
 
def dnds_wrapper(seqs,freqs):
    # print('-')
    # print(len(seqs))
    total_freq=sum(freqs)
    major,non_majors=get_major_sequence(seqs,freqs)
    dnds=0 
    prot_count=0
    proteins=defaultdict(int)
    i=0
    # print(len(non_majors))
    for seq in seqs:
        prot=translate(seq)
        # print(prot)
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
    # print(proteins,prot_count)
    # print('-')
    return dnds,proteins,prot_count

def get_adj(DM,thr_dist,seqnum):
    return 1*(DM <= thr_dist) - np.eye(seqnum)    
    
def atchley_old(proteins,total_reads):
    atchley_dict={'A':[-0.591,-1.302,-0.733,1.570,-0.146],'C':[1.343,0.465,-0.862,-1.020,-0.255],'D':[1.050,0.302,-3.656,-0.259,-3.242],'E':[1.357,-1.453,1.477,0.113,-0.837],'F':[-1.006,-0.590,1.891,-0.397,0.412],'G':[-0.384,1.652,1.330,1.045,2.064],'H':[0.336,-0.417,-1.673,-1.474,-0.078],'I':[-1.239,-0.547,2.131,0.393,0.816],'K':[1.831,-0.561,0.533,-0.277,1.648],'L':[-1.019,-0.987,-1.505,1.266,-0.912],'M':[-0.663,-1.524,2.219,-1.005,1.212],'N':[0.945,0.828,1.299,-0.169,0.933],'P':[0.189,2.081,-1.628,0.421,-1.392],'Q':[0.931,-0.179,-3.005,-0.503,-1.853],'R':[1.538,-0.055,1.502,0.440,2.897],'S':[-0.228,1.399,-4.760,0.670,-2.647],'T':[-0.032,0.326,2.213,0.908,1.313],'V':[-1.337,-0.279,-0.544,1.242,-1.262],'W':[-0.595,0.009,0.672,-2.128,-0.184],'Y':[0.260,0.830,3.097,-0.838,1.512]}
    a=[]
    sd=np.zeros(5) #standard deviation
    wa=np.zeros(5) #weighted average
    for seq in proteins:
        rel_freq=proteins[seq]/total_reads    
        for char in seq:
            v=atchley_dict[char]    
            wa[0]+=v[0]*rel_freq
            wa[1]+=v[1]*rel_freq
            wa[2]+=v[2]*rel_freq
            wa[3]+=v[3]*rel_freq
            wa[4]+=v[4]*rel_freq
            a.append(v)
            if char=='*':
                sys.exit("Error! Stop codon in protein sequences, exiting")
    atchley_array=np.array(a)
    # print(np.shape(atchley_array))
    # import ipdb
    # ipdb.set_trace()
    for i in range(5):
        sd[i]=np.std(atchley_array[:,i])
        
    return sd,wa

def atchley(proteins,rel_freq): #TODO: fix, atchley_dists4==atchley_dists_cat
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
    # print(x)
    # input()
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

def get_std_dist(dvec):
    return np.std(dvec,ddof=1)

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
    
def david_and_pavel_epistasis(seqs,freqs,ent_array,seqnum,seqlen,total_reads):
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
                for a in range(seqnum):
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
   
def main(files):
    all_params=['mean_dist','std_dev','meanConsensus','pelin_nuc_entropy','trans_mut','dnds_val','pca_comps','kol_complexity','freq_corr','s_metric','cluster_coeff','phacelia_score','protein_kmer_entropy','protein_nuc_entropy','atchley_st0','atchley_st1','atchley_st2','atchley_st3','atchley_st4','atchley_wa0','atchley_wa1','atchley_wa2','atchley_wa3','atchley_wa4','cv_dist','max_dist','haplo_freq','pelin_haplo_freq','inscape_kmer_entropy','pelin_kmer_entropy','nuc_div','nuc_entropy','mutato','one_step_entropy','num_1step_components','dumb_epistasis','degree_assortativity','degree_entropy','prot_ratio','david_epis','pavel_epis','david_prot_epis','pavel_prot_epis','prot_inscape_entropy']
    my_params=['mean_dist','std_dev','meanConsensus','trans_mut','dnds_val','pca_comps','kol_complexity','freq_corr','s_metric','cluster_coeff','phacelia_score','protein_nuc_entropy','atchley_st0','atchley_st1','atchley_st2','atchley_st3','atchley_st4','atchley_wa0','atchley_wa1','atchley_wa2','atchley_wa3','atchley_wa4','atchley_dists0','atchley_dists1','atchley_dists2','atchley_dists3','atchley_dists4','atchley_dists_cat','atchley_std_rowsum','cv_dist','max_dist','haplo_freq','pelin_haplo_freq','nuc_div','nuc_entropy','mutato','one_step_entropy','num_1step_components','degree_assortativity','degree_entropy','prot_ratio','prot_inscape_entropy']
    for k in range(10):
        if k!=0:
            my_params.append('inscape_nuc_kmer_'+str(k+1))
    for k in range(10):
        if k!=0:
            my_params.append('inscape_prot_kmer_'+str(k+1))
    for k in range(10):
        my_params.append('pelin_nuc_kmer_'+str(k+1))
    for k in range(10):
        my_params.append('pelin__prot_kmer_'+str(k+1))
    print('file,'+','.join(my_params))
    num_samples=len(files)
    for i in range(num_samples):
        file=files[i]
        #print(file)
        ###############setup###############
        preseqs=get_good_seqs(file)
        # for seq in preseqs:
            # print(len(seq),preseqs[seq])
            # print(seq)
        if len(preseqs)==0:
            print(file+',reading frame error')
            continue
        ali=align(preseqs)
        # for seq in ali:
            # print(len(seq),ali[seq])
            # print(seq)
        seqlen=len(list(ali.keys())[0])
        seqs=remove_blanks(ali,seqlen)
        seqlen=len(list(seqs.keys())[0]) #may have changed if a blank was removed from alignment
        # print(len(seqs))
        # input()
        if type(seqs)==bool:
            print(file+',error!')
            continue
        onlyseqs=list(seqs.keys())
        onlyfreqs=list(seqs.values())
        seqnum=len(seqs)
        if seqnum==0:
            print(file+'error')
            continue
        total_reads=float(sum(onlyfreqs))
        # print(total_reads)
        rel_freq=np.divide(list(onlyfreqs),total_reads,dtype='float')
        major_sequence,non_majors=get_major_sequence(seqs,onlyfreqs)
        dvec,DM=get_dvec(onlyseqs,seqnum,seqlen) 
        adj_1=get_adj(DM,1,seqnum)
        g=nx.from_numpy_matrix(adj_1)
        adj_2=get_adj(DM,2,seqnum)
        ###############calculate features###############
        ent_vec=calc_ordered_frequencies(seqnum,seqlen,seqs,True)
        trans_mut=get_transver_mut(onlyseqs,seqlen)  
        try:
            pca_comps=get_pca_components(seqs,seqnum,seqlen)  
        except:
            pca_comps='nan'
        kol_complexity=kolmogorov_wrapper(seqs,seqlen)  
        s_metric=get_s_metric(g,seqnum)
        cluster_coeff=get_cluster_coeff(adj_1,seqnum,g)
        freq_corr=get_freq_corr(adj_2,onlyfreqs)   
        max_dist=np.max(dvec)
        phacelia_score=phacelia_API(file)
        dnds_val,proteins,prot_count=dnds_wrapper(seqs,onlyfreqs) #FIX
        prot_ratio=prot_count/seqnum #3 #FIX
        # print(len(proteins))
        # atchley_st,atchley_wa=atchley(proteins,total_reads)
        sd,wa,dists,stdrow=atchley(proteins,rel_freq)
        prot_seqsonly=list(proteins.keys())
        prot_freqs=list(proteins.values())
        protnum=len(proteins)
        protlen=len(prot_seqsonly[0])
        protein_nuc_entropy=nuc_entropy_inscape(proteins,protnum,protlen)
        one_step_entropy,num_comp=calc_1step_entropy(adj_1,onlyfreqs)
        haplo_freq=entropy(rel_freq,base=2)
        nuc_div=nuc_div_inscape(onlyfreqs,DM,seqlen,seqnum)
        nuc_entropy=sum(ent_vec)/len(ent_vec)#nuc_entropy_inscape
        pelin_haplo_freq=entropy(rel_freq,base=2)/log(seqnum,2)
        pelin_kmer_protein_house=[]
        pelin_kmer_nuc_house=[]
        inscape_kmer_protein_house=[]
        inscape_kmer_nuc_house=[]
        for k in range(1,11):
            if k!=1:
                inscape_kmer_nuc_house.append(kmer_entropy_inscape(seqs,k))
                inscape_kmer_protein_house.append(kmer_entropy_inscape(proteins,k))
            pelin_kmer_nuc_house.append(kmer_entropy_pelin(onlyseqs,seqlen,k))
            pelin_kmer_protein_house.append(kmer_entropy_pelin(prot_seqsonly,protlen,k))
        mutato=mutation_freq(seqs,onlyseqs,onlyfreqs,seqlen,major_sequence,non_majors)
        mean_dist=np.mean(dvec)  
        std_dev=get_std_dist(dvec)
        meanConsensus=nuc44_consensus(onlyseqs,seqlen,seqnum)  
        # pelin_nuc_entropy=kmer_entropy_pelin(onlyseqs,seqlen,1) #        
        cv_dist=std_dev/float(mean_dist)
        # dumb_epistasis=calc_dumb_epistasis(std_dev,onlyseqs,seqlen,seqnum)
        # david_epis,pavel_epis=david_and_pavel_epistasis(onlyseqs,onlyfreqs,list(ent_vec),seqnum,seqlen,total_reads)
        prot_ent_vec=calc_ordered_frequencies(protnum,protlen,proteins,True)
        prot_inscape_entropy=sum(prot_ent_vec)/len(prot_ent_vec)
        # david_prot_epis,pavel_prot_epis=david_and_pavel_epistasis(prot_seqsonly,prot_freqs,list(prot_ent_vec),protnum,protlen,sum(prot_freqs))
        if sum(sum(adj_1))==0:
            degree_assortativity=0
            degree_entropy=0
        else:
            degree_assortativity=degree_corell(g) #1
            degree_entropy=degree_distribution(g) #2
        # s=[mean_dist,std_dev,meanConsensus,pelin_nuc_entropy,trans_mut,dnds_val,pca_comps,kol_complexity,freq_corr,s_metric,cluster_coeff,phacelia_score,protein_kmer_entropy,protein_nuc_entropy,atchley_st[0],atchley_st[1],atchley_st[2],atchley_st[3],atchley_st[4],atchley_wa[0],atchley_wa[1],atchley_wa[2],atchley_wa[3],atchley_wa[4],cv_dist,max_dist,haplo_freq,pelin_haplo_freq,inscape_kmer_entropy,pelin_kmer_entropy,nuc_div,nuc_entropy,mutato,one_step_entropy,num_comp,dumb_epistasis,degree_assortativity,degree_entropy,prot_ratio,david_epis,pavel_epis,david_prot_epis,pavel_prot_epis,prot_inscape_entropy]
        s=[mean_dist,std_dev,meanConsensus,trans_mut,dnds_val,pca_comps,kol_complexity,freq_corr,s_metric,cluster_coeff,phacelia_score,protein_nuc_entropy,sd[0],sd[1],sd[2],sd[3],sd[4],wa[0],wa[1],wa[2],wa[3],wa[4],dists[0],dists[1],dists[2],dists[3],dists[4],dists[5],stdrow,cv_dist,max_dist,haplo_freq,pelin_haplo_freq,nuc_div,nuc_entropy,mutato,one_step_entropy,num_comp,degree_assortativity,degree_entropy,prot_ratio,prot_inscape_entropy]
        for i in inscape_kmer_nuc_house:
            s.append(i)
        for i in inscape_kmer_protein_house:
            s.append(i)
        for i in pelin_kmer_nuc_house:
            s.append(i)
        for i in pelin_kmer_protein_house:
            s.append(i)
        print(trimfh(file)+','+','.join(map(str,s)))
        
if __name__=='__main__':
    files=[]
    for file in os.listdir(os.getcwd()):
        if file.endswith('fas') or file.endswith('fasta') or file.endswith('fa'):
            files.append(file)
    main(files)

