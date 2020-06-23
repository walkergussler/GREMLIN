#!/usr/bin/env python

__author__ = 'David S. Campo, Ph.D.'
'''
one_step_definition. Program for calculating large one-step networks.
1. It requires as input the location of the aligned fasta files. All sequences are assumed to be different and each sequence 
    has its frequency at the end of its name, preceded by a underscore (e.g. >seq1_765)
2. It calculates the connected components with a user-defined distance limit (defaul 1, one step ) using the entropy trick (most variable positions are evaluated first).
3. For each large component (frequency higher than user-specified threshold, default 5% of all reads), it saves a .csv file, where for every pair of sequences, it shows 
    the hamming distance, if the distance is lower or equal than the user-defined distance threshold, the path distance, 
    thh frequency of node 1 and the frequency of node 2.
    It also saves a .net file (for pajek), a .vec file with the square of the frequencies (for pajek) and a csv with the square of the frequencies (for gephi)

module load Python/2.7.3

With default parameters:
python one_step_definition.py -i input_folder_location -o output_folder_location

Changing the minimal frequency of the large components (-t; default 0.05) 
minimal step hamming distance threshold (-d; default 1):
Save only the biggest component (-s; default 0, no, 1 yes)

'''

import time, os, re, math, glob
import numpy as np
import optparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tempfile import NamedTemporaryFile
from subprocess import check_call
from Bio.Alphabet import generic_dna
from scipy.sparse.csgraph import connected_components, csgraph_from_dense, shortest_path
from sys import exit
from collections import defaultdict

def align(file):
    # print(type(seqs),len(seqs))
    seqs=parse_input(file)
    with NamedTemporaryFile(delete=False, mode='w') as seqdump:
        catname=seqdump.name
        for seq in seqs:
            seqdump.write('>seq_'+str(seqs[seq])+'\n'+str(seq)+'\n')
    with NamedTemporaryFile(delete=False) as aligned:
        alignname=aligned.name
        check_call(['mafft', '--quiet', '--auto', '--thread', '20', '--preservecase', catname], stdout=aligned)
    os.unlink(catname)
    seqs=parse_input(alignname)
    if type(seqs)==bool:
        print('error parsing aligned seqs!')
        exit()
    return seqs

def parse_input(input): #get sequences from a file
    seqs=defaultdict(int)
    with open(input,'r') as f:
        for record in SeqIO.parse(f,'fasta'):
            freq = int(record.id.split('_')[-1])
            seq=record.seq.upper()
            seqs[seq]+=freq
    return seqs
    
def calc_distance_matrix(finalSeqs): #calculate distance matrix from the 1-step list
    from pyseqdist import hamming
    l=len(finalSeqs)
    arr=np.zeros([l,l])
    hdist=hamming(finalSeqs,finalSeqs,ignore_gaps=False)
    for id in range(len(hdist)):
        item=hdist[id]
        arr[:,id]=item[:,0]
    return arr    

def get_adj(DM,thr_dist,seqnum):
    return 1*(DM <= thr_dist) - np.eye(seqnum)    
    
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
        # print(i,freq)
    max_comp=max(comps_freqlist)
    # print(comp_list)
    # print(comps_freqlist)
    # print(comps_info)
    # if list(comps_freqlist).count(max_comp)!=1:
        # print(comps_freqlist)
        # exit('ambiguous as to which component is biggest! Investigate')
    # print('biggest comp percentage='+str(max_comp*100)+'%')
    return comps_freqlist,list(comps_freqlist).index(max_comp),comps_info

def smaller_adj(adj,allowed):
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
                
def process_component(file,only_seqs,only_freqs,component,output_dir,comps_info,adj,save_all_comps,dists):
    if save_all_comps:
        out_1 = output_dir + '/' + file + '_comp' + str(component) + '_links.csv'
        out_2 = output_dir + '/' + file + '_comp' + str(component) + '.fas'
    else:
        out_1 = output_dir + '/' + file + '_links.csv'
        out_2 = output_dir + '/' + file + '.fas'
    nodes_real_names=comps_info[component]
    adj_comp,comp_size=smaller_adj(adj,nodes_real_names)
    sparseMatrixComp = csgraph_from_dense(adj_comp)
    path_dists = shortest_path(sparseMatrixComp, method='auto', directed=False, return_predecessors=False, unweighted=True, overwrite=False)    
    #save link files (.csv and _pajek.net)    
        # for p in range(comp_num-1):
            # for q in range(p+1, comp_num):
                # outputHandle1.write(str(p) + ',' + str(q) + ',' + str(finalDist[p][q]) + ',' + str(adj_comp[p][q]) + ',' + str(path_dists[p][q]) +  ',' + str(freqSub[p]) + ',' + str(freqSub[q]) + '\n')
                # print(str(p) + ',' + str(q) + ',' + str(finalDist[p][q]) + ',' + str(adj_comp[p][q]) + ',' + str(path_dists[p][q]) +  ',' + str(freqSub[p]) + ',' + str(freqSub[q]))
    with open(out_1, "w") as outputHandle1:    
        for p in range(comp_size-1):
            for q in range(p+1, comp_size):
                # print(p,q)
                realp=nodes_real_names[p]
                realq=nodes_real_names[q]
                outputHandle1.write(str(p) + ',' + str(q) + ',' + str(dists[realp,realq]) + ',' + str(adj_comp[p][q]) + ',' + str(path_dists[p][q]) +  ',' + str(only_freqs[realp]) + ',' + str(only_freqs[realq]) + '\n')
                    
    #save .fas file
    comp_seqs=[]
    n = 0
    for k in nodes_real_names:
        print(k)
        haplotype = str(only_seqs[k])
        n += 1
        seq_id = str(n) + '_' + str(only_freqs[k])
        seq_dna = Seq(haplotype,generic_dna)
        seqFASTA = SeqRecord(seq_dna, id = seq_id, name = "", description = "")        
        comp_seqs.append(seqFASTA)
    with open(out_2, "w") as outputHandle2:
        SeqIO.write(comp_seqs, outputHandle2, "fasta")

def main(input_dir,output_dir,dist_lim,allowed_fraction_base,save_all_comps):
    # Mark the start time
    startTime = time.time()
    old_time=time.time()
    OUTPUT_METADATA='big_comp_stats.csv'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, OUTPUT_METADATA), "w") as outputHandle:
        outputHandle.write('file_name,seqnum,reads,Total_number_components,number_big_components,frequency_threshold,total_big_frequency,work_time \n')
    #sorted file list
    fileList = sorted(glob.glob(os.path.join(input_dir, '*.fa*')))
    for file in fileList:
        print(file)
        seqs = align(file)
        only_seqs=list(seqs.keys())
        only_freqs=list(seqs.values())
        total_reads=sum(only_freqs)
        allowed_fraction=total_reads*allowed_fraction_base
        rel_freqs=np.divide(only_freqs,total_reads)
        seqnum = len(seqs)
        seqlen = len(only_seqs[0])
        d_mat=calc_distance_matrix(only_seqs)
        adj=get_adj(d_mat,dist_lim,seqnum)
        comps_freqlist,major_comp,comps_info=get_comp_freqs(adj,only_freqs)
        num_comps=0
        accepted_freq=sum(comps_freqlist[comps_freqlist>allowed_fraction])/total_reads
        # if max(comps_freqlist)<allowed_fraction:
            # exit('major component (%.2f) is less than allowed fraction (%.2f)' (max(comps_freqlist),allowed_fraction))
        for i in range(len(comps_freqlist)):
            if comps_freqlist[i]>allowed_fraction:
                num_comps+=1
                if save_all_comps:
                    process_component(trimfh(file),only_seqs,only_freqs,i,output_dir,comps_info,adj,True,d_mat)
                else:
                    if i==major_comp:
                        process_component(trimfh(file),only_seqs,only_freqs,i,output_dir,comps_info,adj,False,d_mat)

        # save stats
        new_time=time.time()
        work_time=new_time-old_time
        old_time=time.time()
        with open(os.path.join(output_dir, OUTPUT_METADATA), "a") as outputHandle:
            outputHandle.write(file + ',' + str(seqnum) + ',' + str(total_reads) + ',' + str(len(comps_freqlist)) + ',' + str(num_comps) + ',' + str(allowed_fraction) + ',' + str(accepted_freq) + ',' + str(work_time) + '\n')
    #mark the end time    
    endTime = time.time()
    work_time =  endTime - startTime
    print('script took %.2f seconds to proces this folder' % work_time)
    
if __name__ == '__main__':
    argument_parser = optparse.OptionParser()            
    argument_parser.add_option('-i', metavar='IDIR', action='store', type=str, dest='input_directory', default='', help='Input file')
    argument_parser.add_option('-o', metavar='ODIR', action='store', type=str, dest='output_directory', default='echlin1', help='Output directory')
    argument_parser.add_option('-d', action='store', dest='dist_lim', type=int, default= 1, help='dist_lim')
    argument_parser.add_option('-t', action='store', dest='allowed_fraction', type=float, default= 0.05, help='allowed_fraction')
    argument_parser.add_option('-s', action='store', dest='save_all_comps', type=int, default= 0, help='save only biggest')
    options, args = argument_parser.parse_args()
    input_dir = options.input_directory
    output_dir = options.output_directory    
    allowed_fraction = options.allowed_fraction    
    dist_lim = options.dist_lim
    save_all_comps = options.save_all_comps

    main(input_dir,output_dir,dist_lim,allowed_fraction,save_all_comps)
    # os.chdir(output_dir)
    # try:
        # os.system('mkdir fasta linkscsv trash stats_dir')
    # except:
        # os.system('rm -r fasta linkscsv trash stats_dir')
        # os.system('mkdir fasta linkscsv trash stats_dir')
    # os.system('mv *vec *net *gephi* trash')
    # os.system('mv *csv stats_dir')
    # os.system('mv stats_dir/big_comp_stats.csv .')
    # os.system('mv *fas fasta')
    
    