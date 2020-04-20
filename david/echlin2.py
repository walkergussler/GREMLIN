#!/usr/bin/env python

__author__ = 'David S. Campo, Ph.D.'
'''
one_step_stats. Program for calculating network statistics of one-step networks. 
1.Input is an edge list in csv format. The list was produced by one_step_definition.py and includes: node1, node2, hamming distance, one-step logical, path distances, frequency of node 1 and frequency of node 2.  
2. Output is .csv file with the summary of several topological properties of each network

usage:
python one_step_stats.py -i input_folder_location -o output_folder_location


'''


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
            hTemp.append(i*math.log(np.divide(1, j, dtype = float),2))
            
        colorEntropy.append(sum(hTemp))
    
    graphEntropy = min(colorEntropy)
    return graphEntropy

def entropyCalc(freqM):#different results than scipy.stats.entropy - how to resolve?
    productVectorM = 0
    for i in freqM:
        if i > 0:
            productVectorM = productVectorM + (i*math.log(i, 2))
            print(i,productVectorM)
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
        degreeBox.append(i)
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
    vonEntropyBox = np.divide(entropyCalc(eValsLapNonZeroBoxNorm), math.log(boxNodes,2), dtype = float)
    degreeEntropyBox = np.divide(entropyCalc(degreeNormBox), math.log(boxNodes,2), dtype = float)
    KSEntropyBox = np.divide(math.log(spectralRadiusAdjBox, 2), math.log(boxNodes-1,2), dtype = float)
    motifEntropyBox = np.divide(entropyCalc(freqMBox), math.log(len(freqMBox),2), dtype = float)
    popEntropyBox = np.divide(entropyCalc(eigenCentBox), math.log(boxNodes,2), dtype = float)
    graphEntropyBox = np.divide(graphEntropyCalc(colorsBox), math.log(boxNodes,2), dtype = float)
    
    return edgePBox, radiusBox, kCoreBox, degreeAssortBox, diameterPathBox, avgPathDistBox, nEdgesBox, globalEfficiencyBox, avgDegreeBox, maxDegreeBox, spectralRadiusAdjBox, spectralGapBox, scaledSpectralRadiusBox, colorNumBox, avgClustCoeffBox, freqMBox, motifEntropyBox, vonEntropyBox, graphEntropyBox, popEntropyBox, KSEntropyBox, degreeEntropyBox

def fractalCalc(dist, nodeNum):
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

def process_links(links):
    print('file_name,comp_size,n_edges,edgeP,radius,kCore,degreeAssort,corr_path_hamm_dist,RMSE_path_hamm_dist,avg_degree,max_degree,mean_hamm_dist,mean_path_dist,diameter_Path,diameter_Hamm,corr_degree_freq,corr_eigen_cent_freq,corr_close_cent_freq,corr_bet_cent_freq,corrPageRankfreq,genetic_load,CV_freq,localOptFrac,viable_fraction,scaledSpectralRadius,spectralRadiusAdj,spectralGap,spectralRadiusHamm,popEntropy,VonEntropy,KSEntropy,degreeEntropy,average_clustering_coeficcient,global_efficiency,motif_1_star,motif_2_path,motif_3_cycle,motif_4_tailed_triangle,motif_5_envelope,motif_6_clique,graphEntropy,motif_entropy,colorNum,pathDb,pathDbConstant,pathRSquare,pathModularity,pathModularityConstant,pathModularityRSquare,diameterPathSelf,avgPathDistSelf,nEdgesSelf,edgePSelf,globalEfficiencySelf,avgDegreeSelf,maxDegreeSelf,spectralRadiusAdjSelf,spectralGapSelf,popEntropySelf,scaledSpectralRadiusSelf,colorNumSelf,avgClustCoeffSelf,vonEntropySelf,KSEntropySelf,degreeEntropySelf,graphEntropySelf,motifEntropySelf,freqMBoxSelf\n')
    G = nx.Graph() #G is adjacency matrix
    node1 = []
    node2 = []
    hammDist  = []
    pathDist = []
    freqDict = {}
    for row in links:
        node1Temp = int(row[0])
        node2Temp = int(row[1])
        hammDistTemp = int(float(row[2]))
        pathDistTemp = int(float(row[4]))
        freqDict[node1Temp] = float(row[5])
        freqDict[node2Temp] = float(row[6])
        node1.append(node1Temp)
        node2.append(node2Temp)
        hammDist.append(hammDistTemp)
        pathDist.append(pathDistTemp)
        if pathDistTemp == 1:
            G.add_edge(node1Temp, node2Temp)
    
    pairsNum = len(node1)
    nodeList = range(nodes)
    ## node frequencies
    freqCount = np.zeros(nodes)
    for k in nodeList:
        freqCount[k] = freqDict[k]
    del freqDict

    #constants
    posNum = 15
    letters = 2
    u = 0.000115

    if nodes <= 3:
        print("Sample %s has too few(%i) nodes" %(file,nodes))
    else:
        #print 'correlations'
        ## distance properties
        diameterHamm = max(hammDist)
        avgHammDist = np.mean(hammDist)
        corrPathHammDist = np.corrcoef(hammDist, pathDist)
        RMSEPathHammDist = mean_squared_error(hammDist, pathDist)
        del hammDist, 
        degreeRaw = list(G.degree())
        degree = []
        for i in degreeRaw:
            degree.append(i)
        degreeNorm = np.divide(degree, np.sum(degree), dtype = float)
        eigenCentRaw = list(nx.eigenvector_centrality_numpy(G, weight=None).values())
        eigenCent = np.divide(eigenCentRaw, sum(eigenCentRaw), dtype = float)
        closeCent = list(nx.closeness_centrality(G).values())
        betCent = list(nx.betweenness_centrality(G).values())
        pageRank = list(nx.pagerank_numpy(G).values())
        # correlations
        corrDegreeFreq = np.corrcoef(freqCount, degree)
        corrEigenCentFreq = np.corrcoef(freqCount, eigenCent)
        corrCloseCentFreq = np.corrcoef(freqCount, closeCent)
        corrBetCentFreq = np.corrcoef(freqCount, betCent)
        corrPageRankfreq = np.corrcoef(freqCount, pageRank)                
        edgeP, radius, kCore, degreeAssort, diameterPath, avgPathDist, nEdges, globalEfficiency, avgDegree, maxDegree, spectralRadiusAdj, spectralGap, scaledSpectralRadius, colorNum, avgClustCoeff, freqM, motifEntropy, vonEntropy, graphEntropy, popEntropy, KSEntropy, degreeEntropy = boxStats(G)
        spectralRadiusHamm = 1#max(abs(eValsH))

        nReads = sum(freqCount)
        freqCountRel = np.divide(freqCount, nReads, dtype = float)
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
                #print freqCount[n0], freqCount[n1]
                if freqCountRel[n0] > freqCountRel[n1]:
                   flagMax = 1
                if freqCountRel[n0] < freqCountRel[n1]:
                   flagMin = 1                          
                   
            if flagMax == 1 and flagMin == 0:
                #rint n0, n1, freqCount[n0], freqCount[n1]
                localOptNum = localOptNum + 1 
        localOptFrac = np.divide(localOptNum, nodes, dtype = float)          
        
        vs = 1 - uPosNum * (1 - (np.divide(degree, neighbors, dtype = float)))
        viableFraction = sum(vs*eigenCent)
        del freqCount, eigenCent, freqCountRel, d1, vs

        # Fractal dimension
        #print 'fractal'
        fractalDb, fractalRSquare, fractalDbConstant, fractalModularity, fractalModularityConstant, fractalModularityRSquare, diameterPathSelf, avgPathDistSelf, nEdgesSelf, edgePSelf, radiusSelf, kCoreSelf, degreeAssortSelf, globalEfficiencySelf, avgDegreeSelf, maxDegreeSelf, spectralRadiusAdjSelf, spectralGapSelf, popEntropySelf,scaledSpectralRadiusSelf, colorNumSelf, avgClustCoeffSelf, vonEntropySelf, KSEntropySelf, degreeEntropySelf, graphEntropySelf, motifEntropySelf, freqMBoxSelf  = fractalCalc(pathDist, nodes)    
     
        #save summary file
        print('we did it')
        print(str(fileName1) + ',' +  str(nodes) + ',' +  str(nEdges) + ',' +  str(edgeP) + ',' + str(radius) + ',' + str(kCore) + ',' + str(degreeAssort) + ',' +  str(corrPathHammDist[0][1]) +  ',' +  str(RMSEPathHammDist) +  ',' +  str(avgDegree) + ',' +  str(maxDegree) +  ',' +  str(avgHammDist) + ',' +  str(avgPathDist) + ',' +  str(diameterPath) + ',' +  str(diameterHamm) + ',' +  str(corrDegreeFreq[0][1]) + ',' +  str(corrEigenCentFreq[0][1]) + ',' +  str(corrCloseCentFreq[0][1]) + ',' +  str(corrBetCentFreq[0][1]) + ',' +  str(corrPageRankfreq[0][1]) + ',' +  str(geneticLoad) + ',' +  str(CV) + ',' +  str(localOptFrac) + ',' +  str(viableFraction) +  ',' + str(scaledSpectralRadius) + ',' +  str(spectralRadiusAdj) + ',' + str(spectralGap) + ',' +  str(spectralRadiusHamm) + ',' +  str(popEntropy) + ',' +  str(vonEntropy) + ',' +  str(KSEntropy)  + ',' +  str(degreeEntropy) + ',' +  str(avgClustCoeff) + ',' +  str(globalEfficiency) + ',' +  str(100*freqM[0]) + ',' +  str(100*freqM[1]) + ',' +  str(100*freqM[2]) + ',' +  str(100*freqM[3]) + ',' +  str(100*freqM[4]) + ',' +  str(100*freqM[5]) + ',' +  str(graphEntropy) + ',' +  str(motifEntropy) +  ',' +  str(colorNum) + ',' +  str(fractalDb) + ',' +  str(fractalDbConstant) + ',' +  str(fractalRSquare) + ',' +  str(fractalModularity) + ',' +  str(fractalModularityConstant) + ',' +  str(fractalModularityRSquare) + ',' +  str(diameterPathSelf) + ',' +  str(avgPathDistSelf) + ',' +  str(nEdgesSelf) + ',' +  str(edgePSelf) + ',' +  str(globalEfficiencySelf) + ',' +  str(avgDegreeSelf) + ',' +  str(maxDegreeSelf) + ',' +  str(spectralRadiusAdjSelf) + ',' + str(spectralGapSelf) + ',' +  str(popEntropySelf)+ ',' +  str(scaledSpectralRadiusSelf) + ',' +  str(colorNumSelf) + ',' +  str(avgClustCoeffSelf) + ',' +  str(vonEntropySelf)+ ',' +  str(KSEntropySelf)+','  +  str(degreeEntropySelf) + ',' +  str(graphEntropySelf) + ',' +  str(motifEntropySelf) + ',' +  str(freqMBoxSelf)     + '\n')

if __name__=="__main__":
    for file in os.listdir(os.getcwd()):
        if file.endswith("links.csv"):
            print(file)
            process_links(file)
        