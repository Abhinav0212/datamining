import sys
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

outputFile = "output.csv"
outputFile_1 = "output_wine.csv"
outputFile_2 = "output_wine_off_the_shelf.csv"

def minMaxNormalize(ratio_data_matrix,max_arr,min_arr):

    ran_arr = max_arr - min_arr
    for r in range(0,len(ratio_data_matrix)):
        ratio_data_matrix[r] = (ratio_data_matrix[r] - min_arr) / ran_arr
    return ratio_data_matrix

def preprocessData(inputData,id):
    if(id==1):
        clusterFeatures = inputData[:,1:-2].astype(np.float)
    else:
        clusterFeatures = inputData[:,1:-1].astype(np.float)
    # normalizedClusterFeatures = minMaxNormalize(clusterFeatures,clusterFeatures.max(0),clusterFeatures.min(0))
    # return normalizedClusterFeatures
    return clusterFeatures

def assignRandomRow(processedData):
    return random.randint(0,processedData.shape[0]-1)

def getInitialPoints(processedData,k):
    centroids = np.zeros((k,processedData.shape[1]), dtype=np.float)
    for i in range(0,k):
        row = assignRandomRow(processedData)
        centroids[i] = processedData[row]
    return centroids

def euclidianDistance(row1, row2):
    difference = row1-row2
    dist = math.sqrt(np.dot(difference,difference.transpose()))
    return dist

def assignClusters(processedData,centroids,k):
    clusterNum=  np.zeros((processedData.shape[0]), dtype=np.int)
    for i in range(0, processedData.shape[0]):
        clusterDist = -1
        for j in range(0,k):
            dist = euclidianDistance(processedData[i], centroids[j])
            if(dist<clusterDist or clusterDist==-1):
                clusterDist = dist
                clusterNum[i] = (j+1)
    return clusterNum

def findFarthestPoints(centroids,processedData,clusterClass,k):
    newPoints= int(k - len(np.unique(clusterClass)))
    datapointDistances =  np.zeros((processedData.shape[0]), dtype=[('dist',float),('id',int)])
    for i in range(0, processedData.shape[0]):
        clusterVal = clusterClass[i]-1
        datapointDistances[i][0] = euclidianDistance(processedData[i], centroids[clusterVal])
        datapointDistances[i][1] = i
    datapointDistances = np.sort(datapointDistances,order='dist')[::-1]
    return datapointDistances[0:newPoints]

def updateCentroids(processedData,clusterClass,k):
    newCentroids = np.zeros((k,processedData.shape[1]), dtype=np.float)
    flag = False
    emptyClusters = []
    for j in range(0,k):
        totalCount = 0
        for i in range(0, processedData.shape[0]):
            if(clusterClass[i]==j+1):
                newCentroids[j] = newCentroids[j] + processedData[i,:]
                totalCount = totalCount + 1
        if(totalCount!=0):
            newCentroids[j] = newCentroids[j] / totalCount
        # clusters with no data points
        else:
            flag = True
            emptyClusters.append(j)
    if(flag):
        centroidsForEmptyCluster = findFarthestPoints(newCentroids,processedData,clusterClass,k)
        for val in range(0, len(emptyClusters)):
            index = centroidsForEmptyCluster[val][1]
            newCentroids[emptyClusters[val]] = processedData[index]
    return newCentroids

def kMeansCluster(processedData, k):
    centroids = getInitialPoints(processedData,k)
    check = True
    counter = 1
    while(check):
        clusterClass = assignClusters(processedData,centroids,k)
        newCentroids = updateCentroids(processedData,clusterClass,k)
        counter+=1
        if((newCentroids==centroids).all()):
            check = False
        else:
            centroids = newCentroids
    print counter
    print centroids
    return clusterClass

def estimateError(clusterClass,processedData,k):
    centroids = updateCentroids(processedData,clusterClass,k)
    globalCentroid = processedData.sum(0)/processedData.shape[0]
    sseCLusterTotal = 0
    seeTotal = 0
    sseCluster =  [0 for i in range(0,k)]
    clusterCount = [0 for i in range(0,k)]
    for i in range(0, processedData.shape[0]):
        clusterVal = clusterClass[i]-1
        dist = euclidianDistance(processedData[i], centroids[clusterVal])
        sseCLusterTotal = sseCLusterTotal + (dist*dist)
        sseCluster[clusterVal] += (dist*dist)
        clusterCount[clusterVal]+=1

        dist2 = euclidianDistance(processedData[i], globalCentroid)
        seeTotal = seeTotal + (dist2*dist2)
    print "Total Cluster SSE",sseCLusterTotal

    ssbTotal = 0
    for i in range(0,k):
        print "Cluster",i,"SSE",sseCluster[i]
        dist = euclidianDistance(globalCentroid, centroids[i])
        ssbTotal = ssbTotal + (clusterCount[i]*dist*dist)
    print "Total SSE",seeTotal
    print "SSB",(seeTotal-sseCLusterTotal)
    return sseCLusterTotal

def computeDistanceMatrix(processedData):
    rowSize = processedData.shape[0]
    colSize = processedData.shape[1]
    rowDuplicateMatrix = np.repeat([processedData], rowSize, axis=0)
    colDuplicate = np.repeat([processedData], rowSize, axis=1)
    colDuplicateMatrix = colDuplicate.reshape(rowSize, rowSize, colSize)
    diff = rowDuplicateMatrix - colDuplicateMatrix
    distanceMatrix = np.sqrt((diff**2).sum(2))
    return distanceMatrix

def computeSilhouetteWidth(clusterAssigned,processedData,k):
    distanceMatrix = computeDistanceMatrix(processedData)
    totalSilhouetteCoeff = 0
    totalCount = 0
    for i in range(0,k):
        indices = np.argwhere(clusterAssigned==i+1).flatten()
        silhouetteIntraCluster = distanceMatrix[indices,:][:,indices].sum(1)
        silhouetteIntraCluster = silhouetteIntraCluster/(len(silhouetteIntraCluster)-1)

        silhouetteInterCluster = np.zeros((len(indices),k-1), dtype=np.float)
        for j in range(0,k):
            col_indices = np.argwhere(clusterAssigned==j+1).flatten()
            if(j<i):
                silhouetteInterCluster[:,j] = distanceMatrix[indices,:][:,col_indices].mean(1)
            elif(j>i):
                silhouetteInterCluster[:,j-1] = distanceMatrix[indices,:][:,col_indices].mean(1)
        silhouetteInterCluster = silhouetteInterCluster.min(1)
        silhoutteCoefficient = (silhouetteInterCluster - silhouetteIntraCluster)/np.maximum(silhouetteInterCluster,silhouetteIntraCluster)
        print "Silhoutte Coefficient",i,np.sum(silhoutteCoefficient)/len(silhoutteCoefficient)
        totalSilhouetteCoeff += np.sum(silhoutteCoefficient)
        totalCount += len(silhoutteCoefficient)
    print "Total Silhoutte Coefficient",totalSilhouetteCoeff/totalCount
    return totalSilhouetteCoeff/totalCount

def assignClassToCluster(clusterAssigned,k,actualClass):
    clusterClass =  np.zeros(len(clusterAssigned), dtype=np.int)
    totalClasses = np.unique(actualClass)
    for i in range(0,k):
        tempCluster = actualClass[clusterAssigned==i+1]
        classes, counts = np.unique(tempCluster, return_counts = True)
        maxClass = classes[np.argmax(counts)]
        clusterClass[clusterAssigned==i+1] = maxClass
    return clusterClass

def calculateConfusionMatrix(classAssigned,actualClass):
    totalClasses = np.unique(actualClass)
    for i in range(0,len(totalClasses)):
        print '{0: >6}'.format(totalClasses[i]),
    print ""
    confusion_matrix =  np.zeros((len(totalClasses),len(totalClasses)), dtype=np.float)
    for i in range(0,len(totalClasses)):
        tempClasses = classAssigned[actualClass==i+1]
        classes, counts = np.unique(tempClasses, return_counts = True)
        for j in range(0,len(classes)):
            confusion_matrix[i][classes[j]-1] = counts[j]

    total_examples = confusion_matrix.sum(axis=1)
    confusion_matrix = confusion_matrix/total_examples[:,None]
    width = 5
    for row in range(0, len(confusion_matrix)):
        print totalClasses[row],
        for elem in confusion_matrix[row]:
            print '{:{width}.{prec}f}'.format(elem, width=5, prec=4),
        print
    return confusion_matrix

def scatterPlot(processedData,clusterAssigned,actualClass,k):
    totalClasses = np.unique(actualClass)
    colors = ['r', 'b', 'g', 'y', 'm']
    markers = ['*', 'o', '^', '+']
    for classes in totalClasses:
        for i in range(0,k):
            dataToPlot = processedData[np.logical_and(clusterAssigned==i+1,actualClass==classes)]
            plt.scatter(dataToPlot[:,0],dataToPlot[:,1],color=colors[i],marker=markers[classes-1])
    plt.show()

def calculateGiniIndex(clusterQuality,threshold):
    total = len(clusterQuality)
    posCount = float(len(clusterQuality[clusterQuality>threshold]))/total
    negCount = 1-posCount
    giniIndex = 1 - (posCount*posCount + negCount*negCount)
    return giniIndex

def validateUsingQuality(quality,clusterAssigned,k):
    clusterGini =  np.zeros(k, dtype=np.float)
    clusterQualityMean =  np.zeros(k, dtype=np.float)
    clusterQualitySDev =  np.zeros(k, dtype=np.float)
    totalGini = 0
    dataSize = len(clusterAssigned)
    print "Cluster Numbers",
    for i in range(0,k):
        clusterQuality = quality[clusterAssigned==i+1]
        clusterGini[i]=calculateGiniIndex(clusterQuality,5)
        totalGini += clusterGini[i]*len(clusterQuality)/dataSize
        clusterQualityMean[i]=clusterQuality.mean()
        clusterQualitySDev[i]=clusterQuality.std()
        print '{0: >11}'.format(i),
    print ""
    print "Cluster Gini Index",clusterGini
    print "Cluster Qual Mean ",clusterQualityMean
    print "Cluster Qual StDev",clusterQualitySDev
    print "Total Gini",totalGini
    return totalGini

def printClusters(clusterAssigned):
    f1 = open(outputFile, 'w+')
    f1.write("ID,Cluster Number\n")
    for rows in range(0,len(clusterAssigned)):
        f1.write(str(rows+1)+","+str(clusterAssigned[rows])+"\n")
    f1.close()

def writeWineData(k,SSE,Sil,Gini):
    f1 = open(outputFile_1, 'a+')
    # f1 = open(outputFile_2, 'a+')
    f1.write(str(k)+","+str(SSE)+","+str(Sil)+","+str(Gini)+"\n")
    f1.close()

def clusterData(inputFileName,k):
    convertId = lambda x: float(x.strip("\""))
    convertClass = lambda x: 2.0 if x=='"High"' else 1.0
    if(inputFileName=="wine_quality-red.csv"):
        inputData = np.genfromtxt(inputFileName,delimiter = ',',skip_header=1,converters={0: convertId, 13: convertClass})
        processedData = preprocessData(inputData,1)
        quality = inputData[:,-2].astype(np.int)
    else:
        inputData = np.genfromtxt(inputFileName,delimiter = ',',skip_header=1,converters={0: convertId})
        processedData = preprocessData(inputData,2)

    actualClass = inputData[:,-1].astype(np.int)
    totalClasses = np.unique(actualClass)

    clusterAssigned = kMeansCluster(processedData,k)

    # clusterOffTheShelf = KMeans(n_clusters=k).fit(processedData)
    # clusterAssigned = clusterOffTheShelf.labels_ + 1

    printClusters(clusterAssigned)
    # SSE for the original data
    print "\nTrue SSE for",inputFileName[:-4]
    estimateError(actualClass,processedData,len(totalClasses))
    print "\nEstimated SSE for",inputFileName[:-4],"k =",k
    SSE = estimateError(clusterAssigned,processedData,k)
    # print "\nSilhouette width for",inputFileName[:-4],"k =",k
    # Sil = computeSilhouetteWidth(clusterAssigned,processedData,k)
    #
    # classAssigned = assignClassToCluster(clusterAssigned,k,actualClass)
    # print "\nConfusion Matrix for\n",inputFileName[:-4],"k =",k
    # calculateConfusionMatrix(classAssigned,actualClass)
    # print ""
    # if(inputFileName!="wine_quality-red.csv"):
    #     scatterPlot(processedData,clusterAssigned,actualClass,k)
    # else:
    #     Gini = validateUsingQuality(quality,clusterAssigned,k)
    #     writeWineData(k,SSE,Sil,Gini)

if __name__ == "__main__":
    inputFileName = sys.argv[1]
    k = sys.argv[2]
    clusterData(inputFileName,int(k))
