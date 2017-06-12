import numpy as np
import sys

trainFile = "./data/train.txt"
testFile = "./data/test.txt"
outputFile = "./result/output.txt"

def readFile(fileName):
    fileData = np.genfromtxt(fileName,delimiter = ' ',dtype=object)
    dimensionValues = fileData[:,:-1].astype(np.float)
    actualClass = fileData[:,-1]
    return dimensionValues,actualClass

def calculateGaussianParams(gaussianValues):
    mean = gaussianValues.mean()
    std = gaussianValues.std()
    return mean,std

def createGaussianClassifier(fileName):
    gaussianDim1 = {}
    gaussianDim2 = {}
    probVowel = {}
    dimensionValues,actualVowel = readFile(fileName)
    totalClasses, counts = np.unique(actualVowel,return_counts = True)
    # print totalClasses, counts
    i = 0
    for vowel in totalClasses:
        gaussianDim1[vowel] = calculateGaussianParams(dimensionValues[actualVowel==vowel,0])
        gaussianDim2[vowel] = calculateGaussianParams(dimensionValues[actualVowel==vowel,1])
        probVowel[vowel] = float(counts[i])/counts.sum()
        i+=1
    return (gaussianDim1,gaussianDim2,probVowel)

def calcGaussProb(gaussianDim,val):
    mean, std = gaussianDim
    power = -np.power((val-mean),2)/(2*np.power(std,2))
    prob = np.exp(power)/(std*np.sqrt(2*3.1417))
    return prob

def calculateConfusionMatrix(classAssigned,actualClass):
    predictedClass = np.array(classAssigned)
    totalClasses = np.unique(actualClass)
    print "  ",
    for vowel in totalClasses:
        print vowel,
    print ""
    confusion_matrix =  {}
    for vowel in totalClasses:
        confusion_matrix[vowel] = {}
        tempClasses = predictedClass[actualClass==vowel]
        print vowel,
        for vowel2 in totalClasses:
            val = len(tempClasses[tempClasses==vowel2])
            print '{0: >2}'.format(val),
        print

def printClusters(clusterAssigned):
    f1 = open(outputFile, 'w+')
    f1.write("ID,Cluster Number\n")
    for rows in range(0,len(clusterAssigned)):
        f1.write(str(rows+1)+","+str(clusterAssigned[rows])+"\n")
    f1.close()

def predictVowels(gaussianClassifier,fileName):
    gaussianDim1,gaussianDim2,probVowel = gaussianClassifier
    dimensionValues,actualVowel = readFile(fileName)
    predictedVowel = []
    correct = 0.0
    misclassified = {}
    for rows in range(0,len(dimensionValues)):
        maxProb = 0
        finalVowel = ""
        for vowel in probVowel:
            part1 = calcGaussProb(gaussianDim1[vowel],dimensionValues[rows,0])
            part2 = calcGaussProb(gaussianDim2[vowel],dimensionValues[rows,1])
            part3 = probVowel[vowel]
            prob = part1 * part2 * part3
            if(prob > maxProb):
                maxProb = prob
                finalVowel = vowel
        predictedVowel.append(finalVowel)
        if(finalVowel==actualVowel[rows]):
            correct+=1
        else:
            if finalVowel in misclassified:
                misclassified[finalVowel]+=1
            else:
                misclassified[finalVowel]=1
    print correct/len(dimensionValues)
    calculateConfusionMatrix(predictedVowel,actualVowel)
    return predictedVowel


if __name__ == "__main__":
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    gaussianClassifier = createGaussianClassifier(trainFile)
    predictedVowel = predictVowels(gaussianClassifier,testFile)
