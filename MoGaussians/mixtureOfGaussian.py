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

def calcGaussProb(mean, std, val):
    power = -np.power((val-mean),2)/(2*np.power(std,2))
    prob = np.exp(power)/(std*np.sqrt(2*3.1417))
    return prob

def getInitialMOG(gaussianValues, mixtureNumber):
    mean = gaussianValues.mean()
    std = gaussianValues.std()
    mixOfGauss = []
    # intial values for the mixtures
    for i in range(0,mixtureNumber):
        prior = 1.0/mixtureNumber;
        if i%2==0:
            mean1 = mean - (float(i)/20)*std
        else:
            mean1 = mean + (float(i)/20)*std
        mixOfGauss.append([prior,mean1,std])
    return mixOfGauss

def expectationStep(gaussianValues,mixOfGauss):
    probValues = np.zeros((len(mixOfGauss), len(gaussianValues)))
    cost = 0
    for i in range(0,len(mixOfGauss)):
        prior, mean, std = mixOfGauss[i]
        for j in range(0, len(gaussianValues)):
            probValues[i,j] = prior * calcGaussProb(mean, std, gaussianValues[j])

    temp = probValues.sum(0)
    cost = (np.log(temp)).sum()
    probValues = probValues/probValues.sum(0)
    return probValues,-cost

def maximizationStep(gaussianValues,probValueGauss):
    mixOfGauss = []
    size = len(gaussianValues)
    N = probValueGauss.sum()
    for i in range(0,len(probValueGauss)):
        newVal = probValueGauss[i]*gaussianValues
        Ni = probValueGauss[i].sum()
        mean = newVal.sum()/Ni
        std = np.sqrt(((newVal*gaussianValues).sum()/Ni) - (mean*mean))
        prior = Ni/N
        mixOfGauss.append([prior,mean,std])
    return mixOfGauss

def calculateMOGParams(gaussianValues, mixtureNumber):
    mixOfGauss = getInitialMOG(gaussianValues, mixtureNumber)
    cost = 0
    converged = False
    i=0
    while not converged:
        probValueGauss, newCost = expectationStep(gaussianValues,mixOfGauss)
        newMixOfGauss = maximizationStep(gaussianValues,probValueGauss)
        if(cost-newCost<0.01 and i!=0):
            converged = True
        mixOfGauss = newMixOfGauss
        cost = newCost
        i+=1
    return mixOfGauss

def createGaussianClassifier(fileName,mixtureNumber):
    gaussianDim1 = {}
    gaussianDim2 = {}
    probVowel = {}
    dimensionValues,actualVowel = readFile(fileName)
    totalClasses, counts = np.unique(actualVowel,return_counts = True)
    i = 0
    for vowel in totalClasses:
        gaussianDim1[vowel] = calculateMOGParams(dimensionValues[actualVowel==vowel,0], mixtureNumber)
        gaussianDim2[vowel] = calculateMOGParams(dimensionValues[actualVowel==vowel,1], mixtureNumber)
        probVowel[vowel] = float(counts[i])/counts.sum()
        i+=1
    return (gaussianDim1,gaussianDim2,probVowel)

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

def calcVowelProb(mixOfGauss, val):
    totalProb = 0.0
    for i in range(0,len(mixOfGauss)):
        prior, mean, std = mixOfGauss[i]
        totalProb += prior * calcGaussProb(mean, std, val)
    return totalProb

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
            part1 = calcVowelProb(gaussianDim1[vowel],dimensionValues[rows,0])
            part2 = calcVowelProb(gaussianDim2[vowel],dimensionValues[rows,1])
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
    printClusters(predictedVowel)
    print correct/len(dimensionValues)
    calculateConfusionMatrix(predictedVowel,actualVowel)
    return predictedVowel

if __name__ == "__main__":
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    mixtureNumber = sys.argv[3]
    gaussianClassifier = createGaussianClassifier(trainFile,int(mixtureNumber))
    predictedVowel = predictVowels(gaussianClassifier,testFile)
