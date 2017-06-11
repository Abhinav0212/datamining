import sys
import numpy as np
import math

game_metadata = [3,3,3,3,2,2,2,2,2]

def preprocessData(file):
    data = np.genfromtxt(file,delimiter = ',',dtype=np.float)
    return data

def calculateEntropy(class_column,weights):
    total_classes = game_metadata[len(game_metadata)-1]
    class_distribution = np.zeros(total_classes,dtype=np.float)
    for row in range(0,len(class_column)):
        if(class_column[row]==1):
            class_distribution[1]+=weights[row]
        else:
            class_distribution[0]+=weights[row]
    class_distribution = class_distribution / class_distribution.sum()
    entropy = 0
    for value in class_distribution:
        entropy = entropy - (value*math.log(value))
    return [entropy, np.argmax(class_distribution),np.max(class_distribution)]

def estimateInformationGain(processed_data,column,total_columns,weights):
    total_feature_values = game_metadata[column]
    feature_distribution = np.zeros(total_feature_values,dtype=np.float)
    feature_decision = np.zeros(total_feature_values,dtype=np.float)
    feature_decision_prob = np.zeros(total_feature_values,dtype=np.float)
    split_classes = {}
    split_weights = {}

    for feature in range(0,total_feature_values):
        split_classes[feature]=[]
        split_weights[feature]=[]

    for row in range(0,len(processed_data)):
        feature_value = int(processed_data[row][column])
        feature_distribution[feature_value]+=weights[row]
        split_classes[feature_value].append(processed_data[row][total_columns-1])
        split_weights[feature_value].append(weights[row])
    feature_distribution = feature_distribution / feature_distribution.sum()

    information_before_split = calculateEntropy(processed_data[:,total_columns-1],weights)[0]
    information = 0
    for feature in range(0,total_feature_values):
        entropy_result = calculateEntropy(split_classes[feature],split_weights[feature])
        entropy = entropy_result[0]
        feature_decision[feature] = entropy_result[1]
        feature_decision_prob[feature] = entropy_result[2]
        information = information + (feature_distribution[feature]*entropy)
    gain = information_before_split - information
    return [gain,feature_decision,feature_decision_prob]

def determineBestSplit(processed_data,weights):
    total_columns = processed_data.shape[1]
    max_gain = -1
    best_feature = -1
    best_feature_decision = []
    best_feature_decision_prob = []
    for column in range(0,total_columns-1):
        gain_result = estimateInformationGain(processed_data,column,total_columns,weights)
        gain = gain_result[0]
        if(gain>max_gain):
            max_gain = gain
            best_feature = column
            best_feature_decision = gain_result[1]
            best_feature_decision_prob = gain_result[2]
    return best_feature, best_feature_decision, best_feature_decision_prob

def learnWeights(predicted_class,processed_data,weights):
    error = 0.0
    total_columns = processed_data.shape[1]
    size = len(predicted_class)
    for row in range(0,size):
        if(predicted_class[row]!=processed_data[row][total_columns-1]):
            error+=weights[row]
    for row in range(0,size):
        if(predicted_class[row]==processed_data[row][total_columns-1]):
            weights[row] = weights[row] * error / (1-error)
    weights = weights / weights.sum()
    z = math.log((1-error)/error)
    return [weights,z]

def trainEnsembleClassifier(train_file_name,ensemble_number):
    processed_data = preprocessData(train_file_name)
    num_examples = len(processed_data)
    weights = np.ones(num_examples,dtype=np.float)*(1.0/num_examples)
    classifiers = []
    classifier_weights = []
    for i in range(0,ensemble_number):
        DecisionStump = determineBestSplit(processed_data,weights)
        training_predictions = predictClass(processed_data,DecisionStump)
        new_weights = learnWeights(training_predictions,processed_data,weights)
        weights = new_weights[0]
        z = new_weights[1]
        classifiers.append(DecisionStump)
        classifier_weights.append(z)
    return [classifiers,classifier_weights]

def predictClass(processed_data,DecisionStump):
    feature_number = DecisionStump[0]
    feature_decision = DecisionStump[1]
    prediction = np.zeros(len(processed_data),dtype=np.float)
    for row in range(0,len(processed_data)):
        decision = feature_decision[int(processed_data[row][feature_number])]
        if decision==0:
            prediction[row] = -1
        else:
            prediction[row] = 1
    return prediction

def calculateAccuracy(predicted_class,processed_data):
    correct_count = 0.0
    total_columns = processed_data.shape[1]
    size = len(predicted_class)
    for row in range(0,size):
        if(predicted_class[row]==processed_data[row][total_columns-1]):
            correct_count+=1
    return correct_count*100/size

def testEnsembleClassifier(ensembleClassifier,test_file_name):
    processed_data = preprocessData(test_file_name)
    classifiers = ensembleClassifier[0]
    classifier_weights = ensembleClassifier[1]
    predicted_class = np.zeros(len(processed_data),dtype=np.float)
    for i in range(0,len(classifiers)):
        predicted_class += (predictClass(processed_data,classifiers[i])*classifier_weights[i])
    for row in range(0,len(processed_data)):
        if(predicted_class[row]<0):
            predicted_class[row] = -1
        else:
            predicted_class[row] = 1
    print "accuracy :",calculateAccuracy(predicted_class,processed_data)

if __name__ == "__main__":
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    ensemble_number = sys.argv[3]
    ensembleClassifier = trainEnsembleClassifier(train_file_name,int(ensemble_number))
    testEnsembleClassifier(ensembleClassifier,test_file_name)
