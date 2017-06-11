import sys
import numpy as np
import math

game_metadata = [3,3,3,3,2,2,2,2,2]

def preprocessData(file):
    data = np.genfromtxt(file,delimiter = ',',dtype=np.float)
    return data

def calculateEntropy(class_column):
    total_classes = game_metadata[len(game_metadata)-1]
    class_distribution = np.zeros(total_classes,dtype=np.float)
    for row in range(0,len(class_column)):
        if(class_column[row]==1):
            class_distribution[1]+=1
        else:
            class_distribution[0]+=1
    class_distribution = class_distribution / class_distribution.sum()
    entropy = 0
    for value in class_distribution:
        entropy = entropy - (value*math.log(value))
    return [entropy, np.argmax(class_distribution),np.max(class_distribution)]

def estimateInformationGain(processed_data,column,total_columns):
    total_feature_values = game_metadata[column]
    feature_distribution = np.zeros(total_feature_values,dtype=np.float)
    feature_decision = np.zeros(total_feature_values,dtype=np.float)
    feature_decision_prob = np.zeros(total_feature_values,dtype=np.float)
    split_classes = {}

    for feature in range(0,total_feature_values):
        split_classes[feature]=[]

    for row in range(0,len(processed_data)):
        feature_value = int(processed_data[row][column])
        feature_distribution[feature_value]+=1
        split_classes[feature_value].append(processed_data[row][total_columns-1])
    feature_distribution = feature_distribution / feature_distribution.sum()

    information_before_split = calculateEntropy(processed_data[:,total_columns-1])[0]
    information = 0
    for feature in range(0,total_feature_values):
        entropy_result = calculateEntropy(split_classes[feature])
        entropy = entropy_result[0]
        feature_decision[feature] = entropy_result[1]
        feature_decision_prob[feature] = entropy_result[2]
        information = information + (feature_distribution[feature]*entropy)
    gain = information_before_split - information
    return [gain,feature_decision,feature_decision_prob]

def determineBestSplit(processed_data):
    total_columns = processed_data.shape[1]
    max_gain = -1
    best_feature = -1
    best_feature_decision = []
    best_feature_decision_prob = []
    for column in range(0,total_columns-1):
        gain_result = estimateInformationGain(processed_data,column,total_columns)
        gain = gain_result[0]
        if(gain>max_gain):
            max_gain = gain
            best_feature = column
            best_feature_decision = gain_result[1]
            best_feature_decision_prob = gain_result[2]
    return best_feature, best_feature_decision, best_feature_decision_prob

def trainDecisionStump(train_file_name):
    processed_data = preprocessData(train_file_name)
    DecisionStump = determineBestSplit(processed_data)
    return DecisionStump

def predictClass(processed_data,DecisionStump):
    feature_number = DecisionStump[0]
    feature_decision = DecisionStump[1]
    feature_decision_prob = DecisionStump[2]
    prediction = np.zeros(len(processed_data),dtype=np.float)
    confidence = 0.0
    for row in range(0,len(processed_data)):
        val = int(processed_data[row][feature_number])
        decision = feature_decision[val]
        confidence+=feature_decision_prob[val]
        if decision==0:
            prediction[row] = -1
        else:
            prediction[row] = 1
    print "average probability :",confidence/len(processed_data)
    return prediction

def calculateAccuracy(predicted_class,processed_data):
    correct_count = 0.0
    total_columns = processed_data.shape[1]
    size = len(predicted_class)
    for row in range(0,size):
        if(predicted_class[row]==processed_data[row][total_columns-1]):
            correct_count+=1
    return correct_count*100/size

def testDecisionStump(DecisionStump,test_file_name):
    processed_data = preprocessData(test_file_name)
    predicted_class = predictClass(processed_data,DecisionStump)
    print "accuracy :",calculateAccuracy(predicted_class,processed_data)

if __name__ == "__main__":
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    DecisionStump = trainDecisionStump(train_file_name)
    testDecisionStump(DecisionStump,test_file_name)
