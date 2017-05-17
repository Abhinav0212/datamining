import sys
import numpy as np
import math
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
# n - nominal, o - ordinal, i - interval, r - ratio

income_preprocess = ['r','n','r','o','n','n','n','n','n','r','r','r','n']
output_name_1 = 'result/Income_euclid_output.csv'
output_name_2 = 'result/Income_cosine_output.csv'

def splitData(data):
    reduced_data = np.delete(data, [0,4,15], 1)
    nominal_columns = []
    ratio_columns = []
    ordinal_columns = []
    for val in range(0,len(income_preprocess)):
        if income_preprocess[val]=='r':
            ratio_columns.append(val)
        elif income_preprocess[val]=='n':
            nominal_columns.append(val)
        elif income_preprocess[val]=='o':
            ordinal_columns.append(val)

    ratio_data = reduced_data[:,ratio_columns].astype(np.float)
    nominal_data = reduced_data[:,nominal_columns]
    ordinal_data = reduced_data[:,ordinal_columns].astype(np.int)
    return [ratio_data,nominal_data,ordinal_data]

def min_max_normalize(ratio_data_matrix,max_arr,min_arr):

    ran_arr = max_arr - min_arr

    for r in range(0,len(ratio_data_matrix)):
        ratio_data_matrix[r] = (ratio_data_matrix[r] - min_arr) / ran_arr
    return ratio_data_matrix

def preprocess_data(train_data,test_data):

    processed_train_data = splitData(train_data)
    processed_test_data = splitData(test_data)

    ratio_train_data = processed_train_data[0]
    ratio_train_data[:,2] = np.log(ratio_train_data[:,2]+1)
    ratio_train_data[:,3] = np.log(ratio_train_data[:,3]+1)
    max_arr = ratio_train_data.max(0)
    min_arr =  ratio_train_data.min(0)
    processed_train_data[0] = min_max_normalize(ratio_train_data,max_arr,min_arr)

    ratio_test_data = processed_test_data[0]
    ratio_test_data[:,2] = np.log(ratio_test_data[:,2]+1)
    ratio_test_data[:,3] = np.log(ratio_test_data[:,3]+1)
    processed_test_data[0] = min_max_normalize(ratio_test_data,max_arr,min_arr)

    return [processed_train_data,processed_test_data]

def get_classes(data):
    total_class_values = []
    col_id_class = len(data[0])-1
    for rows in range(0,len(data)):
        actual_class = data[rows][col_id_class]
        if(actual_class not in total_class_values):
            total_class_values.append(actual_class)
    return total_class_values

def nominal_similarity(row1,row2):
    penalty = 0
    nominal_dist = 0
    for c in range(0,len(row1)):
        if(row1[c]=='" ?"' or row1[c]=='" ?"'):
            penalty = penalty + 1
        elif(row1[c]==row2[c]):
            nominal_dist = nominal_dist + 1
    return (nominal_dist,penalty)

def get_ordinal_range(ordinal_data):
    max_ordinal_arr = ordinal_data.max(0)
    min_ordinal_arr =  ordinal_data.min(0)
    ran_ordinal_arr = max_ordinal_arr - min_ordinal_arr
    return ran_ordinal_arr

def ordinal_similarity(row1, row2, ran_arr):

    ordinal_dist = 0
    for c in range(0,len(row1)):
            ordinal_dist = ordinal_dist + (float(abs(row1[c]-row2[c]))/ran_arr[c])
    return (len(row1) - ordinal_dist)

def euclidian_similarity(row1, row2):
    difference = row1-row2
    dist = math.sqrt(np.dot(difference,difference.transpose()))
    return (1/(1+dist))

def cosine_similarity(row1, row2):
    product = np.dot(row1,row2.transpose())
    row1_dist = math.sqrt(np.dot(row1,row1.transpose()))
    row2_dist = math.sqrt(np.dot(row2,row2.transpose()))
    return(product/(row1_dist*row2_dist))

def write_to_file(predicted_class, data, distanceMeasure, k):
    if(distanceMeasure==1):
        f1 = open(output_name_1, 'w+')
    else:
        f1 = open(output_name_2, 'w+')
    f1.write("ID,Actual Class, Predicted Class, Posterior probability\n")

    col_id_class = len(data[0])-1
    for rows in range(0,len(data)):
        f1.write(str(rows)+","+data[rows][col_id_class]+","+predicted_class[rows][0]+","+str(float(predicted_class[rows][1])/k)+"\n")
    f1.close()

def assign_class(sorted_similarity_matrix,k):
    predicted_class = []
    for rows in range(0,len(sorted_similarity_matrix)):
        count_matrix = {}
        max_class = ""
        max_count = 0

        for ct in range(0,k):
            class_nearest= sorted_similarity_matrix[rows][ct][2]
            if(class_nearest not in count_matrix):
                count_matrix[class_nearest] = 0
            count_matrix[class_nearest] = count_matrix[class_nearest] + 1
            if(count_matrix[class_nearest]>max_count):
                max_class = class_nearest
                max_count = count_matrix[class_nearest]

        predicted_class.append([max_class,max_count])
    return predicted_class

def print_output(assigned_class,data,k):
    col_id_class = len(data[0])-1
    for rows in range(0,len(data)):
        print rows,"\t",data[rows][col_id_class],"\t",
        print assigned_class[rows][0],"\t",float(assigned_class[rows][1])/k

def calculate_classfication_error(assigned_class,data,distanceMeasure,k):
    correct = 0
    col_id_class = len(data[0])-1
    for rows in range(0,len(data)):
        if(assigned_class[rows][0]==data[rows][col_id_class]):
            correct = correct + 1
    error = (1-(float(correct)/len(data)))
    print "classification error :",error

def calculate_confusion_matrix(assigned_class,data,class_list):
    total_class_values = class_list
    col_id_class = len(data[0])-1
    for val in class_list:
        print val,
    print ""
    confusion_matrix =  np.zeros((len(total_class_values),len(total_class_values)), dtype=np.float)
    for rows in range(0,len(data)):
        actual_class = total_class_values.index(data[rows][col_id_class])
        predicted_class = total_class_values.index(assigned_class[rows][0])
        confusion_matrix[actual_class][predicted_class] += 1

    # print confusion_matrix
    total_examples = confusion_matrix.sum(axis=1)
    print confusion_matrix/total_examples[:,None]
    return confusion_matrix

def calculate_cost_measures(confusion_matrix,distanceMeasure,k):
    true_pos = confusion_matrix[0][0]
    false_neg = confusion_matrix[0][1]
    false_pos = confusion_matrix[1][0]
    true_neg = confusion_matrix[1][1]
    print "True Positive",true_pos
    print "False Negative",false_neg
    print "False Positive",false_pos
    print "True Negative",true_neg
    print "Accuracy",float(true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
    precision = float(true_pos)/(true_pos+false_pos)
    print "Precision",precision
    recall = float(true_pos)/(true_pos+false_neg)
    print "Recall",recall
    F_index = 2*(precision*recall)/(precision+recall)
    print "F score",F_index

def plot_roc_curve(predicted_class_matrix,test_data,class_list):
    confidence_score = []
    actual_class = []
    positive = class_list[0]
    col_id_class = len(test_data[0])-1
    for rows in range(0,len(test_data)):
        if(predicted_class_matrix[rows][0]==positive):
            confidence_score.append(predicted_class_matrix[rows][1])
        else:
            confidence_score.append(1-predicted_class_matrix[rows][1])
        if(test_data[rows][col_id_class]==positive):
            actual_class.append(1)
        else:
            actual_class.append(0)
    fpr, tpr, thresholds = roc_curve(actual_class,confidence_score)
    roc_auc = auc(fpr,tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def trainOffTheShelfClassifier(train_data,normalized_train_data,k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    col_id_class = len(train_data[0])-1
    class_labels = train_data[:,col_id_class]
    neigh.fit(normalized_train_data, class_labels)
    return neigh

def testOffTheShelfClassifier(neigh,test_data,normalized_test_data,class_list):
    col_id_class = len(test_data[0])-1
    class_labels = test_data[:,col_id_class]
    predicted_class = neigh.predict(normalized_test_data)
    # write_to_file(predicted_class, test_data, 1, k)
    print "classification error :",(1-neigh.score(normalized_test_data,class_labels))
    # calculate_confusion_matrix(predicted_class,test_data,class_list)

def useOffTheShelfClassifier(train_data,normalized_train_data,test_data,normalized_test_data,k,class_list):
    neigh = trainOffTheShelfClassifier(train_data,normalized_train_data,k)
    testOffTheShelfClassifier(neigh,test_data,normalized_test_data,class_list)

def evaluate_results(sorted_similarity,test_data,distanceMeasure,k,class_list):
    predicted_class = assign_class(sorted_similarity,k)
    write_to_file(predicted_class, test_data, distanceMeasure, k)
    calculate_classfication_error(predicted_class,test_data,distanceMeasure,k)
    confusion_matrix = calculate_confusion_matrix(predicted_class,test_data,class_list)
    # plot_roc_curve(predicted_class,test_data,class_list)
    calculate_cost_measures(confusion_matrix,distanceMeasure,k)
    # print_output(predicted_class,data,k)

def estimateSimilarity(train_file_name,test_file_name, distanceMeasure, k):
    train_data = np.genfromtxt(train_file_name,delimiter = ',',dtype=object,skip_header=1)
    class_list = get_classes(train_data)
    test_data = np.genfromtxt(test_file_name,delimiter = ',',dtype=object,skip_header=1)
    processed_data = preprocess_data(train_data,test_data)
    processed_train_data = processed_data[0]
    processed_test_data = processed_data[1]
    col_id_class = len(train_data[0])-1

    ratio_train_data = processed_train_data[0]
    nominal_train_data = processed_train_data[1]
    ordinal_train_data =  processed_train_data[2]
    ratio_test_data = processed_test_data[0]
    nominal_test_data = processed_test_data[1]
    ordinal_test_data =  processed_test_data[2]

    ratio_weight = ratio_train_data.shape[1]
    nominal_weight = nominal_train_data.shape[1]
    ordinal_weight = ordinal_train_data.shape[1]
    total_weight = ratio_weight + nominal_weight + ordinal_weight

    ordinal_range = get_ordinal_range(ordinal_train_data)

    # For using off-the-shelf classificer TODO need to convert the non numeric features to numbers
    # normalized_train_data = np.concatenate((ratio_train_data,nominal_train_data,ordinal_train_data),axis=1)
    # normalized_test_data = np.concatenate((ratio_test_data,nominal_test_data,ordinal_test_data),axis=1)
    # useOffTheShelfClassifier(train_data,normalized_train_data,test_data,normalized_test_data,k,class_list)

    similarity_matrix = np.zeros((len(ratio_test_data),len(ratio_train_data)), dtype=[('dist',np.float),('id',np.int),('class',object)])


    for r in range(0,len(ratio_test_data)):
        for r1 in range(0,len(ratio_train_data)):
            ratio_row1 = ratio_test_data[r]
            ratio_row2 = ratio_train_data[r1]
            nominal_row1 = nominal_test_data[r]
            nominal_row2 = nominal_train_data[r1]
            ordinal_row1 = ordinal_test_data[r]
            ordinal_row2 = ordinal_train_data[r1]

            nominal_result = nominal_similarity(nominal_row1,nominal_row2)
            penalty = nominal_result[1]
            nominal_similar = nominal_result[0]

            ordinal_similar = ordinal_similarity(ordinal_row1,ordinal_row2,ordinal_range)

            if(distanceMeasure==1):
                ratio_similar = ratio_weight * euclidian_similarity(ratio_row1, ratio_row2)
            else:
                ratio_similar = ratio_weight * cosine_similarity(ratio_row1, ratio_row2)

            final_similarity = (ratio_similar + nominal_similar + ordinal_similar)/(total_weight-penalty)
            similarity_matrix[r][r1] = (final_similarity,r1,train_data[r1][col_id_class])

    sorted_similarity = np.sort(similarity_matrix, order='dist')[:,::-1]
    evaluate_results(sorted_similarity,test_data,distanceMeasure,k,class_list)



if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print "Usage: python knnIris.py <training_file> <testing_file> <distance_measure> [<value_of_k>] "
    if (len(sys.argv) > 4):
        k = sys.argv[4]
    else:
        k=5
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    distanceMeasure = sys.argv[3]
    estimateSimilarity(train_file_name,test_file_name,int(distanceMeasure), int(k))
