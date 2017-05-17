import sys
import numpy as np
import math
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

# n - nominal, o - ordinal, i - interval, r - ratio
iris_preprocess = ['r','r','r','r','n']
output_name_1 = 'result/Iris_euclid_output.csv'
output_name_2 = 'result/Iris_cosine_output.csv'

def min_max_normalize(ratio_data_matrix,max_arr,min_arr):
    ran_arr = max_arr - min_arr
    for r in range(0,len(ratio_data_matrix)):
        ratio_data_matrix[r] = (ratio_data_matrix[r] - min_arr) / ran_arr
    return ratio_data_matrix

def preprocess_data(train_data,test_data):

    train_data_sans_class = train_data[:,:4].astype(np.float)
    test_data_sans_class = test_data[:,:4].astype(np.float)
    max_arr = train_data_sans_class.max(0)
    min_arr =  train_data_sans_class.min(0)
    normalized_train_data = min_max_normalize(train_data_sans_class,max_arr,min_arr)
    normalized_test_data = min_max_normalize(test_data_sans_class,max_arr,min_arr)

    return (normalized_train_data,normalized_test_data)

def get_classes(data):
    total_class_values = []
    col_id_class = len(data[0])-1
    for rows in range(0,len(data)):
        actual_class = data[rows][col_id_class]
        if(actual_class not in total_class_values):
            total_class_values.append(actual_class)
    return total_class_values

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
        # This is for writing the output of the off-the-shelf classifier
        # f1.write(str(rows)+","+data[rows][col_id_class]+","+predicted_class[rows]+"\n")
    f1.close()

def assign_class(sorted_similarity_matrix,k):
    predicted_class = []
    for rows in range(0,len(sorted_similarity_matrix)):
        count_matrix = Counter()
        max_class = ""
        max_count = 0

        for ct in range(0,k):
            count_matrix[sorted_similarity_matrix[rows][ct][2]]+=1

        predicted_class.append(count_matrix.most_common(1)[0])
    return predicted_class

def print_output(assigned_class,data,k):
    col_id_class = len(data[0])-1
    for rows in range(0,len(data)):
        print rows,"\t",data[rows][col_id_class],"\t\t",
        print assigned_class[rows][0],"\t\t",float(assigned_class[rows][1])/k

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
        # This is for the off-the-shelf classifier
        # predicted_class = total_class_values.index(assigned_class[rows])
        confusion_matrix[actual_class][predicted_class] += 1

    # print confusion_matrix
    total_examples = confusion_matrix.sum(axis=1)
    print confusion_matrix/total_examples[:,None]
    return confusion_matrix

def evaluate_results(sorted_similarity_matrix,test_data,distanceMeasure,k,class_list):
    predicted_class_matrix = assign_class(sorted_similarity_matrix,k)
    write_to_file(predicted_class_matrix, test_data, distanceMeasure, k)
    calculate_classfication_error(predicted_class_matrix,test_data,distanceMeasure,k)
    confusion_matrix = calculate_confusion_matrix(predicted_class_matrix,test_data,class_list)
    # print_output(predicted_class,data,k)

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
    error = (1-neigh.score(normalized_test_data,class_labels))
    print "classification error :",error
    calculate_confusion_matrix(predicted_class,test_data,class_list)

def useOffTheShelfClassifier(train_data,normalized_train_data,test_data,normalized_test_data,k,class_list):
    neigh = trainOffTheShelfClassifier(train_data,normalized_train_data,k)
    testOffTheShelfClassifier(neigh,test_data,normalized_test_data,class_list)

def estimateSimilarity(train_file_name,test_file_name, distanceMeasure, k):
    train_data = np.genfromtxt(train_file_name,delimiter = ',',dtype=object,skip_header=1)
    class_list = get_classes(train_data)
    test_data = np.genfromtxt(test_file_name,delimiter = ',',dtype=object,skip_header=1)
    normalized_data = preprocess_data(train_data,test_data)
    normalized_train_data = normalized_data[0]
    normalized_test_data = normalized_data[1]

    col_id_class = len(train_data[0])-1

    # To call the off-the-shelf classifier
    # useOffTheShelfClassifier(train_data,normalized_train_data,test_data,normalized_test_data,k,class_list)
    # To call the off-the-shelf classifier without preprocessing the data
    # useOffTheShelfClassifier(train_data,train_data[:,:4].astype(np.float),test_data,test_data[:,:4].astype(np.float),k,class_list)

    similarity_matrix = np.zeros((len(normalized_test_data),len(normalized_train_data)), dtype=[('dist',np.float),('id',np.int),('class',object)])

    for r in range(0,len(normalized_test_data)):
        for r1 in range(0,len(normalized_train_data)):
            row1 = normalized_test_data[r]
            row2 = normalized_train_data[r1]

            if(distanceMeasure==1):
                similarity_val = euclidian_similarity(row1, row2)
            else:
                similarity_val = cosine_similarity(row1, row2)
            similarity_matrix[r][r1] = (similarity_val,r1,train_data[r1][col_id_class])

    sorted_similarity_matrix = np.sort(similarity_matrix, order='dist')[:,::-1]
    evaluate_results(sorted_similarity_matrix,test_data,distanceMeasure,k,class_list)

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
