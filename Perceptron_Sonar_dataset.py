#importing packages
from csv import reader
from random import randrange

#create a function to laod csv data
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

#create a function to convert string column to float
def str_column_to_float(dataset):
    for row in dataset:
        for column in range(len(row)-1):
            row[column] = float(row[column].strip())

#create a function to convert the last column to integer. M=1 and R=0
def str_column_to_int(dataset):
    for row in dataset:
        row[-1] = 1 if row[-1] == 'M' else 0

#create a function that slpits the dataset into train and test.Take 166 rows for training and 44 rows for testing. Selectoin is random.return train and test dataset
def train_test_split(dataset):
    train = list()
    train_size = 166
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

#define prediction function
def predict(row, weights):
    activation = weights[0]
    #calculating output
    for i in range(len(row)-1):
        activation +=weights[i+1]*row[i]
    #converting output to binary
    return 1.0 if activation >=0.0 else 0.0

#define the function to estimate the weights
def perceptron(train, l_rate, n_epoch):
    #initialize the weights
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i+1] = weights[i+1] + l_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights

#define a function to test the model on the test dataset
def perceptron_algorithm(train, test, l_rate, n_epoch):
    predictions = list()
    weights = perceptron(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return(predictions)

#write a function to calculate the accuracy of the model
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

#create a main function which loads data and calls all the functions
def main():
    #load and prepare data
    filename = 'sonar.csv'
    dataset = load_csv(filename)
    str_column_to_float(dataset)
    str_column_to_int(dataset)
    #split data
    train, test = train_test_split(dataset)
    #define parameters
    l_rate = 0.01
    n_epoch = 500
    #train model
    predictions = perceptron_algorithm(train, test, l_rate, n_epoch)
    actual = [row[-1] for row in test]
    accuracy = accuracy_metric(actual, predictions)
    print('Accuracy: %.3f' % accuracy)

#Observations:
#The perceptron is trained to adjust weights to classify the dataset. The weights are adjusted number of times over the dataset.
#The weights are adjusted to minimize the error. The error is calculated as the difference between the predicted output and the target output.
#The weights are adjusted using the learning rate. The learning rate is the rate at which the weights are adjusted.
#The weights are adjusted in the direction of the error. The weights are adjusted using the formula:
#weights[i+1] = weights[i+1] + row[i]*l_rate*error
#The bias is adjusted using the formula:
#weights[0]=weights[0] + l_rate*error

#Limitations:
#The perceptron is a linear classifier. It can only classify linearly separable datasets.
#The perceptron can only classify binary datasets.
#The perceptron can only classify datasets with a single decision boundary.

#Conclusion
#The perceptron can minimize the error to a certain value. Minimum error can be obtained by decreasing the learning rate and increasing the number of epochs.
#After obtaining the minimum error the results are saturated i.e. there is no effect of decreasing learning rate and increasing epoches.
  
    
