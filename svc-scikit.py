import time
import pandas as pd
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def evaluateModel(model, modelName, trainX, trainY, testX, testY):
    """
    :param model: neural network model object
    :param modelName: neural network name (string)
    :param trainX: training data input
    :param trainY: training data answers
    :param testX: testing data input
    :param testY: testing data answers
    :return: trainingTime, testingTime, confusionMatrix, accuracy
    """
    # training
    startTraining = time.time()
    model.fit(trainX, trainY)
    stopTraining = time.time()
    trainingTime = stopTraining - startTraining
    print(modelName, " Training Time:", trainingTime)

    startTest = time.time()
    preds = model.predict(testX)
    stopTest = time.time()
    testingTime = stopTest - startTest

    accuracy = model.score(testX, testY)

    print(modelName, " Testing Time: ", testingTime)
    print(modelName, " Accuracy: ", accuracy)
    print("")

    confusionMatrix = confusion_matrix(testY, preds, labels=["B", "M"])
    display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,
                                     display_labels=model.classes_)
    display.plot()
    plt.title(modelName + " Neural Network Confusion Matrix")
    plt.show()

    return trainingTime, testingTime, confusionMatrix, accuracy

# reads in dataset
df = pd.read_csv('breastcancerdataset.csv')

# dataframe includes unnamed columns
# accesses all columns that are unnamed and removes them
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# sets target to column titled "diagnosis"
y = df.loc[:,["diagnosis"]]

# deletes diagnosis column from dataframe
del df["diagnosis"]

# data split 70-30
trainX, testX, trainY, testY = train_test_split(df, y, test_size=0.3)

# makes testY and trainY into 1D array as expected by predict and train functions
testY = testY.values.ravel()
trainY = trainY.values.ravel()

# scales data to 0-1 range
# including a scaler has shown to improve network accuracy by over 30%
scaler = MinMaxScaler()

# transforms data
# target y does not need to be scaled because it is binary ("b" or "m")
trainX = scaler.fit_transform(trainX)
testX = scaler.fit_transform(testX)

# visualize some parameters of data
xAxis = df.loc[:,['radius_mean']]
plt.hist(xAxis)
plt.ylabel("Probability")
plt.xlabel("Radius Mean")
plt.title("Radius Mean Histogram")
plt.show(block=False)
plt.pause(2)
plt.close()

xAxis = df.loc[:,['texture_mean']]
plt.hist(xAxis)
plt.ylabel("Probability")
plt.xlabel("Texture Mean")
plt.title("Texture Mean Histogram")
plt.show(block=False)
plt.pause(2)
plt.close()

xAxis = df.loc[:,['perimeter_mean']]
plt.hist(xAxis)
plt.ylabel("Probability")
plt.xlabel("Perimeter Mean")
plt.title("Perimeter Mean Histogram")
plt.show(block=False)
plt.pause(2)
plt.close()

# svc network
svc = SVC(verbose=0, max_iter=10000)
svcResults = evaluateModel(svc, "SVC", trainX, trainY, testX, testY)

# logistic regression network
logreg = LogisticRegression()
logregResults = evaluateModel(logreg, "Logistic Regression", trainX, trainY,
                              testX, testY)

# kneighborsclassifier
kneighclassifier = KNeighborsClassifier()
kneighResults = evaluateModel(kneighclassifier, "K Neighbors Classifier",
                              trainX, trainY, testX, testY)