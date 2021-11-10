import time
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

    # testing
    startTest = time.time()
    preds = model.predict(testX)
    stopTest = time.time()
    testingTime = stopTest - startTest

    accuracy = model.score(testX, testY)

    confusionMatrix = confusion_matrix(testY, preds, labels=["B", "M"])
    display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,
                                     display_labels=model.classes_)

    falsePositive = confusionMatrix[0, 1] / len(testY)
    falseNegative = confusionMatrix[1, 0] / len(testY)

    display.plot()
    plt.title(modelName + " Confusion Matrix (Close to Continue)")
    plt.show(block=True)

    networkResults = [modelName, trainingTime, testingTime, accuracy, falsePositive, falseNegative]
    return networkResults, preds

def threshNet(model1preds, model2preds, model3preds, threshold, testY):
    threshNetName = "ThreshNet" + str(threshold)

    threshNetPreds = []
    for i in range(len(model1preds)):
        malignantCount = 0

        if model1preds[i] == "M":
            malignantCount += 1

        if model2preds[i] == "M":
            malignantCount += 1

        if model3preds[i] == "M":
            malignantCount += 1

        # if there are more malignant decisions from all the models than
        # threshold, counts as malignant
        if malignantCount >= threshold:
            threshNetPreds.append("M")
        else:
            threshNetPreds.append("B")

    # calculates accuracy
    numCorrectPreds = 0
    for i in range(len(threshNetPreds)):
        if threshNetPreds[i] == testY[i]:
            numCorrectPreds += 1

    accuracy = numCorrectPreds / len(threshNetPreds)

    confusionMatrix = confusion_matrix(testY, threshNetPreds, labels=["B", "M"])
    display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,
                                     display_labels=["B", "M"])
    display.plot()
    plt.title(threshNetName + " Confusion Matrix (Close to Continue)")
    plt.show(block=True)

    falsePositive = confusionMatrix[0, 1] / len(testY)
    falseNegative = confusionMatrix[1, 0] / len(testY)

    return threshNetName, "N/A", "N/A", accuracy, falsePositive, falseNegative

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

# make testY and trainY into 1D array as expected by predict and train functions
testY = testY.values.ravel()
trainY = trainY.values.ravel()

# scales data to 0-1 range
# scaler has shown to improve network accuracy by over 30%
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
plt.title("Radius Mean Histogram (Close to Continue)")
plt.show(block=True)

xAxis = df.loc[:,['texture_mean']]
plt.hist(xAxis)
plt.ylabel("Probability")
plt.xlabel("Texture Mean")
plt.title("Texture Mean Histogram (Close to Continue)")
plt.show(block=True)

xAxis = df.loc[:,['perimeter_mean']]
plt.hist(xAxis)
plt.ylabel("Probability")
plt.xlabel("Perimeter Mean")
plt.title("Perimeter Mean Histogram (Close to Continue)")
plt.show(block=True)

# support vector classification
svc = SVC(verbose=0, max_iter=10000)
svcResults, svcPreds = evaluateModel(svc, "SVC", trainX, trainY, testX, testY)

# logistic regression
logreg = LogisticRegression()
logregResults, logregPreds = evaluateModel(logreg, "Logistic Regression", trainX, trainY,
                              testX, testY)

# kneighborsclassifier
kneighclassifier = KNeighborsClassifier()
kneighResults, kneighPreds = evaluateModel(kneighclassifier, "K Neighbors Classifier",
                              trainX, trainY, testX, testY)

threshNet1Results = threshNet(svcPreds, logregPreds, kneighPreds, 1, testY)
threshNet2Results = threshNet(svcPreds, logregPreds, kneighPreds, 2, testY)
threshNet3Results = threshNet(svcPreds, logregPreds, kneighPreds, 3, testY)

# compile all neural network statistics
networkResults = [svcResults, logregResults, kneighResults,
                  threshNet1Results, threshNet2Results, threshNet3Results]

# create statistic in dataframe to print with table-like appearance
networkTable = pd.DataFrame(networkResults,
                            columns=["Network", "Training Time",
                                     "Testing Time", "Accuracy",
                                     "False Positive", "False Negative"])

# formatting pandas dataframe
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

print("")
print("Result Comparison: ")
print(networkTable)
