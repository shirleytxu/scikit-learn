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

"""
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
"""

# svc network
svc = SVC(verbose=0, max_iter=10000)

# training
startTraining = time.time()
svc.fit(trainX, trainY)
stopTraining = time.time()
svcTrainingTime = stopTraining-startTraining
print("SVC Training Time:", svcTrainingTime)

startTest = time.time()
svcPreds = svc.predict(testX)
stopTest = time.time()
svcTestingTime = stopTest-startTest

svcConfusionMatrix = confusion_matrix(testY, svcPreds, labels=["B", "M"])
display = ConfusionMatrixDisplay(confusion_matrix=svcConfusionMatrix, display_labels=svc.classes_)
display.plot()
plt.title("SVC Neural Network Confusion Matrix")
plt.show()

print("SVC Testing Time: ", svcTestingTime)
print("SVC Accuracy:     ", svcPreds)
print("")

# logistic regression network
logreg = LogisticRegression()

# training
startTraining = time.time()
logreg.fit(trainX, trainY)
stopTraining = time.time()
logregTrainingTime = stopTraining-startTraining
print("LogReg Training Time:", logregTrainingTime)

# testing
startTest = time.time()
logregPreds = logreg.score(testX, testY)
stopTest = time.time()
logregTestingTime = stopTest-startTest
print("LogReg Testing Time: ", logregTestingTime)
print("LogReg Accuracy:     ", logregPreds)
print("")

# kneighborsclassifier
kneighclassifier = KNeighborsClassifier()

# training
startTraining = time.time()
kneighclassifier.fit(trainX, trainY)
stopTraining = time.time()
kneighTrainingTime = stopTraining-startTraining
print("KNeighborsClassifier Training Time:", kneighTrainingTime)

# testing
startTest = time.time()
kneighPreds = kneighclassifier.score(testX, testY)
stopTest = time.time()
kneighTestingTime = stopTest-startTest
print("KNeighborsClassifier Testing Time: ", kneighTestingTime)
print("KNeighborsClassifier Accuracy:     ", kneighPreds)

