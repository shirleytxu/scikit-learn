import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('breastcancerdataset.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# set target
y = df.loc[:,["diagnosis"]]
del df["diagnosis"]

# data 70-30 split
trainX, testX, trainY, testY = train_test_split(df, y, test_size=0.3)
scaler = MinMaxScaler()
# transform data
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

# training
startTraining = time.time()
svc.fit(trainX, trainY.values.ravel())
stopTraining = time.time()
svcTrainingTime = stopTraining-startTraining
print("SVC Training Time:", svcTrainingTime)

# testing
startTest = time.time()
svcPreds = svc.score(testX, testY)
stopTest = time.time()
svcTestingTime = stopTest-startTest
print("SVC Testing Time: ", svcTestingTime)
print("SVC Accuracy:     ", svcPreds)
print("")

logreg = LogisticRegression()

# training
startTraining = time.time()
logreg.fit(trainX, trainY.values.ravel())
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