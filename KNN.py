import csv
import math
import time
from pathlib import Path
import pandas as pd

TRAINING_RECORDS = 60000
TESTING_RECORDS = 1
FEATURES = 784  # 28 * 28
K_MAX = 9
NUM_CLASSES = 10

class Data:
    def __init__(self):
        self.ID = 0
        self.att = []
        self.label = ""
        self.distance = 0.0
        self.rank = 0
        self.predictedLabels = [""] * (K_MAX + 1)

class Labels:
    def __init__(self):
        self.labels = [""] * 10
        self.count = [0.0] * (K_MAX + 1)

class KNNClassifier:
    def __init__(self):
        self.TrainingData = []
        self.TestData = []
        self.LabelData = Labels()
        self.label_predicted = ""
        self.features = 0
        self.records = 0
        self.testings = 0
        self.labels = 0
        self.start = 0
        self.end = 0
        self.usingTime = 0.0
        self.init()
        self.start = time.time()
        self.KNN_classifier()
        self.end = time.time()
        self.usingTime = (self.end - self.start)
        print("\n", self.usingTime)

    def init(self):
        labels = 0
        current_directory = Path(__file__).resolve().parent
        label_file_path = current_directory / "handwritten_labelDatas.txt"
        train_file_path = current_directory / "mnist_train.csv"
        test_file_path = current_directory / "mnist_test.csv"
        with open(label_file_path) as g:
            labels = int(g.readline())
            self.LabelData.labels = g.readline().split()

        with open(train_file_path) as f:
            self.records = TRAINING_RECORDS
            self.features = FEATURES
            f.readline()  # Skip the first line
            index = 1
            for line in f:
                fields = line.strip().split(',')
                input_data = Data()
                input_data.ID = index
                input_data.label = fields[0]
                input_data.att = [int(x) for x in fields[1:]]
                index += 1
                self.TrainingData.append(input_data)

        with open(test_file_path) as test:
            test.readline()  # Skip the first line
            self.testings = TESTING_RECORDS
            index = 1
            for line in test:
                if index > self.testings:
                    return
                fields = line.strip().split(',')
                input_data = Data()
                input_data.ID = index
                input_data.label = fields[0]
                input_data.att = [int(x) for x in fields[1:]]
                index += 1
                self.TestData.append(input_data)

    def viewData(self, data, records):
        print("so luong mau:", records)
        for i in range(min(records, 1)):
            print(f"{data[i].ID:4d}", end="    ")
            for j in range(len(data[i].att)):
                print(f"{data[i].att[j]:5.1f}", end="   ")
            print()
            print(len(data[i].att))
            print("    ", data[i].label)
            print()

    def distanceComputing(self, testData):
        for i in range(self.records):
            sum_squared_diff = sum((a - b) ** 2 for a, b in zip(testData.att, self.TrainingData[i].att))
            distance = math.sqrt(sum_squared_diff)
            self.TrainingData[i].distance = distance

    def sorting(self):
        for i in range(self.records - 1):
            minIndex = i
            for j in range(i + 1, self.records):
                if self.TrainingData[j].distance < self.TrainingData[minIndex].distance:
                    minIndex = j
            self.TrainingData[i], self.TrainingData[minIndex] = self.TrainingData[minIndex], self.TrainingData[i]

    def KNN_classifier(self):
        max_count = 0
        max_index = 0
        #for i in range(self.testings):
        for i in range (0, 1):
            self.distanceComputing(self.TestData[i])
            self.sorting()
            for j in range(self.labels):
                self.LabelData.count[j] = 0.0
            for z in range(1, K_MAX + 1):
                for l in range(self.labels):
                    if self.TrainingData[z].label == self.LabelData.labels[l]:
                        self.LabelData.count[l] += 1
                        break
                if z % 2 != 0:
                    self.setLabel(i, z)
        print()
        # self.viewResult()
        # self.confusionMatrix()

    def setLabel(self, temp, k):
        index = 0
        max_value = self.LabelData.count[0]
        for l in range(1, self.labels):
            if self.LabelData.count[l] > max_value:
                index = l
        self.TestData[temp].predictedLabels[k] = self.LabelData.labels[index]

    def viewResult(self):
        print(" ID | k = 1 | k = 3 | k = 5 | k = 7 | k = 9 | true Label|")
        for j in range(self.testings):
            print(f"{j:3d} |", end="")
            for i in range(1, K_MAX + 1, 2):
                print(f"{self.TestData[j].predictedLabels[i]:5s}  |", end="")
            print(f" {self.TestData[j].label:5s}     |")
            print("--------------------------------------------------------|")

    def confusionMatrix(self):
        confusionMatrix = [[0] * self.labels for _ in range(self.labels)]
        GTIndex = 0
        PLIndex = 0
        corrects = 0
        print(" PL = Predicted Label ")
        print(" GL = Ground Truth ")
        for k in range(1, K_MAX + 1, 2):
            print(f"with k = {k}")
            for i in range(self.labels):
                for j in range(self.labels):
                    confusionMatrix[i][j] = 0
            corrects = 0
            for i in range(self.testings):
                for j in range(self.labels):
                    if self.TestData[i].label == self.LabelData.labels[j]:
                        GTIndex = j
                    if self.TestData[i].predictedLabels[k] == self.LabelData.labels[j]:
                        PLIndex = j
                confusionMatrix[GTIndex][PLIndex] += 1
            self.printLine()
            print(f"total = {self.testings:3d}|", end="")
            for i in range(self.labels):
                print(f" PL = {self.LabelData.labels[i]:5s}  |", end="")
            print()
            self.printLine()
            for i in range(self.labels):
                print(f" GT = {self.LabelData.labels[i]:5s}  |", end="")
                for j in range(self.labels):
                    print(f"    {confusionMatrix[i][j]:2d}   |", end="")
                print()
                self.printLine()
            for i in range(self.labels):
                corrects += confusionMatrix[i][i]
            print(f"accuracy = {corrects / self.testings:.2f} \n \n")

    def printLine(self):
        print("------------|", end="")
        for _ in range(NUM_CLASSES):
            print("---------|", end="")
        print()

if __name__ == "__main__":
    NB = KNNClassifier()