import math
import time
from pathlib import Path
import pandas as pd

TRAIN_RECORDS = 60000
TEST_RECORDS = 10000
NUM_CLASSES = 10
FEATURES = 784
FEATURE_VALUES = 256

class NaiveBayes:
    def __init__(self):
        self.train_datas, self.train_labels, self.test_datas, self.test_labels = [], [], [], []
        self.prior, self.likelihoods, self.confusionMatrix = [], [], []
        self.start, self.end, self.time_use = 0, 0, 0

        self.init()
        self.start = time.time()
        self.train()
        self.predict_logarit()
        self.CM()
        self.end = time.time()
        self.time_use = self.end - self.start
        print(self.time_use)

    def init(self):
        current_directory = Path(__file__).resolve().parent
        train_file_path = current_directory / "mnist_train.csv"
        test_file_path = current_directory / "mnist_test.csv"

        self.train_data = pd.read_csv(train_file_path)
        self.test_data = pd.read_csv(test_file_path)

        self.train_labels = self.train_data.iloc[:, 0].astype(int)
        self.train_datas = self.train_data.iloc[:, 1:].values.astype(int)

        self.test_labels = self.test_data.iloc[:, 0].astype(int)
        self.test_datas = self.test_data.iloc[:, 1:].values.astype(int)

        self.confusionMatrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]

    def train(self):
        self.prior = [0] * NUM_CLASSES

        for label in self.train_labels:
            self.prior[label] += 1

        for i in range(NUM_CLASSES):
            self.prior[i] /= TRAIN_RECORDS

        self.likelihoods = [[[0] * FEATURE_VALUES for _ in range(FEATURES)] for _ in range(NUM_CLASSES)]
        class_counts = [0] * NUM_CLASSES

        for i in range(TRAIN_RECORDS):
            label = self.train_labels[i]
            for j in range(FEATURES):
                index = self.train_datas[i][j]
                self.likelihoods[label][j][index] += 1
            class_counts[label] += 1

        for i in range(NUM_CLASSES):
            for j in range(FEATURES):
                for k in range(FEATURE_VALUES):
                    if self.likelihoods[i][j][k] == 0:
                        self.likelihoods[i][j][k] = 1e-7
                    else:
                        self.likelihoods[i][j][k] /= class_counts[i]

    def predict_logarit(self):
        correct = 0
        sum_errors = 0

        for i in range(TEST_RECORDS):
            max_prob = -float('inf')
            max_class = -1

            for j in range(NUM_CLASSES):
                prob = math.log(self.prior[j])
                for k in range(FEATURES):
                    index = self.test_datas[i][k]
                    prob += math.log(self.likelihoods[j][k][index])

                if prob > max_prob:
                    max_prob = prob
                    max_class = j

            self.confusionMatrix[self.test_labels[i]][max_class] += 1

            if max_class == self.test_labels[i]:
                correct += 1
            else:
                sum_errors += 1

        print()
        print("so ket qua sai :", sum_errors)
        print("so ket qua dung:", correct)
        accuracy = correct / TEST_RECORDS
        print("Accuracy:", accuracy)

    def CM(self):
        print("Ground Truth = GT")
        print("Predicted Label = PL")
        self.printLine()
        print(f"{TEST_RECORDS:3d}|", end="")
        for i in range(NUM_CLASSES):
            print(f" PL = {i:2d}  |", end="")
        print()
        self.printLine()

        for i in range(NUM_CLASSES):
            print(f" GT = {i:2d}     |", end="")
            for j in range(NUM_CLASSES):
                print(f"   {self.confusionMatrix[i][j]:4d}  |", end="")
            print()
            self.printLine()

    def printLine(self):
        print("------------|", end="")
        for _ in range(NUM_CLASSES):
            print("---------|", end="")
        print()

if __name__ == "__main__":
    NB = NaiveBayes()
