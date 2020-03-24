import numpy as np


class MyLinearRegression:

    def __init__(self):
        self.learning_rate = 0.000001

    def fit(self, records, target):
        self.weights = np.random.random(len(records[0]))

        for epoch in range(500):
            print(epoch)
            for r, record in enumerate(records):
                p = self.predict_one(record)
                y = target[r]
                error = y - p

                for i, x_i in enumerate(record):
                    self.weights[i] += self.learning_rate * error * x_i

    def predict(self, features):
        predictions = []
        for record in features:
            prediction = self.predict_one(record)
            predictions.append(prediction)

        return predictions

    def predict_one(self, record):
        sum = 0
        for i, x_i in enumerate(record): # i indice x_i valore di record
            sum += x_i * self.weights[i]

        return sum


class MyLogisticRegression(MyLinearRegression):

    def predict_one(self, record):
        line = super().predict_one(record)
        sigmoid = 1 / ( 1 + np.e**-line )

        if sigmoid < 0.5:
            return 0.0
        else:
            return 1.0



