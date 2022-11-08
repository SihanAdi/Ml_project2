import math

import numpy as np


class SGD:
    def __init__(self, alpha, times):
        self.Lose = None
        self.w = None
        self.alpha = alpha
        self.times = times

    # w=[b,w]
    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        # print(y)
        # print(x.shape[0])
        self.w = np.zeros(2)
        self.Lose = []
        count = 0
        for i in range(self.times):
            count+=1
            y_pre = np.dot(x, self.w[1]) + self.w[0]
            error = y_pre - y
            self.Lose.append(np.sum(error ** 2) / x.shape[0])
            self.w[0] += self.alpha * (np.sum(error * 2) / x.size)
            self.w[1] += self.alpha * (np.dot(x.T, error) * 2 / x.size)
            if ((np.sum(error * 2) / x.size) == 0) and ((np.dot(x.T, error) * 2 / x.size) == 0):
                print(count)
                break


    def predict(self, x):
        result = np.dot(x, self.w[1]) + self.w[0]

        return result




