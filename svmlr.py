import math
import numpy as np
from sklearn.utils import shuffle


class svmLr:
    def __init__(self, alpha, C, times):
        self.Lose = None
        self.w = None
        self.alpha = alpha
        self.C = C
        self.times = times

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        self.w = np.zeros(len(x[0]))
        self.Lose = []
        count = 1
        pre_loss = 0
        while count <= self.times:
            count += 1
            x, y = shuffle(x, y)
            for i in range(len(x)):
                d = 1 - (y[i] * np.dot(x[i], self.w))
                new_w = np.zeros(len(self.w))
                if d <= 0:
                    new_w += self.w
                else:
                    new_w += self.w - self.C * x[i] * y[i]
                self.w -= self.alpha * new_w
            d = 1 - (y * np.dot(x, self.w))
            for j in range(len(d)):
                if d[j] < 0:
                    d[j] = 0
            loss = 1/2 * np.dot(self.w, self.w) + self.C * np.sum(d) / len(x)
            print("loss is:", loss)
            if pre_loss != 0:
                if math.isclose(pre_loss, loss, abs_tol=(0.0001 * pre_loss)):
                    print("Recursive convergence:", count)
                    break

            pre_loss = loss

    def predict(self, x):
        print("w: ", self.w)
        result = np.zeros(x.shape[0])
        for i in range(len(result)):
            result[i] = np.sign(np.dot(x[i], self.w))

        return result



